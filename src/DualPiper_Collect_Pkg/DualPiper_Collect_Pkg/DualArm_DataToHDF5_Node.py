import os
import argparse
import time
import copy
import cv2
import h5py
import numpy as np
import dm_env
import collections
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pynput import keyboard
import queue
import threading
from typing import Dict
from types import SimpleNamespace

# 全局队列，用于存储键盘事件
key_event_queue = queue.Queue()
def on_press_terminal(key):
    """
    键盘事件处理函数，将有效按键存入队列
    支持的按键：
    'c': 启动数据收集
    'x': 终止当前轮次数据收集
    's': 保存当前收集的数据
    'r': 清除数据收集完成标志位
    """
    try:
        if key.char in ['c', 'x', 's', 'r']:    # 只处理有效按键
            key_event_queue.put(key.char)
    except AttributeError:
        pass
    
def save_data(args, timesteps, actions, dataset_path):
    """
    将收集到的机器人数据(观测和动作)保存到HDF5文件。
    Args:
        args: 命令行参数，包含相机名称、是否使用深度图像等。
        timesteps(list):存储 dm_env.TimeStep对象的列表,包含机器人观测数据。
        actions(list): 存储机器人实际发出的动作的列表。
        dataset_path(str): 数据集保存路径（不包含.hdf5后缀)
        
    """
    # 数据字典
    data_size = len(actions)

    # 初始化数据字典，用于存储不同类型的数据
    data_dict = {
        # 机器人关节位置、速度和力矩（来自观测）
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        # 实际发送的机器人动作
        '/action': [],
    }
    
    
    # 为每个相机添加图像和深度图像（如果使用）的存储列表
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        # 使用深度图，排除三种相机
        if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:   
            data_dict[f'/observations/images_depth/{cam_name}'] = []
            
    # 遍历timesteps和actions，将数据填充到data_dict中
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0) # 取出动作
        ts = timesteps.pop(0)   # 取出对应的TimeStep（前一帧的观测）

        # 填充观测数据  # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 填充动作数据
        data_dict['/action'].append(action)

        # 填充图像数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
    
    t0 = time.time()    # 记录保存开始时间
    
    # 使用 h5py 库创建 HDF5 文件并写入数据
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # 设置文件属性
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        root.attrs['sim'] = False           # 标记是否为仿真数据
        root.attrs['compress'] = False      # 标记图像是否压缩
        root.attrs['data_size'] = data_size # 记录数据总帧数
        
        # 创建 observations 族
        obs = root.create_group('observations')
        # 创建 images 组，并为每个相机创建数据集
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            # 图像数据集，uint8，彩色图像3通道，设置 chunking 提高读写效率
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
            
        # 如果使用深度图像，创建 images_depth 组，并为非 wrist 相机创建深度图像数据集
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                # 深度图像数据集，uint16，单通道深度图像，设置 chunking 提高读写效率
                if cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
                    _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                                   chunks=(1, 480, 640), )
                    
        # 创建 qpos, qvel, effort 和 action 数据集
        _ = obs.create_dataset('qpos', (data_size, 14))      # 关节位置，14个关节
        _ = obs.create_dataset('qvel', (data_size, 14))      # 关节速度，14个关节
        _ = obs.create_dataset('effort', (data_size, 14))    # 关节力矩，14个关节
        _ = root.create_dataset('action', (data_size, 14))   # 动作，14维
        
        # 将 data_dict 中的数据写入 HDF5 文件
        for name, array in data_dict.items():
            root[name][...] = array
    
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)
    

class RosOperator(Node):
    """
    ROS节点,用于收集机器人数据(图像 关节状态)并保存到HDF5文件。
    通过订阅ROS话题获取数据,并提供数据同步和存储功能。
    """
    def __init__(self):
        """
            初始化 RosOperator 节点。

            Args:
            args: 命令行参数，包含ROS话题名词、相机名称等。
        """
        super().__init__('collect_data_node')  # 初始化 ROS 节点，命名为'collect_data_node'
        
        # --- Declare parameters and provide defaults ---
        # 基本开关与路径
        self.declare_parameter('use_front_camera', False)
        self.declare_parameter('dataset_dir', '/home/zzq/Desktop/piper_arm_ws/data')
        self.declare_parameter('task_name', 'pick_GreenOnBlue')
        self.declare_parameter('episode_idx', 23)
        self.declare_parameter('max_timesteps', 1000)
        # camera_names 支持 list 或逗号分隔字符串
        self.declare_parameter('camera_names', ['cam_top','cam_left'])
        # 话题
        self.declare_parameter('img_front_topic', '/front_camera/front_camera/color/image_raw')
        self.declare_parameter('img_top_topic', '/top_camera/top_camera/color/image_raw')
        self.declare_parameter('img_leftwrist_topic', '/leftwrist_camera/wrist_camera/color/image_raw')
        self.declare_parameter('img_front_depth_topic', '/front_camera/front_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('img_top_depth_topic', '/top_camera/top_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('img_leftwrist_depth_topic', '/leftwrist_camera/wrist_camera/aligned_depth_to_color/image_raw')
        # 机械臂关节状态话题
        self.declare_parameter('left_master_arm_topic', 'left/master_joint_states')
        self.declare_parameter('left_slave_arm_topic', 'left/slave_joint_states')
        self.declare_parameter('right_master_arm_topic', 'right/master_joint_states')
        self.declare_parameter('right_slave_arm_topic', 'right/slave_joint_states')
        # 其它配置
        self.declare_parameter('use_depth_image', False)
        self.declare_parameter('frame_rate', 30)

        # --- Read parameters into a simple namespace for backward-compatibility with existing code ---
        p = lambda name: self.get_parameter(name).value  # helper
        # camera_names: 如果是字符串则按逗号分割
        raw_cam = self.get_parameter('camera_names').value
        if isinstance(raw_cam, str):
            camera_names = [c.strip() for c in raw_cam.split(',') if c.strip()]
        else:
            camera_names = list(raw_cam)

        args = SimpleNamespace(
            use_front_camera=p('use_front_camera'),
            dataset_dir=p('dataset_dir'),
            task_name=p('task_name'),
            episode_idx=p('episode_idx'),
            max_timesteps=p('max_timesteps'),
            camera_names=camera_names,

            img_front_topic=p('img_front_topic'),
            img_top_topic=p('img_top_topic'),
            img_leftwrist_topic=p('img_leftwrist_topic'),
            img_front_depth_topic=p('img_front_depth_topic'),
            img_top_depth_topic=p('img_top_depth_topic'),
            img_leftwrist_depth_topic=p('img_leftwrist_depth_topic'),

            left_master_arm_topic=p('left_master_arm_topic'),
            left_slave_arm_topic=p('left_slave_arm_topic'),
            right_master_arm_topic=p('right_master_arm_topic'),
            right_slave_arm_topic=p('right_slave_arm_topic'),
            use_depth_image=p('use_depth_image'),
            frame_rate=p('frame_rate'),
        )
        self.args = args
        self.bridge = CvBridge()    # 图像转换桥，用于将 ROS Image 消息转换为 OpenCV 图像格式
        self.camera_names = self.args.camera_names
        
        # 图像和关节状态队列，用于存储从ROS话题接收到的数据，maxlen限制队列大小，防止内存溢出
        self.img_queues = {
            # 'cam_front': deque(maxlen=50),
            'cam_left': deque(maxlen=50),
            'cam_top': deque(maxlen=50),
            # 'cam_front_depth': deque(maxlen=50),
            # 'cam_wrist_depth': deque(maxlen=50),
            # 'cam_top_depth': deque(maxlen=50),
            'left/master_arm': deque(maxlen=50), # 左边机械臂关节状态
            'left/slave_arm': deque(maxlen=50), # 左边机械臂关节状态
            'right/master_arm': deque(maxlen=50),  # 右边机械臂关节状态
            'right/slave_arm': deque(maxlen=50),  # 右边机械臂关节状态
        }
        self.img_top_sub = self.create_subscription(Image, args.img_top_topic,
                                                    self.img_top_callback, 50)
        self.img_front_sub = self.create_subscription(Image, args.img_front_topic,
                                                    self.img_front_callback, 50)   
        self.img_leftwrist_sub = self.create_subscription(Image, args.img_leftwrist_topic,
                                                    self.img_leftwrist_callback, 50)
        self.left_master_arm_sub = self.create_subscription(JointState, args.left_master_arm_topic,
                                                       self.left_master_arm_callback, 50)
        self.left_slave_arm_sub = self.create_subscription(JointState, args.left_slave_arm_topic,
                                                       self.left_slave_arm_callback, 50)
        self.right_master_arm_sub = self.create_subscription(JointState, args.right_master_arm_topic,
                                                      self.right_master_arm_callback, 50)
        self.right_slave_arm_sub = self.create_subscription(JointState, args.right_slave_arm_topic,
                                                      self.right_slave_arm_callback, 50)
        
        # 如果启用深度图像，则创建深度图像的订阅者
        if self.args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image, args.img_front_depth_topic,
                                                                self.img_front_depth_callback, 50)

            self.img_top_depth_sub = self.create_subscription(Image, args.img_top_depth_topic,
                                                              self.img_top_depth_callback, 50)
        
        self.key = None  # 存储当前按下的键盘键
        # 启动两个线程，一个用于处理键盘输入，一个用于处理主逻辑（数据收集、保存）
        self.term_thread = threading.Thread(target=self.terminal_thread)
        self.key_process_thread = threading.Thread(target=self.key_process_loop)
        self.term_thread.start()
        self.key_process_thread.start()
        
        self.timesteps, self.actions = [], []   # 存储收集到的 TimeStep 对象和动作
        self.collecting_data = False    # 数据收集标志位
        
        print(f'\033[32m The Node of Robot data collection. \033[0m')
        print(f'\033[32m Press "c" to start collecting data. \033[0m')
        print(f'\033[32m Press "x" to stop collecting data. \033[0m')
        print(f'\033[32m Press "s" to save collecting data. \033[0m')

    def key_process_loop(self):
        """
            独立的线程函数，用于监听键盘事件并将有效按键放入队列。
        """
        listener = keyboard.Listener(on_press=on_press_terminal)    # 创建键盘监听器
        listener.start()    # 启动监听器
        while rclpy.ok():   # 持续监听，直到 ROS 关闭
            try:
                self.key = key_event_queue.get(timeout=0.1) # 从队列中获取键盘事件，设置超时防止阻塞
            except queue.Empty:
                pass    # 队列为空时跳过
            
    def terminal_thread(self):
        """
            独立的线程函数，处理数据收集、停止和保存的逻辑，响应键盘事件。
        """
        episode_idx = self.args.episode_idx # 当前数据收集的 episode 索引
        while rclpy.ok():
            if self.key == 'c' and not self.collecting_data:
                # 按下 'c' 且当前不在收集数据，则开始收集数据
                self.get_logger().info(f'Start collecting data for episode {episode_idx}')
                self.process()  # 调用 process 方法开始数据流处理
                
            elif self.key == 's' and self.collecting_data:
                # 按下 's' 且当前正在收集数据，则保存数据
                self.get_logger().info(f'To Save collecting data to file for episode {episode_idx}')
                dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)    # 如果数据集目录不存在则创建
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
                save_data(self.args, self.timesteps, self.actions, dataset_path)
                self.collecting_data = False    # 清除收集数据标志位
                episode_idx += 1    # 增加 episode 索引
            elif self.key == 'r':
                # 按下 'r'，清除收集完成标志位，重置数据列表
                self.get_logger().info(f'Clear the flag of collectted data')
                self.timesteps = []
                self.actions = []
                self.collecting_data = False    # 重置数据收集标志位
                self.key = None # 清除当前键，避免重复触发
    def get_frame(self, use_arm=True):
        """
        从订阅的ROS话题队列中获取同步的图像和关节状态数据。
        该函数会找到所有所需数据队列中最近的时间戳，并丢弃所有早于该时间戳的数据，
        以确保返回的数据是同步的。

        Args:
            use_arm (bool): 是否需要同步机械臂关节状态数据。

        Returns:
            dict or False: 包含同步后的图像、深度图像（如果启用）、主从机械臂关节状态的字典，
                           如果任何所需队列为空则返回False。
        """
        def _get_latest_time(keys):
            """
            获取 keys 对应的所有队列中最新的数据时间戳，并返回最小的时间（确保所有数据点的同步）。
            """
            # 遍历所有指定的队列，获取每个队列中最新数据的时间戳，并返回这些时间戳中的最小值
            # 这样可以确保后续同步时，所有数据都至少在该最小时间戳之后
            return min(
                self.img_queues[k][-1].header.stamp.sec + 1e-9 * self.img_queues[k][-1].header.stamp.nanosec
                for k in keys if self.img_queues[k] # 确保队列不为空
            ) if keys else None # 如果 keys 为空，则返回 None
        
        def _sync_queues(sync_time, keys):
            """
            清除 keys 对应的队列中所有早于 sync_time 的数据，确保所有数据在 sync_time 之后。
            """
            for k in keys:
                # 对于每个队列，循环弹出队列头部的数据，直到队列为空或队列头部数据的时间戳大于等于sync_time
                while self.img_queues[k] and (
                        self.img_queues[k][0].header.stamp.sec + 1e-9 * self.img_queues[k][
                    0].header.stamp.nanosec < sync_time
                ):
                    self.img_queues[k].popleft()

        # 根据配置构建需要同步的队列键列表
        keys = []
        if 'cam_front' in self.args.camera_names:
            keys.append('cam_front')
            if self.args.use_depth_image:
                keys.append('cam_front_depth')
        if 'cam_left' in self.args.camera_names:
            keys.append('cam_left')
        if 'cam_top' in self.args.camera_names:
            keys.append('cam_top')
            if self.args.use_depth_image:
                keys.append('cam_top_depth')

        if use_arm:
            keys.extend(['left/master_arm', 'left/slave_arm', 'right/master_arm', 'right/slave_arm'])

        # 如果任何一个所需队列为空，则无法进行同步，返回 False
        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False
        
        # 获取所有队列中最近的时间戳作为同步时间
        sync_time = _get_latest_time(keys)
        if sync_time is None:
            return False
        
        # 同步所有队列，丢弃早于 sync_time 的数据
        _sync_queues(sync_time, keys)

        # 检查同步后是否有队列变为空，如果变空说明数据不足，再次返回False
        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False

        # 从队列中取出最新的图像数据
        images = {}
        for key in self.args.camera_names:
            if key in self.img_queues and self.img_queues[key]:
                # 将 ROS Image 消息转换为 OpenCV 图像格式
                images[key] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(), desired_encoding='passthrough')
        
        # 从队列中取出最新的深度图像数据（如果启用）
        images_depth = None
        if self.args.use_depth_image:
            images_depth = {}
            for key in keys:
                if 'depth' in key and self.img_queues[key]:
                    name = key.rsplit('_', 1)[0]    # 从深度话题名称中提取相机名称
                    images_depth[name] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(),
                                                                   desired_encoding='16UC1')

        # 从队列中取出最新的机械臂关节状态数据
        left_master_arm, left_slave_arm, right_master_arm, right_slave_arm = None, None, None, None
        if use_arm:
            if self.img_queues['left/master_arm']:
                left_master_arm = self.img_queues['left/master_arm'].popleft()
            if self.img_queues['left/slave_arm']:
                left_slave_arm = self.img_queues['left/slave_arm'].popleft()
            if self.img_queues['right/master_arm']:
                right_master_arm = self.img_queues['right/master_arm'].popleft()
            if self.img_queues['right/slave_arm']:
                right_slave_arm = self.img_queues['right/slave_arm'].popleft()

        # 返回包含所有同步数据的字典
        return {
            "images": images,
            #"depths": images_depth,
            "left/master_arm": left_master_arm,
            "left/slave_arm": left_slave_arm,
            "right/master_arm": right_master_arm,
            "right/slave_arm": right_slave_arm
        }
        
    # ROS 回调函数，将接收到的 Image 消息添加到对应的队列
    def img_front_callback(self, msg):
        self.img_queues['cam_front'].append(msg)
        # self.img_front_deque.append(msg)

    def img_leftwrist_callback(self, msg):
        self.img_queues['cam_left'].append(msg)

    def img_top_callback(self, msg):
        self.img_queues['cam_top'].append(msg)

    # ROS 回调函数，将接收到的深度图像消息添加到对应的队列
    def img_front_depth_callback(self, msg):
        self.img_queues['cam_front_depth'].append(msg)

    def img_top_depth_callback(self, msg):
        self.img_queues['cam_top_depth'].append(msg)

    # ROS 回调函数，将接收到的 JointState 消息添加到对应的队列
    def left_master_arm_callback(self, msg):
        self.img_queues['left/master_arm'].append(msg)
    def left_slave_arm_callback(self, msg):
        self.img_queues['left/slave_arm'].append(msg)
    def right_master_arm_callback(self, msg):
        self.img_queues['right/master_arm'].append(msg)
    def right_slave_arm_callback(self, msg):
        self.img_queues['right/slave_arm'].append(msg)
        
    def process(self):
        """
        数据收集的主循环。
        在该循环中，节点以设定的帧率获取同步数据，构建dm_env.TimeStep对象，
        并将观测和动作添加到相应的列表中。
        """
        self.timesteps = []     # 清空 timesteps 列表
        self.actions = []       # 清空 actions 列表

        # 初始化图像字典和深度图像字典（如果使用），用于存储当前帧的图像数据
        # 初始时填充随机数据，确保字典结构完整
        image_dict = {name: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                      for name in self.args.camera_names}
        image_depth_dict = {
            name: np.random.random((480, 640, 1))
            for name in self.args.camera_names
            if name not in ['cam_wrist','cam_left', 'cam_right']
        } if self.args.use_depth_image else {}
        # image_depth_dict = {name: np.random.random((480, 640, 1)) for name in
        #                     self.args.camera_names} if self.args.use_depth_image else {}

        count, rate, print_flag = 0, self.create_rate(self.args.frame_rate), True
        
        # 数据收集循环，直到 ROS 关闭或达到最大时间步数
        while rclpy.ok() and (count <= self.args.max_timesteps):
            result = self.get_frame()    # 获取同步后的数据
            if not result:
                # 如果数据未准备好（队列为空或无法同步），则打印错误消息并等待
                if print_flag:
                    self.get_logger().error('state not ready yet')
                rate.sleep()    # 按照设定的帧率等待
                continue
            count += 1  # 帧计数增加
            left_master_arm, left_slave_arm = result['left/master_arm'],result['left/slave_arm']    # 获取左边机械臂关节状态
            right_master_arm, right_slave_arm = result['right/master_arm'],result['right/slave_arm']    # 获取左边机械臂关节状态
            # 更新图像字典和深度图像字典为当前帧的实际数据
            for i, name in enumerate(self.camera_names):
                image_dict[name] = result['images'][name].copy()
                if self.args.use_depth_image and name != ('cam_left' or 'cam_right'):   # 腕部相机深度图像可能不规则或不需要
                    image_depth_dict[name] = result['depths'][name].copy()

            # 构建 dm_env.TimeStep 的观测字典
            obs = collections.OrderedDict()
            obs['images']=copy.deepcopy(image_dict) # 图像数据
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(image_depth_dict)   # 深度图像数据
            left_position = np.array(left_slave_arm.position)[:7]
            left_velocity = np.array(left_slave_arm.velocity)[:7]
            left_effort = np.array(left_slave_arm.effort)[:7]
            right_position = np.array(right_slave_arm.position)[:7]
            right_velocity = np.array(right_slave_arm.velocity)[:7]
            right_effort = np.array(right_slave_arm.effort)[:7]
            positon = np.concatenate((left_position, right_position), axis=0)   # 合并左右机械臂关节位置
            velocity = np.concatenate((left_velocity, right_velocity), axis=0)  # 合并左右机械臂关节速度
            effort = np.concatenate((left_effort, right_effort), axis=0)        # 合并左右机械臂关节力矩

            obs['qpos'] = positon  # 从机械臂关节位置
            obs['qvel'] = velocity  # 从机械臂关节速度
            obs['effort'] = effort  # 从机械臂关节力矩

            if count == 1:
                # 第一帧数据作为 dm_env.TimeStep 的 FIRST 类型
                ts = dm_env.TimeStep(   # 统一强化学习环境的交互接口
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                self.timesteps.append(ts)
                continue    # 继续下一帧数据收集
            
            # 后续数据作为 dm_env.StepType.MID 类型
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)
            left_position = np.array(left_master_arm.position)[:7]  # 获取左边主机械臂关节位置
            right_position = np.array(right_master_arm.position)[:7]  # 获取右边主机械臂关节位置
            combined_position = np.concatenate((left_position, right_position), axis=0)  # 合并左右机械臂关节位置
            self.actions.append(np.array(combined_position))  # 记录主机械臂关节位置作为动作
            self.timesteps.append(ts)   # 记录 TimeStep

            self.get_logger().info(f'Frame data:{count}')   # 打印当前帧数
            rate.sleep()    # 按照设定的帧率等待
            if count > self.args.max_timesteps:
                self.get_logger().info('Stop collecting data,frame_index:{}'.format(count - 1))
                self.collecting_data =True
                break
            if self.key == 'x':
                # 如果按下 'x'键，则停止当前轮次的数据收集
                self.get_logger().info('Stop collecting data,frame_index:{}'.format(count - 1))
                self.collecting_data =True
                break

        # 打印收集到的 timesteps 和 actions 的数量
        print("len(timesteps): ", len(self.timesteps))
        print("len(actions)  : ", len(self.actions))
        
# def get_arguments():
#     """
#     解析命令行参数。
#     """
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--use_front_camera', action='store_true', default=False, )
#     parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
#                         default="/home/zzq/Desktop/piper_arm_ws/data", required=False)
#     parser.add_argument('--task_name', action='store', type=str, help='Task name.',
#                         default="pick_GreenOnBlue", required=False)
#     parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
#                         default=0, required=False)
#     parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
#                         default=500, required=False)
#     parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
#                         # default=[ 'cam_front','cam_wrist', 'cam_top'], required=False)
#                         default=['cam_top'], required=False)   # 默认只使用腕部和顶部相机
#
#     # 相机话题名词，默认值可以根据实际情况修改
#     parser.add_argument('--img_leftwrist_topic', action='store', type=str, help='img_front_topic',
#                         default='/leftwrist_camera/wrist_camera/color/image_raw', required=False)
#     parser.add_argument('--img_top_topic', action='store', type=str, help='img_top_topic',
#                         default='/top_camera/top_camera/color/image_raw', required=False)
#     # parser.add_argument('--img_wrist_topic', action='store', type=str, help='img_wrist_topic',
#     #                     default='/wrist_camera/wrist_camera/color/image_raw', required=False)
#     # parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
#     #                     default='/front_camera/front_camera/aligned_depth_to_color/image_raw', required=False)
#     parser.add_argument('--img_top_depth_topic', action='store', type=str, help='img_top_depth_topic',
#                         default='/top_camera/top_camera/aligned_depth_to_color/image_raw', required=False)
#     parser.add_argument('--img_leftwrist_depth_topic', action='store', type=str, help='img_wrist_depth_topic',
#                         default='/leftwrist_camera/wrist_camera/aligned_depth_to_color/image_raw', required=False)
#
#     # 机械臂关节状态话题名词
#     parser.add_argument('--left_master_arm_topic', action='store', type=str,
#                         default='left/master_joint_states', required=False)
#     parser.add_argument('--left_slave_arm_topic', action='store', type=str,
#                         default='left/slave_joint_states', required=False)
#     parser.add_argument('--right_master_arm_topic', action='store', type=str,
#                         default='right/master_joint_states', required=False)
#     parser.add_argument('--right_slave_arm_topic', action='store', type=str,
#                         default='right/slave_joint_states', required=False)
#
#     parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
#                         default=False, required=False)  # 默认不使用深度图像
#     parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
#                         default=30, required=False)     # 默认帧率为50Hz
#
#     args = parser.parse_args()
#     return args


def main(args=None):
    """
    主函数，初始化ROS，创建RosORosOperatorperator节点，并启动ROS事件循环。
    """
    from rclpy.utilities import remove_ros_args
    import sys
    rclpy.init(args=args)    # 初始化 ROS 客户端库
    # args = get_arguments()  # 获取命令行参数
    ros_operator = RosOperator()    # 创建 RosOperator 节点实例

    # 不过，在rclpy.spin()之后，这个listener.stop()和listener.join()可能不会被执行，
    # 因为rclpy.spin()是阻塞的。更常见的方式是在node的destroy_node方法中处理线程的关闭。
    listener = keyboard.Listener(on_press=on_press_terminal)
    # listener.start() # 如果在ros_operator内部已经启动，这里可以注释掉，避免重复启动

    rclpy.spin(ros_operator)    # 启动 ROS 事件循环，阻塞指导节点被关闭或 ROS 被关闭
    
    ros_operator.destroy_node() # 销毁节点，释放资源
    listener.stop()     # 停止键盘监听器
    listener.join()     # 等待键盘监听器线程结束
    rclpy.shutdown()    # 关闭 ROS 客户端库

if __name__ == '__main__':
    main()

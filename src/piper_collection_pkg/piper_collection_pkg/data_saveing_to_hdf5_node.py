import os
import argparse
import time
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
key_event_queue = queue.Queue()
def on_press_terminal(key):
    """键盘事件处理函数，只是把按键存入队列"""
    try:
        '''
            c->x->r->c->s
            c代表启动收集数据节点，x代表终止当前轮次收集数据，s代表保存数据，r代表清除收集完成标志位
        '''
        if key.char in ['c','x','s','r']:  # 只处理有效按键
            key_event_queue.put(key.char)
    except AttributeError:
        pass
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)

    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)

        for cam_name in args.camera_names:

            # print()
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
    t0 = time.time()

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['data_size'] = data_size
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )

        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )


        _ = obs.create_dataset('qpos', (data_size, 8))
        _ = obs.create_dataset('qvel', (data_size, 8))
        _ = obs.create_dataset('effort', (data_size, 8))
        _ = root.create_dataset('action', (data_size, 8))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('collect_data_node')  # 传入节点名称
        self.bridge = CvBridge()
        self.slave_arm_deque = deque(maxlen = 2000)
        self.master_arm_deque = deque(maxlen = 2000)
        self.img_front_deque = deque(maxlen = 2000)
        self.img_front_depth_deque = deque(maxlen = 2000)
        self.img_wrist_deque = deque(maxlen = 2000)
        self.img_wrist_depth_deque = deque(maxlen = 2000)
        self.args=args

        self.img_front_sub = self.create_subscription(Image,args.img_front_topic,
                                                self.img_front_callback,10)
        self.img_wrist_sub =self.create_subscription(Image,args.img_wrist_topic,
                                                     self.img_wrist_callback,10)
        if args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image,args.img_front_depth_topic,
                                                          self.img_front_depth_callback,10)
            self.img_wrist_depth_sub = self.create_subscription(Image,args.img_wrist_depth_topic,
                                                          self.img_wrist_depth_callback,10)
        self.master_arm_sub = self.create_subscription(JointState,args.master_arm_topic,
                                                        self.master_arm_callback,10)
        self.slave_arm_sub = self.create_subscription(JointState,args.slave_arm_topic,
                                                      self.slave_arm_callback,10)
        self.key = None # 键盘输入
        self.stop_collecting = False
        self.terminal_thread = threading.Thread(target=self.terminal_thread)
        self.key_process_thread = threading.Thread(target=self.key_process_loop)
        self.terminal_thread.start()
        self.key_process_thread.start()

        self.timesteps, self.actions = [], []
        self.collecting_data = False

        print(f'\033[32m  The Node of Robot data collection. \033[0m')
        print(f'\033[32m　Press "c" to start collecting data. \033[0m')
        print(f'\033[32m　Press "x" to stop collecting data. \033[0m')
        print(f'\033[32m　Press "s" to save collecting data. \033[0m')
        # pass
    def key_process_loop(self):
        listener = keyboard.Listener(on_press=on_press_terminal)
        listener.start()
        while rclpy.ok():
            try:
                self.key = key_event_queue.get(timeout=0.1)
            except queue.Empty:
                pass
    def terminal_thread(self):
        episode_idx = self.args.episode_idx
        while rclpy.ok():
            if self.key == 'c' and not self.collecting_data:
                self.get_logger().info(f'Start collecting data for episode {episode_idx}')
                self.timesteps, self.actions = self.process()
                self.collecting_data = True

            elif self.key == 's' and self.collecting_data:
                self.get_logger().info(f'To Save collecting data to file for episode {episode_idx}')
                dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
                save_data(self.args, self.timesteps, self.actions, dataset_path)
                self.collecting_data = False
                episode_idx += 1
            elif self.key == 'r':
                self.get_logger().info(f'Clear the flag of collectted data')
                self.collecting_data = False
                self.key=None
    def get_frame(self):
        if len(self.img_front_deque) == 0 or (self.args.use_depth_image and len(self.img_front_depth_deque) == 0):
            return False
        if len(self.img_wrist_deque) == 0 or (self.args.use_depth_image and len(self.img_wrist_depth_deque) == 0):
            return False
        if len(self.master_arm_deque) == 0 or len(self.slave_arm_deque) == 0:
            return False

        latest_front_img_time = self.img_front_deque[-1].header.stamp.sec+1e-9*self.img_front_deque[-1].header.stamp.nanosec
        latest_wrist_img_time = self.img_wrist_deque[-1].header.stamp.sec + 1e-9*self.img_wrist_deque[-1].header.stamp.nanosec
        latest_master_time = self.master_arm_deque[-1].header.stamp.sec + 1e-9 * self.master_arm_deque[
            -1].header.stamp.nanosec
        latest_slave_time = self.slave_arm_deque[-1].header.stamp.sec + 1e-9 * self.slave_arm_deque[
            -1].header.stamp.nanosec

        sync_time = min(latest_front_img_time,
                        latest_wrist_img_time,
                        latest_master_time,
                        latest_slave_time)

        while self.img_front_deque and (
            self.img_front_deque[0].header.stamp.sec + 1e-9 * self.img_front_deque[0].header.stamp.nanosec < sync_time):
            self.img_front_deque.popleft()

        while self.img_wrist_deque and (
            self.img_wrist_deque[0].header.stamp.sec + 1e-9 * self.img_wrist_deque[0].header.stamp.nanosec < sync_time):
            self.img_wrist_deque.popleft()

        while self.master_arm_deque and (
            self.master_arm_deque[0].header.stamp.sec + 1e-9 * self.master_arm_deque[0].header.stamp.nanosec < sync_time):
            self.master_arm_deque.popleft()

        while self.slave_arm_deque and (
                self.slave_arm_deque[0].header.stamp.sec + 1e-9 * self.slave_arm_deque[0].header.stamp.nanosec < sync_time):
            self.slave_arm_deque.popleft()
        if not self.img_wrist_deque or not self.img_front_deque or not self.master_arm_deque or not self.slave_arm_deque:
            return False
        img_front_msg = self.img_front_deque.popleft()
        img_wrist_msg = self.img_wrist_deque.popleft()
        img_wrist = self.bridge.imgmsg_to_cv2(img_wrist_msg,
                                        desired_encoding ='passthrough')
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg,
                                        desired_encoding ='passthrough')
        img_front_depth = None
        img_wrist_depth = None
        if self.args.use_depth_image and self.img_wrist_depth_deque and self.img_front_depth_deque:
            img_wrist_depth_msg = self.img_wrist_depth_deque.popleft()
            img_front_depth_msg = self.img_front_depth_deque.popleft()
            img_wrist_depth = self.bridge.imgmsg_to_cv2(img_wrist_depth_msg, desired_encoding='16UC1')
            img_front_depth = self.bridge.imgmsg_to_cv2(img_front_depth_msg, desired_encoding='16UC1')

        master_arm = self.master_arm_deque.popleft()
        slave_arm = self.slave_arm_deque.popleft()

        return img_front,img_wrist,img_front_depth,img_wrist_depth, master_arm, slave_arm
    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)
    def img_front_depth_callback(self, msg):
        self.img_front_depth_deque.append(msg)
    def img_wrist_callback(self, msg):
        self.img_wrist_deque.append(msg)
    def img_wrist_depth_callback(self, msg):
        self.img_wrist_depth_deque.append(msg)
    def master_arm_callback(self, msg):
        self.master_arm_deque.append(msg)
    def slave_arm_callback(self, msg):
        self.slave_arm_deque.append(msg)
    def process(self):
        timesteps = []
        actions = []
        image_front =np.random.randint(0,255,size=(480 ,640 ,3) ,dtype = np.uint8)
        image_wrist =np.random.randint(0,255,size=(480 ,640 ,3) ,dtype = np.uint8)

        image_front_depth =np.random.random(size=(480 ,640 ,1))
        image_wrist_depth =np.random.random(size=(480 ,640 ,1))

        image_dict = {self.args.camera_names[0]: image_front,
                      self.args.camera_names[1]: image_wrist}
        image_depth_dict = {self.args.camera_names[0]: image_front_depth,
                            self.args.camera_names[1]: image_wrist_depth}
        count = 0
        rate =self.create_rate(self.args.frame_rate)
        print_flag = True

        while rclpy.ok() and (count < self.args.max_timesteps + 1):
            result = self.get_frame()
            if not result:
                if print_flag:
                    self.get_logger().error('state not ready yet')
                rate.sleep()
                continue
            count+=1
            img_front,img_wrist,img_front_depth,img_wrist_depth,master_arm,slave_arm = result

            image_dict[self.args.camera_names[0]] = img_front
            image_dict[self.args.camera_names[1]] = img_wrist
            image_depth_dict[self.args.camera_names[0]] = img_front_depth
            image_depth_dict[self.args.camera_names[1]] = img_wrist_depth
            obs = collections.OrderedDict()
            obs['images'] = {self.args.camera_names[0]: image_dict[self.args.camera_names[0]],
                             self.args.camera_names[1]: image_dict[self.args.camera_names[1]]}
            if self.args.use_depth_image:
                obs['images_depth'] = {self.args.camera_names[0]: image_depth_dict[self.args.camera_names[0]],
                                       self.args.camera_names[1]: image_depth_dict[self.args.camera_names[1]]}

            obs['qpos'] = np.array(slave_arm.position)
            obs['qvel'] = np.array(slave_arm.velocity)
            obs['effort'] = np.array(slave_arm.effort)

            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            action =np.array(master_arm.position)
            actions.append(action)
            timesteps.append(ts)
            self.get_logger().info(f'Frame data:{count}')
            rate.sleep()
            if self.key == 'x':
                # self.stop_collecting = False
                self.get_logger().info('Stop collecting data,frame_index:{}'.format(count-1))
                break

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        # self.collecting_data = False
        return timesteps,actions
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="/home/zzq/Desktop/piper_arm_ws/data/", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="pick_hxy_to_box", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=400, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_front','cam_wrist'], required=False)
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/front_camera/color/image_raw', required=False)
    parser.add_argument('--img_wrist_topic',action='store',type=str,help='img_wrist_topic',
                        default='/wrist_camera/wrist_camera/color/image_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/front_camera/front_camera/depth/image_rect_raw', required=False)
    parser.add_argument('--img_wrist_depth_topic',action='store',type=str,help='img_wrist_depth_topic',
                        default='/wrist_camera/wrist_camera/depth/image_rect_raw',required=False)
    parser.add_argument('--master_arm_topic', action = 'store',type=str,default='/master_joint_states', required=False)
    parser.add_argument('--slave_arm_topic', action = 'store',type=str,default='/joint_states', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    args = parser.parse_args()
    return args
def main():
    rclpy.init()
    args = get_arguments()
    ros_operator = RosOperator(args)
    listener = keyboard.Listener(on_press=on_press_terminal)
    rclpy.spin(ros_operator)
    ros_operator.destroy_node()
    listener.stop()
    listener.join()
    rclpy.shutdown()
if __name__ == '__main__':
    main()

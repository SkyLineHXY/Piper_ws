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
key_event_queue = queue.Queue()
def on_press_terminal(key):
    """键盘事件处理函数，只是把按键存入队列"""
    try:
        '''
            c->x->r->c->s
            c代表启动收集数据节点，x代表终止当前轮次收集数据，s代表保存数据，r代表清除收集完成标志位
        '''
        if key.char in ['c', 'x', 's', 'r']:  # 只处理有效按键
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
        if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
            data_dict[f'/observations/images_depth/{cam_name}'] = []
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)  # 动作  当前动作
        ts = timesteps.pop(0)  # 前一帧
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
            if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
        cv2.imshow('save-data', ts.observation['images']['cam_top'])
        cv2.waitKey(1)
    t0 = time.time()

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
                if cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
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
        self.args = args
        self.bridge = CvBridge()
        self.camera_names = self.args.camera_names

        self.img_queues = {
            'cam_front': deque(maxlen=50),
            'cam_wrist': deque(maxlen=50),
            'cam_top': deque(maxlen=50),
            'cam_front_depth': deque(maxlen=50),
            'cam_wrist_depth': deque(maxlen=50),
            'cam_top_depth': deque(maxlen=50),
            'master_arm': deque(maxlen=50),
            'slave_arm': deque(maxlen=50),
        }
        self.img_front_sub = self.create_subscription(Image, args.img_front_topic,
                                                      self.img_front_callback, 50)
        self.img_wrist_sub = self.create_subscription(Image, args.img_wrist_topic,
                                                      self.img_wrist_callback, 50)
        self.img_top_sub = self.create_subscription(Image, args.img_top_topic,
                                                    self.img_top_callback, 50)
        self.master_arm_sub = self.create_subscription(JointState, args.master_arm_topic,
                                                       self.master_arm_callback, 50)
        self.slave_arm_sub = self.create_subscription(JointState, args.slave_arm_topic,
                                                      self.slave_arm_callback, 50)
        if self.args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image, args.img_front_depth_topic,
                                                                self.img_front_depth_callback, 50)
            # self.img_wrist_depth_sub = self.create_subscription(Image, args.img_wrist_depth_topic,
            #                                                     self.img_wrist_depth_callback, 50)
            self.img_top_depth_sub = self.create_subscription(Image, args.img_top_depth_topic,
                                                              self.img_top_depth_callback, 50)

        self.key = None  # 键盘输入
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
    def key_process_loop(self):
        listener = keyboard.Listener(on_press=on_press_terminal)
        listener.start()
        while rclpy.ok():
            try:
                self.key = key_event_queue.get(timeout=0.1)
            except queue.Empty:
                pass
    def terminal_thread(self):
        # timesteps =[]
        # actions =[]
        episode_idx = self.args.episode_idx
        while rclpy.ok():
            if self.key == 'c' and not self.collecting_data:
                self.get_logger().info(f'Start collecting data for episode {episode_idx}')
                self.process()
                self.collecting_data = True
            elif self.key == 's' and self.collecting_data:
                self.get_logger().info(f'To Save collecting data to file for episode {episode_idx}')
                dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
                save_data(self.args, self.timesteps,self.actions, dataset_path)
                self.collecting_data = False
                episode_idx += 1
            elif self.key == 'r':
                self.get_logger().info(f'Clear the flag of collectted data')
                self.timesteps = []
                self.actions = []
                self.collecting_data = False
                self.key = None
    def get_frame_v2(self, use_arm=True):
        def _get_latest_time(keys):
            """
            获取 keys 对应的所有队列中最新的数据时间戳，并返回最小的时间（确保所有数据点的同步）。
            """
            return min(
                self.img_queues[k][-1].header.stamp.sec + 1e-9 * self.img_queues[k][-1].header.stamp.nanosec
                for k in keys if self.img_queues[k]
            ) if keys else None
        def _sync_queues(sync_time, keys):
            """
            清除 keys 对应的队列中所有早于 sync_time 的数据，确保所有数据在 sync_time 之后。
            """
            for k in keys:
                while self.img_queues[k] and (
                        self.img_queues[k][0].header.stamp.sec + 1e-9 * self.img_queues[k][
                    0].header.stamp.nanosec < sync_time
                ):
                    self.img_queues[k].popleft()
        keys = []
        if 'cam_front' in self.args.camera_names:
            keys.append('cam_front')
            if self.args.use_depth_image:
                keys.append('cam_front_depth')
        if 'cam_wrist' in self.args.camera_names:
            keys.append('cam_wrist')
        if 'cam_top' in self.args.camera_names:
            keys.append('cam_top')
            if self.args.use_depth_image:
                keys.append('cam_top_depth')
        if use_arm:
            keys.extend(['master_arm', 'slave_arm'])
        # 如果任何队列为空，则返回 False
        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False
        sync_time = _get_latest_time(keys)
        if sync_time is None:
            return False
        _sync_queues(sync_time, keys)
        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False
        images = {}
        for key in self.args.camera_names:
            if key in self.img_queues and self.img_queues[key]:
                images[key] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(), desired_encoding='passthrough')
        images_depth = None
        if self.args.use_depth_image:
            images_depth = {}
            for key in keys:
                if 'depth' in key and self.img_queues[key]:
                    name = key.rsplit('_', 1)[0]
                    images_depth[name] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(),
                                                                   desired_encoding='16UC1')

        master_arm, slave_arm = None, None
        if use_arm:
            if self.img_queues['master_arm']:
                master_arm = self.img_queues['master_arm'].popleft()
            if self.img_queues['slave_arm']:
                slave_arm = self.img_queues['slave_arm'].popleft()

        return {
            "images": images,
            "depths": images_depth,
            "master_arm": master_arm,
            "slave_arm": slave_arm
        }

    def img_front_callback(self, msg):
        self.img_queues['cam_front'].append(msg)
        # self.img_front_deque.append(msg)

    def img_wrist_callback(self, msg):
        self.img_queues['cam_wrist'].append(msg)
    def img_top_callback(self, msg):
        self.img_queues['cam_top'].append(msg)
    def img_front_depth_callback(self, msg):
        self.img_queues['cam_front_depth'].append(msg)
    def img_top_depth_callback(self, msg):
        self.img_queues['cam_top_depth'].append(msg)
    def master_arm_callback(self, msg):
        self.img_queues['master_arm'].append(msg)
    def slave_arm_callback(self, msg):
        self.img_queues['slave_arm'].append(msg)
    def process(self):
        self.timesteps = []
        self.actions = []
        image_dict = {name: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for name in self.args.camera_names}
        image_depth_dict = {
            name: np.random.random((480, 640, 1))
            for name in self.args.camera_names
            if name not in ['cam_wrist','cam_wrist_left', 'cam_wrist_right']
        } if self.args.use_depth_image else {}
        # image_depth_dict = {name: np.random.random((480, 640, 1)) for name in
        #                     self.args.camera_names} if self.args.use_depth_image else {}
        count, rate, print_flag = 0, self.create_rate(self.args.frame_rate), True
        while rclpy.ok() and (count < self.args.max_timesteps + 1):
            result = self.get_frame_v2()
            if not result:
                if print_flag:
                    self.get_logger().error('state not ready yet')
                rate.sleep()
                continue
            count += 1
            images, depths, master_arm, slave_arm = result['images'],result['depths'],result['master_arm'],result['slave_arm']
            for i, name in enumerate(self.camera_names):
                image_dict[name] = images[name].copy()
                if self.args.use_depth_image and name != 'cam_wrist':
                    image_depth_dict[name] = depths[name].copy()

            obs = collections.OrderedDict()
            obs['images']=copy.deepcopy(image_dict)
            obs['qpos'] = np.array(slave_arm.position)
            obs['qvel'] = np.array(slave_arm.velocity)
            obs['effort'] = np.array(slave_arm.effort)
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(image_depth_dict)
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                self.timesteps.append(ts)
                continue
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)
            self.actions.append(np.array(master_arm.position))
            self.timesteps.append(ts)
            self.get_logger().info(f'Frame data:{count}')
            rate.sleep()
            if self.key == 'x':
                self.get_logger().info('Stop collecting data,frame_index:{}'.format(count - 1))
                break
        print("len(timesteps): ", len(self.timesteps))
        print("len(actions)  : ", len(self.actions))



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_front_camera', action='store_true', default=False, )
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="/home/zzq/Desktop/piper_arm_ws/data/", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="pick_blue_object_to_box", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=[ 'cam_front','cam_wrist', 'cam_top'], required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/front_camera/color/image_raw', required=False)
    parser.add_argument('--img_top_topic', action='store', type=str, help='img_top_topic',
                        default='/top_camera/top_camera/color/image_raw', required=False)
    parser.add_argument('--img_wrist_topic', action='store', type=str, help='img_wrist_topic',
                        default='/wrist_camera/wrist_camera/color/image_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/front_camera/front_camera/aligned_depth_to_color/image_raw', required=False)
    parser.add_argument('--img_top_depth_topic', action='store', type=str, help='img_top_depth_topic',
                        default='/top_camera/top_camera/aligned_depth_to_color/image_raw', required=False)
    parser.add_argument('--img_wrist_depth_topic', action='store', type=str, help='img_wrist_depth_topic',
                        default='/wrist_camera/wrist_camera/depth/image_rect_raw', required=False)
    parser.add_argument('--master_arm_topic', action='store', type=str, default='/master_joint_states', required=False)
    parser.add_argument('--slave_arm_topic', action='store', type=str, default='/joint_states', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=True, required=False)
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

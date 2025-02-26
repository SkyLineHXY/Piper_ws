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
import sys
import cv2

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
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
    t0 = time.time()

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        root.attrs['sim'] = False
        root.attrs['compress'] = False

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
        self.img_deque = deque(maxlen = 2000)
        self.img_depth_deque = deque(maxlen = 2000)
        self.args=args

        self.img_sub = self.create_subscription(Image,args.img_topic,
                                                self.img_callback,10)
        if args.use_depth_image:
            self.img_depth_sub = self.create_subscription(Image,args.img_depth_topic,
                                                          self.img_depth_callback,10)

        self.master_arm_sub = self.create_subscription(JointState,args.master_arm_topic,
                                                        self.master_arm_callback,10)
        self.slave_arm_sub = self.create_subscription(JointState,args.slave_arm_topic,
                                                      self.slave_arm_callback,10)

    def get_frame(self):
        if len(self.img_deque) == 0 or (self.args.use_depth_image and len(self.img_depth_deque) == 0):
            return False
        if len(self.master_arm_deque)==0 or len(self.slave_arm_deque)==0:
            return False

        latest_img_time = self.img_deque[-1].header.stamp.sec+1e-9*self.img_deque[-1].header.stamp.nanosec
        latest_master_time = self.master_arm_deque[-1].header.stamp.sec + 1e-9 * self.master_arm_deque[
            -1].header.stamp.nanosec
        latest_slave_time = self.slave_arm_deque[-1].header.stamp.sec + 1e-9 * self.slave_arm_deque[
            -1].header.stamp.nanosec
        sync_time = min(latest_img_time, latest_master_time, latest_slave_time)

        while self.img_deque and (
            self.img_deque[0].header.stamp.sec + 1e-9 * self.img_deque[0].header.stamp.nanosec < sync_time):
            self.img_deque.popleft()

        while self.master_arm_deque and (
            self.master_arm_deque[0].header.stamp.sec + 1e-9 * self.master_arm_deque[0].header.stamp.nanosec < sync_time):
            self.master_arm_deque.popleft()

        while self.slave_arm_deque and (
                self.slave_arm_deque[0].header.stamp.sec + 1e-9 * self.slave_arm_deque[0].header.stamp.nanosec < sync_time):
            self.slave_arm_deque.popleft()
        if not self.img_deque or not self.master_arm_deque or not self.slave_arm_deque:
            return False

        img_msg = self.img_deque.popleft()
        img = self.bridge.imgmsg_to_cv2(img_msg,desired_encoding ='passthrough')
        img_depth = None

        if self.args.use_depth_image and self.img_depth_deque:
            img_depth_msg = self.img_depth_deque.popleft()
            img_depth = self.bridge.imgmsg_to_cv2(img_depth_msg, desired_encoding='passthrough')

        master_arm = self.master_arm_deque.popleft()
        slave_arm = self.slave_arm_deque.popleft()

        return img, img_depth, master_arm, slave_arm

    def img_callback(self, msg):
        self.img_deque.append(msg)

    def img_depth_callback(self, msg):
        self.img_depth_deque.append(msg)

    def master_arm_callback(self, msg):
        self.master_arm_deque.append(msg)

    def slave_arm_callback(self, msg):
        self.slave_arm_deque.append(msg)

    def process(self):
        timesteps = []
        actions = []
        image =np.random.randint(0 ,255 ,size=(480 ,640 ,3) ,dtype = np.uint8)
        image_dict = {self.args.camera_names[0]: image}
        count = 0
        rate =self.create_rate(self.args.frame_rate)
        print_flag = True

        while rclpy.ok() and (count < self.args.max_timesteps + 1):
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count+=1
            img,img_depth,master_arm,slave_arm = result

            image_dict[self.args.camera_names[0]] = img

            obs = collections.OrderedDict()
            obs['images'] = img
            if self.args.use_depth_image:
                obs['images_depth'] = {self.args.camera_names[0]: img_depth}

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

            print('Frame data:',count)
            rate.sleep()
        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps,actions
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="/home/zzq/Desktop/Piper_ws/data/", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="piper_aloha", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=2, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=300, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_realsense'], required=False)
    parser.add_argument('--img_topic', action='store', type=str, help='img_topic',
                        default='/camera/camera/color/image_raw', required=False)
    parser.add_argument('--img_depth_topic', action='store', type=str, help='img_depth_topic',
                        default='/camera/camera/depth/image_rect_raw', required=False)
    parser.add_argument('--master_arm_topic', action = 'store',type=str,default='/joint_states', required=False)
    parser.add_argument('--slave_arm_topic', action = 'store',type=str,default='/joint_states', required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=15, required=False)

    args = parser.parse_args()
    return args


def main():
    rclpy.init()
    args = get_arguments()

    ros_operator = RosOperator(args)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(ros_operator)

    try:
        def spin_thread():
            executor.spin()

        import threading
        thread =threading.Thread(target=spin_thread)
        thread.start()
        
        timesteps,actions = ros_operator.process()
        
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(dataset_dir, f"episode_{args.episode_idx}")
        save_data(args,timesteps,actions,dataset_path)

    finally:
        executor.shutdown()
        ros_operator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

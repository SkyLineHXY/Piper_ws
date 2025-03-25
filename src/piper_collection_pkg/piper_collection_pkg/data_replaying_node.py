#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image,JointState
from geometry_msgs.msg import Twist
import threading
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm

class RosOperator(Node):
    def __init__(self,args):
        super().__init__('Replay_data_Node')
        self.args = args
        self.bridge = CvBridge()

        # self.img_pub = self.create_publisher(Image, self.args.img_topic, 10)
        self.slave_arm_pub =self.create_publisher(JointState, args.slave_arm_topic, 10)
        self.master_arm_pub =self.create_publisher(JointState, args.master_arm_topic,10)

        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        dataset_name = f'episode_{args.episode_idx}.hdf5'
        dataset_path = os.path.join(dataset_dir, dataset_name)

        self.qpos, self.qvels, self.efforts, self.actions, self.image_dicts = self.load_hdf5(dataset_path)

        self.plot_qpos_action(self.qpos,self.actions,output_dir= os.path.join(dataset_dir,'output'))

        self.stitch_and_save_videos(image_dict=self.image_dicts,output_path=os.path.join(dataset_dir,'output',f'stitched_video_episode{self.args.episode_idx}.mp4'))

        # self.joint_state_msg = JointState()
        # self.joint_state_msg.name=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        # self.twist_msg = Twist()
        # if args.only_pub_master:
        #     self.last_action = [0.6,
        #                         0.3,
        #                         -0.17,
        #                         -1,
        #                         0.15,
        #                         1.34,
        #                         0.04]
        #     self.publisher_master_only_thread = threading.Thread(target=self.publish_master_only)
        #     self.publisher_master_only_thread.start()
        # else:
        #     self.current_idx = 0
        #     self.publisher_all_thread = threading.Thread(target=self.publish_all)
        #     self.publisher_all_thread.start()

    def publish_master_only(self):
        rate = self.create_rate(self.args.frame_rate)
        while rclpy.ok():
            try:
                action = self.actions.pop(0)
                new_actions = np.linspace(self.last_action, action, 20)
                for act in new_actions:
                    timestamp = self.get_clock().now().to_msg()
                    self.joint_state_msg.header.stamp = timestamp
                    self.joint_state_msg.position = act[:].tolist()
                    self.master_arm_pub.publish(self.joint_state_msg)
            finally:
                self.get_logger().info("All actions published, shutting down...")
                rclpy.shutdown()
            rate.sleep()
    def publish_all(self):
        rate = self.create_rate(self.args.frame_rate)
        while rclpy.ok():
            if self.current_idx >= len(self.actions):
                self.get_logger().info("Replay completed")
                rclpy.shutdown()
                return
            i = self.current_idx
            self.get_logger().info(f"Publishing frame {i + 1}/{len(self.actions)}")

            cam_names = list(self.image_dicts.keys())
            images = []
            for cam in [cam_names[0]]:
                img = self.image_dicts[cam][i][:,:,[2,1,0]]
                images.append(img)

            timestamp = self.get_clock().now().to_msg()

            self.joint_state_msg.header.stamp = timestamp

            self.joint_state_msg.position = self.actions[i][:7].tolist()
            self.master_arm_pub.publish(self.joint_state_msg)

            # 发布图像
            # self.img_pub.publish(self.bridge.cv2_to_imgmsg(images[0], "rgb8"))

            self.current_idx += 1
            rate.sleep()
        pass
    def load_hdf5(self,dataset_path):

        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit()

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            compressed = root.attrs.get('compress', False)
            qpos = root['/observations/qpos'][:,:7]
            qvel = root['/observations/qvel'][:,:7]
            action = root['/action'][:,:7]
            if 'effort' in root['/observations/'].keys():
                effort = root['/observations/effort'][:,:7]
            else:
                effort = None
            image_dict = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
            # images = root['/observations/images/cam_realsense'][()]

            if compressed:
                compress_len = root['/compress_len'][()]
        # 图像压缩
        if compressed:
            for cam_id, cam_name in enumerate(image_dict.keys()):
                # un-pad and uncompress
                padded_compressed_image_list = image_dict[cam_name]
                image_list = []
                for frame_id, padded_compressed_image in enumerate(
                        padded_compressed_image_list):  # [:1000] to save memory
                    image_len = int(compress_len[cam_id, frame_id])
                    compressed_image = padded_compressed_image
                    image = cv2.imdecode(compressed_image, 1)
                    image_list.append(image)
                image_list[cam_name] = image_list

        return qpos, qvel, effort, action, image_dict


    def stitch_and_save_videos(self,image_dict, output_path='stitched_video.mp4'):
        """
            读取 image_dict 并拼接两个摄像头的视频流，保存为单个视频
            :param image_dict: 包含两个摄像头帧的字典 {cam_name: [frames]}
            :param output_path: 输出视频路径
        """
        cam_names = list(image_dict.keys())
        if len(cam_names) < 1:
            raise ValueError("image_dict 必须包含至少一个摄像头的视频流")
        frames_list = [image_dict[cam] for cam in cam_names]
        heights = [frames[0].shape[0] for frames in frames_list]
        widths = [frames[0].shape[1] for frames in frames_list]
        stitched_height = max(heights)
        stitched_width = sum(widths)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 15, (stitched_width, stitched_height))
        for frame_set in zip(*frames_list):
            stitched_frame = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
            x_offset = 0
            for i, frame in enumerate(frame_set):
                h, w, _ = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stitched_frame[:h, x_offset:x_offset + w] = frame
                x_offset += w
            out.write(stitched_frame)
        out.release()
        self.get_logger().info(f"拼接视频已保存到 {output_path}")


    def plot_qpos_action(self,qpos,action,output_dir=None):

        timesteps = range(qpos.shape[0])  # 获取时间步

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # 绘制 qpos 曲线
        for i in range(qpos.shape[1]):
            axs[0].plot(timesteps, qpos[:, i], label=f'Joint {i + 1}')
        axs[0].set_title('Joint Position (qpos) Over Time')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        axs[0].grid(True)
        # 绘制 action 曲线
        for i in range(action.shape[1]):
            axs[1].plot(timesteps, action[:, i], label=f'Action {i + 1}')
        axs[1].set_title('Action Over Time')
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Action Value')
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(os.path.join(output_dir, f'qpos_action_{self.args.episode_idx}.png'))
def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str,help='Dataset dir.',
                        default='/home/zzq/Desktop/piper_arm_ws/data/',required=False)
    parser.add_argument('--task_name',action='store', type=str,help='Task name.',
                        default="pick_hxy_to_box", required=False)
    parser.add_argument('--episode_idx', action='store', type=int,
                        help='Episode index.',default=0, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',required=False,
                        default=['cam_front','cam_wrist'])

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/front_camera/color/image_raw', required=False)
    parser.add_argument('--img_wrist_topic',action='store',type=str,help='img_wrist_topic',
                        default='/wrist_camera/wrist_camera/color/image_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/front_camera/front_camera/depth/image_rect_raw', required=False)
    parser.add_argument('--img_wrist_depth_topic',action='store',type=str,help='img_wrist_depth_topic',
                        default='/wrist_camera/wrist_camera/depth/image_rect_raw',required=False)

    parser.add_argument('--master_arm_topic', action='store', type=str, help='master_arm_topic',
                        default='joint_ctrl_single', required=False)
    parser.add_argument('--slave_arm_topic', action='store', type=str, help='slave_arm_topic',
                        default='/joint_states', required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--only_pub_master', action='store_true', help='only_pub_master', required=False)
    args = parser.parse_args()
    return args
    # # 生成qpos和qvel曲线
        # timesteps = np.arange(len(qpos))
        # plt.figure(figsize=(12, 6))
        #
        # # 绘制qpos曲线
        # plt.subplot(2, 1, 1)
        # for i in range(qpos.shape[1]):
        #     plt.plot(timesteps, qpos[:, i], label=f'Joint {i + 1}')
        #
        # plt.title('Joint Positions (qpos)')
        # plt.xlabel('Timesteps')
        # plt.ylabel('Position (rad)')
        # plt.legend()
        #
        # plt.subplot(2, 1, 2)
        #
        # for i in range(qvel.shape[1]):
        #     plt.plot(timesteps, qvel[:, i], label=f'Joint {i + 1}')
        #
        # plt.title('Joint Velocities (qvel)')
        # plt.xlabel('Timestep')
        # plt.ylabel('Velocity (rad/s)')
        # plt.legend()
        #
        # plot_path = os.path.join(output_path, 'joint_states_plot.png')
        # plt.tight_layout()
        # plt.savefig(plot_path)
        # plt.close()
        # print(f"Saved joint states plot to {plot_path}")

        # video_path = os.path.join(output_path, 'image_stream.mp4')
        # height, width, _ = images[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        # # cv2.imshow('image', images[0])
        # # cv2.imshow('image2', images[250])
        # # cv2.waitKey(0)
        # for img in tqdm(images, desc="Writing video"):
        #     # cv2.imshow('img', img)
        #     # cv2.waitKey(0)
        #
        #     video_writer.write(img)
        # video_writer.release()
        # print(f"Saved video to {video_path}")

def main(args=None):
    rclpy.init(args=None)
    if args is None:
        args = get_arguments()

    node = RosOperator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == '__main__':
    main()
    # dataset_dir = '/home/zzq/Desktop/Piper_ws/data/piper_aloha/'
    # dataset_file = 'episode_2.hdf5'
    # dataset_path = os.path.join(dataset_dir, dataset_file)
    # output_path = os.path.join(dataset_dir, 'output')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # load_hdf5(dataset_path, output_path)
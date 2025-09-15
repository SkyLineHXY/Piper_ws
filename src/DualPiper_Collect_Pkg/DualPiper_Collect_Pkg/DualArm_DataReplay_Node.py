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
    """
    ROS 2 节点，用于回放 HDF5 数据集中的机器人轨迹和图像数据。
    支持加载关节位置、速度、力矩、动作和多种相机图像（包括深度图）。
    提供数据可视化（关节位置和动作曲线）和视频拼接功能。
    """
    def __init__(self,args):
        """
        初始化 RosOperator 节点。

        Args:
            args: 命令行参数，包含数据集路径、任务名称、 episode索引、ROS topic 名称等。
        """
        super().__init__('Replay_data_Node') # 初始化 ROS 2 节点，命名为 'Replay_data_Node'
        self.args = args
        self.bridge = CvBridge() # 初始化 CvBridge，用于 ROS Image 消息和 OpenCV 图像之间的转换

        # 创建 ROS 2 发布者
        # self.img_pub = self.create_publisher(Image, self.args.img_topic, 10) # 图像发布者（当前被注释掉）
        self.slave_arm_pub =self.create_publisher(JointState, args.slave_arm_topic, 10) # 从机械臂关节状态发布者
        self.master_arm_pub =self.create_publisher(JointState, args.master_arm_topic,10) # 主机械臂关节状态发布者

        # 构建数据集路径
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        dataset_name = f'episode_{args.episode_idx}.hdf5'
        dataset_path = os.path.join(dataset_dir, dataset_name)
        
        
        # 从 HDF5 文件加载数据
        self.qpos, self.qvels, self.efforts, self.actions, self.image_dicts,self.image_depth_dicts= self.load_hdf5(dataset_path)

        # 绘制关节位置和动作曲线并保存
        self.plot_qpos_action(self.qpos,self.actions,output_dir= os.path.join(dataset_dir,'output'))

        # 拼接并保存 RGB 图像视频
        self.stitch_and_save_videos(image_dict=self.image_dicts,output_path=
                                    os.path.join(dataset_dir,'output',f'stitched_video_episode{self.args.episode_idx}.mp4'))
        # 如果存在深度图数据，则拼接并保存深度图视频
        if self.image_depth_dicts is not None:
            self.stitch_and_save_videos(image_dict=self.image_depth_dicts, output_path=os.path.join(dataset_dir, 'output',
                                            f'stitched_video_depth_episode{self.args.episode_idx}.mp4'),mode='depth')

    def load_hdf5(self,dataset_path,use_depth=False):
        """
        从 HDF5 文件加载机器人轨迹数据和图像数据。

        Args:
            dataset_path (str): HDF5 数据集文件的路径。
            use_depth (bool): 是否加载深度图像数据。

        Returns:
            tuple: 包含 qpos, qvel, effort, actions, image_dicts, image_depth_dicts 的元组。
        """
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit() # 如果数据集文件不存在，则退出程序

        with h5py.File(dataset_path, 'r') as root: # 以只读模式打开 HDF5 文件
            is_sim = root.attrs['sim'] # 读取模拟环境属性
            compressed = root.attrs.get('compress', False) # 读取图像是否压缩的属性，默认为 False
            qpos = root['/observations/qpos'][:,:] # 加载关节位置数据，取前7个关节
            qvel = root['/observations/qvel'][:,:] # 加载关节速度数据，取前7个关节
            action = root['/action'][:,:] # 加载动作数据，取前7个动作
            if 'effort' in root['/observations/'].keys(): # 检查是否存在力矩数据
                effort = root['/observations/effort'][:,:] # 加载力矩数据，取前7个关节
            else:
                effort = None # 如果不存在，设置为 None
            image_dict = dict() # 初始化图像字典

            for cam_name in root[f'/observations/images/'].keys(): # 遍历所有摄像头名称
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()] # 加载每个摄像头的图像数据
            # images = root['/observations/images/cam_realsense'][()] # 示例，加载特定摄像头的图像

            if use_depth: # 如果需要加载深度图
                image_depth_dict = dict() # 初始化深度图像字典
                for cam_name in root[f'/observations/images_depth/'].keys(): # 遍历所有摄像头的深度图名称
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][()] # 加载每个摄像头的深度图像数据
            else:
                image_depth_dict = None # 如果不需要，设置为 None

            if compressed: # 如果图像被压缩
                compress_len = root['/compress_len'][()] # 加载压缩长度信息

        # 图像解压缩 (如果图像是压缩的)
        if compressed:
            # Note: The original code has a logical error here.
            # It reassigns image_list[cam_name] = image_list, which would overwrite the list of images
            # for the current camera with the list of *all* images processed so far.
            # It should likely be image_dict[cam_name] = image_list.
            # However, since the code snippet related to `publish_all` is commented out,
            # this part might not be actively used for ROS publishing but rather for the video stitching.
            # For the purpose of providing detailed comments, I will comment on the existing logic
            # and note the potential improvement.
            for cam_id, cam_name in enumerate(image_dict.keys()):
                padded_compressed_image_list = image_dict[cam_name] # 获取当前摄像头的压缩图像列表
                image_list = [] # 用于存储解压缩后的图像
                for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # 遍历每一帧压缩图像
                    # image_len = int(compress_len[cam_id, frame_id]) # 获取当前帧的实际压缩长度 (此行在当前逻辑中未使用)
                    compressed_image = padded_compressed_image # 实际的压缩图像数据
                    image = cv2.imdecode(compressed_image, 1) # 使用 OpenCV 解码图像 (1 表示加载彩色图像)
                    image_list.append(image) # 将解压缩后的图像添加到列表中
                image_dict[cam_name] = image_list # 更新 image_dict 中的图像列表 (修正: 原代码为 image_list[cam_name] = image_list)

        return qpos, qvel, effort, action, image_dict,image_depth_dict

    def stitch_and_save_videos(self,image_dict, output_path='stitched_video.mp4',mode='rgb'):
        """
        读取 image_dict 并拼接多个摄像头的视频流，可选择是否包含深度图，保存为单个视频。

        Args:
            image_dict (dict): 包含摄像头帧的字典 {cam_name: [frames]}。
            output_path (str): 输出视频文件的完整路径。
            mode (str): 选择视频类型 ("rgb" 或 "depth")，用于处理图像格式。
        """
        if mode not in ["rgb", "depth"]:
            raise ValueError("mode 必须是 'rgb' 或 'depth'") # 检查模式是否有效
        cam_names = list(image_dict.keys()) # 获取所有摄像头名称
        if len(cam_names) < 1:
            raise ValueError("image_dict 必须包含至少一个摄像头的视频流") # 检查是否包含摄像头数据

        frames_list = [image_dict[cam] for cam in cam_names] # 获取所有摄像头的帧列表
        if mode == "depth": # 如果是深度图模式
            for i in range(len(frames_list)):
                # 对深度图像进行归一化到 0-255 范围，并转换为 8 位无符号整数
                frames_list[i] = [cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                     for frame in frames_list[i]]
                # 将单通道深度图转换为 BGR 格式（OpenCV 视频写入通常需要三通道）
                frames_list[i] = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames_list[i]]

        # 计算拼接后视频的尺寸
        heights = [frames[0].shape[0] for frames in frames_list] # 获取所有摄像头的图像高度
        widths = [frames[0].shape[1] for frames in frames_list] # 获取所有摄像头的图像宽度
        stitched_height = max(heights) # 拼接后视频的高度取最大图像高度
        stitched_width = sum(widths) # 拼接后视频的宽度为所有图像宽度之和

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 视频编码器，mp4v 是 MPEG-4 编码
        out = cv2.VideoWriter(output_path, fourcc, 30, (stitched_width, stitched_height)) # 创建 VideoWriter 对象，帧率为 30

        # 逐帧拼接图像并写入视频
        for frame_set in zip(*frames_list): # 使用 zip 将所有摄像头的同一时间步的帧打包
            stitched_frame = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8) # 创建空白的拼接帧
            x_offset = 0 # 横向偏移量，用于放置下一张图像
            for i, frame in enumerate(frame_set):
                h, w, _ = frame.shape # 获取当前帧的高度和宽度
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将 BGR 转换为 RGB (matplotlib 显示通常为 RGB)
                stitched_frame[:h, x_offset:x_offset + w] = frame # 将当前帧放置到拼接帧的相应位置
                x_offset += w # 更新横向偏移量
            out.write(stitched_frame) # 将拼接好的帧写入视频
        out.release() # 释放视频写入器资源
        self.get_logger().info(f"拼接视频已保存到 {output_path}") # 记录视频保存信息

    def plot_qpos_action(self,qpos,action,output_dir=None):
        """
        绘制关节位置 (qpos) 和动作 (action) 随时间变化的曲线，并保存为图片。

        Args:
            qpos (numpy.ndarray): 关节位置数据。
            action (numpy.ndarray): 动作数据。
            output_dir (str, optional): 图片保存目录。如果为 None，则不保存。
        """
        timesteps = range(qpos.shape[0]) # 获取时间步（帧数）

        fig, axs = plt.subplots(2, 1, figsize=(10, 8)) # 创建一个包含两个子图的 Matplotlib 图形

        # 绘制 qpos 曲线
        for i in range(qpos.shape[1]): # 遍历每个关节
            axs[0].plot(timesteps, qpos[:, i], label=f'Joint {i + 1}') # 绘制关节位置曲线
        axs[0].set_title('Joint Position (qpos) Over Time') # 设置子图标题
        axs[0].set_xlabel('Timestep') # 设置 X 轴标签
        axs[0].set_ylabel('Position') # 设置 Y 轴标签
        axs[0].legend() # 显示图例
        axs[0].grid(True) # 显示网格

        # 绘制 action 曲线
        for i in range(action.shape[1]): # 遍历每个动作维度
            axs[1].plot(timesteps, action[:, i], label=f'Action {i + 1}') # 绘制动作曲线
        axs[1].set_title('Action Over Time') # 设置子图标题
        axs[1].set_xlabel('Timestep') # 设置 X 轴标签
        axs[1].set_ylabel('Action Value') # 设置 Y 轴标签
        axs[1].legend() # 显示图例
        axs[1].grid(True) # 显示网格

        plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
        plt.show() # 显示图形

        if output_dir is not None: # 如果指定了输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) # 如果目录不存在，则创建
            fig.savefig(os.path.join(output_dir, f'qpos_action_{self.args.episode_idx}.png')) # 保存图形为 PNG 图片

def get_arguments():
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 包含所有解析参数的对象。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str,help='Dataset dir.',
                                default='/home/zzq/Desktop/piper_arm_ws/data/',required=False)
    parser.add_argument('--task_name',action='store', type=str,help='Task name.',
                                default="pick_GreenOnBlue", required=False)
    parser.add_argument('--episode_idx', action='store', type=int,
                                help='Episode index.',default=0, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',required=False,
                                default=['cam_top','cam_left']) # 列表默认值在 argparse 中通常作为字符串处理，这里需要注意实际使用时是否解析为列表

    parser.add_argument('--img_left_wrist_topic', action='store', type=str, help='img_front_topic',
                                default='/leftwrist_camera/wrist_camera/color/image_raw', required=False)
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


def main(args=None):
    """
    主函数，初始化 ROS 2 并运行 Replay_data_Node。
    """
    rclpy.init(args=None) # 初始化 ROS 2 客户端库
    if args is None:
        args = get_arguments() # 如果没有传入参数，则从命令行解析

    node = RosOperator(args) # 创建 RosOperator 节点实例
    try:
        rclpy.spin(node) # 启动 ROS 2 事件循环，节点将在此处运行，直到被中断或关闭
    except KeyboardInterrupt:
        pass # 捕获键盘中断异常 (Ctrl+C)，允许程序优雅退出
    finally:
        node.destroy_node() # 销毁节点，释放资源
        rclpy.shutdown() # 关闭 ROS 2 客户端库

if __name__ == '__main__':
    main()
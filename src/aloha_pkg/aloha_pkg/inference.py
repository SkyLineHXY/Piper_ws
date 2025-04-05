import torch
import numpy as np
import os
import pickle
import argparse
from collections import OrderedDict
from einops import rearrange
from cv_bridge import CvBridge
from constants import FPS
import rclpy
from collections import deque
from sensor_msgs.msg import JointState, Image
import time
import multiprocessing
import threading
from utils import * # helper functions
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import copy

stop_flag = False
#动作块平滑操作
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('Aloha_interface_Node')
        self.args = args
        self.camera_names =self.args.camera_names
        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.img_queues = {
            'cam_front': deque(maxlen=50),
            'cam_wrist': deque(maxlen=50),
            'cam_top': deque(maxlen=50),
            'cam_front_depth': deque(maxlen=50),
            # 'cam_wrist_depth': deque(maxlen=50),
            'cam_top_depth': deque(maxlen=50),
            'slave_arm': deque(maxlen=50),
        }

        # self.img_front_deque = deque(maxlen=50)
        # self.img_wrist_deque = deque(maxlen=50)
        # self.img_front_depth_deque = deque(maxlen =50)
        # self.img_wrist_depth_deque = deque(maxlen=50)
        # self.slave_arm_deque = deque(maxlen=50)

        self.slave_arm_pub = self.create_publisher(JointState, self.args.slave_arm_cmd_topic, 10)

        self.img_front_sub = self.create_subscription(Image,self.args.img_front_topic,
                                                      self.img_front_callback,10,callback_group=self.reentrant_callback_group)
        self.img_wrist_sub = self.create_subscription(Image,self.args.img_wrist_topic,
                                                     self.img_wrist_callback,10,callback_group=self.reentrant_callback_group)
        self.img_top_sub = self.create_subscription(Image, args.img_top_topic,
                                                    self.img_top_callback, 50)

        self.slave_arm_sub = self.create_subscription(JointState,args.slave_arm_topic,
                                                      self.slave_arm_callback,10,callback_group=self.reentrant_callback_group)
        if self.args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image,self.args.img_front_depth_topic,
                                                          self.img_front_depth_callback,10)
            self.img_wrist_depth_sub = self.create_subscription(Image,self.args.img_wrist_depth_topic,
                                                          self.img_wrist_depth_callback,10)


        self.initialize_model()
        self.inference_actions= []  #存储动作块
        self.timer_model_interface_threading = threading.Thread(target=self.model_inference)
        self.timer_model_interface_threading.start()

    def initialize_model(self):

        set_seed(1000)
        self.config = get_model_config(self.args)
        self.policy = make_policy(self.config['policy_class'], self.config['policy_config'])
        # ckpt_path = os.path.join('/home/zzq/Desktop/piper_arm_ws/policy_step_30000_seed_0.ckpt')
        ckpt_path = os.path.join(self.config['ckpt_dir'], self.config['ckpt_name'])
        # ckpt_path = '/home/zzq/Desktop/piper_arm_ws/policy_step_30000_seed_0.ckpt'
        state_dict = torch.load(ckpt_path, weights_only=False,map_location='cuda:0')
        new_state_dict = {}
        stats_path = os.path.join(self.config['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            if self.args.policy_class in ["ACT", "CNNMLP"]:
                self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
                self.post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
            elif self.args.policy_class == "Diffusion":
                self.pre_process = lambda s_qpos: (s_qpos - stats['action_min']) / (
                        stats['action_max'] - stats['action_min']) * 2 - 1
                self.post_process = lambda a: (a + 1) / 2 * (stats['action_max'] - stats['action_min']) + stats['action_min']
            else:
                NotImplementedError
        new_state_dict = {k: v for k, v in state_dict.items() if k not in [
            "model.is_pad_head.weight", "model.is_pad_head.bias",
            "model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]}
        if not self.policy.deserialize(new_state_dict):
            print("Checkpoint path does not exist")
            return False

        self.kalman_filters = [Kalman1D() for _ in range(self.config['policy_config']['action_dim'])]

        self.policy.cuda()
        self.policy.eval()
        # 发布基础的姿态
        self.initial_pose = [-0.0857, 0.558, -0.604, 0.055, 0.88, 0.32, 0.045]
        self.slave_arm_publish(self.initial_pose)
        input("Enter any key to continue :")
    def slave_arm_callback(self,msg):
        self.img_queues['slave_arm'].append(msg)
        # self.slave_arm_deque.append(msg)
    def img_front_callback(self,msg):

        self.img_queues['cam_front'].append(msg)
        # self.img_front_deque.append(msg)
    def img_top_callback(self,msg):
        self.img_queues['cam_top'].append(msg)
    def img_wrist_callback(self,msg):
        self.img_queues['cam_wrist'].append(msg)

    def img_top_depth_callback(self,msg):
        self.img_queues['cam_top_depth'].append(msg)
    def img_front_depth_callback(self,msg):
        self.img_queues['cam_front_depth'].append(msg)

    def slave_arm_publish(self,joint):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = joint
        self.slave_arm_pub.publish(msg)
    def get_frame_v2(self):
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

        keys.extend(['slave_arm'])
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
                    images_depth[name] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft())
        slave_arm = self.img_queues['slave_arm'].popleft()
        return {
            "images": images,
            "depths": images_depth,
            "slave_arm": slave_arm
        }

        return img_front,img_wrist,img_front_depth,img_wrist_depth, slave_arm
    def model_inference(self):
        def update_action_buffer(action_buffer, action_chunk, action_t):
            """
            for example:
            chunk_size = 5  # 缓冲区和 action_chunk 都是长度为 5
            action_buffer = [0, 0, 0, 0, 0]  # 初始缓冲区
            action_chunk = [1, 2, 3, 4, 5]   # 要写入的新动作
            action_t = 3  # 当前时间步

            计算起始和终止索引
            start_idx = 3 % 5 = 3
            end_idx = (3 + 5) % 5 = 3

            action_buffer[3:] = action_chunk[:2]  # [3:] 是索引 3 和 4，写入 [1, 2]
            => action_buffer = [0, 0, 0, 1, 2]

            action_buffer[:3] = action_chunk[2:]  # [:3] 是索引 0、1、2，写入 [3, 4, 5]
            => action_buffer = [3, 4, 5, 1, 2]
            """
            start_idx = action_t % chunk_size
            end_idx = (start_idx + chunk_size) % chunk_size
            action_buffer[start_idx:] = action_chunk[:chunk_size - start_idx]
            action_buffer[:end_idx] = action_chunk[chunk_size - start_idx:]
            return action_buffer

        rate =self.create_rate(30)
        actions_smotion_list = []
        actions_list = []
        max_publish_step = self.config['episode_len']
        chunk_size = self.config['policy_config']['chunk_size']
        all_time_actions = np.zeros(
            (max_publish_step, max_publish_step + chunk_size, self.config['policy_config']['action_dim'])
        ) if self.config['temporal_agg'] else None
        k = 0.1 #temporal_agg 参数
        t=0
        image_dict={}
        image_depth_dict = {}
        # pre_action = np.array(self.initial_pose)
        # action_buffer = np.zeros([chunk_size, self.config['policy_config']['action_dim']])
        while rclpy.ok() and t<max_publish_step:
            result = self.get_frame_v2()
            # result = self.get_frame()`
            if not result:
                self.get_logger().info("No frame received")
                rate.sleep()
                continue

            images, depths, slave_arm = result['images'], result['depths'],result['slave_arm']
            for i, name in enumerate(self.camera_names):
                image_dict[name] = images[name].copy()
                if self.args.use_depth_image and name != 'cam_wrist':
                    image_depth_dict[name] = depths[name].copy()
            obs = OrderedDict()
            obs['images']=copy.deepcopy(image_dict)
            obs['qpos'] = np.array(slave_arm.position[:7])
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(image_depth_dict)

            # obs['qvel'] = np.array(slave_arm.velocity)
            # obs['effort'] = np.array(slave_arm.effort)

            # img_front, img_wrist, img_front_depth, img_wrist_depth, slave_arm = result
            # obs = OrderedDict({
            #     'images': {
            #         self.args.camera_names[0]: img_front,
            #         self.args.camera_names[1]: img_wrist
            #     },
            #     'qpos': np.array(slave_arm.position)[:7]
            # })
            # if self.args.use_depth_image:
            #     obs['images_depth'] = {
            #         self.args.camera_names[0]: img_front_depth,
            #         self.args.camera_names[1]: img_wrist_depth}
            qpos = self.pre_process(obs['qpos'])
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            depth_image = get_depth_image(obs, self.args.camera_names) if self.args.use_depth_image else None
            curr_images = get_image(obs, self.args.camera_names)
            import time
            start_time = time.time()
            if self.args.policy_class == 'ACT':
                actions_tensor = self.policy(qpos_tensor, curr_images, depth_image=depth_image)
            elif self.args.policy_class == 'Diffusion':
                actions_tensor = self.policy(qpos_tensor, curr_images)
            end_time = time.time()
            print("model cost time: ", end_time - start_time)
            action_chunk_curr=actions_tensor.cpu().detach().numpy()
            action_chunk_curr = np.reshape(action_chunk_curr,
                                           (action_chunk_curr.shape[1], action_chunk_curr.shape[-1]))  # 当前时刻推理的动作块
            if self.config['temporal_agg']:
                all_time_actions[[t], t:t + chunk_size] = action_chunk_curr
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights[:, np.newaxis]
                raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                action = self.post_process(raw_action[0]).tolist()
                self.slave_arm_publish(action)

                actions_smotion_list.append(action)
                action_src = self.post_process(action_chunk_curr[0]).tolist()
                actions_list.append(action_src)
            else:
                smoothed_action_chunk = np.zeros_like(action_chunk_curr)
                for i in range(action_chunk_curr.shape[0]):  # 遍历 chunk 中每个动作
                    for j in range(7):  # 每个自由度
                        smoothed_action_chunk[i, j] = self.kalman_filters[j].update(action_chunk_curr[i, j])
                action_chunk_curr = smoothed_action_chunk
                for action in action_chunk_curr:
                    action = self.post_process(action).tolist()
                    self.slave_arm_publish(action)
                    rate.sleep()


            # self.inference_actions.append(all_actions.cpu().detach().numpy())
            rate.sleep()
            t += 1
        actions_smotion_list =np.array(actions_smotion_list)
        actions_list = np.array(actions_list)
        T, N = actions_smotion_list.shape  # 获取时间步长和关节数量

        import matplotlib.pyplot as plt
        # # 创建子图
        fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)
        # # 绘制每个关节的数据
        for i in range(N):
            axes[i].plot(actions_smotion_list[:, i], label="actions_smotion", linestyle="-")
            axes[i].plot(actions_list[:, i], label="actions", linestyle="--")
            axes[i].set_ylabel(f"Joint {i + 1}")
            axes[i].legend()
        axes[-1].set_xlabel("Time step")
        plt.suptitle("Joint Actions Over Time")
        plt.tight_layout()
        plt.show()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/front_camera/color/image_raw', required=False)
    parser.add_argument('--img_wrist_topic',action='store',type=str,help='img_wrist_topic',
                        default='/wrist_camera/wrist_camera/color/image_raw', required=False)
    parser.add_argument('--img_top_topic', action='store', type=str,
                        help='img_top_topic',default='/top_camera/top_camera/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/front_camera/front_camera/depth/image_rect_raw', required=False)
    parser.add_argument('--img_wrist_depth_topic',action='store',type=str,help='img_wrist_depth_topic',
                        default='/wrist_camera/wrist_camera/depth/image_rect_raw',required=False)
    parser.add_argument('--slave_arm_cmd_topic', action = 'store',type=str,default='/slave_piper_joint_ctrl', required=False)
    parser.add_argument('--slave_arm_topic', action='store', type=str, default='/joint_states', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class',
                        default='ACT', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=100,required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim',
                        default=512,required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward',
                        default=3200,required=False)
    parser.add_argument('--camera_names', action='store', type=list, help='camera_names',
                        default=['cam_front','cam_wrist','cam_top'],required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], required=False)
    parser.add_argument('--temporal_agg',action='store', type=bool, default=False, required=False)
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str,default='/media/zzq/ZZQ_SSD/ZZQ/trained_weight/dellrtx2080ti/piper_aloha/pick_blue_object_to_box/ACT/dellrtx2080_pick_blue_object_to_box_ACT_2025-04-03_23-35-10_numsteps_80000_chunksize_100_latent_dim_128')

                        # default='/home/zzq/Desktop/Open_Source_Project/act-plus-plus/trainings/pick_hxy_to_box/ACT/dellrtx5000_pick_hxy_to_box_ACT_2025-03-21_12-41-26_numsteps_40000_chunksize_100_latent_dim_128/', help='ckpt_dir')
    parser.add_argument('--ckpt_name', action='store', type=str, default='policy_step_52000_seed_0.ckpt',help='ckpt_name')
    parser.add_argument('--max_publish_step', action='store', type=int, default=1000, help='max_publish_step')
    parser.add_argument('--is_eval',action='store',type=bool,default=True,help='is_eval')
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)
    parser.add_argument('--state_dim', action='store', type=int, default=7, help='state_dim',)
    parser.add_argument('--use_actions_interpolation',action='store',type=bool,default=True, help='use_actions_interpolation')
    args = parser.parse_args()
    return args
def main():
    rclpy.init()
    args = get_arguments()
    ros_operator = RosOperator(args)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_operator)
    executor.spin()
    ros_operator.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()
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


stop_flag = False
#动作块平滑操作
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('Aloha_interface_Node')
        self.args = args
        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.img_front_deque = deque(maxlen=2000)
        self.img_wrist_deque = deque(maxlen=2000)
        self.img_front_depth_deque = deque(maxlen = 2000)
        self.img_wrist_depth_deque = deque(maxlen=2000)
        self.slave_arm_deque = deque(maxlen=2000)

        self.slave_arm_pub = self.create_publisher(JointState, self.args.slave_arm_cmd_topic, 10)

        self.img_front_sub = self.create_subscription(Image,self.args.img_front_topic,
                                                      self.img_front_callback,10,callback_group=self.reentrant_callback_group)
        self.img_wrist_sub = self.create_subscription(Image,self.args.img_wrist_topic,
                                                     self.img_wrist_callback,10,callback_group=self.reentrant_callback_group)
        self.slave_arm_sub = self.create_subscription(JointState,args.slave_arm_topic,
                                                      self.slave_arm_callback,10,callback_group=self.reentrant_callback_group)


        if self.args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image,self.args.img_front_depth_topic,
                                                          self.img_front_depth_callback,10)
            self.img_wrist_depth_sub = self.create_subscription(Image,self.args.img_wrist_depth_topic,
                                                          self.img_wrist_depth_callback,10)

        self.initialize_model()
        self.inference_actions= []#存储动作块
        self.timer_model_interface_threading = threading.Thread(target=self.model_inference)
        # self.piper_driver_threading  = threading.Thread(target=self.piper_driver)
        self.timer_model_interface_threading.start()
        # self.piper_driver_threading.start()
        # self.timer_model_interface = self.create_timer(1.0 / FPS, callback=self.model_inference,callback_group=self.reentrant_callback_group)

    def initialize_model(self):
        set_seed(1000)
        self.config = get_model_config(self.args)
        self.policy = make_policy(self.config['policy_class'], self.config['policy_config'])
        ckpt_path = os.path.join(self.config['ckpt_dir'], self.config['ckpt_name'])
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
        self.policy.cuda()
        self.policy.eval()
        # 发布基础的姿态
        initial_pose = [-0.0857, 0.558, -0.604, 0.055, 0.88, 0.2, 0.045]
        self.slave_arm_publish(initial_pose)
        input("Enter any key to continue :")
    def slave_arm_callback(self,msg):
        self.slave_arm_deque.append(msg)
    def slave_arm_publish(self,joint):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = joint
        self.slave_arm_pub.publish(msg)
    def get_model_config(args):
        set_seed(1)
    def img_front_callback(self,msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)
    def img_wrist_callback(self,msg):
        if len(self.img_wrist_deque) >= 2000:
            self.img_wrist_deque.popleft()
        self.img_wrist_deque.append(msg)
    def img_front_depth_callback(self,msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)
    def img_wrist_depth_callback(self,msg):
        if len(self.img_wrist_depth_deque) >= 2000:
            self.img_wrist_depth_deque.popleft()
        self.img_wrist_depth_deque.append(msg)
    def get_frame(self):
        if len(self.img_front_deque) == 0 or (self.args.use_depth_image and len(self.img_front_depth_deque) == 0):
            return False
        if len(self.img_wrist_deque) == 0 or (self.args.use_depth_image and len(self.img_wrist_depth_deque) == 0):
            return False
        if len(self.slave_arm_deque) == 0 :
            return False
        latest_front_img_time = self.img_front_deque[-1].header.stamp.sec + 1e-9 * self.img_front_deque[-1].header.stamp.nanosec
        latest_wrist_img_time = self.img_wrist_deque[-1].header.stamp.sec + 1e-9 * self.img_wrist_deque[-1].header.stamp.nanosec
        latest_slave_time = self.slave_arm_deque[-1].header.stamp.sec + 1e-9 * self.slave_arm_deque[-1].header.stamp.nanosec
        sync_time = min(latest_front_img_time,
                        latest_wrist_img_time,
                        latest_slave_time)
        while self.img_front_deque and (
            self.img_front_deque[0].header.stamp.sec + 1e-9 * self.img_front_deque[0].header.stamp.nanosec < sync_time):
            self.img_front_deque.popleft()

        while self.img_wrist_deque and (
            self.img_wrist_deque[0].header.stamp.sec + 1e-9 * self.img_wrist_deque[0].header.stamp.nanosec < sync_time):
            self.img_wrist_deque.popleft()

        while self.slave_arm_deque and (
                self.slave_arm_deque[0].header.stamp.sec + 1e-9 * self.slave_arm_deque[0].header.stamp.nanosec < sync_time):
            self.slave_arm_deque.popleft()
        if not self.img_wrist_deque or not self.img_front_deque  or not self.slave_arm_deque:
            return False

        img_front_msg = self.img_front_deque.popleft()
        img_wrist_msg = self.img_wrist_deque.popleft()
        img_wrist = self.bridge.imgmsg_to_cv2(img_wrist_msg,desired_encoding ='passthrough')
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg,desired_encoding ='passthrough')
        img_front_depth = None
        img_wrist_depth = None

        if self.args.use_depth_image and self.img_wrist_depth_deque and self.img_front_depth_deque:
            img_wrist_depth_msg = self.img_wrist_depth_deque.popleft()
            img_front_depth_msg = self.img_front_depth_deque.popleft()
            img_wrist_depth = self.bridge.imgmsg_to_cv2(img_wrist_depth_msg, desired_encoding='16UC1')
            img_front_depth = self.bridge.imgmsg_to_cv2(img_front_depth_msg, desired_encoding='16UC1')
        slave_arm = self.slave_arm_deque.popleft()

        return img_front,img_wrist,img_front_depth,img_wrist_depth, slave_arm
    def model_inference(self):
        rate =self.create_rate(10)
        actions_smotion_list = []
        actions_list = []
        max_publish_step = self.config['episode_len']
        chunk_size = self.config['policy_config']['chunk_size']
        all_time_actions = np.zeros(
            (max_publish_step, max_publish_step + chunk_size, self.config['policy_config']['action_dim'])
        ) if self.config['temporal_agg'] else None
        k = 0.1 #temporal_agg 参数
        t=0
        while rclpy.ok() and t<max_publish_step:

            result = self.get_frame()
            if not result:
                self.get_logger().info("No frame received")
                rate.sleep()
                continue
            img_front, img_wrist, img_front_depth, img_wrist_depth, slave_arm = result
            obs = OrderedDict({
                'images': {
                    self.args.camera_names[0]: img_front,
                    self.args.camera_names[1]: img_wrist
                },
                'qpos': np.array(slave_arm.position)[:7]
            })
            if self.args.use_depth_image:
                obs['images_depth'] = {
                    self.args.camera_names[0]: img_front_depth,
                    self.args.camera_names[1]: img_wrist_depth}
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
                for action in action_chunk_curr:
                    action = self.post_process(action).tolist()

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


        # action = None
        # max_publish_step = config['episode_len']
        # chunk_size = config['policy_config']['chunk_size']
        # # 创建多个推理进程
        # num_processes = 4  # 根据 CPU 核心数调整
        #
        # t = 0
        # all_time_actions = np.zeros(
        #     (max_publish_step, max_publish_step + chunk_size, self.config['policy_config']['action_dim'])
        # ) if config['temporal_agg'] else None
        # while t < max_publish_step and rclpy.ok():
        #     if len(inference_actions) > 0:
        #         action_chunk_curr = np.reshape(inference_actions.pop(0),(inference_actions.shape[1], inference_actions.shape[-1]))#当前时刻推理的的动作块
        #         if config['temporal_agg']:
        #             #动作ema平滑
        #             all_time_actions[[t], t:t + chunk_size] = action_chunk_curr
        #             actions_for_curr_step = all_time_actions[:, t]
        #             actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        #             actions_for_curr_step = actions_for_curr_step[actions_populated]
        #             k = 0.01
        #             exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        #             exp_weights = exp_weights / exp_weights.sum()
        #             exp_weights = exp_weights[:, np.newaxis]
        #             raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
        #             #动作数据保存
        #             action = self.post_process(raw_action[0]).tolist()
        #             actions_smotion_list.append(action)
        #             action_src = self.post_process(action_chunk_curr[0]).tolist()
        #             actions_list.append(action_src)
        #             self.slave_arm_publish(action)
        #             t += 1
        #         else:
        #             for action in action_chunk_curr:
        #                 action = self.post_process(action).tolist()
        #                 self.slave_arm_publish(action)
        #     rate.sleep()
        # with torch.inference_mode():
        #     while t < max_publish_step and rclpy.ok():
        #         # 每个回合的步数
        #         with inference_lock:
        #             if t >= max_t and config['policy_class'] == 'ACT':
        #                 if inference_actions is not None:
        #                     all_actions = np.reshape(inference_actions, (inference_actions.shape[1], inference_actions.shape[-1]))
        #                     inference_actions = None
        #                     max_t = t + self.args.pos_lookahead_step
        #                     if config['temporal_agg']:
        #                         all_time_actions[[t], t:t + chunk_size] = all_actions
        #                         actions_for_curr_step = all_time_actions[:, t]
        #                         actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        #                         actions_for_curr_step = actions_for_curr_step[actions_populated]
        #                         k = 0.01
        #                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        #                         exp_weights = exp_weights / exp_weights.sum()
        #                         exp_weights = exp_weights[:, np.newaxis]
        #                         raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
        #                         action = self.post_process(raw_action[0]).tolist()
        #                         actions_smotion_list.append(action)
        #                         action_src = self.post_process(all_actions[0]).tolist()
        #                         actions_list.append(action_src)
        #                         self.slave_arm_publish(action)
        #                         t += 1
        #                         rate.sleep()
        #                     else:
        #                         for action in all_actions:
        #                             action = self.post_process(action).tolist()
        #                             self.slave_arm_publish(action)
        #                             rate.sleep()


        # 终止推理线程
        # stop_flag = True
        # inference_thread.join()
        # actions_smotion_list =np.array(actions_smotion_list)
        # actions_list = np.array(actions_list)
        # T, N = actions_smotion_list.shape  # 获取时间步长和关节数量
        #
        # import matplotlib.pyplot as plt
        # # 创建子图
        # fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)
        # # 绘制每个关节的数据
        # for i in range(N):
        #     axes[i].plot(actions_smotion_list[:, i], label="actions_smotion", linestyle="-")
        #     axes[i].plot(actions_list[:, i], label="actions", linestyle="--")
        #     axes[i].set_ylabel(f"Joint {i + 1}")
        #     axes[i].legend()
        # axes[-1].set_xlabel("Time step")
        # plt.suptitle("Joint Actions Over Time")
        # plt.tight_layout()
        # plt.show()
    # Interpolate the actions to make the robot move smoothly

    # def piper_driver(self):
    #     max_publish_step = self.config['episode_len']
    #     chunk_size = self.config['policy_config']['chunk_size']
    #     rate = self.create_rate(30)
    #     t = 0
    #     all_time_actions = np.zeros(
    #         (max_publish_step, max_publish_step + chunk_size, self.config['policy_config']['action_dim'])
    #     ) if self.config['temporal_agg'] else None
    #
    #     while t < max_publish_step and rclpy.ok():
    #         if len(self.inference_actions) > 0:
    #             action_chunk_curr = self.inference_actions.pop(0)
    #             action_chunk_curr = np.reshape(action_chunk_curr,(action_chunk_curr.shape[1], action_chunk_curr.shape[-1]))#当前时刻推理的动作块
    #             if self.config['temporal_agg']:
    #                 #动作ema平滑
    #                 all_time_actions[[t], t:t + chunk_size] = action_chunk_curr
    #                 actions_for_curr_step = all_time_actions[:, t]
    #                 actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    #                 actions_for_curr_step = actions_for_curr_step[actions_populated]
    #                 k = 0.01
    #                 exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    #                 exp_weights = exp_weights / exp_weights.sum()
    #                 exp_weights = exp_weights[:, np.newaxis]
    #                 raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
    #                 action = self.post_process(raw_action[0]).tolist()
    #                 self.slave_arm_publish(action)
    #             else:
    #                 for action in action_chunk_curr:
    #                     action = self.post_process(action).tolist()
    #         rate.sleep()
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/front_camera/color/image_raw', required=False)
    parser.add_argument('--img_wrist_topic',action='store',type=str,help='img_wrist_topic',
                        default='/wrist_camera/wrist_camera/color/image_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/front_camera/front_camera/depth/image_rect_raw', required=False)
    parser.add_argument('--img_wrist_depth_topic',action='store',type=str,help='img_wrist_depth_topic',
                        default='/wrist_camera/wrist_camera/depth/image_rect_raw',required=False)
    parser.add_argument('--slave_arm_cmd_topic', action = 'store',type=str,default='/slave_piper_joint_ctrl', required=False)
    parser.add_argument('--slave_arm_topic', action='store', type=str, default='/joint_states', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    # parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
    #                     default=1, required=False)
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
                        default=['cam_front','cam_wrist'],required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], required=False)
    parser.add_argument('--temporal_agg',action='store', type=bool, default=True, required=False)
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str,
                        default='/home/zzq/Desktop/Open_Source_Project/act-plus-plus/trainings/pick_hxy_to_box/ACT/dellrtx5000_pick_hxy_to_box_ACT_2025-03-21_12-41-26_numsteps_40000_chunksize_100_latent_dim_128/', help='ckpt_dir')
    parser.add_argument('--ckpt_name', action='store', type=str, default='policy_last.ckpt',help='ckpt_name')
    parser.add_argument('--max_publish_step', action='store', type=int, default=300, help='max_publish_step')
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
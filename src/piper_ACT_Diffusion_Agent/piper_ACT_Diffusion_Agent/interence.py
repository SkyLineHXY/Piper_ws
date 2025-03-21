import torch
import numpy as np
import os
import pickle
import argparse
from collections import OrderedDict
from einops import rearrange
from cv_bridge import CvBridge
import rclpy
from collections import deque
from sensor_msgs.msg import JointState, Image
import time
import threading
from utils import * # helper functions
from rclpy.node import Node

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

#动作块平滑操作
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('Aloha_interface_Node')
        self.args = args
        self.bridge = CvBridge()
        self.img_front_deque = deque(maxlen=2000)
        self.img_wrist_deque = deque(maxlen=2000)
        self.img_front_depth_deque = deque(maxlen = 2000)
        self.img_wrist_depth_deque = deque(maxlen=2000)
        self.slave_arm_deque = deque(maxlen=2000)

        # self.arm_publish_lock = threading.Lock()
        # self.arm_publish_lock.acquire()

        self.slave_arm_pub = self.create_publisher(JointState, self.args.slave_arm_cmd_topic, 10)

        self.img_front_sub = self.create_subscription(Image,self.args.img_front_topic,
                                                      self.img_front_callback,10)
        self.img_wrist_sub = self.create_subscription(Image,self.args.img_wrist_topic,
                                                     self.img_wrist_callback,10)
        self.slave_arm_sub = self.create_subscription(JointState,args.slave_arm_topic,
                                                      self.slave_arm_callback,10)
        if self.args.use_depth_image:
            self.img_front_depth_sub = self.create_subscription(Image,self.args.img_front_depth_topic,
                                                          self.img_front_depth_callback,10)
            self.img_wrist_depth_sub = self.create_subscription(Image,self.args.img_wrist_depth_topic,
                                                          self.img_wrist_depth_callback,10)

        self.piper_action_filter = ArmActionFilter(history_length=self.args.chunk_size, smoothing_factor=0.2, filter_type='ema')

        self.interence_thread = threading.Thread(target=self.model_inference)
        self.interence_thread.start()
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
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        set_seed(1000)

        config = get_model_config(self.args)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        policy = make_policy(config['policy_class'], config['policy_config'])
        ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
        state_dict = torch.load(ckpt_path, weights_only=False)
        new_state_dict = {}
        stats_path = os.path.join(config['ckpt_dir'], f'dataset_stats.pkl')
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
                raise ValueError("Unsupported policy class")

        # 过滤不必要的权重
        new_state_dict = {k: v for k, v in state_dict.items() if k not in [
            "model.is_pad_head.weight", "model.is_pad_head.bias",
            "model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]}
        if not policy.deserialize(new_state_dict):
            print("Checkpoint path does not exist")
            return False
        policy.cuda()
        policy.eval()
        # 发布基础的姿态
        initial_pose = [-0.0857, 0.108, 0.039, 0.169, -0.0655, 0.162, 0.024]
        self.slave_arm_publish(initial_pose)
        # time.sleep(2)
        input("Enter any key to continue :")
        # ros_operator.slave_arm_publish(slave1)

        action = None
        max_publish_step = config['episode_len']
        chunk_size = config['policy_config']['chunk_size']

        with torch.inference_mode():
            t,max_t = 0,0
            rate = self.create_rate(self.args.publish_rate)
            all_time_actions = np.zeros(
                (max_publish_step, max_publish_step + chunk_size, config['policy_config']['action_dim'])
            ) if config['temporal_agg'] else None
            if config['temporal_agg']:
                past_actions = []
                # num_queries = chunk_size
            while t < max_publish_step and rclpy.ok():
                # 每个回合的步数
                if t >= max_t:
                    pre_action = action
                    inference_thread = threading.Thread(
                        target=self.model_inference_process,
                        args=(self.args, config, policy, stats, t, pre_action)
                    )
                    inference_thread.start()
                    inference_thread.join()
                    with inference_lock:
                        if inference_actions is not None:
                            action_sequence = np.reshape(inference_actions, (inference_actions.shape[1], inference_actions.shape[-1]))
                            inference_actions = None
                            max_t = t + self.args.pos_lookahead_step
                if config['temporal_agg']:
                    past_actions.append(action_sequence)  # 记录历史动作
                    if len(past_actions) > 5:  # 只保留最近 5 组动作块
                        past_actions.pop(0)
                    action_sequence = temporal_ensemble(past_actions)
                for action in action_sequence:
                    action = self.post_process(action).tolist()
                    self.slave_arm_publish(action)
                    rate.sleep()
                t += 1
    # Interpolate the actions to make the robot move smoothly

    def model_inference_process(self,args,config, policy, stats, t, pre_action):
        global inference_lock, inference_actions, inference_timestep, pre_pos_process

        rate = self.create_rate(args.publish_rate)
        while rclpy.ok():
            result = self.get_frame()
            if not result:
                self.get_logger().info("No frame received")
                rate.sleep()
                continue

            img_front, img_wrist, img_front_depth, img_wrist_depth, slave_arm = result
            obs = OrderedDict({
                'images': {
                    config['camera_names'][0]: img_front,
                    config['camera_names'][1]: img_wrist
                },
                'qpos': np.array(slave_arm.position)[:7]
            })

            if args.use_depth_image:
                obs['images_depth'] = {
                    config['camera_names'][0]: img_front_depth,
                    config['camera_names'][1]: img_wrist_depth}

            qpos = self.pre_process(obs['qpos'])
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            curr_images = get_image(obs, config['camera_names'])
            depth_image = get_depth_image(obs, config['camera_names']) if args.use_depth_image else None

            if config['policy_class']=='ACT':
                all_actions = policy(qpos_tensor,curr_images,depth_image=depth_image)
            elif config['policy_class']=='Diffusion':
                all_actions = policy(qpos_tensor,curr_images)

            with inference_lock:
                inference_actions = all_actions.cpu().detach().numpy()
                inference_timestep = t
            break


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
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=1, required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class',
                        default='ACT', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=30,required=False)
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
                        default='/home/zzq/Desktop/Open_Source_Project/act-plus-plus/trainings/pick_hxy_to_box/ACT/dellrtx5000_pick_hxy_to_box_ACT_2025-03-19_21-17-05_numsteps_40000_chunksize_30_latent_dim_128/', help='ckpt_dir')
    parser.add_argument('--ckpt_name', action='store', type=str, default='policy_best.ckpt',help='ckpt_name')
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
    rclpy.spin(ros_operator)
    ros_operator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
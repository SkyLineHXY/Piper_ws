import numpy as np
import torch
from collections import deque
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from einops import rearrange
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts
    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:  
                    action = root['/action'][:,:7]
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts][:7]
                qvel = root['/observations/qvel'][start_ts][:7]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1
            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]
            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]
            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)
            # normalize image and change dtype to float
            image_data = image_data / 255.0
            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][:,:7]
                qvel = root['/observations/qvel'][:,:7]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][:,:7]
                    # dummy_base_action = np.zeros([action.shape[0], 2])
                    # action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)
    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)
    train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

### env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])
def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose
### helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result
def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images,axis=0)
    curr_image = torch.from_numpy(curr_image /255.0).float().cuda().unsqueeze(0)
    return curr_image
def get_depth_image(observation, camera_names):
    curr_depths = []
    for cam_name in camera_names:
        curr_depths.append(observation['images_depth'][cam_name])
    curr_depths = np.stack(curr_depths, axis=0)
    curr_depths = torch.from_numpy(curr_depths).float().cuda().unsqueeze(0)
    return curr_depths


def interpolate_action(args, prev_action, cur_action):
    """
         对机械臂的目标动作进行插值，使其从 `prev_action` 平滑过渡到 `cur_action`，以满足步长限制。

         参数:
             args: 一个包含参数的对象，要求其中有 `arm_steps_length`，表示各个关节的步长
             prev_action (numpy.ndarray): 之前的动作（关节角度或位置）
             cur_action (numpy.ndarray): 目标动作（关节角度或位置）

         返回:
             numpy.ndarray: 插值后的动作序列，确保过渡平滑
     """
    steps = np.array(args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def get_model_config(args):
    set_seed(1)
    backbone = 'resnet18'
    lr_backbone = 1e-5
    if args.policy_class == 'ACT':

        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
                         'is_eval':args.is_eval,
                         'masks':False,
                         'position_embedding': 'sine',
                         'lr_backbone': lr_backbone,
                         'num_queries': args.chunk_size,
                         'hidden_dim': args.hidden_dim,
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'kl_weight': 10,
                         'backbone': backbone,
                         'nheads': nheads,
                         'camera_names': args.camera_names,
                         'chunk_size': args.chunk_size,
                         'vq': args.use_vq,
                         'vq_class': args.vq_class,
                         'vq_dim': args.vq_dim,
                         'action_dim': 7,
                         'no_encoder': args.no_encoder,
                         'use_robot_base':False,
                         'use_depth_image':False
                         }
    elif args.policy_class == 'Diffusion':
        policy_config = {
                         'lr': lr_backbone,
                         'camera_names': args.camera_names,
                         'action_dim': 7,
                         'observation_horizon': 1,
                         'action_horizon': 1,
                         'prediction_horizon':  args.chunk_size,
                         'num_queries': args.chunk_size,
                         'chunk_size': args.chunk_size,
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': args.use_vq,
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {
                         'masks': False,
                         'lr_backbone': lr_backbone,
                         'backbone' : backbone,
                         'num_queries': 1,
                         'chunk_size': args.chunk_size,
                         'camera_names':  args.camera_names,
                         'use_robot_base':False,
                         'use_depth_image':False,
                         'hidden_dim': args.hidden_dim,
                         'position_embedding': 'sine',
                         'is_eval': args.is_eval,
                         }
    else:
        raise NotImplementedError('Not implemented yet')
    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'episode_len': args.max_publish_step,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names':args.camera_names,
    }
    return config

def make_policy(policy_class,policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


class ArmActionFilter:
    def __init__(self, history_length=5, smoothing_factor=0.1,filter_type='ema'):
        """
        机械臂动作平滑滤波器
        :param history_length: 存储历史动作的步数
        :param smoothing_factor: 平滑系数（仅用于EMA滤波）
        :param filter_type: 选择滤波类型 'ema'（指数加权平均）或 'moving_average'（滑动均值）
        """
        self.previous_action = None  # 记录上一时间步的平滑值
        self.history_length = history_length
        self.smoothing_factor = smoothing_factor
        self.filter_type = filter_type
        self.action_history = deque(maxlen=history_length)

    def update_history(self, new_action):
        """
        更新动作历史
        :param new_action: 新的动作块 (30,7) numpy 数组
        """
        if not isinstance(new_action, np.ndarray) :
            raise ValueError("Invalid action data type")
        self.action_history.append(new_action)
    def smooth_action(self, new_action):
        """
         对当前动作进行平滑处理
         :param new_action: 当前动作 (30,7)
         :return: 平滑后的动作 (30,7)
         """
        self.update_history(new_action)
        if self.filter_type == 'ema':
            return self.ema_smoothing(new_action)


    def ema_smoothing(self, new_action):
        """
        指数加权平均（EMA）滤波
        :param new_action: 当前动作 (30,7)
        :return: 平滑后的动作 (30,7)
        """
        if self.previous_action is None:
            self.previous_action = new_action.copy()
            return new_action  # 没有历史数据，直接返回当前数据
        smoothed_action = np.zeros_like(new_action)
        smoothed_action[0] = self.smoothing_factor * new_action[0] + (1 - self.smoothing_factor) * self.previous_action[0]
        for t in range(1, new_action.shape[0]):  # 在时间轴方向进行EMA平滑
            smoothed_action[t] = self.smoothing_factor * new_action[t] + (1 - self.smoothing_factor) * smoothed_action[
                t - 1]
        self.previous_action = smoothed_action[-1]
        return smoothed_action

def temporal_ensemble(action_blocks, weights=None):
    """
    使用加权平均融合多个动作块，以增强动作的平滑性
    :param action_blocks: (N, 30, 7) 维度的 N 个历史动作块
    :param weights: (N,) 维度的加权系数
    :return: (30, 7) 维度的最终平滑动作块
    """
    action_blocks = np.array(action_blocks)  # 转换为 NumPy 数组
    if weights is None:
        # 使用指数衰减权重（越新的权重越大）
        N = action_blocks.shape[0]
        weights = np.exp(-0.1 * np.arange(N))[::-1]  # 指数衰减
        weights /= np.sum(weights)  # 归一化

    # 计算加权平均
    smoothed_action = np.tensordot(weights, action_blocks, axes=([0], [0]))

    return smoothed_action

import os
import argparse
import pickle
import numpy as np
import random
import time
from termcolor import colored  # 用于为终端输出添加颜色
import h5py  # 用于读取和处理 HDF5 文件
import zarr  # 用于压缩和存储数据
from termcolor import cprint  # 用于带颜色的终端输出
from tqdm import tqdm  # 显示进度条
import argparse  # 处理命令行参数

shape_meta = {
    "obs": {
        "qpos": {"shape": (14,), "type": "low_dim"},
        "qvel": {"shape": (14,), "type": "low_dim"},
        # "cam_front": {"shape": (3, 480, 640), "type": "rgb"},
        "cam_top": {"shape": (3, 480, 640), "type": "rgb"}
    },
    "action": {"shape": (14,), "type": "low_dim"}
}

def convert_dataset(args):
    # 解析参数
    Hdf5Demo_DIR = args.hdf5demo_dir  # 输入 HDF5 数据目录
    ZarrSave_DIR = args.zarrsave_dir  # 输出 Zarr 数据存储目录

    save_img = bool(args.save_img)    # 是否保存图像数据（0 或 1）
    save_depth = bool(args.save_depth)  # 是否保存深度数据（0 或 1）
    use_depth = bool(args.use_depth)
    if not os.path.isdir(Hdf5Demo_DIR):
        raise FileNotFoundError(f"HDF5 input dir not found: {Hdf5Demo_DIR}")
    # 检查并准备保存目录
    if os.path.exists(ZarrSave_DIR):
        # 如果目标目录已经存在，警告用户
        cprint('Data already exists at {}'.format(ZarrSave_DIR), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()  # 从用户获取输入
        user_input = 'y'  # 当前直接硬编码为 "y"
        if user_input == 'y':
            cprint('Overwriting {}'.format(ZarrSave_DIR), 'red')
            os.system('rm -rf {}'.format(ZarrSave_DIR))  # 删除现有目录及其内容
        else:
            cprint('Exiting', 'red')  # 如果用户拒绝覆盖，则退出程序
            return
    os.makedirs(ZarrSave_DIR, exist_ok=True)  # 创建新的目标目录


    demo_files = [f for f in os.listdir(Hdf5Demo_DIR) if f.endswith(".hdf5")]
    if len(demo_files) == 0:
        print("No .hdf5 files found in", Hdf5Demo_DIR)
        return
    demo_files = sorted(demo_files)  # 按文件名排序
    # open zarr group (directory store)
    zroot = zarr.open_group(ZarrSave_DIR, mode='a')
    total_count = 0
    episode_ends = []
    # total_count = 0  # 用于记录总的动作数量
    # color_arrays = []  # 保存图像数据
    # depth_arrays = []  # 保存深度数据
    # cloud_arrays = []  # 保存点云数据
    # state_arrays = []  # 保存状态数据
    # action_arrays = []  # 保存动作数据
    # episode_ends_arrays = []  # 保存每个文件结束的索引

    # helper to create or get zarr dataset (appendable along axis 0)
    def get_or_create(name, shape_sample, dtype, chunkshape=None):
        if name in zroot:
            ds = zroot[name]
            # check compatibility
            if ds.dtype != np.dtype(dtype):
                raise RuntimeError(f"Existing dataset {name} dtype mismatch: {ds.dtype} vs {dtype}")
            return ds
        else:
            # initial shape is (0, ...) so we can resize and append
            init_shape = (0,) + shape_sample[1:]
            if chunkshape is None:
                chunkshape = (1,) + shape_sample[1:]
            ds = zroot.create_dataset(name,
                                      shape=init_shape,
                                      chunks=chunkshape,
                                      dtype=dtype)
            return ds

    # 创建zarr根目录
    root = zarr.group()
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'low_dim']
    for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
        this_data = []
        for file_name in demo_files:
            with h5py.File(os.path.join(Hdf5Demo_DIR, file_name), 'r') as data:
                if key == 'action':
                    arr = data['/action'][:].astype(np.float32)
                    this_data.append(arr)
                elif key in ['qvel','qpos']:
                    arr = data[f'/observations/{key}'][:].astype(np.float32)
                    this_data.append(arr)
            this_data = np.concatenate(this_data, axis=0)
            data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
                overwrite=True
            )
    for demo_file in demo_files:
        # 加载每个 HDF5 文件
        file_path = os.path.join(Hdf5Demo_DIR, demo_file)  # 拼接文件路径

        print("process:", file_path)  # 打印正在处理的文件名
        with h5py.File(file_path, "r") as data:

            if '/action' not in data:
                print(f"Warning: '/action' not in {file_path}, skipping file.")
                continue
            action = data['/action'][:]  # 加载动作数据
            if action.ndim == 1:
                action = action.reshape((action.shape[0], 1))
            n =action.shape[0]
            if '/observations/qpos' not in data or '/observations/qvel' not in data:
                print(f"Warning: qpos/qvel missing in {file_path}, skipping.")
                continue
            qpos = data['/observations/qpos'][:]
            qvel = data['/observations/qvel'][:]
            # consistency checks
            if len(qpos) != n or len(qvel) != n:
                raise RuntimeError(f"Length mismatch in {file_path}: action {n}, qpos {len(qpos)}, qvel {len(qvel)}")
            # --- images ---
            images_per_cam = {}
            if save_img:
                if '/observations/images' not in data:
                    raise RuntimeError(f"Requested save_img but '/observations/images' group not found in {file_path}")
                for cam_name in data['/observations/images'].keys():
                    arr = data[f'/observations/images/{cam_name}'][:]
                    # Expecting (N,H,W,C)
                    if arr.shape[0] != n:
                        raise RuntimeError(f"Image count mismatch for cam {cam_name} in {file_path}: {arr.shape[0]} vs {n}")
                    images_per_cam[cam_name] = arr

            # --- depth images ---
            depths_per_cam = {}
            if use_depth and save_depth:
                if '/observations/images_depth' not in data:
                    print(f"Note: depth requested but '/observations/images_depth' missing in {file_path}; skipping depth for this file.")
                else:
                    for cam_name in data['/observations/images_depth'].keys():
                        darr = data[f'/observations/images_depth/{cam_name}'][:]
                        if darr.shape[0] != n:
                            raise RuntimeError(f"Depth count mismatch for cam {cam_name} in {file_path}: {darr.shape[0]} vs {n}")
                        depths_per_cam[cam_name] = darr

            # ----- create/get zarr datasets (lazy, on first file create) -----
            action_ds = get_or_create('action', action.shape, dtype=action.dtype)
            qpos_ds = get_or_create('observations/qpos', qpos.shape, dtype=qpos.dtype)
            qvel_ds = get_or_create('observations/qvel', qvel.shape, dtype=qvel.dtype)

            # images: one dataset per camera under observations/images/<cam>
            img_ds_map = {}
            if save_img:
                for cam_name,arr in images_per_cam.item():
                    ds_name = f'observations/images/{cam_name}'
                    ds = get_or_create(ds_name,arr.shape,
                                       dtype=arr.dtype,
                                       chunkshape=(1, arr.shape[1], arr.shape[2], arr.shape[3]))

            #depths
            depth_ds_map = {}
            if use_depth and save_depth:
                for cam_name,darr in depths_per_cam.items():
                    ds_name = f'observations/images_depth/{cam_name}'
                    # darr shape (N,H,W)
                    ds = get_or_create(ds_name, darr.shape, dtype=darr.dtype, chunkshape=(1, darr.shape[1], darr.shape[2]))
                    depth_ds_map[cam_name] = ds

            # --- append data by resizing target datasets ---
            # actions
            cur = action_ds.shape[0]
            action_ds.resize(cur + n, axis=0)
            action_ds[cur:cur + n, ...] = action.astype(action_ds.dtype)




            # # qpos, qvel
            # if '/observations/qpos' not in data or '/observations/qvel' not in data:
            #     print(f"Warning: qpos/qvel missing in {file_path}, skipping.")
            #     continue
            # qpos = data['/observations/qpos'][:]
            # qvel = data['/observations/qvel'][:]
            #
            # image_dict = dict() # 初始化图像字典
            # for cam_name in data[f'/observations/images/'].keys(): # 遍历所有摄像头名称
            #     image_dict[cam_name] = data[f'/observations/images/{cam_name}'][()] # 加载每个摄像头的图像数据
            # if use_depth: # 如果需要加载深度图
            #     image_depth_dict = dict() # 初始化深度图像字典
            #     for cam_name in data[f'/observations/images_depth/'].keys(): # 遍历所有摄像头的深度图名称
            #         image_depth_dict[cam_name] = data[f'/observations/images_depth/{cam_name}'][()] # 加载每个摄像头的深度图像数据
            #
            # action = data['/action'][:, :]  # 加载动作数据
            # qpos = data['/observations/qpos'][:,:] # 加载关节位置数据，取前7个关节
            # qvel = data['/observations/qvel'][:,:] # 加载关节速度数据，取前7个关节
            #
            # length = len(qpos)


class Args: pass
args = Args()
args.hdf5demo_dir = "/home/zzq/Desktop/piper_arm_ws/data/pull_late_and_pickplace_obj"
args.zarrsave_dir = "/home/zzq/Desktop/piper_arm_ws/data/pull_late_and_pickplace_obj/zarr"
args.save_img = 1
args.save_depth = 1
args.use_depth = 0

convert_dataset(args)

# if __name__ == '__main':
# #     parser = argparse.ArgumentParser()
# # `   parser.add_argument()
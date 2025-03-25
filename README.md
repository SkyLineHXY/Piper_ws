# 松灵机械臂遥操作指南

本指南介绍如何配置和运行松灵机械臂的遥操作系统，包括依赖安装、CAN 接口使能以及 ROS2 运行流程。

## 1. 安装依赖

在开始使用前，请确保已安装必要的 Python 依赖和 ROS2 相关组件。

### 1.1 安装 Python 依赖
```shell
pip3 install python-can scipy piper_sdk
```


### 1.2 安装 ROS2 依赖
```shell
sudo apt install ros-$ROS_DISTRO-ros2-control
sudo apt install ros-$ROS_DISTRO-ros2-controllers
sudo apt install ros-$ROS_DISTRO-controller-manager
```
## 2. 使能 CAN 接口

### 2.1 查询 CAN 接口的硬件地址

先分别单独连接用于控制机械臂的 USB 转 CAN 模块到电脑。

使用以下命令查询 can0 设备的硬件地址：
```shell
sudo ethtool -i can0 | grep bus
```
记录 bus-info 的数值，例如 1-1.2:1.0。

若 can0 无法识别，请尝试 can1。

注意： 这些硬件接口地址是固定的，一旦记录完成，后续连接机械臂时应保持一致。

### 2.2 配置并使能 CAN 接口

修改 can_config.sh 文件，调整 USB_PORTS 变量，使其匹配查询到的 bus-info 硬件地址。
```shell
# 预定义 USB 端口、目标接口名称及其波特率（适用于多个 CAN 模块）
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS
    USB_PORTS["1-1.2:1.0"]="can_master:1000000"
    USB_PORTS["1-1.3:1.0"]="can_slave:1000000"
fi
```
运行 can_config.sh 以使能 CAN 接口。
```shell
bash can_config.sh
```
## 3. 运行 ROS2 组件

### 3.1 编译工作空间

在 ROS2 工作空间内执行以下命令以编译相关 pkg：
```shell
colcon build
bash install/setup.bash
```

## 3.2 运行松灵主从臂 CAN 通信

启动松灵主从臂 CAN 通信的 ROS2 launch 文件：
```shell
ros2 launch piper_collection_pkg piper_slave_master_system.launch.py
```
3.3 启动主从臂示教节点
```shell
ros2 launch piper_collection_pkg slave_arm_control_node
```
至此，松灵机械臂的遥操作系统已成功配置并启动。


`data_saveing_to_hdf5_node.py`是收集动作episode,并且保存到hdf5文件的节点。
目前只实现了定长度的episode收集，按下键盘c开始收集动作数据，收集完一个episode后，再按下s就保存。

`data_replaying_node.py`是hdf5文件读取节点，并且绘制出曲线和video回放。

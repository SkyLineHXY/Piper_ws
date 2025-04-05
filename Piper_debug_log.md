# Piper机械臂调试记录

官网github仓库链接

``` bash
git clone https://github.com/agilexrobotics/piper_sdk?tab=readme-ov-file
```

## CAN 通信
单臂使能CAN0接口:
``` bash
bash can_activate.sh
```

## C_PiperInterface_V2类常用CAN通信接口程序

### C_PiperInterface_V2类初始化

```python
from piper_sdk import *
import time
piper = C_PiperInterface_V2()
piper.ConnectPort()
time.sleep(0.025) # 需要时间去读取固件反馈帧，否则会反馈-0x4AF
```

### Piper机械臂ENABLE和DISABLE
直接调用官方demo仓库即可

``` python 
def enable_fun(piper:C_PiperInterface_V2, enable:bool):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    loop_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (loop_flag):
        elapsed_time = time.time() - start_time
        print(f"--------------------")
        enable_list = []
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
        if(enable):
            enable_flag = all(enable_list)
            piper.EnableArm(7)
            piper.GripperCtrl(0,1000,0x01, 0)
        else:
            enable_flag = any(enable_list)
            piper.DisableArm(7)
            piper.GripperCtrl(0,1000,0x02, 0)
        print(f"使能状态: {enable_flag}")
        print(f"--------------------")
        if(enable_flag == enable):
            loop_flag = True
            enable_flag = True
        else: 
            loop_flag = False
            enable_flag = False
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print(f"超时....")
            elapsed_time_flag = True
            enable_flag = False
            loop_flag = True
            break
        time.sleep(0.5)
    resp = enable_flag
    print(f"Returning response: {resp}")
    return resp
```



### 机械臂姿态读取接口

end_pose末端位姿读取：

```python
endpose = piper.GetArmEndPoseMsgs()
```

Joint_state关节位置信息读取：

```python
JointMsgs = piper.GetArmJointMsgs() #单位0.001度
```

夹抓信息读取：

```python
GripperMsgs = piper.GetArmGripperMsgs()#主要接口是夹抓的位置angle以及力矩effort
```



### 机械臂姿态控制接口

夹抓控制接口：

```python
piper.GripperCtrl(gripper_angle:int, gripper_effort:int, gripper_code:int, set_zero:int)
```

*Args* ：

```
gripper_angle (int): 夹爪角度,单位 0.001°
gripper_effort (int): 夹爪力矩,单位 0.001N/m
gripper_code (int): 夹爪使能/失能/清除错误
    0x00失能,0x01使能
    0x03/0x02,使能清除错误,失能清除错误
set_zero:(int): 设定当前位置为0点
    0x00无效值
    0xAE设置零点
```

end_pose控制方式

```python
piper.EndPoseCtrl(X:int, Y:int, Z:int, RX:int, RY:int, RZ:int)
```
## Debug日志
### python语法问题
#### 字典赋值问题
字典类型是以哈希表（hash table） 形式存储的，其本质h上是对象的引用，字典的**key**存储的是**value**的**引用**。
采用直接赋值的方法存储方式：
```python
    obs['images'] (新字典) ----> image_dict (原字典) ---->'cam1' -> image1 (旧对象)
                                                        'cam2' -> image2 (旧对象)
```
采用copy.copy()浅拷贝的方法赋值：
```python
    obs['images'] (新字典) ----> 'cam1' -> image1 (旧对象)
                                'cam2' -> image2 (旧对象)
    image_dict (原字典) ----> 'cam1' -> image1 (旧对象)
                             'cam2' -> image2 (旧对象)
```
如果想赋值value的话，就得用`copy.deepcopy()`深拷贝的方法赋值。
```python
    obs['images'] (新字典) ----> 'cam1_new' -> image1 (新对象)
                                'cam2_new' -> image2 (新对象)
    image_dict (原字典) ----> 'cam1' -> image1 (旧对象)
                             'cam2' -> image2 (旧对象)
```
### Pytorch问题
```python
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out:List[NestedTensor] = []
        pos = []
        for name,x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))
        return out, pos
```
这段代码把 backbone 和 position_embedding 放进 nn.Sequential 中，使得self[0] 是 backbone，self[1] 是 position_embedding。
在forward函数中，整体逻辑是：
```
    Input (NestedTensor)
        ↓
    Backbone (e.g., ResNet)
        ↓
    多尺度特征 {layer1, layer2, ...}
        ↓                        ↓
    位置编码器 <------------ 每个特征图
        ↓                        ↓
    位置编码 pos          特征图 out
        ↓                        ↓
            -> 输出： ([NestedTensor], [position_tensor])
```
#### NestedTensor概念
NestedTensor 是 PyTorch 中（尤其是在 torchvision 和 DETR 相关模型中）提出的一种特殊张量结构，
主要用于处理不规则尺寸的图像或特征图，解决标准张量必须是矩形（相同形状）的限制。
其包括如下两部分：
```
NestedTensor = {
    "tensors": padded_tensor,     # shape: (B, C, H_max, W_max)，把所有图像 pad 到统一大小
    "mask": padding_mask          # shape: (B, H_max, W_max)，标记出哪些位置是 padding
}
```
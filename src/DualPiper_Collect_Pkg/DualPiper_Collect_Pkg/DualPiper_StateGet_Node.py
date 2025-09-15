import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from piper_msgs.msg import PiperStatusMsg, PosCmd
import math
from piper_sdk import *
import time
import threading
from geometry_msgs.msg import Pose
# from scipy.spatial.transform import Rotation as RS
from numpy import clip
from rclpy.executors import MultiThreadedExecutor

class DualPiperRosNode(Node):

    def __init__(self) -> None:
        super().__init__('dual_piper_state_get_node')
        
        # --- 1. 声明和获取左右臂的CAN端口参数 ---
        self.declare_parameter('left_can_port', 'can_left')
        self.declare_parameter('right_can_port', 'can_right')
        self.declare_parameter('auto_enable', True)
        self.can_left_port = self.get_parameter('left_can_port').get_parameter_value().string_value
        self.can_right_port = self.get_parameter('right_can_port').get_parameter_value().string_value
        self.auto_enable = self.get_parameter('auto_enable').get_parameter_value().bool_value
        self.get_logger().info(f"Left Arm CAN Port: {self.can_left_port}")
        self.get_logger().info(f"Right Arm CAN Port: {self.can_right_port}")
        self.get_logger().info(f"Auto Enable: {self.auto_enable}")

        # --- 2. 为左右臂分别实例化和连接Piper接口 ---
        self.get_logger().info("Connecting to Left Arm...")
        self.piper_left = C_PiperInterface(can_name=self.can_left_port)
        self.piper_left.ConnectPort()
        self.get_logger().info("Left Arm Connected.")
        
        self.get_logger().info("Connecting to Right Arm...")
        self.piper_right = C_PiperInterface(can_name=self.can_right_port)
        self.piper_right.ConnectPort()
        self.get_logger().info("Right Arm Connected.")
        
        # --- 3. 为左右臂分别创建话题发布者 ---
        # 左臂话题
        self.left_master_joint_pub = self.create_publisher(JointState, 'left/master_joint_states', 10)
        self.left_slave_joint_pub = self.create_publisher(JointState, 'left/slave_joint_states', 10)
        
        # 右臂话题
        self.right_master_joint_pub = self.create_publisher(JointState, 'right/master_joint_states', 10)
        self.right_slave_joint_pub = self.create_publisher(JointState, 'right/slave_joint_states', 10)

        # --- 4. 为左右臂分别创建JointState消息对象 ---
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        
        # 左臂消息
        self.left_master_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        self.left_slave_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        
        # 右臂消息
        self.right_master_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        self.right_slave_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        
        # --- 5. 使能和线程启动 ---
        self.__enable_flag_left = False
        self.__enable_flag_right = False
        
        #     if self.auto_enable:
        if self.auto_enable:
            self.enable_dual_arms()

        self.publisher_joint_thread = threading.Thread(target=self.publisher_joint_loop)
        self.publisher_joint_thread.start()
    
    def enable_dual_arms(self):
        """
        尝试使能左右两个机械臂，直到成功或超时。
        """
        self.get_logger().info("Attempting to enable dual arms...")
        timeout = 10  # 增加超时时间以确保两个臂都有足够时间响应
        start_time = time.time()
        while not (self.__enable_flag_left and self.__enable_flag_right):
            if time.time() - start_time > timeout:
                self.get_logger().error("Timeout reached while enabling arms. Exiting.")
                exit(1)
            # --- 检查和使能左臂 ---
            if not self.__enable_flag_left:
                # 检查左臂所有6个电机是否都已使能
                left_status = self.piper_left.GetArmLowSpdInfoMsgs()
                is_left_enabled = all([
                    left_status.motor_1.foc_status.driver_enable_status,
                    left_status.motor_2.foc_status.driver_enable_status,
                    left_status.motor_3.foc_status.driver_enable_status,
                    left_status.motor_4.foc_status.driver_enable_status,
                    left_status.motor_5.foc_status.driver_enable_status,
                    left_status.motor_6.foc_status.driver_enable_status
                ])
                if is_left_enabled:
                    self.__enable_flag_left = True
                    self.get_logger().info("Left Arm Enabled Successfully.")
                else:
                    self.piper_left.EnableArm(7) # 7代表所有关节
                    self.get_logger().info("Sent enable command to Left Arm.")
            # --- 检查和使能右臂 ---
            if not self.__enable_flag_right:
                # 检查右臂所有6个电机是否都已使能
                right_status = self.piper_right.GetArmLowSpdInfoMsgs()
                is_right_enabled = all([
                    right_status.motor_1.foc_status.driver_enable_status,
                    right_status.motor_2.foc_status.driver_enable_status,
                    right_status.motor_3.foc_status.driver_enable_status,
                    right_status.motor_4.foc_status.driver_enable_status,
                    right_status.motor_5.foc_status.driver_enable_status,
                    right_status.motor_6.foc_status.driver_enable_status
                ])
                if is_right_enabled:
                    self.__enable_flag_right = True
                    self.get_logger().info("Right Arm Enabled Successfully.")
                else:
                    self.piper_right.EnableArm(7)
                    self.get_logger().info("Sent enable command to Right Arm.")
                    
            time.sleep(1)
        self.get_logger().info("Sent enable command to Right Arm.")
        
    def publisher_joint_loop(self):
        """
        以200Hz的频率循环发布左右两臂的关节状态。
        """
        rate = self.create_rate(200)  # 200 Hz
        while rclpy.ok():
            current_time = self.get_clock().now().to_msg()
            
            # 发布左臂关节信息
            self._publish_single_arm_joints(
                self.piper_left, 
                self.left_master_joint_pub, 
                self.left_slave_joint_pub,
                self.left_master_joint_states,
                self.left_slave_joint_states,
                current_time
            )
            
            # 发布右臂关节信息
            self._publish_single_arm_joints(
                self.piper_right, 
                self.right_master_joint_pub,
                self.right_slave_joint_pub,
                self.right_master_joint_states,
                self.right_slave_joint_states,
                current_time
            )
            
            rate.sleep()
            
    def _publish_single_arm_joints(self, piper_iface:C_PiperInterface_V2, master_pub, slave_pub, master_msg, slave_msg, stamp):
        """
        辅助函数，用于获取并发布单个机械臂的主从关节状态。

        Args:
            piper_iface: 要操作的 C_PiperInterface 实例 (左臂或右臂)
            master_pub: 主臂关节状态的发布者
            slave_pub: 从臂关节状态的发布者
            master_msg: 用于填充主臂数据JointState消息对象
            slave_msg: 用于填充从臂数据的JointState消息对象
            stamp: 当前时间戳
        """
        try:
            # --- 发布主臂关节状态 ---
            master_msg.header.stamp = stamp
            
            master_joint_data = piper_iface.GetArmJointMsgs()
            master_gripper_data = piper_iface.GetArmGripperMsgs()
            master_speed_data = piper_iface.GetArmHighSpdInfoMsgs()
            # 角度(rad) = (原始值 / 1000) * (pi / 180)
            deg_to_rad = 0.01745329 
            master_msg.position = [
                (master_joint_data.joint_state.joint_1 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_2 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_3 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_4 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_5 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_6 / 1000) * deg_to_rad,
                master_gripper_data.gripper_state.grippers_angle / 1000000,
                0.0
            ]
            master_msg.velocity = [
                master_speed_data.motor_1.motor_speed / 1000,
                master_speed_data.motor_2.motor_speed / 1000,
                master_speed_data.motor_3.motor_speed / 1000,
                master_speed_data.motor_4.motor_speed / 1000,
                master_speed_data.motor_5.motor_speed / 1000,
                master_speed_data.motor_6.motor_speed / 1000,
                0.0, 0.0
            ]
            master_msg.effort = [0.0] * 8
            master_msg.effort[6] = master_gripper_data.gripper_state.grippers_effort / 1000
            master_pub.publish(master_msg)
            
            # --- 发布从臂关节状态 ---
            slave_msg.header.stamp = stamp
            slave_joint_data = piper_iface.GetArmJointCtrl()
            slave_gripper_data = piper_iface.GetArmGripperCtrl()
    
            slave_msg.position = [
                    (slave_joint_data.joint_ctrl.joint_1 / 1000) * deg_to_rad,
                    (slave_joint_data.joint_ctrl.joint_2 / 1000) * deg_to_rad,
                    (slave_joint_data.joint_ctrl.joint_3 / 1000) * deg_to_rad,
                    (slave_joint_data.joint_ctrl.joint_4 / 1000) * deg_to_rad,
                    (slave_joint_data.joint_ctrl.joint_5 / 1000) * deg_to_rad,
                    (slave_joint_data.joint_ctrl.joint_6 / 1000) * deg_to_rad,
                    slave_gripper_data.gripper_ctrl.grippers_angle / 1000000,
                    0.0
                ]
            # 从臂的速度和力矩信息通常与主臂相同或不直接提供，这里暂时复用主臂速度
            slave_msg.velocity = master_msg.velocity
            slave_msg.effort = master_msg.effort
            slave_pub.publish(slave_msg)
            
        except Exception as e:
                self.get_logger().error(f"Failed to publish joint states: {e}")
                
def main(args=None):
    rclpy.init(args=args)
    node = DualPiperRosNode()
    # 使用多线程执行器，因为节点内部使用了独立的线程来发布数据
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down dual piper node.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



            
                
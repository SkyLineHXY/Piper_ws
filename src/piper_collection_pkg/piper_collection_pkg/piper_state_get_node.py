#
# from typing import (
#     Optional,
# )
# import time
# from piper_sdk import *
# if __name__ == "__main__":
#     piper = C_PiperInterface()
#     piper.ConnectPort()
#     while True:
#         joint_0: float = (piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
#         joint_1: float = (piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
#         joint_2: float = (piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
#         print(f"joint_0: {joint_0}, joint_1: {joint_1}, joint_2: {joint_2}")
#         time.sleep(1)
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
from scipy.spatial.transform import Rotation as R  # For Euler angle to quaternion conversion
from numpy import clip
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
class PiperRosNode(Node):
    def __init__(self) -> None:
        super().__init__('piper_state_get_node')
        self.declare_parameter('piper_can_port', 'can0')
        self.declare_parameter('auto_enable', True)
        # self.declare_parameter('gripper_exist', True)
        # self.declare_parameter('rviz_ctrl_flag', False)
        self.piper_can_port = self.get_parameter('piper_can_port').get_parameter_value().string_value
        self.auto_enable = self.get_parameter('auto_enable').get_parameter_value().bool_value
        # self.gripper_exist = self.get_parameter('gripper_exist').get_parameter_value().bool_value
        # self.rviz_ctrl_flag = self.get_parameter('rviz_ctrl_flag').get_parameter_value().bool_value
        self.get_logger().info(f"can_port is {self.piper_can_port}")
        self.get_logger().info(f"auto_enable is {self.auto_enable}")
        # self.get_logger().info(f"gripper_exist is {self.gripper_exist}")
        # self.get_logger().info(f"rviz_ctrl_flag is {self.rviz_ctrl_flag}")

        self.piper = C_PiperInterface(can_name=self.piper_can_port)
        self.piper.ConnectPort()


        self.master_joint_pub = self.create_publisher(JointState, 'master_joint_states', 10)
        self.slave_joint_pub = self.create_publisher(JointState, 'slave_joint_states', 10)
        self.piper_arm_status_pub = self.create_publisher(PiperStatusMsg, 'piper_arm_status', 10)
        self.piper_end_pose_pub = self.create_publisher(Pose, 'piper_end_pose', 10)

        self.slave_joint_states = JointState()
        self.slave_joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7','joint8']
        self.slave_joint_states.position = [0.00] * 8
        self.slave_joint_states.velocity = [0.00] * 8
        self.slave_joint_states.effort = [0.00] * 8

        self.master_joint_states = JointState()
        self.master_joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7','joint8']
        self.master_joint_states.position = [0.0] * 8
        self.master_joint_states.velocity = [0.0] * 8
        self.master_joint_states.effort = [0.0] * 8
        # Enable flag
        self.__enable_flag = False
        self.piper_endpos_msg = Pose()

        self.piper_arm_status_msg = PiperStatusMsg()
        # self.piper_enable()

        self.publisher_joint_thread = threading.Thread(target=self.publisher_joint_loop)
        self.publisher_armstatus_thread = threading.Thread(target=self.publish_armstatus_loop)
        self.publisher_endpose_thread = threading.Thread(target=self.PublishArmEndPose)
        self.publisher_joint_thread.start()
        # self.publisher_armstatus_thread.start()
        # self.publisher_endpose_thread.start()
    def GetEnableFlag(self):
        return self.__enable_flag
    def piper_enable(self):
        enable_flag = False
        timeout = 5
        # Record the time before entering the loop
        start_time = time.time()
        elapsed_time_flag = False
        if self.auto_enable:
            while not (enable_flag):
                elapsed_time = time.time() - start_time
                # 从臂left+right
                print("--------------------")
                enable_flag = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                              self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                              self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                              self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                              self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                              self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
                print("Enable status:", enable_flag)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                if (enable_flag):
                    self.__enable_flag = True
                print("--------------------")
                # Check if the timeout has been exceeded
                if elapsed_time > timeout:
                    print("Timeout....")
                    elapsed_time_flag = True
                    enable_flag = True
                    break
                time.sleep(1)
                pass
                if elapsed_time_flag:
                    print("Automatic enable timeout, exiting program")
                    exit(0)
    def publisher_joint_loop(self):
        rate = self.create_rate(200)  # 200 Hz
        while rclpy.ok():
            self.PublishArmJointAndGripper()
            rate.sleep()
    def publish_armstatus_loop(self):
        rate = self.create_rate(10)
        while rclpy.ok():
            self.PublishArmState()
            rate.sleep()
    def publish_endpose_loop(self):
        rate = self.create_rate(200)  # 200 Hz
        while rclpy.ok():
            self.PublishArmEndPose()
            rate.sleep()
    def PublishArmJointAndGripper(self):
        # master arm joint states publisher

        self.master_joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000

        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.master_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,0.0]  # Example values

        self.master_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0,0.0]  # Example values
        self.master_joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6, 0.0]
        self.master_joint_pub.publish(self.master_joint_states)

        self.slave_joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_0: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1000000
        # effort_6: float = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_effort / 1000
        self.slave_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,0.0]  # Example values
        self.slave_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0, 0.0]  # Example values
        self.slave_joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6, 0.0]
        self.slave_joint_pub.publish(self.slave_joint_states)
    def PublishArmEndPose(self):
        # slave arm end pose publisher
        self.piper_endpos_msg.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        self.piper_endpos_msg.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        self.piper_endpos_msg.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000

        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000

        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        self.piper_endpos_msg.orientation.x = quaternion[0]
        self.piper_endpos_msg.orientation.y = quaternion[1]
        self.piper_endpos_msg.orientation.z = quaternion[2]
        self.piper_endpos_msg.orientation.w = quaternion[3]
        self.piper_end_pose_pub.publish(self.piper_endpos_msg)
    def PublishArmState(self):
        #发送从臂的状态
        self.piper_arm_status_msg.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        self.piper_arm_status_msg.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        self.piper_arm_status_msg.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        self.piper_arm_status_msg.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        self.piper_arm_status_msg.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        self.piper_arm_status_msg.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        self.piper_arm_status_msg.err_code = self.piper.GetArmStatus().arm_status.err_code
        self.piper_arm_status_msg.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        self.piper_arm_status_msg.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        self.piper_arm_status_msg.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        self.piper_arm_status_msg.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        self.piper_arm_status_msg.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        self.piper_arm_status_msg.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        self.piper_arm_status_msg.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        self.piper_arm_status_msg.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        self.piper_arm_status_msg.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        self.piper_arm_status_msg.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        self.piper_arm_status_msg.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        self.piper_arm_status_msg.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6

        self.piper_arm_status_pub.publish(self.piper_arm_status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PiperRosNode()
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
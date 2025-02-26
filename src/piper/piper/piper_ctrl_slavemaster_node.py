#!/usr/bin/env python3
# -*-coding:utf8-*-
# This file controls a single robotic arm node and handles the movement of the robotic arm with a gripper.
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import argparse
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R  # For Euler angle to quaternion conversion
from numpy import clip


class PiperRosNode(Node):

    def __init__(self) -> None:
        super().__init__('piper_ctrl_single_node')
        # ROS parameters
        self.declare_parameter('slave_can_port', 'can_slave')
        self.declare_parameter('master_can_port', 'can_master')
        self.declare_parameter('auto_enable', True)
        self.declare_parameter('gripper_exist', True)
        self.declare_parameter('rviz_ctrl_flag', False)
        self.slave_can_port = self.get_parameter('slave_can_port').get_parameter_value().string_value
        self.master_can_port = self.get_parameter('master_can_port').get_parameter_value().string_value

        self.auto_enable = self.get_parameter('auto_enable').get_parameter_value().bool_value
        self.gripper_exist = self.get_parameter('gripper_exist').get_parameter_value().bool_value
        self.rviz_ctrl_flag = self.get_parameter('rviz_ctrl_flag').get_parameter_value().bool_value

        self.get_logger().info(f"slave_can_port is {self.slave_can_port}")
        self.get_logger().info(f"master_can_port is {self.master_can_port}")
        self.get_logger().info(f"auto_enable is {self.auto_enable}")
        self.get_logger().info(f"gripper_exist is {self.gripper_exist}")
        self.get_logger().info(f"rviz_ctrl_flag is {self.rviz_ctrl_flag}")

        self.master_joint_pub = self.create_publisher(JointState, 'master_joint_states', 10)
        self.master_arm_status_pub = self.create_publisher(PiperStatusMsg, 'master_arm_status', 10)
        self.master_end_pose_pub = self.create_publisher(Pose, 'master_end_pose', 10)

        self.slave_joint_pub = self.create_publisher(JointState, 'slave_joint_states', 10)
        self.slave_arm_status_pub = self.create_publisher(PiperStatusMsg, 'slave_arm_status', 10)
        self.slave_end_pose_pub = self.create_publisher(Pose, 'slave_end_pose', 10)
        # Service
        # self.motor_srv = self.create_service(Enable, 'enable_srv', self.handle_enable_service)

        # Joint
        self.master_joint_states = JointState()
        self.master_joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7','joint8']
        self.master_joint_states.position = [0.0] * 8
        self.master_joint_states.velocity = [0.0] * 8
        self.master_joint_states.effort = [0.0] * 8

        self.slave_joint_states = JointState()
        self.slave_joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7','joint8']
        self.slave_joint_states.position = [0.0] * 8
        self.slave_joint_states.velocity = [0.0] * 8
        self.slave_joint_states.effort = [0.0] * 8

        # Enable flag
        self.__enable_flag = False
        # Create piper class and open CAN interface
        self.master_piper = C_PiperInterface(can_name=self.master_can_port)
        self.slave_piper = C_PiperInterface(can_name=self.slave_can_port)
        self.master_piper.ConnectPort()
        self.slave_piper.ConnectPort()

        # Subscription
        self.create_subscription(PosCmd, 'slave_piper_pos_ctrl', self.Slave_pos_callback, 1)
        self.create_subscription(JointState, 'slave_piper_joint_ctrl', self.Slave_joint_callback, 1)
        self.create_subscription(Bool, 'slave_enable_flag', self.Slave_enable_callback, 1)

        self.publisher_thread = threading.Thread(target=self.publish_thread)
        self.publisher_thread.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def publish_thread(self):
        rate = self.create_rate(200)  # 200 Hz
        enable_flag = False
        timeout = 5
        # Record the time before entering the loop
        start_time = time.time()
        elapsed_time_flag = False
        while rclpy.ok():
            if(self.auto_enable):
                while not (enable_flag):
                    elapsed_time = time.time() - start_time
                    #从臂left+right
                    print("--------------------")
                    enable_flag = self.slave_piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                        self.slave_piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                        self.slave_piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                        self.slave_piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                        self.slave_piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                        self.slave_piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status

                    print("Enable status:", enable_flag)
                    self.slave_piper.EnableArm(7)
                    self.slave_piper.GripperCtrl(0, 1000, 0x01, 0)

                    if(enable_flag):
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
            if (elapsed_time_flag):
                print("Automatic enable timeout, exiting program")
                exit(0)
            self.PublishArmState()
            self.PublishArmJointAndGripper()
            self.PublishArmEndPose()
            rate.sleep()
    def PublishArmState(self):
        slave_arm_status = PiperStatusMsg()
        master_arm_status = PiperStatusMsg()

        #发送从臂的状态
        slave_arm_status.ctrl_mode = self.slave_piper.GetArmStatus().arm_status.ctrl_mode
        slave_arm_status.arm_status = self.slave_piper.GetArmStatus().arm_status.arm_status
        slave_arm_status.mode_feedback = self.slave_piper.GetArmStatus().arm_status.mode_feed
        slave_arm_status.teach_status = self.slave_piper.GetArmStatus().arm_status.teach_status
        slave_arm_status.motion_status = self.slave_piper.GetArmStatus().arm_status.motion_status
        slave_arm_status.trajectory_num = self.slave_piper.GetArmStatus().arm_status.trajectory_num
        slave_arm_status.err_code = self.slave_piper.GetArmStatus().arm_status.err_code
        slave_arm_status.joint_1_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        slave_arm_status.joint_2_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        slave_arm_status.joint_3_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        slave_arm_status.joint_4_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        slave_arm_status.joint_5_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        slave_arm_status.joint_6_angle_limit = self.slave_piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        slave_arm_status.communication_status_joint_1 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        slave_arm_status.communication_status_joint_2 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        slave_arm_status.communication_status_joint_3 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        slave_arm_status.communication_status_joint_4 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        slave_arm_status.communication_status_joint_5 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        slave_arm_status.communication_status_joint_6 = self.slave_piper.GetArmStatus().arm_status.err_status.communication_status_joint_6

        self.slave_arm_status_pub.publish(slave_arm_status)

        #发送主臂的状态
        master_arm_status.ctrl_mode = self.master_piper.GetArmStatus().arm_status.ctrl_mode
        master_arm_status.arm_status = self.master_piper.GetArmStatus().arm_status.arm_status
        master_arm_status.mode_feedback = self.master_piper.GetArmStatus().arm_status.mode_feed
        master_arm_status.teach_status = self.master_piper.GetArmStatus().arm_status.teach_status
        master_arm_status.motion_status = self.master_piper.GetArmStatus().arm_status.motion_status
        master_arm_status.trajectory_num = self.master_piper.GetArmStatus().arm_status.trajectory_num
        master_arm_status.err_code = self.master_piper.GetArmStatus().arm_status.err_code
        master_arm_status.joint_1_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        master_arm_status.joint_2_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        master_arm_status.joint_3_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        master_arm_status.joint_4_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        master_arm_status.joint_5_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        master_arm_status.joint_6_angle_limit = self.master_piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        master_arm_status.communication_status_joint_1 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        master_arm_status.communication_status_joint_2 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        master_arm_status.communication_status_joint_3 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        master_arm_status.communication_status_joint_4 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        master_arm_status.communication_status_joint_5 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        master_arm_status.communication_status_joint_6 = self.master_piper.GetArmStatus().arm_status.err_status.communication_status_joint_6

        self.master_arm_status_pub.publish(master_arm_status)

    def PublishArmJointAndGripper(self):
        # slave arm joint states publisher
        self.slave_joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_0: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.slave_piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.slave_piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        vel_0: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.slave_piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.slave_piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.slave_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,0.0]  # Example values
        self.slave_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0,0.0]  # Example values
        self.slave_joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6,0.0]

        self.slave_joint_pub.publish(self.slave_joint_states)

        #master arm joint states publisher
        self.master_joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_0: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.master_piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.master_piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000

        vel_0: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.master_piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.master_piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.master_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,0.0]  # Example values
        self.master_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0,0.0]  # Example values
        self.master_joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6,0.0]
        self.master_joint_pub.publish(self.master_joint_states)

    def PublishArmEndPose(self):
        slave_endpos_msg = Pose()
        master_endpos_msg = Pose()

        # slave arm end pose publisher
        slave_endpos_msg.position.x = self.slave_piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        slave_endpos_msg.position.y = self.slave_piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        slave_endpos_msg.position.z = self.slave_piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.slave_piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.slave_piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.slave_piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        slave_endpos_msg.orientation.x = quaternion[0]
        slave_endpos_msg.orientation.y = quaternion[1]
        slave_endpos_msg.orientation.z = quaternion[2]
        slave_endpos_msg.orientation.w = quaternion[3]
        self.slave_end_pose_pub.publish(slave_endpos_msg)


        # master arm end pose publisher
        master_endpos_msg.position.x = self.master_piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        master_endpos_msg.position.y = self.master_piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        master_endpos_msg.position.z = self.master_piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.master_piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.master_piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.master_piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        master_endpos_msg.orientation.x = quaternion[0]
        master_endpos_msg.orientation.y = quaternion[1]
        master_endpos_msg.orientation.z = quaternion[2]
        master_endpos_msg.orientation.w = quaternion[3]
        self.master_end_pose_pub.publish(master_endpos_msg)

    def Slave_pos_callback(self,pos_data):
        """
            Callback function for subscribing to the end effector pose

            Args:
                pos_data (): The position data
        """
        factor = 180 / 3.1415926
        self.get_logger().info(f"Received PosCmd:")
        self.get_logger().info(f"x: {pos_data.x}")
        self.get_logger().info(f"y: {pos_data.y}")
        self.get_logger().info(f"z: {pos_data.z}")
        self.get_logger().info(f"roll: {pos_data.roll}")
        self.get_logger().info(f"pitch: {pos_data.pitch}")
        self.get_logger().info(f"yaw: {pos_data.yaw}")
        self.get_logger().info(f"gripper: {pos_data.gripper}")
        self.get_logger().info(f"mode1: {pos_data.mode1}")
        self.get_logger().info(f"mode2: {pos_data.mode2}")
        x = round(pos_data.x*1000) * 1000
        y = round(pos_data.y*1000) * 1000
        z = round(pos_data.z*1000) * 1000
        rx = round(pos_data.roll*1000*factor)
        ry = round(pos_data.pitch*1000*factor)
        rz = round(pos_data.yaw*1000*factor)
        if(self.GetEnableFlag()):
            self.slave_piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.slave_piper.MotionCtrl_2(0x01, 0x00, 2)
            self.slave_piper.EndPoseCtrl(x, y, z, rx, ry, rz)
            gripper = round(pos_data.gripper * 1000 * 1000)
            if pos_data.gripper > 80000:
                gripper = 80000
            if pos_data.gripper < 0:
                gripper = 0
            if self.gripper_exist:
                self.slave_piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)

    def Slave_joint_callback(self, joint_data):
        factor = 57324.840764  # 1000*180/3.14

        self.get_logger().info(f"Received Joint States:")
        self.get_logger().info(f"joint_0: {joint_data.position[0]}")
        self.get_logger().info(f"joint_1: {joint_data.position[1]}")
        self.get_logger().info(f"joint_2: {joint_data.position[2]}")
        self.get_logger().info(f"joint_3: {joint_data.position[3]}")
        self.get_logger().info(f"joint_4: {joint_data.position[4]}")
        self.get_logger().info(f"joint_5: {joint_data.position[5]}")
        joint_0 = round(joint_data.position[0]*factor)
        joint_1 = round(joint_data.position[1]*factor)
        joint_2 = round(joint_data.position[2]*factor)
        joint_3 = round(joint_data.position[3]*factor)
        joint_4 = round(joint_data.position[4]*factor)
        joint_5 = round(joint_data.position[5]*factor)
        if (len(joint_data.position) >= 7):
            self.get_logger().info(f"joint_6: {joint_data.position[6]}")
            joint_6 = round(joint_data.position[6]*1000*1000)
            if(self.rviz_ctrl_flag):
                joint_6 = joint_6 * 2
            joint_6 = clip(joint_6, 0, 80000)
        else: joint_6 = 0
        if (self.GetEnableFlag()):
            if(joint_data.velocity != []):
                all_zeros = all(v == 0 for v in joint_data.velocity)
            else:
                all_zeros = True
            if not all_zeros:
                lens = len(joint_data.velocity)
                if lens == 7:
                    vel_all = clip(round(joint_data.velocity[6]), 1, 100)
                    self.get_logger().info(f"vel_all: {vel_all}")
                    self.slave_piper.MotionCtrl_2(0x01, 0x01, vel_all)

                else:
                    self.slave_piper.MotionCtrl_2(0x01, 0x01, 20)
            else:
                self.slave_piper.MotionCtrl_2(0x01, 0x01, 20)

            self.slave_piper.JointCtrl(joint_0, joint_1, joint_2,
                                        joint_3, joint_4, joint_5)

            if self.gripper_exist:
                if len(joint_data.effort) >= 7:
                    gripper_effort = clip(joint_data.effort[6], 0.5, 3)
                    self.get_logger().info(f"gripper_effort: {gripper_effort}")
                    if not math.isnan(gripper_effort):
                        gripper_effort = round(gripper_effort * 1000)
                    else :
                        self.get_logger().warning("Gripper effort is NaN, using default value.")
                        gripper_effort = 0
                    self.slave_piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)
                else:
                    self.slave_piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def Slave_enable_callback(self,enable_flag: Bool):
        """
        Callback function for enabling the robotic arm
            Args:
                enable_flag (): Boolean flag
        """
        self.get_logger().info(f"Received enable flag:")
        self.get_logger().info(f"enable_flag: {enable_flag.data}")

        if enable_flag.data:
            self.__enable_flag = True
            self.slave_piper.EnableArm(7)
            if self.gripper_exist:
                self.slave_piper.GripperCtrl(0, 1000, 0x01, 0)

        else:
            self.__enable_flag = False
            self.slave_piper.DisableArm(7)
            if self.gripper_exist:
                self.slave_piper.GripperCtrl(0, 1000, 0x00, 0)


def main(args = None):
    rclpy.init(args=args)
    piper_slavemaster_node = PiperRosNode()
    try:
        rclpy.spin(piper_slavemaster_node)
    except KeyboardInterrupt:
        pass
    finally:
        piper_slavemaster_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
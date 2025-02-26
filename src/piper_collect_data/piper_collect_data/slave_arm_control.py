import collections
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

class RosOperator(Node):
    def __init__(self) -> None:
        super().__init__('slavearm_ctrl_node')
        self.master_arm_sub = self.create_subscription(JointState,
                                                       'master_joint_states',
                                                        self.master_arm_callback,
                                                       10)

        self.slave_arm_pub = self.create_publisher(JointState,'slave_piper_joint_ctrl',10)


    def master_arm_callback(self, msg):
        master_arm_joint_state = msg
        # joint_0 : float = master_arm_joint_state.position[0]
        # joint_1 : float = master_arm_joint_state.position[1]
        # joint_2 : float = master_arm_joint_state.position[2]
        # joint_3 : float = master_arm_joint_state.position[3]
        # joint_4 : float = master_arm_joint_state.position[4]
        # joint_5 : float = master_arm_joint_state.position[5]
        # joint_6 : float = master_arm_joint_state.position[6]
        #
        # vel_0 : float = master_arm_joint_state.velocity[0]
        # vel_1 : float = master_arm_joint_state.velocity[1]
        # vel_2 : float = master_arm_joint_state.velocity[2]
        # vel_3 : float = master_arm_joint_state.velocity[3]
        # vel_4 : float = master_arm_joint_state.velocity[4]
        # vel_5 : float = master_arm_joint_state.velocity[5]
        # vel_6 : float = master_arm_joint_state.velocity[6]
        eff_6 : float = master_arm_joint_state.effort[6]
        slave_arm_joint_state = JointState()
        slave_arm_joint_state.header.stamp = master_arm_joint_state.header.stamp
        slave_arm_joint_state.name = ['joint0',
                                      'joint1',
                                      'joint2',
                                      'joint3',
                                      'joint4',
                                      'joint5',
                                      'joint6',]
        slave_arm_joint_state.position = master_arm_joint_state.position[:-1]
        # slave_arm_joint_state.velocity = master_arm_joint_state.velocity[0.0]*7
        slave_arm_joint_state.velocity = [0.0] * 7
        # slave_arm_joint_state.velocity = master_arm_joint_state.velocity[:-1]
        # slave_arm_joint_state.velocity = [abs(x *1000) for x in slave_arm_joint_state.velocity]
        slave_arm_joint_state.effort = [0.0] * 7
        slave_arm_joint_state.effort[-1] = eff_6
        self.slave_arm_pub.publish(slave_arm_joint_state)

def main(args=None):
    rclpy.init(args=args)
    ros_operator = RosOperator()
    rclpy.spin(ros_operator)
    ros_operator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
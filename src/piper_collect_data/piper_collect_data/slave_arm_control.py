import collections
import time
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
import threading
class RosOperator(Node):
    def __init__(self) -> None:
        super().__init__('slavearm_ctrl_node')
        self.master_arm_sub = self.create_subscription(JointState,
                                                       'master_joint_states',
                                                        self.master_arm_callback,
                                                       10)

        self.slave_arm_pub = self.create_publisher(JointState,'slave_piper_joint_ctrl',10)
        self.slave_arm_joint_state = JointState()

        self.slave_arm_joint_state.name = ['joint0',
                                      'joint1',
                                      'joint2',
                                      'joint3',
                                      'joint4',
                                      'joint5',
                                      'joint6',]
        self.publisher_thread = threading.Thread(target=self.publish_thread)
        # self.publisher_thread.start()
        self.publisher_start =False
    def master_arm_callback(self, msg):
        if not self.publisher_start:
            self.publisher_start = True
            self.publisher_thread.start()
            # self.publisher_thread = threading.Thread(target=self.publish_thread)
        master_arm_joint_state = msg
        eff_6 : float = master_arm_joint_state.effort[6]
        self.slave_arm_joint_state.header.stamp = master_arm_joint_state.header.stamp
        self.slave_arm_joint_state.position = master_arm_joint_state.position[:-1]
        # slave_arm_joint_state.velocity = master_arm_joint_state.velocity[0.0]*7
        self.slave_arm_joint_state.velocity = [0.0] * 7
        # slave_arm_joint_state.velocity = master_arm_joint_state.velocity[:-1]
        # slave_arm_joint_state.velocity = [abs(x *1000) for x in slave_arm_joint_state.velocity]
        self.slave_arm_joint_state.effort = [0.0] * 7
        self.slave_arm_joint_state.effort[-1] = eff_6
        # self.slave_arm_pub.publish(slave_arm_joint_state)
    def publish_thread(self):
        rate =self.create_rate(200)
        time.sleep(3)
        while rclpy.ok():
            self.slave_arm_pub.publish(self.slave_arm_joint_state)
            rate.sleep()
def main(args=None):
    rclpy.init(args=args)
    ros_operator = RosOperator()
    rclpy.spin(ros_operator)
    ros_operator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
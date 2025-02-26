import collections
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

class RosOperator(Node):
    def __init__(self, args):

        self.master_arm_sub = self.create_subscription(JointState,args.master_arm_topic,
                                                        self.master_arm_callback,10)
        # self.slave_arm_pub = self.create_publisher()

    # def master_arm_callback(self, msg):

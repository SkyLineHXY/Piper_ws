import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import tf_transformations
import numpy as np
from piper_msgs.msg import PiperStatusMsg, PosCmd
angle_to_rad = np.pi/180
class StaticTFBroadcaster(Node):
    def __init__(self):
        super().__init__('pub_grasp_link')
        # 创建StaticTransformBroadcaster对象
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.Piper_Pose = PosCmd()
        # 定义静态TF变换
        self.static_transform = TransformStamped()

        # 设置父坐标系和子坐标系
        self.static_transform.header.frame_id = 'link6'  # 父坐标系
        self.static_transform.child_frame_id = 'grasp_link'  # 子坐标系

        # 设置变换的平移部分 (x, y, z)
        self.static_transform.transform.translation.x = 0.0  # 例如，x方向偏移0.1米
        self.static_transform.transform.translation.y = 0.0
        self.static_transform.transform.translation.z = 0.135


        quat = tf_transformations.quaternion_from_euler(-np.pi, -np.pi / 2, 0.0)  # 例如，无旋转
        self.static_transform.transform.rotation.x = quat[0]
        self.static_transform.transform.rotation.y = quat[1]
        self.static_transform.transform.rotation.z = quat[2]
        self.static_transform.transform.rotation.w = quat[3]

        # 设置时间戳
        self.static_transform.header.stamp = self.get_clock().now().to_msg()

        self.static_transform2 = TransformStamped()
        # 设置父坐标系和子坐标系
        self.static_transform2.header.frame_id = 'camera_link'  # 父坐标系
        self.static_transform2.child_frame_id = 'object_0'  # 子坐标系

        # 设置变换的平移部分 (x, y, z)
        self.static_transform2.transform.translation.x = 0.181  # 例如，x方向偏移0.5米
        self.static_transform2.transform.translation.y = -0.011
        self.static_transform2.transform.translation.z = -0.038

        quat2 = tf_transformations.quaternion_from_euler(0.737, 0.095, -0.105)  # 例如，无旋转
        self.static_transform2.transform.rotation.x = quat2[0]
        self.static_transform2.transform.rotation.y = quat2[1]
        self.static_transform2.transform.rotation.z = quat2[2]
        self.static_transform2.transform.rotation.w = quat2[3]
        # 设置时间戳
        self.static_transform2.header.stamp = self.get_clock().now().to_msg()
        # 发布两个静态TF变换
        self.tf_broadcaster.sendTransform([self.static_transform])

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
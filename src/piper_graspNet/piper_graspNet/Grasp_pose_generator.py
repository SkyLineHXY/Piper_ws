import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf_transformations import *
from cv_bridge import CvBridge
from piper_msgs.msg import PiperStatusMsg, PosCmd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import tf2_ros
import threading
import sys
import struct
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import os
GraspNet_path ='/home/zzq/Desktop/Open_Source_Project/graspnet-baseline'

sys.path.append(os.path.join(GraspNet_path, 'dataset'))
sys.path.append(os.path.join(GraspNet_path, 'utils'))
sys.path.append(GraspNet_path)
import graspnet_baseline



class GraspNet_Subscriber(Node):

    def __init__(self):
        super().__init__('realsense_subscriber')

        self.point_subcription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10)

        self.grasp_detect_srv = self.create_service(Trigger,
                                                '/Grasp_Pose_Get',
                                                self.grasp_detect_callback)

        self.bridge = CvBridge()

        # self.Piper_EndPose_Pub = self.create_publisher(PosCmd,"/end_pose",1)
        self.cv_rgb = None
        self.cv_depth = None
        self.Point_ros = None
        self.point_cloud_data = None
        self.gn = graspnet_baseline.GraspNetBaseLine(checkpoint_path='/home/zzq/Desktop/Open_Source_Project/GraspNet_Pointnet2_PyTorch1.13.1/logs/log_rs/checkpoint.tar',)
        self.gg=None

        self.grasp_transform = tf2_ros.TransformBroadcaster(self)
        self.base_cam_transform = None
        self.tf_pub_timer = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_reader_thread=threading.Thread(target=self.read_transform_thread)
        self.transform_reader_thread.start()

    def read_transform_thread(self):
        rate = self.create_rate(2)

        while rclpy.ok():
            self.read_tf_transform()
            rate.sleep()
    def read_tf_transform(self):
        try:
            self.base_cam_transform = self.tf_buffer.lookup_transform('base_link', 'camera_depth_optical_frame',rclpy.time.Time()) # 查询最新可用的变换)
        except TransformException as ex:
            self.get_logger().error(f"Could not transform camera_link to base_link: {ex}")
    def grasp_detect_callback(self, request, response):
        #获取抓取位姿服务请求
        if self.point_cloud_data is None:
            response.success = False
            response.message = "No point cloud data available."
            self.get_logger().warn("No point cloud data available.")
            return response
        if self.base_cam_transform is None:
            response.success = False
            response.message = "No base_link to camera_link transform available."
            self.get_logger().warn("No base_link to camera_link transform available.")
            return response
        self.gg = self.generate_grasp_pose(self.point_cloud_data)

        tf_pub_timer = self.create_timer(1, self.GraspNet_timer)
        response.success = True
        response.message = "Grasp pose generated and TF published."

        return response
    def generate_grasp_pose(self, point_cloud_data: PointCloud2):
        #调用graspnet生成抓取姿态
        points = []
        colors = []
        for point in point_cloud2.read_points(point_cloud_data, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            points.append([x, y, z])
            rgb_float = point[3]
            packed_rgb = struct.pack('f', rgb_float)# 使用struct将浮点型的rgb解码成int
            rgb_int = struct.unpack('I', packed_rgb)[0]
            r = (rgb_int >> 16) & 0x0000ff
            g = (rgb_int >> 8) & 0x0000ff
            b = rgb_int & 0x0000ff
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # 归一化到[0,1]

        np_points = np.array(points, dtype=np.float32)
        np_colors = np.array(colors, dtype=np.float32)
        # 转换为Open3D点云格式
        o3d_origin_cloud = o3d.geometry.PointCloud()
        o3d_origin_cloud.points = o3d.utility.Vector3dVector(np_points)
        o3d_origin_cloud.colors = o3d.utility.Vector3dVector(np_colors)
        # o3d.visualization.draw_geometries([o3d_origin_cloud])
        self.gg = self.gn.predict(o3d_origin_cloud)

        print(self.gg)
        return self.gg

    def pointcloud_callback(self, msg:PointCloud2):
        self.point_cloud_data = msg

    def GraspNet_timer(self):
        i=0
        for grasp in self.gg:

            grasp_from_camera_pose = np.eye(4)
            grasp_from_camera_pose[:3,:3]=grasp.rotation_matrix
            grasp_from_camera_pose[:3,3]=grasp.translation
            camera_from_base_pose = np.eye(4)
            base_cam_transform_quat =np.array([self.base_cam_transform.transform.rotation.x,
                                              self.base_cam_transform.transform.rotation.y,
                                              self.base_cam_transform.transform.rotation.z,
                                              self.base_cam_transform.transform.rotation.w])
            base_cam_transform_rotation = R.from_quat(base_cam_transform_quat).as_matrix()
            camera_from_base_pose[:3,:3]=base_cam_transform_rotation
            camera_from_base_pose[:3,3]=self.base_cam_transform.transform.translation
            grasp_base_pose = camera_from_base_pose @ grasp_from_camera_pose
            grasp_base_translation = grasp_base_pose[:3,3]
            grasp_base_rotation = grasp_base_pose[:3,:3]

            # print(grasp_from_base_pose)
            transform = TransformStamped()

            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "camera_depth_optical_frame"
            transform.child_frame_id = f"object_{i}"


            transform.transform.translation.x = float(grasp.translation[0])
            transform.transform.translation.y = float(grasp.translation[1])
            transform.transform.translation.z = float(grasp.translation[2])

            rotation = R.from_matrix(grasp.rotation_matrix)
            quaternion = rotation.as_quat()  # 返回的顺序是 (x, y, z, w)

            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]

            transform2 = TransformStamped()
            transform2.header.stamp = self.get_clock().now().to_msg()

            transform2.header.frame_id = "base_link"
            transform2.child_frame_id = f"object_{i}_in_base"
            transform2.transform.translation.x = grasp_base_translation[0]
            transform2.transform.translation.y = grasp_base_translation[1]
            transform2.transform.translation.z = grasp_base_translation[2]

            quaternion_graspbase = R.from_matrix(grasp_base_rotation).as_quat()

            transform2.transform.rotation.x =quaternion_graspbase[0]
            transform2.transform.rotation.y =quaternion_graspbase[1]
            transform2.transform.rotation.z =quaternion_graspbase[2]
            transform2.transform.rotation.w =quaternion_graspbase[3]


            self.grasp_transform.sendTransform([transform,transform2])
            i += 1
def main(args=None):
    rclpy.init(args=args)
    GraspPose_Subscriber = GraspNet_Subscriber()
    rclpy.spin(GraspPose_Subscriber)
    GraspPose_Subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
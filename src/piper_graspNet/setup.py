from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'piper_graspNet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzq',
    maintainer_email='2391384401@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "Grasp_pose_generator_Node = piper_graspNet.Grasp_pose_generator:main",
            "PickandPlace_Control_Node = piper_graspNet.Grasp:main",
            "Grasplink_tf_pub_Node =piper_graspNet.grasplink_tf_pub:main"
        ],
    },
)

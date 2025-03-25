from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'piper_collection_pkg'
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzq',
    maintainer_email='2391384401@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts':[
            'piper_state_get_node = piper_collection_pkg.piper_state_get_node:main',
            'slave_arm_control_node = piper_collection_pkg.slave_arm_follow_control:main',
            'piper_slave_master_ctrl = piper_collection_pkg.piper_ctrl_slavemaster_node:main',
        ],
    },
)

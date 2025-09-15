from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'DualPiper_Collect_Pkg'

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
        'console_scripts': [
            'DualPiper_StateGet_Node = DualPiper_Collect_Pkg.DualPiper_StateGet_Node:main ',
            'DualArm_DataToHDF5_Node=DualPiper_Collect_Pkg.DualArm_DataToHDF5_Node:main'
        ],
    },
)

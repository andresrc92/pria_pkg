from setuptools import find_packages, setup

package_name = 'pria'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andres',
    maintainer_email='andres@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'commander = pria.robot_commander:main',
            'inference = pria.robot_inference:main',
            'usb_camera = pria.usb_camera:main',
            'data_collection = pria.data_collection:main',
            'object_detection = pria.object_detection_node:main',
            'gripper_command = pria.gripper_command:main',
        ],
    },
)

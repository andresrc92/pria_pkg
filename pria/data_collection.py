import rclpy
import rclpy.logging
from rclpy.node import Node

import random
import math
import numpy as np
import os
import yaml
import time
from datetime import datetime

import cv2
from cv_bridge import CvBridge

from std_msgs.msg import String, Int32
from tf2_msgs.msg import TFMessage
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO

from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform, TransformStamped, Quaternion

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from message_filters import TimeSynchronizer

from pria.rotations import *

class DataCollection(Node):

    def __init__(self):
        super().__init__('data_collection')
        
        self.setio = self.create_client(SetIO, '/io_and_status_controller/set_io')
        # while not self.setio.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()
        self.send_request()
        self.start_time = rclpy.time.Time()

        self.declare_parameter('sim',True)
        self.sim_parameter = self.get_parameter('sim').get_parameter_value().bool_value
        
        self.declare_parameter('folder','folder')
        self.directory = self.get_parameter('folder').get_parameter_value().string_value

        self.image_count = 0

        self.image_height = 240
        self.image_width = 320

        self.param_exist = True

        if self.directory == 'folder':
            self.get_logger().error("Folder parameter not set.")
            raise SystemExit

        self.data_paths = ['flat_no_twist','flat_twist','cone_no_twist','cone_twist']
        self.data_index = 0

        self.parent = os.getcwd()
        self.path = os.path.join(self.parent,self.directory)

        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass
        
        self.get_logger().info('Dataset folder in {}'.format(self.path))


        self.io_subscription = self.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_listener_callback,
            10)
        self.io_subscription  # prevent unused variable warning

        self.rgb_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.rgb_listener_callback,
            10)
        self.rgb_subscription  # prevent unused variable warning

        self.bridge = CvBridge()
       
        self.publisher_ = self.create_publisher(String, '/urscript_interface/script_command', 10)
        self.gripper_pub = self.create_publisher(Int32, 'gripper', 10)

        timer_period = 0.3 # seconds

        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.point_count = 0
        self.inMotion = False
        self.inMotionPrev = False
        self.arrivedToPoint = False

        self.state_machine = 0
        self.index = 0

        self.gt = {}

        self.initial_pose = Transform()
        self.initial_matrix = np.eye(4)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)


    def send_request(self):
        """
        This function uses the SetIO service to put DO1 to Low before starting
        """
        self.req.fun = 1 # FUN_SET_DIGITAL_OUT
        self.req.pin = 0
        self.req.state = 0.0
        self.future = self.setio.call_async(self.req)

    def timer_callback(self):
        """
        This is called every ${timer_period} seconds
        """
        # print("Maquina de estados: ", self.state_machine)
        if self.state_machine == 0:

            if self.publish_first_pose() < 0:
                print("Waiting for transform...")
                return
            
            if self.inMotion == False and len(self.gt) == 1:
                self.state_machine = 1
            
        if self.state_machine == 1:

            match self.data_index:
                case 0:
                    self.get_logger().info('Flat surface with no twist')
                    self.points = self.generate_flat_points(60)
                case 1:
                    self.get_logger().info('Flat surface with twist')
                    self.points = self.generate_flat_points(60, twist=True)
                case 2:
                    self.get_logger().info('Cone surface with no twist')
                    self.points = self.generate_cone_points(80)
                case 3:
                    self.get_logger().info('Cone surface with twist')
                    self.points = self.generate_cone_points(80, twist=True)

            self.generate_paths(self.data_paths[self.data_index])
            self.limit = len(self.points)

            if self.inMotion == False:
                self.state_machine = 2

        if self.state_machine == 2:
            if self.point_count < self.limit:
                self.handle_next_pose(self.point_count)
                self.state_machine = 3
            else:
                self.state_machine = 4

        if self.state_machine == 3:
            if self.arrivedToPoint:
                self.state_machine = 2
                self.arrivedToPoint = False
                self.point_count += 1
        
        if self.state_machine == 4:
            self.back_to_first_pose()

            print("Took ", self.image_count, " images")
            print("Finish time ", datetime.now())
            
            self.gt['initial_pose']['total_images'] = self.image_count

            with open(self.gt_path, 'w') as file:
                yaml.dump(self.gt, file, default_flow_style=False)
            
            self.gt = {}

            if self.data_index < len(self.data_paths) - 1:
                self.data_index += 1
                self.state_machine = 5
                self.point_count = 0
                self.image_count = 0
            else:
                raise SystemExit
        
        if self.state_machine == 5:
            if self.inMotion == False:
                self.state_machine = 0
        

    def generate_paths(self, dataset_path):
        p = os.path.join(self.path, dataset_path)
        try:
            os.mkdir(p)
        except FileExistsError:
            pass

        self.img_path = os.path.join(self.path, dataset_path, 'imgs')
        try:
            os.mkdir(self.img_path)
        except FileExistsError:
            pass

        self.gt_path = os.path.join(self.path, dataset_path, 'gt.yaml')

    def io_listener_callback(self, msg):
        """
        This function checks DO1 Falling and Rising edges
        """

        # robot motion flag
        self.inMotion = msg.digital_out_states[0].state

        if not self.inMotion and self.inMotionPrev:
            # Falling edge
            self.arrivedToPoint = True

        self.inMotionPrev = self.inMotion

        # gripper control in Isaac Sim
        if msg.digital_out_states[1].state:
            self.gripper_pub.publish(Int32(data=1))
        else:
            self.gripper_pub.publish(Int32(data=-1))

    def rgb_listener_callback(self, msg):
        """
        This function listens to rgb topic
        """
        # self.get_logger().info('I heard: "%s"' % msg.height)
        image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        # self.color_filter(self.cv_image)    
        img_time = msg.header.stamp

        if self.state_machine == 2 or self.state_machine == 3:

            if self.sim_parameter:
                wrist = 'wrist_3_link_sim'
            else:
                wrist = 'wrist_3_link'

            translation, rotation = self.lookup_transform_('base_link_inertia', wrist)

            actual_matrix = self.create_transformation_matrix(rotation, translation)
            actual_inverse = np.linalg.inv(actual_matrix)

            
            pose_relative_to_start = np.dot(actual_inverse,self.initial_matrix)
            

            T = pose_relative_to_start[:3,3].tolist()
            r = Rotations()
            r.from_matrix(pose_relative_to_start[0:3,0:3])
            Q = r.as_quat().tolist()


            self.publish_tf(np.concatenate((T, Q)),wrist,'gt')

            if np.sum(Q) != 0 and np.sum(T) != 0:
                self.gt.update({
                    self.image_count: {
                        'translation':T,
                        'rotation':Q
                    }
                })
                
                filename = '{}.png'.format(self.image_count)

                resized = cv2.resize(image, (self.image_width, self.image_height))
                self.capture_and_save_image(resized, os.path.join(self.img_path, filename))
                # self.state = 0
                if self.image_count == 0:
                    print("Starting time ", datetime.now())
                print("Image ", self.image_count, " Point ", self.point_count, "/", len(self.points), end='\r')
                self.image_count += 1


    def color_filter(self, frame):
        """
        This function is a test of blue filtering with OpenCV
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
        # Threshold of blue in HSV space 
        lower_blue = np.array([60, 35, 140]) 
        upper_blue = np.array([180, 255, 255]) 
    
        # preparing the mask to overlay 
        mask = cv2.inRange(hsv, lower_blue, upper_blue) 
        
        # The black region in the mask has the value of 0, 
        # so when multiplied with original image removes all non-blue regions 
        result = cv2.bitwise_and(frame, frame, mask = mask) 
    
        cv2.imshow('frame', frame) 
        cv2.imshow('mask', mask) 
        cv2.imshow('result', result)
        cv2.waitKey(1)
    
    def capture_and_save_image(self, frame, filename):
        """
        Save the frame as a JPEG file
        Takes arguments (frame=self.cv_image, filename)
        """
        cv2.imwrite(filename, frame)

    def create_transformation_matrix(self, quaternion, translation):
        """
        Create a transformation matrix from a quaternion and translation vector.
        
        Args:
        quaternion (list or np.ndarray): The quaternion [x, y, z, w]
        translation (list or np.ndarray): The translation vector [tx, ty, tz]

        Returns:
        np.ndarray: The 4x4 transformation matrix
        """
        x, y, z, w = quaternion
        r = Rotations()
        r.from_quat(x,y,z,w)

        R = r.as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    def publish_first_pose(self):
        """
        Publish a static frame with respect to the robot base of the first pose
        """
        if self.sim_parameter:
            wrist = 'wrist_3_link_sim'
        else:
            wrist = 'wrist_3_link'

        translation, rotation = self.lookup_transform_('base_link_inertia', wrist)

        if np.sum(translation) == 0:
            print("Error: Transform not found")
            return -1

        # if (self.initial_pose.translation.x == 0 or self.initial_pose.translation.y == 0 or self.initial_pose.translation.z == 0):   
        self.initial_pose.translation.x = translation[0]
        self.initial_pose.translation.y = translation[1]
        self.initial_pose.translation.z = translation[2]

        r = Rotations()
        r.from_quat(rotation[0],rotation[1],rotation[2],rotation[3])
        self.initial_rotvec = r.as_rotvec()
        self.initial_matrix = self.create_transformation_matrix(rotation, translation)

        self.gt.update({
            'initial_pose': {
                'translation':translation,
                'rotation':rotation,
                'total_images': self.image_count,
                'height':self.image_height,
                'width':self.image_width
            }
        })
            
        self.publish_tf(np.concatenate((translation, rotation)),'base_link_inertia','initial_pose', True)

        return 0


    def generate_flat_points(self, r = 50, twist=False):
        x = np.arange(-r, r, 5) / 1000
        y = np.arange(-r, r, 5) / 1000

        points = []

        for i in range(len(x)):

            if twist:
                r = Rotations()
                r.from_euler(0, 0, random.randrange(-50, 50, 1) / 100 )
                q = r.as_quat()
            else:
                q = [0.0, 0.0, 0.0, 1.0]
                
            z = 0.0

            if i % 2 == 0:
                points.append([x[i], y[0], z, q[0], q[1], q[2], q[3]])
                points.append([x[i], y[-1], z, q[0], q[1], q[2], q[3]])
            else:
                points.append([x[i], y[-1], z, q[0], q[1], q[2], q[3]])
                points.append([x[i], y[0], z, q[0], q[1], q[2], q[3]])
        
        points.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        for i, point in enumerate(points):
            self.publish_tf(point, 'initial_pose', '{}'.format(i))
        
        return points


    def generate_cone_points(self, r = 50, z = 150.0, twist=False):
        x = np.arange(-r, r, 35) / 1000
        y = np.arange(-r, r, 35) / 1000

        z /= 1000
        z *= -1

        points = []

        for i in x:
            for j in y:

                if twist:
                    r = Rotations()
                    r.from_euler(0, 0, random.randrange(-50, 50, 1) / 100 )
                    q = r.as_quat()
                else:
                    q = [0.0,0.0,0.0,1.0]

                points.append([i,j,z,q[0],q[1],q[2],q[3]])
                points.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        
        for i, point in enumerate(points):
            self.publish_tf(point, 'initial_pose', '{}'.format(i))

        return points


    def handle_next_pose(self, index):
        t_next_pose = self.points[index][:3]
        q_next_pose = self.points[index][3:]

        self.publish_tf(self.points[index], 'initial_pose', 'point_{}'.format(index))
        # self.get_logger().info('offset transform {} {} {}'.format(t_next_pose[0],t_next_pose[1],t_next_pose[2]))

        next_pose_matrix = self.create_transformation_matrix(q_next_pose, t_next_pose)

        base_to_next_matrix = np.dot(self.initial_matrix,next_pose_matrix)

        r = Rotations()
        r.from_matrix(base_to_next_matrix[0:3,0:3])
        next_rotation = r.as_rotvec()

        self.send_urscript(base_to_next_matrix[:3,3], next_rotation)

    def send_urscript(self, translation, rotation):
        """
        Using the primary interface to send URScripts programs to move the robot
        """
        msg = String()
        msg.data = """def my_prog():\nset_digital_out(0, True)\nmovej(p[{},{},{},{},{},{}], a=0.1, v=0.08, r=0)\nset_digital_out(0, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
        self.publisher_.publish(msg)
        # self.get_logger().info(msg.data)

    def back_to_first_pose(self):
        self.send_urscript(self.initial_matrix[:3,3], self.initial_rotvec)

    def publish_tf(self, pose, head, child, static=False):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = head
        t.child_frame_id = child       
        
        # We create the translation
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = pose[2]

        t.transform.rotation.x = pose[3]
        t.transform.rotation.y = pose[4]
        t.transform.rotation.z = pose[5]
        t.transform.rotation.w = pose[6]

        # Send the transformation
        if static:
            self.tf_static_broadcaster.sendTransform(t)
        else:
            self.tf_broadcaster.sendTransform(t)

    def lookup_transform_(self, to_frame_rel, from_frame_rel, time=rclpy.time.Time()):
        """
        Search for transformations between two specific frames.
        
        If array is True, it returns a translation and rotation array (w = q[3])
        """

        try:
            t = self.tf_buffer.lookup_transform(
            to_frame_rel,
            from_frame_rel,
            time,
            timeout=rclpy.duration.Duration(seconds=1.0))

            translation = [t.transform.translation.x, t.transform.translation.y,t.transform.translation.z]
            rotation = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

            #todo: put this somewhere else
            if (self.initial_pose.translation.x == 0 or self.initial_pose.translation.y == 0 or self.initial_pose.translation.z == 0):   
                self.initial_pose = t.transform
                r = Rotations()
                r.from_object(t.transform.rotation)
                self.initial_rotvec = r.as_rotvec()
                self.initial_matrix = self.create_transformation_matrix(rotation, translation)
                self.publish_first_pose()
                self.gt.update({
                    'initial_pose': {
                        'translation':translation,
                        'rotation':rotation,
                        'total_images': self.image_count,
                        'height':self.image_height,
                        'width':self.image_width
                    }
                })
                # with open('./test.yaml', 'w') as file:
                #     yaml.dump(self.gt, file,  default_flow_style=False)
            #
            
            # print("C ", t.header.stamp, "\n")
            return translation, rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return [0.0,0.0,0.0], [0.0,0.0,0.0,0.0]


def main(args=None):
    rclpy.init(args=args)

    robot_commander = DataCollection()

    try:
        rclpy.spin(robot_commander)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info('done')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot_commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
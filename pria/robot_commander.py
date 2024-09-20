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

from std_msgs.msg import String
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

class RobotCommander(Node):

    def __init__(self):
        super().__init__('robot_commander')
        
        self.setio = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.setio.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()
        self.send_request()
        self.start_time = rclpy.time.Time()
        
        self.declare_parameter('folder','folder')
        self.directory = self.get_parameter('folder').get_parameter_value().string_value

        self.declare_parameter('images_count',0)
        self.image_count_max = self.get_parameter('images_count').get_parameter_value().integer_value
        self.image_count = 0

        self.points = np.empty((self.image_count_max,7),dtype=float)

        self.image_height = 240
        self.image_width = 320

        self.param_exist = True
        if self.directory == 'folder':
            print("Folder parameter not set.")
            self.param_exist = False

        if self.image_count_max < 1:
            print("Image count not set.")
            self.param_exist = False

        if self.param_exist:
            self.parent = os.getcwd()
            self.path = os.path.join(self.parent,self.directory)
            self.img_path = os.path.join(self.path, 'imgs')

            try:
                os.mkdir(self.path)
            except FileExistsError:
                pass

            try:
                os.mkdir(self.img_path)
            except FileExistsError:
                pass

            self.gt_path = os.path.join(self.path,'gt.yaml')
            
            self.get_logger().info('Dataset folder in {}'.format(self.path))

        # self.tf_subscription = self.create_subscription(
        #     TFMessage,
        #     'tf',
        #     self.tf_listener_callback,
        #     10)
        # self.tf_subscription  # prevent unused variable warning

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

        timer_period = 01.1 # seconds

        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.inMotion = False
        self.inMotionPrev = False

        self.state = 0
        self.index = 0
        self.gt = {}

        self.initial_pose = Transform()
        self.initial_matrix = np.eye(4)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # while self.publish_first_pose() < 0:
        #     print("Waiting for transform.")
        self.once = True
        # self.generate_close_points(60)
        self.generate_cone_points(80)

    def send_request(self):
        """
        This function uses the SetIO service to put DO1 to Low before starting
        """
        self.req.fun = 1 # FUN_SET_DIGITAL_OUT
        self.req.pin = 1
        self.req.state = 0.0
        self.future = self.setio.call_async(self.req)

    def timer_callback(self):
        """
        This is called every ${timer_period} seconds
        """
        if self.once:
            if self.publish_first_pose() == 0:
                self.once = False

        limit = 2
        limit = len(self.close_points)

        if (self.state == 0 or self.state == 2) and self.i < limit and not self.once:
            self.handle_next_pose(self.i)
            print("Movedd " + str(self.state) + " \n")
            self.state = 1

        if self.i >= limit:
            # and exit
            self.back_to_first_pose()
            print("Took ", self.image_count, " images")
            print("Finish time ", datetime.now())
            self.gt['initial_pose']['total_images'] = self.image_count
            with open(self.gt_path, 'w') as file:
                yaml.dump(self.gt, file, default_flow_style=False)        
            raise SystemExit

        # if self.once:
        #     if self.publish_first_pose() == 0:
        #         self.once = False
        #         self.generate_points()

        # elif self.param_exist:
        #     if self.image_count_max > self.image_count:
        #         if not self.inMotion and self.state == 0:
        #             self.handle_next_pose(self.image_count)
        #             self.state = 1
        #     else:
        #         #go back to initial pose
        #         self.back_to_first_pose()
        #         print("Finish time ", datetime.now())
                
        #         #save yaml file with ground truth poses
        #         with open(self.gt_path, 'w') as file:
        #             yaml.dump(self.gt, file,  default_flow_style=False)

        #         for i in range(5):
        #             os.system('spd-say "your program has finished"')
        #             time.sleep(3)

        #         # and exit
        #         raise SystemExit
        #     return
        
        # else:
        #     print("doing nothing")

    def tf_listener_callback(self, msg):
        """
        This function listens to tf
        """
        total_transforms = len(msg.transforms)
        for transform in msg.transforms:
            # self.get_logger().info('I heard: "%s"' % transform.header.frame_id)
            break

    def io_listener_callback(self, msg):
        """
        This function checks DO1 Falling and Rising edges
        """

        self.inMotion = True if msg.digital_out_states[1].state == True else False

        # if self.inMotion and not self.inMotionPrev:
            # Rising edge

        if not self.inMotion and self.inMotionPrev:
            # Falling edge
            self.state = 2
            self.i += 1

        self.inMotionPrev = self.inMotion
        # self.get_logger().info('In motion "%s"' % self.inMotion)

    def rgb_listener_callback(self, msg):
        """
        This function listens to rgb topic
        """
        # self.get_logger().info('I heard: "%s"' % msg.height)
        image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        # self.color_filter(self.cv_image)    
        img_time = msg.header.stamp

        if self.state == 2 or (self.state == 1 and self.image_count == 0) or True:
            # Get current transform for the image ground truth
            # print("A", msg.header.stamp)
            # print("B", self.get_clock().now())
            # print("AA ", rclpy.time.Time())

            #If using Simulation get tf 1 second ago to compensate
            # T, Q = self.lookup_transform_('wrist_3_link', 'initial_pose')
            # T, Q = self.lookup_transform_('wrist_3_link_sim', 'initial_pose')
            T, Q = self.lookup_transform_('base_link_inertia','wrist_3_link_sim')

            actual_matrix = self.create_transformation_matrix(Q,T)
            actual_inverse = np.linalg.inv(actual_matrix)

            
            pose_relative_to_start = np.dot(actual_inverse,self.initial_matrix)
            

            T = pose_relative_to_start[:3,3].tolist()
            r = Rotations()
            r.from_matrix(pose_relative_to_start[0:3,0:3])
            Q = r.as_quat().tolist()


            self.publish_tf(np.concatenate((T, Q)),'wrist_3_link_sim','gt')

            # print("Q ", Q)

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
                print("Image ", self.image_count, " Point ", self.i, "/", len(self.close_points), end='\r')
                self.image_count += 1

        else:
            #save yaml file with ground truth poses
            print("Finish time ", datetime.now())
            with open(self.gt_path, 'w') as file:
                yaml.dump(self.gt, file, default_flow_style=False)            
            raise SystemExit


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
        # translation, rotation = self.lookup_transform_('base_link_inertia', 'wrist_3_link')
        translation, rotation = self.lookup_transform_('base_link_inertia', 'wrist_3_link_sim')

        if np.sum(translation) == 0:
            return -1
        
        print("Found transform!")

        if (self.initial_pose.translation.x == 0 or self.initial_pose.translation.y == 0 or self.initial_pose.translation.z == 0):   
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
                    'total_images': self.image_count_max,
                    'height':self.image_height,
                    'width':self.image_width
                }
            })
            
        self.publish_tf(np.concatenate((translation, rotation)),'base_link_inertia','initial_pose', True)

        return 0

    # def generate_points(self, z_far=10,z_step=1,r_max=100,r_min=100):

    #     if self.image_count_max < 1:
    #         print("Image count not set")
    #         return

    #     zd = z_far / z_step # Interval
    #     z_array = np.arange(z_step) * zd + zd / 2

    #     r_array = r_min + (r_max - r_min) * (np.arange(z_step) / z_step)
    #     a = np.power(r_array,2)
    #     a_percentage = a / np.sum(a)

    #     points_index = 0

    #     images_per_step = self.image_count_max * a_percentage
    #     images_per_step = np.round(images_per_step)

    #     images_per_step[0] += self.image_count_max - np.sum(images_per_step)


    #     for i in range(z_step):

    #         # print(int(images_per_step[0]))

    #         for j in range(int(images_per_step[i])):
    #             x = random.randrange(-r_array[i], r_array[i], 1) / 1000
    #             y = random.randrange(-r_array[i], r_array[i], 1) / 1000
    #             z = - random.randrange(z_array[i] - zd / 2, z_array[i] + zd / 2, 1) / 1000

    #             # Create a quaternion from euler angles
    #             r = Rotations()
    #             r.from_euler(0, 0, random.randrange(-30, 30, 1) / 100 )
    #             q = r.as_quat()


    #             self.points[points_index] = np.array([x,y,z,q[0],q[1],q[2],q[3]])
    #             # print(self.points[points_index])
    #             self.publish_tf(self.points[points_index], 'initial_pose', 'point_{}'.format(points_index))
    #             points_index += 1
        
    #     self.points[-1] = [0.0,0.0,0.0,0.0,0.0,0.0,1.0]
    #     self.points = self.points[(-self.points[:,2]).argsort()]

    #     self.points_list = self.points.tolist()
    #     self.points = np.array(self.sort_points(self.points_list))


    def generate_close_points(self, r = 50):
        x = np.arange(-r, r, 5) / 1000
        y = np.arange(-r, r, 5) / 1000

        self.close_points = []

        for i in range(len(x)):

            r = Rotations()
            r.from_euler(0, 0, random.randrange(-50, 50, 1) / 100 )
            q = r.as_quat()
            q = [0.0,0.0,0.0,1.0]
            z = 0.0

            if i % 2 == 0:
                self.close_points.append([x[i], y[0],z,q[0],q[1],q[2],q[3]])
                self.close_points.append([x[i], y[-1],z,q[0],q[1],q[2],q[3]])
            else:
                self.close_points.append([x[i], y[-1],z,q[0],q[1],q[2],q[3]])
                self.close_points.append([x[i], y[0],z,q[0],q[1],q[2],q[3]])
        
        self.close_points.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0])

        # for i, point in enumerate(self.close_points):
        #     self.publish_tf(point, 'initial_pose', '{}'.format(i))


    def generate_cone_points(self, r = 50, z = 150.0, twist=False):
        x = np.arange(-r, r, 35) / 1000
        y = np.arange(-r, r, 35) / 1000

        z /= 1000
        z *= -1

        self.close_points = []

        for i in x:
            for j in y:

                if twist:
                    r = Rotations()
                    r.from_euler(0, 0, random.randrange(-50, 50, 1) / 100 )
                    q = r.as_quat()
                else:
                    q = [0.0,0.0,0.0,1.0]

                self.close_points.append([i,j,z,q[0],q[1],q[2],q[3]])
                self.close_points.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        
        for i, point in enumerate(self.close_points):
            self.publish_tf(point, 'initial_pose', '{}'.format(i))



    def handle_next_pose(self, index):
        # t_next_pose = self.points[index,:3]
        # q_next_pose = self.points[index,3:]
        t_next_pose = self.close_points[index][:3]
        q_next_pose = self.close_points[index][3:]

        self.publish_tf(self.close_points[index], 'initial_pose', 'point_{}'.format(index))
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
        msg.data = """def my_prog():\nset_digital_out(1, True)\nmovej(p[{},{},{},{},{},{}], a=0.1, v=0.08, r=0)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
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
                        'total_images': self.image_count_max,
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

    robot_commander = RobotCommander()

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
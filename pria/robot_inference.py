import rclpy
import rclpy.logging
from rclpy.node import Node

import random
import math
import numpy as np
import os
import yaml

import cv2
from cv_bridge import CvBridge

from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO

from pria.rotations import *
from pria.sim_model import *

from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform, TransformStamped, Quaternion

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class RobotInferencer(Node):

    def __init__(self):
        super().__init__('robot_inferencer')
        
        self.setio = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.setio.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()
        self.send_request()
        self.start_time = rclpy.time.Time()
        
        # self.declare_parameter('folder','folder')
        # self.directory = self.get_parameter('folder').get_parameter_value().string_value

        # self.declare_parameter('images_count',0)
        # self.image_count_max = self.get_parameter('images_count').get_parameter_value().integer_value
        # self.image_count = 0

        self.image_height = 240
        self.image_width = 320

        # self.parent = os.getcwd()
        # self.path = os.path.join(self.parent,self.directory)
        # self.img_path = os.path.join(self.path, 'imgs')
        # try:
        #     os.mkdir(self.path)
        # except FileExistsError:
        #     pass

        # try:
        #     os.mkdir(self.img_path)
        # except FileExistsError:
        #     pass

        # self.gt_path = os.path.join(self.path,'gt.yaml')
        
        # self.get_logger().info('Dataset folder in {}'.format(self.path))

        self.tf_subscription = self.create_subscription(
            TFMessage,
            'tf',
            self.tf_listener_callback,
            10)
        self.tf_subscription  # prevent unused variable warning

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
        # self.cap = cv2.VideoCapture(0)

        self.publisher_ = self.create_publisher(String, '/urscript_interface/script_command', 10)
        timer_period = 1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.inMotion = False
        self.inMotionPrev = False
        self.homed = False

        # self.state = 0
        self.state = 1
        self.index = 0
        self.gt = {}

        self.initial_pose = Transform()
        self.initial_matrix = np.eye(4)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # self.trainer = Trainer("vibrator_no_twist", 2, 100) # el que anda
        # self.trainer = Trainer("extrusion", 2, 100) # el que anda
        # self.trainer = Trainer("cup_drill_cone_twist", 2, 100) # el que anda
        # self.z_end = self.trainer.read_meta_data()
        # self.trainer.open_model()
        # # self.trainer.open_normalization()
        # self.trainer.open_batch_normalization()

        # self.flat_model = Trainer("cup_drill_twist", 2, 100)
        self.flat_model = Trainer("black_cube_2", 2, 100)
        self.flat_model.open_model()
        self.flat_model.open_batch_normalization()

        # self.move_to_first_pose()
        # print("Initial height ", self.z_end)

        # self.send_home()

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
        return


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
            # self.state = 2

        if not self.inMotion and self.inMotionPrev:
            self.homed = True
            # Falling edge

        self.inMotionPrev = self.inMotion
        # self.get_logger().info('In motion "%s"' % self.inMotion)

    def rgb_listener_callback(self, msg):
        """
        This function listens to rgb topic
        """
        # self.get_logger().info('I heard: "%s"' % msg.height)
        image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        # self.color_filter(image)

        # pose = self.trainer.infer_from_image(image)
        # pose = self.flat_model.infer_from_image(image)

        if (self.homed or True) and not self.inMotion:
            # if self.state == 0:
                # pose = self.trainer.infer_from_image(image)
                # self.handle_predicted_pose(pose)
            if self.state == 1 or True:
                pose = self.flat_model.infer_from_image(image)
                print(pose[:3])
                self.handle_predicted_pose(pose)
            elif self.state == 2 and not self.inMotion:
                self.state = 3
                print("Done!")
                # self.send_grip()
                

            

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
        x,y,z,w = quaternion
        r = Rotations()
        r.from_quat(x,y,z,w)
        R = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T
    
    def handle_predicted_pose(self, prediction,):
        t_current, q_current = self.lookup_transform('base_link_inertia', 'wrist_3_link_sim', True)
        
        t_next_pose = 1 * prediction[:3]
        if self.state == 1:
            # t_next_pose = -1 * prediction[:3] #only for vibrator
            t_next_pose[2] = 0
        # t_next_pose[2] = 0.0
        # print("Trans_pred ", t_next_pose)

        q_next_pose = prediction[3:]
        # q_next_pose[3] *= -1

        # q_next_pose = [0,0,0,1]

        # self.publish_tf(np.concatenate((t_next_pose, q_next_pose)), 'wrist_3_link', 'predicted_pose')
        
        next_pose_matrix = self.create_transformation_matrix(q_next_pose,t_next_pose)
        current_pose_matrix = self.create_transformation_matrix(q_current,t_current)

        base_to_next_matrix = np.dot(current_pose_matrix,next_pose_matrix)

        r = Rotations()
        r.from_matrix(base_to_next_matrix[0:3,0:3])
        q_ = r.as_quat()
        t_ = base_to_next_matrix[:3,3]
        prediction_ = np.concatenate((t_, q_))

        self.publish_tf(prediction_, 'base_link_inertia', 'robot_command')

        dist = np.power(prediction[:3],2)
        dist = np.sum(dist)

        # r.from_quat(q_next_pose[0],q_next_pose[1],q_next_pose[2],q_next_pose[3])

        # r.from_quat(0,0,0,1)

        # If robot is close enough, stop sending commands

        # print(prediction[:3])
        # print(dist)
        if not self.inMotion:

            # print("State ", self.state)
            if self.state == 0:
                # print(dist, t_[2])
                if dist > 0.0001 and t_[2] > 0.35:
                    # if t_[2] > (self.z_end - 0.05):
                    self.send_urscript(t_, r.as_rotvec())
                else:
                    self.state = 1

            elif self.state == 1:
                if dist > 0.0001:
                    self.send_urscript(t_, r.as_rotvec())
                else:
                    self.state = 2
            
        # else:
        #     raise SystemExit

    def publish_tf(self, pose, head, child):
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
        self.tf_broadcaster.sendTransform(t)


    def move_to_first_pose(self):
        pose = self.flat_model.get_initial_pose()

        r = Rotations()
        r.from_array(pose[3:])

        self.send_urscript(pose[:3], r.as_rotvec())

    def send_urscript(self, translation, rotation):
        """
        Using the primary interface to send URScripts programs to move the robot
        """

        msg = String()
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation.x,translation.y, translation.z)
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation[0],translation[1], translation[2])
        msg.data = """def my_prog():\nset_digital_out(1, True)\nmovej(p[{},{},{},{},{},{}], a=0.8, v=0.1, r=0)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
        self.publisher_.publish(msg)
        # self.get_logger().info(msg.data)
        # self.i += 1

    def send_grip(self):
        msg = String()

        with open('grip.script', 'r') as f:
            msg.data = f.read()

        self.publisher_.publish(msg)

    def send_home(self):
        msg = String()

        with open('iniyial.script', 'r') as f:
            msg.data = f.read()

        self.publisher_.publish(msg)


    def send_urscript_tool(self, translation, rotation):
        """
        Using the primary interface to send URScripts programs to move the robot
        """

        msg = String()
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation.x,translation.y, translation.z)
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation[0],translation[1], translation[2])
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nmovej(p[{},{},{},{},{},{}], a=0.8, v=0.1, r=0)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
        msg.data = """def my_prog():\nset_digital_out(1, True)\nspeedl([{},{},{},{},{},{}], a=0.2, t=1)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
        self.publisher_.publish(msg)
        # self.get_logger().info(msg.data)
        # self.i += 1

    def lookup_transform(self, to_frame_rel, from_frame_rel,array=True):
        """
        Search for transformations between two specific frames.
        
        If array is True, it returns a translation and rotation array (w = q[3])
        """

        try:
            t = self.tf_buffer.lookup_transform(
            to_frame_rel,
            from_frame_rel,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0))

            translation = [t.transform.translation.x, t.transform.translation.y,t.transform.translation.z]
            rotation = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

            if array:
                return translation, rotation
            else:
                return t.transform.translation, t.transform.rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return [0,0,0],[0,0,0,1]
    
    def print_transformation_matrix(self, H):
        a, b, c, d = H[0,:]
        e, f, g, h = H[1,:]
        i, j, k, l = H[2,:]
        m, n, o, p = H[3,:]
        self.get_logger().info('transformation matrix: \n [[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]'.format(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p))

    
    def print_rotation_matrix(self, R):
        a, b, c = R[0,:]
        d, e, f = R[1,:]
        g, h, i = R[2,:]
        self.get_logger().info('rotation matrix: \n [[{},{},{}],\n[{},{},{}],\n[{},{},{}]]'.format(a,b,c,d,e,f,g,h,i))

def main(args=None):
    rclpy.init(args=args)

    robot_inferencer = RobotInferencer()

    try:
        rclpy.spin(robot_inferencer)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info('done')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot_inferencer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
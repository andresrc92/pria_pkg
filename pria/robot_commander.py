import rclpy
from rclpy.node import Node

import random
import math
import numpy as np
import os

import cv2
from cv_bridge import CvBridge

from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO

from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform, TransformStamped, Quaternion

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster



class RobotCommander(Node):

    def __init__(self):
        super().__init__('robot_commander')

        self.setio = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.setio.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()
        self.send_request()

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

        self.path = dir_path = os.path.dirname(os.path.realpath(__file__))
        self.state = 0

        self.initial_pose = Transform()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def send_request(self):
        self.req.fun = 1 # FUN_SET_DIGITAL_OUT
        self.req.pin = 1
        self.req.state = 0.0
        self.future = self.setio.call_async(self.req)


    def tf_listener_callback(self, msg):
        total_transforms = len(msg.transforms)
        for transform in msg.transforms:
            # self.get_logger().info('I heard: "%s"' % transform.header.frame_id)
            break

    def io_listener_callback(self, msg):
        self.inMotion = True if msg.digital_out_states[1].state == True else False

        if self.inMotion and not self.inMotionPrev:
            # Rising edge
            self.state = 0
            
        # if not self.inMotion and self.inMotionPrev:
            # Falling edge

        self.inMotionPrev = self.inMotion


        # self.get_logger().info('In motion "%s"' % self.inMotion)

    def rgb_listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.height)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        # self.color_filter(self.cv_image)


    def color_filter(self, frame):
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

    def timer_callback(self):
        translation, rotation = self.lookup_transform('base', 'tool0')

        if not self.inMotion and self.state == 0:
            self.state = 1
            self.handle_next_pose([self.initial_pose.translation.x,self.initial_pose.translation.y,self.initial_pose.translation.z], [self.initial_pose.rotation.x,self.initial_pose.rotation.y,self.initial_pose.rotation.z,self.initial_pose.rotation.w])

        # if self.cv_image.any():
        #     ret = cv2.imwrite('tes.jpg', self.cv_image)
        #     self.get_logger().info('Saving image %s' % ret)

        # if not self.inMotion:
        #     self.handle_next_pose()
        #     translation, rotation = self.lookup_transform('next_pose', 'base')
        #     self.send_urscript(translation, rotation)
    
    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def quaternion_from_euler(self, ai, aj, ak):
        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = math.cos(ai)
        si = math.sin(ai)
        cj = math.cos(aj)-0.173735
        sj = math.sin(aj)
        ck = math.cos(ak)
        sk = math.sin(ak)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        q = np.empty((4, ))
        q[0] = cj*sc - sj*cs # w
        q[1] = cj*ss + sj*cc # x
        q[2] = cj*cs - sj*sc # y
        q[3] = cj*cc + sj*ss # z

        return q

    def rotation_vector_from_quaternion(self, quaternion):
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z

        # Calculate the angle of rotation
        angle = 2 * np.arccos(w)

        # Calculate the sin of half the angle
        s = np.sqrt(1 - w * w)

        if s < 1e-6:  # If s is close to zero, the direction of the axis is not important
            x_axis = x
            y_axis = y
            z_axis = z
        else:
            x_axis = x / s
            y_axis = y / s
            z_axis = z / s

        # Return the rotation vector (angle * axis)
        rotation_vector = np.array([x_axis, y_axis, z_axis]) * angle
        return rotation_vector
    
    def multiply_quaternions(self, q1, q2):
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return [w, x, y, z]

    def handle_next_pose(self, translation, rotation):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base'
        t.child_frame_id = 'next_pose'

        # We create the translation
        t.transform.translation.x = random.random() * 0.1 + translation[0]
        t.transform.translation.y = random.random() * 0.1 + translation[1]
        t.transform.translation.z = 0.0 + translation[2]

        # Create a quaternion from euler angles
        q_next_pose = self.quaternion_from_euler(0, 0, random.random() * 0.2 )
        # aux = q_next_pose[0]
        # q_next_pose[0] = q_next_pose[3]
        # q_next_pose[3] = aux
        
        q = self.multiply_quaternions(rotation, q_next_pose)

        t.transform.rotation.w = q[0]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]

        # q_ = Quaternion()
        # q_.x, q_.y, q_.z, q_.w = q[0], q[1], q[2], q[3]
        # euler = self.euler_from_quaternion(q_)

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        self.lookup_transform('base', 'next_pose')

        # self.get_logger().info(self.rotation_vector_from_quaternion(t.transform.rotation))
        self.send_urscript(t.transform.translation, t.transform.rotation)

    def send_urscript(self, translation, rotation):
        r, p, y = self.euler_from_quaternion(rotation)

        msg = String()
        msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation.x,translation.y, translation.z)
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing..')
        # self.i += 1

    def lookup_transform(self, to_frame, from_frame):
        to_frame_rel = to_frame
        from_frame_rel = from_frame

        try:
            t = self.tf_buffer.lookup_transform(
            to_frame_rel,
            from_frame_rel,
            rclpy.time.Time())

            if (self.initial_pose.translation.x == 0 or self.initial_pose.translation.y == 0 or self.initial_pose.translation.z == 0):   
                self.get_logger().info('first pose empty')
                self.initial_pose = t.transform

            translation = [t.transform.translation.x, t.transform.translation.y,t.transform.translation.z]
            rotation = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            rotation_euler = self.euler_from_quaternion(t.transform.rotation)
            rot_vec = self.rotation_vector_from_quaternion(t.transform.rotation)

            # self.get_logger().info('translation x: {} y: {} z: {}'.format(translation[0], translation[1], translation[2]))
            # self.get_logger().info('rotation q x: {} y: {} z: {} w: {}'.format(rotation[0], rotation[1], rotation[2], rotation[3]))
            # self.get_logger().info('rotation r: {} p: {} y: {}'.format(rotation_euler[0], rotation_euler[1], rotation_euler[2]))
            self.get_logger().info('rotation from {} r: {} p: {} y: {}'.format(from_frame, rot_vec[0], rot_vec[1], rot_vec[2]))
            self.get_logger().info(' ')

            # self.handle_next_pose()
            return translation, rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

def main(args=None):
    rclpy.init(args=args)

    robot_commander = RobotCommander()

    rclpy.spin(robot_commander)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot_commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
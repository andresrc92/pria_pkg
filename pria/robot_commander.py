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

from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform, TransformStamped, Quaternion

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster




class Rotations ():
    def __init__(self):
        self.me = True

    def from_quat(self,x,y,z,w):
        """
        Defines a rotation from Quaternion taking W as last argument
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # print(self.x, self.y, self.z, self.w)

    def from_euler(self,x,y,z):
        """
        Defines a rotation from Euler RPY angles
        """
        x /= 2.0
        y /= 2.0
        z /= 2.0
        ci = np.cos(x)
        si = np.sin(x)
        cj = np.cos(y)
        sj = np.sin(y)
        ck = np.cos(z)
        sk = np.sin(z)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        self.x = cj*sc - sj*cs
        self.y = cj*ss + sj*cc
        self.z = cj*cs - sj*sc
        self.w = cj*cc + sj*ss

    def from_rotvec(self, x, y, z):
        """
        Convert a rotation vector to a quaternion.
        
        Args:
        x (float): The x component of the rotation vector
        y (float): The y component of the rotation vector
        z (float): The z component of the rotation vector

        Returns:
        np.ndarray: The corresponding quaternion [qx, qy, qz, qw]
        """
        rotation_vector = np.array([x, y, z])
        theta = np.linalg.norm(rotation_vector)
        
        if np.isclose(theta, 0):
            # If the angle is close to zero, return the identity quaternion
            return np.array([0, 0, 0, 1])
        
        axis = rotation_vector / theta
        sin_half_theta = np.sin(theta / 2)
        self.x = axis[0] * sin_half_theta
        self.y = axis[1] * sin_half_theta
        self.z = axis[2] * sin_half_theta
        self.w = np.cos(theta / 2)

    def from_object(self, obj):
        self.x = obj.x
        self.y = obj.y
        self.z = obj.z
        self.w = obj.w

    def from_matrix(self, m):
        """
        Convert a rotation matrix to a quaternion.
        
        Args:
        R (np.ndarray): The rotation matrix (3x3)

        Returns:
        np.ndarray: The corresponding quaternion [qx, qy, qz, qw]
        """
        assert m.shape == (3, 3), "R must be a 3x3 matrix"
        
        # Calculate the trace of the matrix
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5/t
            q[0] = (m[2,1] - m[1,2]) * t
            q[1] = (m[0,2] - m[2,0]) * t
            q[2] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t
        
        self.x = q[0]
        self.y = q[1]
        self.z = q[2]
        self.w = q[3]

    def as_quat(self):
        """
        Returns the rotation as Quaternion array
        with order [x,y,z,w]
        """
        return np.array([self.x, self.y, self.z, self.w])
    
    def as_euler(self):
        """
        Returns the rotation as Euler angles
        with order [roll, pitch, yaw]
        """
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def as_rotvec(self):
        """
        Returns the rotation as Rotations vector
        with order [x, y, z]
        """

        # Calculate the angle of rotation
        angle = 2 * np.arccos(self.w)

        # Calculate the sin of half the angle
        s = np.sqrt(1 - self.w * self.w)

        if s < 1e-6:  # If s is close to zero, the direction of the axis is not important
            x_axis = self.x
            y_axis = self.y
            z_axis = self.z
        else:
            x_axis = self.x / s
            y_axis = self.y / s
            z_axis = self.z / s

        # Return the rotation vector (angle * axis)
        rotation_vector = np.array([x_axis, y_axis, z_axis]) * angle
        return rotation_vector
    
    def as_matrix(self):        
        """
        Returns the rotations as a rotation matrix.
        """
        R = np.array([
            [2*(self.w**2+self.x**2)-1, 2*(self.x*self.y-self.w*self.z), 2*self.x*self.z + 2*self.y*self.w],
            [2*self.x*self.y + 2*self.z*self.w, 2*(self.w**2+self.y**2)-1, 2*self.y*self.z - 2*self.x*self.w],
            [2*self.x*self.z - 2*self.y*self.w, 2*self.y*self.z + 2*self.x*self.w, 2*(self.w**2 + self.z**2)-1]
        ])
        return R
    
    def multiply_quaternions(self, q1, q2):
        """
        Quaternions multiplication.
        
        Quaternions must be arrays with the 'w' value at the last index position
        q[0] = q.x, q[1] = q.y, q[2] = q.z, q[3] = q.w

        """
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return [x, y, z, w]

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

        self.parent = os.getcwd()
        self.path = os.path.join(self.parent,self.directory)
        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass

        self.gt_path = os.path.join(self.path,'gt.yaml')
        
        self.get_logger().info('Dataset folder in {}'.format(self.path))

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
        # translation, rotation = self.lookup_transform('base', 'tool0')
        translation, rotation = self.lookup_transform('base_link_inertia', 'wrist_3_link', True)

        if self.image_count_max > self.image_count:
            if not self.inMotion and self.state == 0:
                self.state = 1
                self.handle_next_pose()
        else:
            #go back to initial pose
            self.back_to_first_pose()
            
            #save yaml file with ground truth poses
            with open(self.gt_path, 'w') as file:
                yaml.dump(self.gt, file,  default_flow_style=False)

            # and exit
            raise SystemExit

        # if self.cv_image.any():
        #     ret = cv2.imwrite('tes.jpg', self.cv_image)
        #     self.get_logger().info('Saving image %s' % ret)

        # if not self.inMotion:
        #     self.handle_next_pose()
        #     translation, rotation = self.lookup_transform('next_pose', 'base')
        #     self.send_urscript(translation, rotation)

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

        if self.inMotion and not self.inMotionPrev:
            # Rising edge
            self.state = 2

        # if not self.inMotion and self.inMotionPrev:
            # Falling edge

        self.inMotionPrev = self.inMotion
        # self.get_logger().info('In motion "%s"' % self.inMotion)

    def rgb_listener_callback(self, msg):
        """
        This function listens to rgb topic
        """
        # self.get_logger().info('I heard: "%s"' % msg.height)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        # self.color_filter(self.cv_image)
        if self.state == 2:
            T, R = self.lookup_transform('initial_pose', 'wrist_3_link')
            # filename = '{},{},{},{},{},{},{}.png'.format(T[0],T[1],T[2],R[0],R[1],R[2],R[3])
            self.gt.update({
                self.image_count: {
                    'translation':T,
                    'rotation':R
                }
            })
            filename = '{}.png'.format(self.image_count)
            self.capture_and_save_image(self.cv_image, os.path.join(self.path, filename))
            self.state = 0
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
        x,y,z,w = quaternion
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
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link_inertia'
        t.child_frame_id = 'initial_pose'
        t.transform = self.initial_pose

        self.tf_static_broadcaster.sendTransform(t)

    def handle_next_pose(self):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'initial_pose'
        t.child_frame_id = 'next_pose'

        # We create the translation
        t.transform.translation.x = random.randrange(-90, 90, 1) / 1000
        t.transform.translation.y = random.randrange(-90, 90, 1) / 1000
        t.transform.translation.z = 0.0 #random.randrange(0, -300, -1) / 1000 #

        t_next_pose = [t.transform.translation.x,t.transform.translation.y,t.transform.translation.z]
        # self.get_logger().info('offset transform {} {} {}'.format(t_next_pose[0],t_next_pose[1],t_next_pose[2]))

        # Create a quaternion from euler angles
        r = Rotations()
        r.from_euler(0, 0, random.randrange(-30, 30, 1) / 100 )
        q_next_pose = r.as_quat()

        next_pose_matrix = self.create_transformation_matrix(q_next_pose, t_next_pose)
        # self.print_transformation_matrix(next_pose_matrix)

        base_to_next_matrix = np.dot(self.initial_matrix,next_pose_matrix)
        # self.print_transformation_matrix(base_to_next_matrix)
        
        q = q_next_pose

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        r = Rotations()
        r.from_matrix(base_to_next_matrix[0:3,0:3])
        next_rotation = r.as_rotvec()

        # self.get_logger().info(t_next_pose)
        # if translation != -1 and rotation != -1:
        # self.send_urscript(t_next_pose, q_aux)
        self.send_urscript(base_to_next_matrix[:3,3], next_rotation)

    def back_to_first_pose(self):
        self.send_urscript(self.initial_matrix[:3,3], self.initial_rotvec)

    def send_urscript(self, translation, rotation):
        """
        Using the primary interface to send URScripts programs to move the robot
        """

        msg = String()
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation.x,translation.y, translation.z)
        # msg.data = """def my_prog():\nset_digital_out(1, True)\nrv=rpy2rotvec([{},{},{}])\nmovej(p[{},{},{},rv[0],rv[1],rv[2]], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(r,p,y,translation[0],translation[1], translation[2])
        msg.data = """def my_prog():\nset_digital_out(1, True)\nmovej(p[{},{},{},{},{},{}], a=1.2, v=0.25, r=0)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
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
                        'rotation':rotation
                    }
                })
                # with open('./test.yaml', 'w') as file:
                #     yaml.dump(self.gt, file,  default_flow_style=False)
            #

            if array:
                return translation, rotation
            else:
                return t.transform.translation, t.transform.rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return -1, -1
    
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
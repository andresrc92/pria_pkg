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

from std_msgs.msg import String, Float32
from tf2_msgs.msg import TFMessage
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO

from pria.rotations import *
from pria.pytorch_training_script import *

from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform, TransformStamped, Quaternion

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import time


class GraspingTask(Node):

    def __init__(self):
        super().__init__('grasping_task')
        
        self.setio = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.setio.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()
        self.send_request()
        self.start_time = rclpy.time.Time()
        

        self.declare_parameter('sim',True)
        self.sim_parameter = self.get_parameter('sim').get_parameter_value().bool_value
        
        self.declare_parameter('folder','folder')
        self.directory = self.get_parameter('folder').get_parameter_value().string_value

        self.gripper_pub = self.create_publisher(Float32, 'gripper', 10)

        self.image_height = 240
        self.image_width = 320

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

        self.state = 3
        self.index = 0
        self.trajectories = {}

        self.initial_pose = Transform()
        self.initial_matrix = np.eye(4)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # self.generate_cone_points()
        self.state_machine = 0

        self.object_dir = os.path.join("dataset", self.directory)
        
        self.img_path = os.path.join(self.object_dir, 'trajectories_twist')

        try:
            os.mkdir(self.img_path)
        except FileExistsError:
            pass

        self.grip_script_dir = os.path.join(self.object_dir,"gripper_sim.script")
        self.coarse_model_dir = os.path.join(self.object_dir,"cone_twist")
        self.fine_model_dir = os.path.join(self.object_dir,"flat_twist")

        # Approach stage
        self.coarse_model = Trainer(self.coarse_model_dir, 100)
        self.coarse_model.open_model()
        self.coarse_model.open_batch_normalization()

        # Pose refinement stage
        self.fine_model = Trainer(self.fine_model_dir, 100)
        self.fine_model.open_model()
        self.fine_model.open_batch_normalization()

        self.reached_goal = False
        self.got_first_pose = False
        self.trajectories_path = "trajectories.yaml" # os.path.join(self.model_folder, "loss.npy")

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

        self.req.fun = 1 # FUN_SET_DIGITAL_OUT
        self.req.pin = 0
        self.req.state = 0.0
        self.future = self.setio.call_async(self.req)

    def timer_callback(self):
        """
        This is called every ${timer_period} seconds
        """
        print(self.state_machine)
        if self.state_machine == 0:
            
            if self.publish_first_pose() < 0:
                print("Waiting for transform ... ")
                return
            
            if not self.inMotion:
                self.state_machine = 1

        elif self.state_machine == 1:
            if not self.inMotion:
                self.state_machine = 2
        
        elif self.state_machine == 2:
            if self.reached_goal:
                self.reached_goal = False
                self.state_machine = 3

        elif self.state_machine == 3:
            if not self.inMotion:
                self.state_machine = 4

        elif self.state_machine == 4:
            if self.reached_goal:
                self.reached_goal = False
                self.state_machine = 5

        elif self.state_machine == 5:
            self.send_grip()
            self.state_machine = 6
        
        elif self.state_machine == 6:
            if self.inMotion:
                self.state_machine = 7

        elif self.state_machine == 7:
            if not self.inMotion:
                self.back_to_first_pose()
                time.sleep(0.5)
                raise SystemExit

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
        # Robot motion flag
        self.inMotion = msg.digital_out_states[1].state

        # Gripper control
        if msg.digital_out_states[0].state:
            self.gripper_pub.publish(Float32(data=0.1))
        else:
            self.gripper_pub.publish(Float32(data=-0.1))


    def rgb_listener_callback(self, msg):
        """
        This function listens to rgb topic
        """
        
        # self.get_logger().info('I heard: "%s"' % msg.height)
        image = self.bridge.imgmsg_to_cv2(msg, 'rgba8')
        image2save = self.bridge.imgmsg_to_cv2(msg, 'bgra8')

        h, w, c = image.shape
        if w != 320 or h != 240:
            im = cv2.resize(image, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
            # im2save = cv2.resize(image2save, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        else:
            im = image
            # im2save = image2save
        
        im_array = np.asarray(im)[:, :, :3] / 255

        if not self.inMotion:

            if self.state_machine == 2:

                predicted_pose = self.coarse_model.infer_from_image(im_array)
                print("Coarse model: ", predicted_pose[2])

                if self.dist(predicted_pose[:3]) < 0.009:
                    self.reached_goal = True
                else:
                    # self.trajectory_plane.append(T)
                    self.handle_predicted_pose(predicted_pose)
    
            elif self.state_machine == 4:

                predicted_pose = self.fine_model.infer_from_image(im_array)
                print("Fine model: ", self.dist(predicted_pose[:3]))

                if self.dist(predicted_pose[:3]) < 0.007:
                    self.reached_goal = True
                else:
                    # self.trajectory_plane.append(T)
                    self.handle_predicted_pose(predicted_pose)

            
    def dist(self,vector):
        p = np.power(vector,2)
        d = np.sum(p)
        return np.sqrt(d)
    
    def update_trajectories(self, index):
        self.trajectories.update({
            self.index:
                {
                    'cono': self.trajectory_cone,
                    'plano': self.trajectory_plane
                }
        })
                
    def save_trajectories(self):
        path = os.path.join(self.img_path, self.trajectories_path)
        with open(path, 'w') as file:
            yaml.dump(self.trajectories, file, default_flow_style=False)     
            

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
    
    def handle_predicted_pose(self, prediction):

        if self.sim_parameter:
            wrist = 'wrist_3_link_sim'
        else:
            wrist = 'wrist_3_link'

        t_current, q_current = self.lookup_transform_('base_link_inertia', wrist)
        
        t_next_pose = 0.4 * prediction[:3]
        if self.state == 1:
            t_next_pose[2] = 0.0

        q_next_pose = prediction[3:]
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

        self.send_urscript(t_, r.as_rotvec())


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

            if (self.initial_pose.translation.x == 0 or self.initial_pose.translation.y == 0 or self.initial_pose.translation.z == 0):   
                self.initial_pose = t.transform
                r = Rotations()
                r.from_object(t.transform.rotation)
                self.initial_rotvec = r.as_rotvec()
                self.initial_matrix = self.create_transformation_matrix(rotation, translation)
                self.got_first_pose = True
                self.publish_first_pose()


            return translation, rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return [0.0,0.0,0.0], [0.0,0.0,0.0,0.0]

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

    def handle_next_pose(self, index, current):
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

        d = self.dist(base_to_next_matrix[:3,3]-current)

        if d > 0.005:
            # print("Distancia a incio de trayectoria ", d)
            self.send_urscript(base_to_next_matrix[:3,3], next_rotation)
            return False        
        else:
            self.send_request()
            return True


    def move_to_first_pose(self,current):
        initial = self.fine_model.get_initial_pose()
       
        d = self.dist(initial[:3]-current)

        if d > 0.001:
            print("Homming..")
            r = Rotations()
            r.from_array(initial[3:])

            self.send_urscript(initial[:3], r.as_rotvec())
        else:
            print("Homed..")
            self.homed = True
            self.send_request()

    
    def back_to_first_pose(self):
        self.send_urscript(self.initial_matrix[:3,3], self.initial_rotvec)

    def send_urscript(self, translation, rotation):
        """
        Using the primary interface to send URScripts programs to move the robot
        """

        msg = String()
        msg.data = """def my_prog():\nset_digital_out(1, True)\nmovej(p[{},{},{},{},{},{}], a=0.8, v=0.1, r=0)\nset_digital_out(1, False)\nend""".format(translation[0],translation[1], translation[2], rotation[0], rotation[1], rotation[2])
        self.publisher_.publish(msg)
        
    def send_grip(self, sim=True):
        msg = String()

        if sim:
            with open('./urscripts/gripper_sim.script', 'r') as f:
                msg.data = f.read()
        else:
            with open('./urscripts/grip.script', 'r') as f:
                msg.data = f.read()

        self.publisher_.publish(msg)

    def send_home(self):
        msg = String()

        with open('initial.script', 'r') as f:
            msg.data = f.read()

        self.publisher_.publish(msg)


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

    robot_inferencer = GraspingTask()

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
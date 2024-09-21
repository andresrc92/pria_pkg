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

        self.generate_cone_points()

        self.path = os.getcwd()
        self.img_path = os.path.join(self.path, 'trajectories')
        try:
            os.mkdir(self.img_path)
        except FileExistsError:
            pass

        # self.trainer = Trainer("vibrator_no_twist", 2, 100) # el que anda
        # self.trainer = Trainer("extrusion", 2, 100) # el que anda
        self.trainer = Trainer("black_cube_cone", 2, 100) # el que anda
        # self.z_end = self.trainer.read_meta_data()
        self.trainer.open_model()
        # self.trainer.open_normalization()
        self.trainer.open_batch_normalization()

        # self.flat_model = Trainer("cup_drill_twist", 2, 100)
        self.flat_model = Trainer("black_cube", 2, 100)
        self.flat_model.open_model()
        self.flat_model.open_batch_normalization()

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

        # if not self.inMotion and self.inMotionPrev:
        #     self.homed = True
            # Falling edge

        self.inMotionPrev = self.inMotion
        # self.get_logger().info('In motion "%s"' % self.inMotion)

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
            im2save = cv2.resize(image2save, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        else:
            im = image
            im2save = image2save
        
        im_array = np.asarray(im)[:, :, :3] / 255

        # plt.imshow(im_array)
        # plt.show()

        # Get trayectory to the object
        T, Q = self.lookup_transform_('base_link_inertia','wrist_3_link_sim')

        # if not self.homed and not self.inMotion and self.got_first_pose or self.state == 3:
        if not self.inMotion and self.got_first_pose and self.state == 3:
            # self.move_to_first_pose(T)
            if self.handle_next_pose(self.index, T):
                filename = "{}_i.png".format(self.index)
                self.capture_and_save_image(im2save, os.path.join(self.img_path, filename))
                self.trajectory_cone = []
                self.trajectory_plane = []
                self.state = 0
                self.homed = True

        if T[2] < 0.16 and self.state < 2:
            self.state = 2
            print("Robot too low.")

        if self.homed and self.got_first_pose:

            if not self.inMotion:

                if self.state == 0:
                    pose = self.trainer.infer_from_image(im_array)

                    if pose[2] > 0.008:
                        self.trajectory_cone.append(T)
                        self.handle_predicted_pose(pose)
                    else:
                        self.state = 1

                elif self.state == 1:
                    pose = self.flat_model.infer_from_image(im_array)

                    if self.dist(pose[:3]) < 0.001:
                        self.reached_goal = True
                        self.state = 2
                    else:
                        self.trajectory_plane.append(T)
                        self.handle_predicted_pose(pose)

                elif self.state == 2:
                    filename = "{}_f.png".format(self.index)
                    self.capture_and_save_image(im2save, os.path.join(self.img_path, filename))
                    self.state = 3
                    self.update_trajectories(self.index)
                    self.index += 1
                    print("Trajectory ", self.index)
                    self.homed = False

            if self.index >= self.max_points:
                self.save_trajectories()
                raise SystemExit

            
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
        t_current, q_current = self.lookup_transform_('base_link_inertia', 'wrist_3_link_sim')
        
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
                # self.gt.update({
                #     'initial_pose': {
                #         'translation':translation,
                #         'rotation':rotation,
                #         'total_images': self.image_count_max,
                #         'height':self.image_height,
                #         'width':self.image_width
                #     }
                # })


            return translation, rotation

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return [0.0,0.0,0.0], [0.0,0.0,0.0,0.0]

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

    def generate_cone_points(self, r = 50, z = 150.0, twist=False):
        x = np.arange(-r, r, r*0.8) / 1000
        y = np.arange(-r, r, r*0.8) / 1000

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

        self.max_points = len(self.close_points)
        
        for i, point in enumerate(self.close_points):
            self.publish_tf(point, 'initial_pose', '{}'.format(i))


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
        initial = self.flat_model.get_initial_pose()
       
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

        with open('initial.script', 'r') as f:
            msg.data = f.read()

        self.publisher_.publish(msg)


    # def lookup_transform(self, to_frame_rel, from_frame_rel,array=True):
    #     """
    #     Search for transformations between two specific frames.
        
    #     If array is True, it returns a translation and rotation array (w = q[3])
    #     """

    #     try:
    #         t = self.tf_buffer.lookup_transform(
    #         to_frame_rel,
    #         from_frame_rel,
    #         rclpy.time.Time(),
    #         timeout=rclpy.duration.Duration(seconds=1.0))

    #         translation = [t.transform.translation.x, t.transform.translation.y,t.transform.translation.z]
    #         rotation = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

    #         if array:
    #             return translation, rotation
    #         else:
    #             return t.transform.translation, t.transform.rotation

    #     except TransformException as ex:
    #         self.get_logger().info(
    #             f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
    #         return [0,0,0],[0,0,0,1]
    
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
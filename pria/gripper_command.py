import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32

from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates

class GripperCommand(Node):

    def __init__(self):
        super().__init__('gripper_command')
        self.gripper_pub = self.create_publisher(Int32, 'gripper', 10)
        self.set_io_client = self.create_client(SetIO, 'set_io')
        self.get_io_sub = self.create_subscription(IOStates, '/io_and_status_controller/io_states', self.io_states_callback, 10)


    def io_states_callback(self, msg):
        if msg.digital_out_states[1].state:
            self.gripper_pub.publish(Int32(data=1))
        else:
            self.gripper_pub.publish(Int32(data=0))

def main():
    rclpy.init()
    gripper_command = GripperCommand()
    rclpy.spin(gripper_command)
    gripper_command.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
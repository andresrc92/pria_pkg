import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32

from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates

class GripperCommand(Node):

    def __init__(self):
        super().__init__('gripper_command')
        self.gripper_pub = self.create_publisher(Float32, 'gripper', 10)
        self.set_io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        self.get_io_sub = self.create_subscription(IOStates, '/io_and_status_controller/io_states', self.io_states_callback, 10)
        self.prev_state = False


    def io_states_callback(self, msg):
        state = msg.digital_out_states[0].state

        if state != self.prev_state:
            print("Gripper state change: ",state)
            self.prev_state = state

        if state:
            self.gripper_pub.publish(Float32(data=0.1))
        else:
            self.gripper_pub.publish(Float32(data=-0.1))

def main():
    rclpy.init()
    gripper_command = GripperCommand()
    rclpy.spin(gripper_command)
    gripper_command.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
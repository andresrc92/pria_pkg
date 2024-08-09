import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/rgb', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.cap = cv2.VideoCapture(2)  # Change the argument if your camera is not at index 0
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            im = cv2.resize(frame, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
            msg = self.bridge.cv2_to_imgmsg(im, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

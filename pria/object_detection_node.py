import torch
from torchvision import models, transforms
from PIL import Image

import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datetime import datetime

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_copilot')
        self.subscription = self.create_subscription(
            ROSImage,
            '/rgb',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(ROSImage, '/rgb_bb', 10)
        self.bridge = CvBridge()
        # Load a pre-trained Fast R-CNN model
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.coco_labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
            'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image = self.transform(cv_image).unsqueeze(0)
            start = datetime.now()
            with torch.no_grad():
                predictions = self.model(image)[0]    
            
            time = datetime.now() - start
            self.get_logger().info(f'Inference time: {time} seconds')
            
            result_image = self.visualize_results(cv_image, predictions)
            
            result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.publisher.publish(result_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def visualize_results(self, cv_image, predictions, threshold=0.5):
        
        for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
            if score >= threshold:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(cv_image, f'{label.item()} {score:.2f}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return cv_image
    
    # Load and transform the image
    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    # Perform object detection
    def detect_objects(self, image_path):
        image = self.load_image(image_path)
        with torch.no_grad():
            predictions = self.model(image)
        
        return predictions[0]


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
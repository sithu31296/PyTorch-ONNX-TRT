from std_msgs.msg import String 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
import rclpy 
import yaml 
import cv2
import torch
import os
import time
from rclpy.node import Node
from threading import Thread, Event
from yolov5.scripts.infer import YOLOV5, FPS
from ament_index_python.packages import get_package_prefix

BASE_PATH = os.path.join(get_package_prefix('yolov5').replace('install', 'src'), 'yolov5')
WEIGHTS_PATH = os.path.join(BASE_PATH, 'scripts/cfg', 'yolov5_352_416.trt')


class YOLOv5Node(Node):
    def __init__(self):
        super().__init__('yolov5_node')
        self.bridge = CvBridge()
        self.current_frame = None
        
        self.get_logger().info('Model Initializing...')
        self.yolo = YOLOV5(WEIGHTS_PATH)
        self.get_logger().info('Model Loaded...')

        self.subscriber = self.create_subscription(Image, '/image', self.callback_image, 5)
        self.subscriber
        self.publisher = self.create_publisher(Image, '/yolov5/vis', 5)
        self.fps = FPS()
        timer_period = 0.01
        timer = self.create_timer(timer_period, self.process_image)

    def process_image(self):
        if self.current_frame is not None:
            self.fps.start()
            image = self.yolo.predict(self.current_frame)
            image_msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
            self.publisher.publish(image_msg)
            self.fps.stop()
            curr_fps = self.fps.get_fps()
            self.get_logger().info(f'Current {curr_fps}')

    def callback_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
        except CvBridgeError as e:
            raise e 
        self.current_frame = cv_image


def main(args=None):
    rclpy.init(args=args)
    main_node = YOLOv5Node()
    rclpy.spin(main_node)
    main_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

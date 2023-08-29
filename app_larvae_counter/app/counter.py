import rclpy
from rclpy.node import Node
import os
import cv2
import threading
from cv_bridge import CvBridge
from .preProcess import PreProcess
from .featureExtraction import FeatureExtraction
from .model import Model
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import json


class Counter(Node):
    def __init__(self):
        self.bridge = CvBridge()
        print("Started!")
        super().__init__('counter')
        self.subscription = self.create_subscription(
            Image,
            'floater/microscope',
            self.image_callback,
            100)
        self.subscription  # prevent unused variable warning
        self.model = Model()
        self.frame_id = ''
        self.publisher_ = self.create_publisher(String, 'counter', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.model = Model()
        self.message_processed = threading.Event()
        self.count = 0

    def image_callback(self, msg):
        frame_id = msg.header.frame_id
        # print("Header id:", frame_id)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image message to OpenCV format
            # print("Image retrieved", cv_image.shape)
            self.count = self.counter(cv_image)
            self.frame_id = frame_id
            self.message_processed.set()
            # count = counter.getCount()

            print("Num larvae:", self.count)
            # cv2.imshow("Microscope Image", cv_image)
            # cv2.waitKey(1)
        except Exception as e:
            print(e)

    def getCount(self):
        return self.count
    
    def timer_callback(self):
        msg = String()        

        data = {
            "frame_id"  : str(self.frame_id),
            "count"     : self.count
        }
        
        # msg.header = header
        msg.data = json.dumps(data)
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
            
    def counter(self, image):

        descBatch = self.extractDescriptors(image)
        counter = 0       

        for descList in descBatch:
            response = self.model.classifyImage(descList)
            counter += response

        return counter

    def extractDescriptors(self, image):

        image = PreProcess.resize(image, 1280, 720)
        # print(image.shape)
        correlationMap = PreProcess.sliding_correlation(image, "src/counter/counter/ref.png")
        # print(correlationMap.shape)
        circles_info = PreProcess.findCenters(correlationMap)
        image = PreProcess.unsharpMask(image)
        equalizedImage = PreProcess.equalize(image)
        descBatch = FeatureExtraction.extract_DESC(image, circles_info)

        return descBatch


def main(args=None):
    rclpy.init(args=args)

    counter = Counter()



    rclpy.spin(counter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    counter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
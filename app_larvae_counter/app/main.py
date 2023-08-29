import rospy
from sensor_msgs.msg import Image
import os
import cv2
from cv_bridge import CvBridge
from .preProcess import PreProcess
from .featureExtraction import FeatureExtraction
from .model import Model

model = Model()


def extractDescriptors(image):

    image = PreProcess.resize(image, 1280, 720)

    
    
    correlationMap = PreProcess.sliding_correlation(image)

    circles_info = PreProcess.findCenters(correlationMap)

    image = PreProcess.unsharpMask(image)
    equalizedImage = PreProcess.equalize(image)
    

    descBatch = FeatureExtraction.extract_DESC(image, circles_info)

    # print(siffBatch)
    return descBatch
   


def counter(image):

    siffBatch = extractDescriptors(image)
    counter = 0

    

    for siftList in siffBatch:
        response = model.classifyImage(siftList)
        counter += response

    return response

    

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image message to OpenCV format
        count = counter(cv_image)
        # cv2.imshow("Microscope Image", cv_image)
        # cv2.waitKey(1)
    except Exception as e:
        print(e)

def loadImages(input_path):

    images = []
    # Check if the input path is a file or a directory
    if os.path.isfile(input_path):  # Single image file
        image = cv2.imread(input_path)
        if image is not None:
            images.append(image)
    elif os.path.isdir(input_path):  # Directory containing multiple images
        for filename in os.listdir(input_path):
            image_path = os.path.join(input_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

    return images


if __name__ == "__main__":

    # rospy.init_node("image_subscriber")  # Initialize ROS node
    # bridge = CvBridge()  # Initialize CvBridge to convert between ROS Image messages and OpenCV images
    
    # # Subscribe to the image topic
    # image_topic = "/floter/microscope"  # Modify this to match your actual topic name
    # rospy.Subscriber(image_topic, Image, image_callback)
    
    # rospy.spin()  # Keep the node running until it's manually stopped


############################################################################################################

    path = "./grouped" + "/_image_004.png"
    # path = "./dataset/other_planktons/single"

    images = loadImages(path)

    for image in images:
        count = counter(image)
        # print(count)

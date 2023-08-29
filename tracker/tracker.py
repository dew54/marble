import rclpy
from rclpy.node import Node
import os
import cv2
import json
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import NavSatFix






class Tracker(Node):
    def __init__(self):
        print("Started!")
        super().__init__('Tracker')


        
        self.depthSub = self.create_subscription(
            PoseWithCovarianceStamped,
            'floater/depth',
            self.depth_callback,
            100)


        self.gnssSub = self.create_subscription(
            NavSatFix,
            'floater/gnss',
            self.gnss_callback,
            100)

        self.counterSub = self.create_subscription(
            String,
            'counter',
            self.counter_callback,
            100)

        self.bufferDepth = []
        self.bufferGnss = []
        self.buffConcentration = []
        self.row = [0, 0, 0, 0, 0]

    def computeConcentration(self, count):
        volume = 0.001
        c = count/volume        
        return c

    
    def computeGrad(self, gnssData, depthData , cData):

        print(gnssData)
        
        lat = gnssData[1][0]
        lon = gnssData[1][1]
        depth = depthData[1]

        dLat = lat - gnssData[0][0]
        dLon = lon - gnssData[0][1]
        dDepth = depth - depthData[0]
        dC = cData[1]-cData[0]
        gradient = [dC/dLat, dC/dLon, dC/dDepth]
        position = [lat, lon, depth]

        return position, gradient



    


    def counter_callback(self, msg):
        data = json.loads(msg.data)
        count = data["count"]
        frame_id = data["frame_id"]
        print("count", frame_id)

        # if len(self.bufferDepth) > 1 and len(self.bufferGnss) > 1:

        depthData = self.get_last_val_with_id(frame_id, self.bufferDepth)
        gnssData = self.get_last_val_with_id(frame_id, self.bufferGnss)

        print("Length depth data", len(self.bufferDepth))
        print("Length gnss data", len(self.bufferGnss))

        if depthData is not None and gnssData is not None:

            print("Entered in processing..")
        
            c = self.computeConcentration(count)        
            self.buffConcentration.append((frame_id, c))
            cData = self.get_last_val_with_id(frame_id, self.buffConcentration)


            current_time = self.get_clock().now()


            position, gradient = self.computeGrad(gnssData, depthData, cData)

            row = [current_time, frame_id, position, gradient]

            print(row)
        else:
            print("Nothing to do")

        


    def depth_callback(self, msg):
        frame_id = msg.header.frame_id
        print("depth", frame_id)
        position_data = msg.pose.pose.position
        z = position_data.z

        self.bufferDepth.append((frame_id, z))



    def gnss_callback(self, msg):
        frame_id = msg.header.frame_id
        print("gnss", frame_id)

        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        print(latitude)

        data = (latitude, longitude, altitude)
        self.bufferGnss.append((frame_id, data))

    def get_last_val_with_id(self, arr, target_id):
        result = []
        count = 0

        for item in reversed(arr):
            if item[0] == target_id:
                result.append(item[1])
                count += 1
                if count == 2:
                    break

        return reversed(result) if len(result) == 2 else None



def main(args=None):
    rclpy.init(args=args)

    tracker = Tracker()



    rclpy.spin(tracker)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
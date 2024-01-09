import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Int32
import hashlib
from rclpy.qos import qos_profile_sensor_data
import image_geometry
import os

class ContourDetectionNode(Node):
    camera_model = None
    cumulative_count = 0

    def __init__(self):
        super().__init__('contour_detection_node')
        self.subscription = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                    self.image_callback, qos_profile=qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                        self.camera_info_callback, 
                                                        qos_profile=qos_profile_sensor_data)
        
        self.cv_bridge = CvBridge()
        self.counts_publisher = self.create_publisher(Int32, 'counts_over_time', 10)
        # Initialize variables to accumulate counts
        self.counts_over_time = []
        self.tracker = cv2.TrackerKCF_create()
        self.current_potholes = []
        self.occupancy_grid = np.zeros((500, 500), dtype=np.uint8)  # Adjust the grid size as needed
        #self.pothole_file = 'pothole_sizes.txt'
        self.pothole_file = 'pothole/result/pothole_sizes.txt'
        file_directory = self.create_file_directory()
        self.pothole_file = os.path.join(file_directory, 'pothole_sizes.txt')
        self.file_handle = open(self.pothole_file, 'a')
            
    def create_file_directory(self):
        file_directory = os.path.dirname(self.pothole_file)
        if not os.path.exists(file_directory):
            os.makedirs(file_directory)
        return file_directory
    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().info(f'Error converting image: {str(e)}')
            return

        # Apply magenta color contour detection
        result = self.detect_magenta_contours(cv_image)
        
        # Display the result
        cv2.imshow('Potholes and Sizes', result)
        cv2.waitKey(1)
        if len(self.counts_over_time) > 0:
            counts_msg = Int32()
            counts_msg.data = self.counts_over_time[-1]
            self.counts_publisher.publish(counts_msg)

    def detect_magenta_contours(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = image.copy()
        unique_pothole_numbers = set()
        contours_count = 0
        unique_contours = set()

        potholes_to_remove = []

        for pothole in self.current_potholes:
            pothole['tracked'] = False

        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 1300:
                    contour_tuple = tuple(contour.flatten())

                    found = False
                    for pothole in self.current_potholes:
                        if cv2.matchShapes(contour, pothole['contour'], cv2.CONTOURS_MATCH_I2, 0.0) < 0.5:
                            found = True
                            pothole['tracked'] = True
                            break

                    if not found:
                        # Check against previously detected contours
                        found_similar_contour = False
                        for pothole in self.current_potholes:
                            if cv2.matchShapes(contour, pothole['contour'], cv2.CONTOURS_MATCH_I2, 0.0) < 0.5:
                                found_similar_contour = True
                                break

                        if not found_similar_contour:
                            # New pothole
                            unique_color = self.hash_color(str(contour))
                            cv2.drawContours(result, [contour], -1, unique_color, 2)

                            center = self.get_contour_center(contour)

                            # Calculate pothole size
                            size = cv2.contourArea(contour)

                            # Store pothole number and size in a text file
                            pothole_number = contours_count + self.cumulative_count
                            with open(f'pothole/pothole_sizes.txt', 'a') as file:
                                file.write(f"Pothole {pothole_number}: Size {size}\n")

                            cv2.putText(result, f"Size: {size}",
                                        (center[0], center[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            contours_count += 1
                            pothole_number = contours_count + self.cumulative_count

                            cv2.putText(result, str(pothole_number),
                                        (center[0], center[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            unique_contours.add(contour_tuple)
                            unique_pothole_numbers.add(pothole_number)
                            self.current_potholes.append({'contour': contour.copy(), 'tracked': True})

        for pothole in self.current_potholes:
            if not pothole['tracked']:
                # Pothole disappeared
                potholes_to_remove.append(pothole)

        for pothole in potholes_to_remove:
            self.current_potholes.remove(pothole)

        total_pothole = len(unique_pothole_numbers)
        print(total_pothole + self.cumulative_count)

        self.cumulative_count += contours_count
        self.counts_over_time.append(self.cumulative_count)
        for pothole_number in unique_pothole_numbers:
            self.file_handle.write(f"Pothole {pothole_number}: Size {size}\n")

        return result


    def get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None

    def get_grid_coordinates(self, point):
        print(f"Original Point: {point}")
        grid_x = int(point[0] / 10)  # Adjust grid cell size as needed
        grid_y = int(point[1] / 10)
        print(f"Grid Coordinates: ({grid_x}, {grid_y})")
        return grid_x, grid_y

    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def hash_color(self, text):
        # Use a hash function to generate a consistent color from the contour's unique identifier
        hashed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        color = (hashed % 256, (hashed // 256) % 256, (hashed // 65536) % 256)
        return color

    def get_counts_over_time(self):
        return self.counts_over_time

def main(args=None):
    rclpy.init(args=args)

    contour_detection_node = ContourDetectionNode()

    rclpy.spin(contour_detection_node)

    counts_over_time = contour_detection_node.get_counts_over_time()
    print(counts_over_time)

    contour_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

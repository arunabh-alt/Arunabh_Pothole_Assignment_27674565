import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Int32  # Change the import statement
import cv2
import numpy as np
from rclpy import qos
import image_geometry

class ContourDetectionNode(Node):
    camera_model = None
    def __init__(self):
        super().__init__('contour_detection_node')
        self.publisher_ = self.create_publisher(Int32, '/contours', 10)  # Change the message type
        self.subscription = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info', self.camera_info_callback, qos_profile=qos.qos_profile_sensor_data)
        self.cv_bridge = CvBridge()
        self.previous_potholes = []
        self.pothole_id = 0

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().info(f'Error converting image: {str(e)}')
            return

        contours = self.detect_magenta_contours(cv_image)

        # Count the number of contours and publish as Int32
        contour_count = len(contours)
        self.publisher_.publish(Int32(data=contour_count))
        cv2.imshow('Contours', cv_image)
        cv2.waitKey(1)
        # Update the occupancy grid based on the detected contours
        # Assume you have an OccupancyGridUpdater instance available
       # self.occupancy_grid_updater.update_occupancy_grid(self.previous_potholes)


    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def detect_magenta_contours(self, image):
        # Convert the image from BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for magenta color in HSV
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])

        lower_color = np.array([0, 0, 47])  # Adjust as needed
        upper_color = np.array([15, 15, 57])  # Adjust as needed

        # Threshold the image to obtain the magenta color regions
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        #mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area or other criteria if needed
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                filtered_contours.append(contour)

        # Compare with previous contours to eliminate duplicates
        self.current_potholes = []
        for contour in filtered_contours:
            contour_center = self.get_contour_center(contour)
            pothole_found = False

            # Check if the contour has already been counted
            for prev_pothole in self.previous_potholes:
                prev_center = prev_pothole['center']
                if self.distance_between_points(contour_center, prev_center) < 10:
                    pothole_found = True
                    break

            if not pothole_found:
                self.pothole_id += 1
                self.current_potholes.append({'id': self.pothole_id, 'contour': contour, 'center': contour_center})

        # Update previous contours
        self.previous_potholes = self.current_potholes
        contours_count = 0 
        # Draw the contours on the original image and assign cumulative IDs
        result = image.copy()
        for pothole in self.current_potholes:
            contour = pothole['contour']
            area = cv2.contourArea(contour)
            if area > 1000:
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                center = pothole['center']
                contours_count += 1
                #cv2.putText(result, contour_size_text, (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(result, str(contours_count), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(contours_count) 
      
        return result
   

    def get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None

    def distance_between_points(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main(args=None):
    rclpy.init(args=args)
    contour_detection_node = ContourDetectionNode()
    rclpy.spin(contour_detection_node)
    contour_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

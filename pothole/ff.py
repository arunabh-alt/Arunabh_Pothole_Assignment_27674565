import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy import qos
from sensor_msgs.msg import CameraInfo, PointCloud2
import image_geometry

class ContourDetectionNode(Node):
    camera_model = None
    cumulative_count = 0
    def __init__(self):
        super().__init__('contour_detection_node')
        self.subscription = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        self.subscription

        self.cv_bridge = CvBridge()

        # Initialize previous contours
        self.previous_potholes = []
        self.pothole_id = 0
        # self.tracker = cv2.TrackerGOTURN_create()
        # self.tracker.read('goturn.prototxt')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().info(f'Error converting image: {str(e)}')
            return

        # Apply magenta color contour detection
        result = self.detect_magenta_contours(cv_image)
        
        # Display the result
        cv2.imshow('Magenta Contours', result)
        cv2.waitKey(1)
    
    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg) 
        camera_matrix = msg.k
        distortion_coefficients = msg.d

        # Print or use camera info data as per your requirement
        self.get_logger().info(f'Camera Info - Camera Matrix: {camera_matrix}, Distortion Coefficients: {distortion_coefficients}')
    
    def detect_magenta_contours(self, image):
        # Convert the image from BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for magenta color in HSV
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])

        # Threshold the image to obtain the magenta color regions
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area or other criteria if needed
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                filtered_contours.append(contour)

        # Draw the contours on the original image
        result = image.copy()

        # Accumulated contours
        accumulated_contours = []
        
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the (x, y) position is the same as previous contours
            if self.is_duplicate_contour(x, y):
                continue
            
            # Add the contour coordinates to the accumulated contours
            accumulated_contours.append((x, y))
            
            # Draw the contour
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # Increment the cumulative count
            self.cumulative_count += 1

            # Display the cumulative count
            cv2.putText(result, str(self.cumulative_count),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Store the current frame's contours as previous contours for the next frame
        self.previous_potholes = accumulated_contours

        # Display the total count of unique contours
        print(f"Accumulated Contours: {self.cumulative_count}")

        return result

    def is_duplicate_contour(self, x, y):
        for prev_x, prev_y in self.previous_potholes:
            # Check if the (x, y) position is the same as previous contours within a threshold
            if abs(x - prev_x) < 10 and abs(y - prev_y) < 10:
                return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = ContourDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
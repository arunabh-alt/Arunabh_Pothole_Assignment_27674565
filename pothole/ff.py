import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Int32
from rclpy.qos import qos_profile_sensor_data
import image_geometry

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

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().info(f'Error converting image: {str(e)}')
            return

        # Apply magenta color contour detection
        result = self.detect_magenta_contours(cv_image)
        
        # Display the result
        cv2.imshow('Magenta Contours', result)
        cv2.waitKey(1)
        if len(self.counts_over_time) > 0:
                counts_msg = Int32()
                counts_msg.data = self.counts_over_time[-1]
                self.counts_publisher.publish(counts_msg)

    def detect_magenta_contours(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_magenta = np.array([0, 0, 47])
        upper_magenta = np.array([15, 15, 57])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            # Calculate the contour area dynamically based on the length of the contour
            contour_area = cv2.contourArea(contour)
            contour_length = cv2.arcLength(contour, True)
            if contour_area > 0.8 * contour_length:
                filtered_contours.append(contour)

        result = image.copy()
        unique_pothole_numbers = set()
        contours_count = 0
        unique_contours = set()

        potholes_to_remove = []

        for pothole in self.current_potholes:
            pothole['tracked'] = False

        if filtered_contours:
            for contour in filtered_contours:
                area = cv2.contourArea(contour)

                if area > 1300:
                    contour_tuple = tuple(contour.flatten())

                    found = False
                    for pothole in self.current_potholes:
                        if cv2.matchShapes(contour, pothole['contour'], cv2.CONTOURS_MATCH_I2, 0.0) < 0.1:
                            found = True
                            pothole['tracked'] = True
                            break

                    if not found:
                        # New pothole
                        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

                        center = self.get_contour_center(contour)

                        contours_count += 1
                        pothole_number = contours_count + self.cumulative_count

                        cv2.putText(result, str(pothole_number),
                                    (center[0], center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        unique_contours.add(contour_tuple)
                        unique_pothole_numbers.add(pothole_number)
                        self.current_potholes.append({'contour': contour, 'tracked': True})

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

        return result

    def get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None

    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

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

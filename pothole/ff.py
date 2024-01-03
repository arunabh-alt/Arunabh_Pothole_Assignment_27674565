import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import image_geometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point

class ContourDetectionNode(Node):
    camera_model = None
    cumulative_count = 0

    def __init__(self):
        super().__init__('contour_detection_node')

        # Subscriptions for camera image and camera info
        self.subscription = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                    self.image_callback, qos_profile=qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                        self.camera_info_callback, 
                                                        qos_profile=qos_profile_sensor_data)

        # Subscription for the occupancy grid map
        self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/map', self.occupancy_grid_callback, 10)

        self.cv_bridge = CvBridge()

        # Initialize variables to accumulate counts
        self.counts_over_time = []
        self.tracker = cv2.TrackerKCF_create()
        self.current_potholes = []

        # Occupancy grid map
        self.occupancy_grid = None

    def occupancy_grid_callback(self, msg):
        # Store the occupancy grid map for later use
        self.occupancy_grid = msg

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

    def detect_magenta_contours(self, image):
        if self.occupancy_grid is None:
            self.get_logger().warn('Occupancy grid not available. Skipping contour detection.')
            return image

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            # Check if the contour is within certain areas in the occupancy grid
            if self.is_contour_in_valid_area(contour):
                filtered_contours.append(contour)

        # Rest of your code for processing filtered contours...

    def is_contour_in_valid_area(self, contour):
        # Check if the contour falls within valid areas in the occupancy grid
        if self.occupancy_grid is not None:
            for point in contour:
                # Convert contour point to map coordinates
                map_point = self.map_coordinates_from_pixel(point[0], self.camera_model)
                
                # Check if the map point is within a valid area in the occupancy grid
                if not self.is_point_in_valid_area(map_point):
                    return False

        return True

    def is_point_in_valid_area(self, map_point):
        # Placeholder logic for simplicity
        if self.occupancy_grid is not None:
            # Assuming occupancy_grid is a 2D array where each cell represents occupancy (0 for free, 100 for occupied)
            # You may need to adapt this based on the actual structure of your occupancy grid
            x_index = int(map_point.x)
            y_index = int(map_point.y)

            if 0 <= x_index < len(self.occupancy_grid) and 0 <= y_index < len(self.occupancy_grid[0]):
                # Check if the cell is not occupied
                return self.occupancy_grid[x_index][y_index] == 0

        return False  # Default to False if occupancy grid is not available or point is outside grid

    def map_coordinates_from_pixel(self, pixel, camera_model):
        # Placeholder logic for simplicity
        u, v = pixel
        ray = camera_model.projectPixelTo3dRay((u, v))
        return Point(x=ray[0], y=ray[1], z=ray[2])

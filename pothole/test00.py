import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy import qos

# OpenCV
import cv2

# ROS libraries
import image_geometry
from tf2_ros import Buffer, TransformListener

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped 
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import  Int32
import math

class OccupancyGridUpdater(Node):
    camera_model = None
    image_depth_ros = None

    visualisation = True
    color2depth_aspect = 1.0
    def __init__(self):
        super().__init__('occupancy_grid_updater')
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        
        self.object_location_pub = self.create_publisher(PoseStamped, '/limo/object_location', 10)

        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                self.image_color_callback, qos_profile=qos.qos_profile_sensor_data)
        self.occupancy_grid = np.zeros((100, 100), dtype=np.float32)  # Adjust the size as needed
        self.cv_bridge = CvBridge()

    def get_tf_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None

    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def image_depth_callback(self, msg):
        self.image_depth_ros = msg

    def image_color_callback(self, msg):
        if self.camera_model is None or self.image_depth_ros is None:
            return

        try:
            image_color = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            image_depth = self.cv_bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # Convert color image to HSV
        hsv_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for magenta in HSV
        magenta_lower = np.array([150, 100, 100])
        magenta_upper = np.array([170, 255, 255])

        # Create a mask to extract the magenta color
        mask = cv2.inRange(hsv_image, magenta_lower, magenta_upper)
        image_mask = cv2.bitwise_and(image_color, image_color, mask=mask)

        # Convert the binary mask to a single-channel image
        image_mask_single_channel = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        M = cv2.moments(image_mask_single_channel)
        if M["m00"] == 0:
            print('No pothole has been detected.')
        else:
            contours, _ = cv2.findContours(image_mask_single_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(contours)
            image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])
            contours_count = len(contours)
            if area > 1000:
                cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
                cv2.putText(image_color, str(contours_count), (int(image_coords[1]), int(image_coords[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Image",image_color)
            cv2.waitKey(1)    
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume contours is a list of detected contours, each represented by a list of points

        # Update the occupancy grid based on the detected contours
        self.update_occupancy_grid(contours)

    def update_occupancy_grid(self, contours):
        # Iterate through detected contours and update occupancy grid
        for contour in contours:
            # Transform contour coordinates if needed
            # Map contour coordinates to grid cells
            contour_cells = np.round(contour).astype(int)

            # Update occupancy grid cells
            self.occupancy_grid[contour_cells[:, 1], contour_cells[:, 0]] = 1.0

        # Publish or use the updated occupancy grid as needed
        # You may want to use a ROS publisher to send the updated map to other nodes

def main(args=None):
    rclpy.init(args=args)
    occupancy_grid_updater = OccupancyGridUpdater()
    rclpy.spin(occupancy_grid_updater)
    occupancy_grid_updater.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

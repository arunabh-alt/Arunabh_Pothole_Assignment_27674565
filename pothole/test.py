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

class ObjectDetector(Node):
    camera_model = None
    image_depth_ros = None

    visualisation = True
    color2depth_aspect = 1.0

    def __init__(self):    
        super().__init__('image_projection_3')
        self.bridge = CvBridge()

        # Subscriptions for camera info and images
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        
        self.object_location_pub = self.create_publisher(PoseStamped, '/limo/object_location', 10)

        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                self.image_color_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', 
                                                self.image_depth_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for pothole count
        self.pothole_count_publisher = self.create_publisher(Int32, '/total_potholes', 10)

        # Counter for the number of color masks
        self.color_mask_counter = 0

        # Store the last detected number of contours
        self.last_num_contours = 0
        self.detected_potholes = []
        self.accumulated_contours = 0
        self.occupancy_grid = np.zeros((480, 640), dtype=np.uint8)

    def update_occupancy_grid(self, contours):
        cv2.drawContours(self.occupancy_grid, contours, -1, 255, thickness=cv2.FILLED)
    def get_tf_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None
    def camera_info_callback(self, data):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def calculate_contour_size(self, contours):
        if not contours:
            return 0.0  # No contours, so size is zero
        contour_area = cv2.contourArea(contours[0])
        scaling_factor = 0.1  
        # Calculate the size in nanometers
        contour_size_nanometers = scaling_factor * np.sqrt(contour_area)

        return contour_size_nanometers

    def image_color_callback(self, data):
        if self.camera_model is None or self.image_depth_ros is None:
            return

        try:
            image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        except CvBridgeError as e:
            print(e)
            return
        num_contours_array = []
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

        # Calculate moments using the single-channel image
        M = cv2.moments(image_mask_single_channel)
        num_contours = 0
        if M["m00"] == 0:
            print('No pothole has been detected.')
            # If magenta is not detected, use the last detected number of contours
            num_contours = self.last_num_contours
        else:
            # Increment the color mask counter
            self.color_mask_counter += 1

            # Find contours in the color mask
            contours, _ = cv2.findContours(image_mask_single_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check for new potholes
            new_potholes = []
            for contour in contours:
                # Calculate the centroid of the pothole
                M = cv2.moments(contour)

                # Check if the area (M["m00"]) is non-zero before performing the division
                if M["m00"] != 0:
                    centroid = (M["m01"] / M["m00"], M["m10"] / M["m00"])
                    # Check if the pothole is close to existing ones
                    is_close_to_existing = any(math.sqrt((centroid[0] - p[0])**2 + (centroid[1] - p[1])**2) < 1 for p in self.detected_potholes)
                    
                    if not is_close_to_existing:
                        new_potholes.append(centroid)
                else:
                    print("Duplicate Potholes Detected")

            # Add new potholes to the list of detected potholes
            self.detected_potholes.extend(new_potholes)

            # Check if there are valid potholes before further processing
            if new_potholes:
                # Draw contours on the original image
                cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
                cv2.drawContours(image_depth, contours, -1, (0, 255, 0), 2)
                # Count the number of contours
                num_contours = len(new_potholes)
                num_contours_array.append(num_contours)
                # Store the last detected number of contours
                self.last_num_contours = num_contours
                #M = cv2.moments(image_mask_single_channel)

                if M["m00"] != 0:
                    # Calculate the centroid of the pothole
                    x_position = M["m01"] / M["m00"]
                    y_position = M["m10"] / M["m00"]
                    image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])
                    camera_coords = self.camera_model.projectPixelTo3dRay((image_coords[1], image_coords[0]))
                    # "map" from color to depth image
                    depth_coords = (image_depth.shape[0]/2 + (image_coords[0] - image_color.shape[0]/2)*self.color2depth_aspect, 
                        image_depth.shape[1]/2 + (image_coords[1] - image_color.shape[1]/2)*self.color2depth_aspect)
                    # get the depth reading at the centroid location
                    depth_value = image_depth[int(depth_coords[0]), int(depth_coords[1])]  # you might need to do some boundary checking first!
                    # Update occupancy grid
                    self.update_occupancy_grid(contours)

                    # Calculate the size of the contours in nanometers
                    contour_size_nanometers = self.calculate_contour_size(contours)
                    # Convert the size to a string for displaying
                    contour_size_text = f"Contour Size: {contour_size_nanometers:.2f} nm"
                    # Put the text on the image
                    
                    # image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])
                    # camera_coords = self.camera_model.projectPixelTo3dRay((image_coords[1], image_coords[0]))

                    # # "map" from color to depth image
                    # depth_coords = (image_depth.shape[0]/2 + (image_coords[0] - image_color.shape[0]/2)*self.color2depth_aspect, 
                    #     image_depth.shape[1]/2 + (image_coords[1] - image_color.shape[1]/2)*self.color2depth_aspect)
                    # # get the depth reading at the centroid location
                    # depth_value = image_depth[int(depth_coords[0]), int(depth_coords[1])]  # you might need to do some boundary checking first!

                    print('image coords: ', image_coords)
                    print('depth coords: ', depth_coords)
                    print('depth value: ', depth_value)

                    # Print the size of the color mask and the color mask counter
                    #print(f"Size of magenta color mask: {M['m00']}")
                    #print(f"Total color masks counted: {self.color_mask_counter}")
                    camera_coords = [x/camera_coords[2] for x in camera_coords]  # adjust the resulting vector so that z = 1
                    camera_coords = [x*depth_value for x in camera_coords]
                    print('camera coords: ', camera_coords)

                    # define a point in camera coordinates
                    object_location = PoseStamped()
                    object_location.header.frame_id = "depth_link"
                    object_location.pose.orientation.w = 1.0
                    object_location.pose.position.x = camera_coords[0]
                    object_location.pose.position.y = camera_coords[1]
                    object_location.pose.position.z = camera_coords[2]

                    # publish so we can see that in rviz
                    self.object_location_pub.publish(object_location)        

                    # print out the coordinates in the odom frame
                    transform = self.get_tf_transform('depth_link', 'odom')
                    p_camera = do_transform_pose(object_location.pose, transform)

                    print('odom coords: ', p_camera.position)

                    if self.visualisation:
                        for contour_count, contour in enumerate(num_contours_array, start=1):
                            #center = contour['center']
                        # draw circles
                            cv2.circle(image_color, (int(image_coords[1]), int(image_coords[0])), 10, 255, -1)
                            cv2.circle(image_depth, (int(depth_coords[1]), int(depth_coords[0])), 5, 255, -1)
                            cv2.putText(image_color, contour_size_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(image_color, str(contour_count), (int(image_coords[1]), int(image_coords[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # resize and adjust for visualisation
                        image_color = cv2.resize(image_color, (0, 0), fx=0.5, fy=0.5)
                        image_depth *= 1.0/10.0  # scale for visualisation (max range 10.0 m)
                            
                        cv2.imshow("image depth", image_depth)
                        cv2.imshow("image color", image_color)
                        #cv2.imshow("image depth", image_depth)
                        cv2.waitKey(1)
                    print(f"Number of pothole detected: {num_contours}")
                    if num_contours > 0:
                            pothole_count_msg = Int32()
                            pothole_count_msg.data = num_contours
                            self.pothole_count_publisher.publish(pothole_count_msg)

                else:
                    print("Warning: Contour with zero area. Skipping further processing.")

                    #image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])

            else:
                print("No valid potholes detected. Skipping further processing.")

            # Print the number of contours

def main(args=None):
    rclpy.init(args=args)
    image_projection = ObjectDetector()
    rclpy.spin(image_projection)
    image_projection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

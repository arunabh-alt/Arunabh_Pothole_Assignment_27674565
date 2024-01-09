import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy import qos
import image_geometry
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
class DepthCameraSubscriber(Node):
    def __init__(self):
        super().__init__('depth_camera_subscriber')
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        
        #self.object_location_pub = self.create_publisher(PoseStamped, '/limo/object_location', 10)

        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                self.depth_callback, qos_profile=qos.qos_profile_sensor_data)
        self.cv_bridge = CvBridge()
        self.camera_model = None
    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
    def depth_callback(self, msg):
        try:
            image_color = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")
            return
        
        hsv_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for greycolor in HSV
        lower_grey = np.array([0, 0, 47])
        upper_grey = np.array([15, 15, 70])

        # Create binary mask for grey color
        grey_mask = cv2.inRange(hsv_image, lower_grey, upper_grey)
        V= cv2.moments(grey_mask)
        if V["m00"] == 0:
            print("Blank space")
        # Find contours
        else:
            contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an image to draw the contours (color image)
            color_image_contours = np.copy(image_color)

            for i, contour in enumerate(contours):
                # Calculate the size of the contour
                contour_size = cv2.contourArea(contour)
                
                # Draw the contour on the color image
                cv2.drawContours(color_image_contours, [contour], -1, (0, 255, 0), 2)

                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    print('No pothole has been detected.')
                else:    
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    image_coords = (cx, cy)
                    
                    # Draw a circle around the contour
                    cv2.circle(color_image_contours, (cx, cy), 10, (0, 0, 255), -1)
                    
                    # Draw the contour count and size inside the circle
                    text = f"{i+1}: Size={contour_size:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
                    text_x = cx - text_size[0] // 2
                    text_y = cy + text_size[1] // 2
                    #cv2.putText(color_image_contours, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

            # Display the color image with contours
            cv2.imshow("Color Contours", color_image_contours)
            cv2.waitKey(1)
            
        



def main(args=None):
    rclpy.init(args=args)

    depth_camera_subscriber = DepthCameraSubscriber()

    rclpy.spin(depth_camera_subscriber)

    depth_camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
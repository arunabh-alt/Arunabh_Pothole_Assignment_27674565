import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from rclpy import qos
from sensor_msgs.msg import PointCloud2
import image_geometry

class ObstacleAvoidanceAndPositionTracking(rclpy.node.Node):
    camera_model = None

    def __init__(self):
        super().__init__('obstacle_avoidance_and_position_tracking')

        # Subscriptions for laser scan, position data, and pothole count
        self.scan_subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        self.position_subscription = self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        
        # Subscriber for pothole count
        self.pothole_count_subscription = self.create_subscription(Int32, 'counts_over_time', self.pothole_count_callback, 10)

        # Publisher for velocity commands
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Variables for position tracking and pothole counting
        self.starting_position = None
        self.positions = []

        # Twist message for velocity commands
        self.twist = Twist()
        self.initial_position = None
        self.is_initial_position_reached = False

        # Declare parameter only once in the constructor
        self.declare_parameter("total_pothole_count", 0)
        self.initial_movement_timer = self.create_timer(10.0, self.stop_initial_movement)

    def scan_callback(self, msg):
        min_distance = min(msg.ranges[int(len(msg.ranges)/2) -10 : int(len(msg.ranges)/2) +10])

        if min_distance < 0.55:  # Threshold distance for obstacle detection
            self.twist.linear.x = 0.0  # Stop forward motion
            self.twist.angular.z = -0.7  # Rotate to avoid obstacle
        else:
            self.twist.linear.x = 0.5  # Move forward
            self.twist.angular.z = 0.4  # No rotation

        self.publisher.publish(self.twist)

        if self.initial_position is None:
            self.initial_position = self.get_current_position()

    def odometry_callback(self, msg):
        if hasattr(self, 'current_position'):
            # Check if initial movement has completed
            if len(self.positions) == 0:
                self.positions.append(msg.pose.pose)
            else:
                # Only append positions after the initial movement
                if self.distance_between_poses(msg.pose.pose, self.positions[-1]) > 0.05:
                    self.positions.append(msg.pose.pose)

                # Now you can check if the current position is close to the initial position
                if self.distance_between_poses(msg.pose.pose, self.initial_position) < 0.05:
                    if not self.is_initial_position_reached:
                        self.get_logger().info("Robot is back at the initial position")
                        self.is_initial_position_reached = True
                    self.twist.linear.x = 0.0
                    self.twist.angular.z= 0.0
                elif self.is_initial_position_reached:
                    self.is_initial_position_reached = False  # Reset the flag when the robot moves away from the initial position
        else:
            # Store initial position during initial movement
            self.current_position = msg.pose.pose

    def distance_between_poses(self, pose1, pose2):
        # Check if either pose is None
        if pose1 is None or pose2 is None:
            return float('inf')  # or any large value to indicate infinite distance

        # Check if the 'position' attribute is present in the poses
        if hasattr(pose1, 'position') and hasattr(pose2, 'position'):
            # Calculate the Euclidean distance between two Pose objects
            return ((pose1.position.x - pose2.position.x)**2 +
                    (pose1.position.y - pose2.position.y)**2 +
                    (pose1.position.z - pose2.position.z)**2)**0.5
        else:
            return float('inf')

    def get_current_position(self):
        return self.current_position

    def camera_info_callback(self, msg):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def pothole_count_callback(self, msg):
        total_pothole_count = msg.data
        if self.is_initial_position_reached:
            self.get_logger().info(f'Total Potholes Detected: {total_pothole_count}')

    def stop_initial_movement(self):
        # Stop the initial movement after 10 seconds
        self.initial_movement_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_and_position_tracking = ObstacleAvoidanceAndPositionTracking()
    rclpy.spin(obstacle_avoidance_and_position_tracking)
    obstacle_avoidance_and_position_tracking.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

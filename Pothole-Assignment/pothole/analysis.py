import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class PotholeAnalysisNode(Node):

    def __init__(self):
        super().__init__('pothole_analysis_node')
        self.pothole_number_subscription = self.create_subscription(
            Int32,
            'pothole_number',
            self.pothole_number_callback,
            10
        )
        self.size_subscription = self.create_subscription(
            Int32,
            'size_real_time',
            self.size_callback,
            10
        )

    def pothole_number_callback(self, msg):
        pothole_number = msg.data
        self.get_logger().info(f'Received Pothole Number: {pothole_number}')

    def size_callback(self, msg):
        size = msg.data
        self.get_logger().info(f'Received Pothole Size: {size} nm')

        # Analyze pothole size
        if size > 4000:
            self.get_logger().info('Bad condition - Large pothole!')
        elif 4000>size and size>2000:
            self.get_logger().info('Medium Size Pothole')
        else:
            self.get_logger().info('Minor pothole.')

def main(args=None):
    rclpy.init(args=args)

    pothole_analysis_node = PotholeAnalysisNode()

    rclpy.spin(pothole_analysis_node)

    pothole_analysis_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import mediapipe as mp


class FollowMe(Node):
    def __init__(self):
        super().__init__('follow_me')
        self.bridge = CvBridge()

        # Subscribing to camera feed
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Publisher for movement
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Debug image publisher
        self.debug_pub = self.create_publisher(Image, '/mediapipe_followme/image', 10)

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)

        self.image_width = 640
        self.center_x = self.image_width // 2  # 160
        self.tolerance_px = 20  # How close to the center before moving forward
        self.prev_linear_x = 0.0

    def image_callback(self, msg):
        twist = Twist()

        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_width = cv_image.shape[1]

        # Process image
        results = self.pose.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                cv_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark
            ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]

            ############################
            # Print left ankle coordinates
            print(f"Normalized (X,Y,Z): ({ankle.x:.3f}, {ankle.y:.3f}, {ankle.z:.3f})")
            print(f"Pixel (X,Y): ({int(ankle.x * self.image_width)}, {int(ankle.y * cv_image.shape[0])})")
            ############################

            ############################
            # Get ankle x and y in pixels
            ankle_x = int(ankle.x * self.image_width)
            ankle_y = int(ankle.y * cv_image.shape[0])

	    # Draw blue circle around ankle
            cv2.circle(cv_image, (ankle_x, ankle_y), 7, (255, 0, 0), 2)

            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.debug_pub.publish(debug_msg)
	    ############################


            offset = ankle_x - self.center_x  # How far from center


            if ankle.visibility < 0.5:
                self.get_logger().info('Ankle not visible. Stopping.')
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            # Angular control: rotate to center the ankle
            elif abs(offset) > self.tolerance_px:
                twist.angular.z = -0.003 * offset  # negative because camera frame
                twist.linear.x = 0.0
                self.get_logger().info(f'Centering ankle: offset={offset}, turning...')
		    
            else:
                # Ankle is centered -> move forward slowly
                twist.angular.z = 0.0
                twist.linear.x = 0.1
                self.get_logger().info(f'Ankle centered. Moving forward.')

        else:
            self.get_logger().info('No landmarks. Stopping.')
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        # Send debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FollowMe()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





#!/usr/bin/env python3

import rospy
import math
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf.transformations
import time
import numpy as np # We'll use this for interpolation

class WaypointNavigator:
    """
    A modular waypoint navigator that simulates human-like movement.
    - Moves fast on straights.
    - Automatically slows down when approaching sharp turns.
    - Uses proportional steering for smoother curves.
    - Adds noise for realism.
    """
    def __init__(self, waypoints):
        rospy.init_node('waypoint_navigator', anonymous=True)
        rospy.loginfo("Starting Human-Like Waypoint Navigator Node...")

        # --- Configuration ---
        self.cmd_vel_topic = '/actor/cmd_vel'
        self.odom_topic = '/actor/odom'
        
        # Modular waypoints, passed in during creation
        self.waypoints = waypoints
        if not self.waypoints:
            rospy.logerr("No waypoints provided! Shutting down.")
            rospy.signal_shutdown("No waypoints")
            return

        # --- NEW: Human-like speed parameters ---
        self.max_linear_speed = 0.6     # Speed on straights (m/s)
        self.min_linear_speed = 0.2        # Speed on corners (m/s)
        self.slowing_distance = 1.5        # How far from a corner to start slowing (m)
        self.corner_threshold_angle = 0.5  # Angle (rad) to define a "corner" (~30 deg)
        self.turn_p_gain = 0.8             # Proportional gain for steering
        
        # --- NEW: Noise parameters ---
        self.linear_noise = 0.0           # Max linear wiggle (+/- m/s)
        self.angular_noise = 0.1           # Max angular drift (+/- rad/s)
        
        # How close to get to a waypoint
        self.distance_threshold = 0.2

        # ROS Publishers and Subscribers
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        # State variables
        self.current_pos = None
        self.current_yaw = 0.0
        self.target_waypoint_index = 0

    def odom_callback(self, msg):
        """Callback to update the model's current position and orientation."""
        self.current_pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.current_yaw = yaw

    def normalize_angle(self, angle):
        """Normalize an angle to be between -pi and pi."""
        if angle > math.pi:
            angle -= 2 * math.pi
        if angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def is_target_a_corner(self, current_index):
        """
        Checks if the waypoint at current_index is a "corner" by
        comparing the angle of the segment leading into it and the
        segment leading out of it.
        """
        # Get indices for the previous, current, and next waypoints
        num_waypoints = len(self.waypoints)
        prev_index = (current_index - 1) % num_waypoints
        next_index = (current_index + 1) % num_waypoints

        # Get the points
        prev_point = self.waypoints[prev_index]
        curr_point = self.waypoints[current_index]
        next_point = self.waypoints[next_index]

        # Calculate the angle of the two segments
        angle_in = math.atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
        angle_out = math.atan2(next_point[1] - curr_point[1], next_point[0] - curr_point[0])

        # Find the difference
        angle_diff = abs(self.normalize_angle(angle_in - angle_out))

        # Return True if the angle difference is sharp enough
        return angle_diff > self.corner_threshold_angle

    def run(self):
        """Main loop to navigate through waypoints."""
        rate = rospy.Rate(10) # 10 Hz
        
        while not rospy.is_shutdown():
            if self.current_pos is None:
                rospy.loginfo("Waiting for odometry data...")
                rate.sleep()
                continue

            # Get the current target waypoint
            target_x, target_y = self.waypoints[self.target_waypoint_index]

            # Calculate distance and angle to the target
            delta_x = target_x - self.current_pos.x
            delta_y = target_y - self.current_pos.y
            distance = math.sqrt(delta_x**2 + delta_y**2)
            angle_to_target = math.atan2(delta_y, delta_x)

            # Check if we've reached the waypoint
            if distance < self.distance_threshold:
                rospy.loginfo(f"Waypoint {self.target_waypoint_index} reached!")
                self.target_waypoint_index = (self.target_waypoint_index + 1) % len(self.waypoints)
                continue

            # --- NEW: Speed Control Logic ---
            # Check if the target we're heading *towards* is a corner
            is_corner = self.is_target_a_corner(self.target_waypoint_index)
            
            # Start at max speed
            current_linear_speed = self.max_linear_speed
            
            # If it's a corner AND we're within slowing distance, slow down
            if is_corner and distance < self.slowing_distance:
                # We'll use numpy's interpolation to smoothly scale our speed
                # from min_speed (at 0m) to max_speed (at slowing_distance)
                current_linear_speed = np.interp(
                    distance,
                    [0, self.slowing_distance],
                    [self.min_linear_speed, self.max_linear_speed]
                )
                # Clamp to the minimum speed
                current_linear_speed = max(self.min_linear_speed, current_linear_speed)

            # --- NEW: Steering Control Logic (Proportional) ---
            # This makes the robot steer *while* it moves
            angle_diff = self.normalize_angle(angle_to_target - self.current_yaw)
            
            move_cmd = Twist()
            # Set angular speed based on how far we need to turn
            move_cmd.angular.z = self.turn_p_gain * angle_diff
            
            # Only move forward if we are generally facing the target
            # This prevents driving sideways or backwards
            if abs(angle_diff) < math.pi / 2: # 90 degrees
                move_cmd.linear.x = current_linear_speed

            # --- NEW: Add Noise for Realism ---
            if move_cmd.linear.x > 0:
                 move_cmd.linear.x += random.uniform(-self.linear_noise, self.linear_noise)
            
            move_cmd.angular.z += random.uniform(-self.angular_noise, self.angular_noise)
            
            # Publish the final command
            self.cmd_pub.publish(move_cmd)
            rate.sleep()

        # Stop the robot on shutdown
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        # --- MODULAR WAYPOINTS ---
        # Define your test paths here and pass them to the navigator

        # Path 1: The simple square
        path_square = [
            (6.0, 6.0),
            (-6.0, 6.0),
            (-6.0, -6.0),
            (6.0, -6.0)
        ]

        # Path 2: A more complex "S" shape
        path_s_curve = [
            (0.0, -4.0),
            (2.0, -4.0),
            (4.0, -2.0),
            (4.0, 0.0),
            (2.0, 2.0),
            (0.0, 4.0),
            (-2.0, 4.0),
            (-4.0, 2.0),
            (-4.0, 0.0),
            (-2.0, -2.0),
        ]

        # Path 1: "Tall" Zig-Zag (Moves up the Y-axis)
        path_tall_zigzag = [
            (0.0, 0.0),
            (2.0, 2.0),
            (-2.0, 4.0),
            (2.0, 6.0),
            (-2.0, 8.0),
            (0.0, 10.0) # Finishes in the middle
        ]

        # Path 2: "Wide" Zig-Zag (Moves across the X-axis)
        path_wide_zigzag = [
            (0.0, 0.0),
            (2.0, 2.0),
            (4.0, -2.0),
            (6.0, 2.0),
            (8.0, -2.0),
            (10.0, 0.0) # Finishes in the middle
        ]
        
        # Path 3: Sharp, tight zig-zag
        path_tight_zigzag = [
            (0.0, 0.0),
            (1.0, 2.0),
            (0.0, 4.0),
            (1.0, 6.0),
            (0.0, 8.0)
        ]

        # Path that works with the "S-Maze" world
        path_s_maze_loop = [
            (4.0, 4.0), # Go to bottom-right corner
            (4.0, 1.0),  # Go left
            (-4.0, 1.0), # Go to bottom-left corner
            (-4.0, -4.0),  # Go up to middle-left
            (4.0, -4.0),  # Go to top-left corner
            (4.0, -3.0),   # Go right
            (-3.0, -3.0),   # Go to top-right corner
            (-3.0, -1.0),
            (3.0,-1.0),
            (3.0,3.0),
            (-4.0,3.0),
            (-4.0,4.0)    # Go to middle-right (near start)
        ]

        path_maze_extra = [(0.0,-5.0),
                            (5.0,-5.0),
                           (0.0,-5.0),
                           (-5.0,-5.0)]
        
        # Select which path to run
        # -----------------------------
        # navigator = WaypointNavigator(path_square)
        
        navigator = WaypointNavigator(path_maze_extra)
        # -----------------------------
        #rospy.sleep(2)
        navigator.run()

    except rospy.ROSInterruptException:
        pass
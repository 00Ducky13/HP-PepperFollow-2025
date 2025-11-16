#!/usr/bin/env python3
import rospy
import os

if __name__ == '__main__':
    try:
        rospy.init_node('test_timer_node', anonymous=True)
        
        test_duration_seconds = 300  # 5 minutes
        
        rospy.loginfo("Test timer started. Running for 5 minutes.")
        
        # This will sleep for the duration, respecting ROS time 
        # (if use_sim_time is true) or wall clock time.
        rospy.sleep(test_duration_seconds)
        
        rospy.loginfo("5 minutes elapsed. Shutting down test.")
        
        # When this node exits, because it's 'required' in the launch
        # file, roslaunch will terminate everything else.
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Timer interrupted.")

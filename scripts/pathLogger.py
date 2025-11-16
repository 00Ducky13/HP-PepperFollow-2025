#!/usr/bin/env python3
import rospy
import csv
import os
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import time
import threading # <-- Added
from std_msgs.msg import Bool # <-- Added

# --- User Configuration ---
# !!_IMPORTANT: Change these to match your model names in Gazebo_!!
MODEL_NAME_PERSON = "scrubs" 
MODEL_NAME_PEPPER = "pepper_MP"
# --------------------------

# Global variables
csv_writer = None
csv_file = None
start_time = None
g_target_acquired = False # <-- Added: Default to False
g_target_status_lock = threading.Lock() # <-- Added: For thread-safe updates

def target_status_callback(msg):
    """
    Callback function for the /pepper/target_status topic.
    Updates the global variable for target acquisition status.
    """
    global g_target_acquired, g_target_status_lock
    
    with g_target_status_lock:
        g_target_acquired = msg.data

def model_states_callback(msg):
    """
    Callback function for the /gazebo/model_states topic.
    Finds the person and pepper models and logs their positions.
    """
    global csv_writer, start_time, g_target_acquired, g_target_status_lock

    try:
        # Find the indices of the models
        person_index = msg.name.index(MODEL_NAME_PERSON)
        pepper_index = msg.name.index(MODEL_NAME_PEPPER)
    except ValueError:
        # One of the models isn't in the list yet
        rospy.logwarn_throttle(5, "Waiting for models '{}' and '{}' to appear in Gazebo...".format(MODEL_NAME_PERSON, MODEL_NAME_PEPPER))
        return

    # Get the poses
    person_pose = msg.pose[person_index]
    pepper_pose = msg.pose[pepper_index]

    # Get the current ROS time
    # If this is the first message, record the start time
    if start_time is None:
        start_time = rospy.get_rostime()

    current_time = (rospy.get_rostime() - start_time).to_sec()

    # Get the current target status in a thread-safe way
    with g_target_status_lock:
        current_target_status = g_target_acquired

    # Write data to CSV
    # [timestamp, person_x, person_y, person_z, pepper_x, pepper_y, pepper_z]
    csv_writer.writerow([
        "{:.4f}".format(current_time),
        "{:.4f}".format(person_pose.position.x),
        "{:.4f}".format(person_pose.position.y),
        "{:.4f}".format(person_pose.position.z),
        "{:.4f}".format(pepper_pose.position.x),
        "{:.4f}".format(pepper_pose.position.y),
        "{:.4f}".format(pepper_pose.position.z),
        current_target_status # <-- Added
    ])

def cleanup_shutdown():
    """
    Called on node shutdown (e.g., Ctrl+C).
    Closes the CSV file.
    """
    global csv_file
    if csv_file:
        rospy.loginfo("Closing CSV log file.")
        csv_file.close()

def main():
    global csv_writer, csv_file
    rospy.init_node('path_logger', anonymous=True)

    # Define the log file path (e.g., in the user's home directory)
    log_filename = "pepperResearch/testing/following/5_headTracking/tracking_log_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))
    log_filepath = os.path.join(os.path.expanduser('~'), log_filename)

    try:
        # Open the CSV file for writing
        csv_file = open(log_filepath, 'w')
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow([
            "timestamp", 
            "person_x", "person_y", "person_z", 
            "pepper_x", "pepper_y", "pepper_z",
            "target_acquired"
        ])

        rospy.loginfo("Logging model positions to: {}".format(log_filepath))

        # Register the shutdown hook
        rospy.on_shutdown(cleanup_shutdown)

        # Subscribe to the model states topic
        rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)

        # --- Added Subscriber ---
        # Subscribe to the target status topic from the perception node
        rospy.Subscriber('/pepper/target_status', Bool, target_status_callback)
        # ------------------------

        # Keep the node alive
        rospy.spin()

    except IOError as e:
        rospy.logerr("Failed to open log file: {}".format(e))
    except rospy.ROSInterruptException:
        pass
    finally:
        # Redundant cleanup, but good practice
        if csv_file:
            csv_file.close()

if __name__ == '__main__':
    main()
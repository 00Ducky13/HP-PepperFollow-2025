#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
import collections
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf2_ros
import tf.transformations
import tf2_geometry_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from threading import Thread
# from collections import deque # deque is already imported via collections

# --- 1. NEW IMPORTS ---
import torch
import torchreid
import torchvision.transforms as T
import torch.nn.functional as F
# ---

class PepperPathFollower:
    def __init__(self):
        rospy.init_node('pepper_path_follower', anonymous=True)
        rospy.loginfo("Starting Pepper Path Following Node...")

        self.target_status_pub = rospy.Publisher('/pepper/target_status', Bool, queue_size=1)
### NEW: Read visualization parameter ###
        # Get the private parameter '~visualize', default to False if not set
        self.enable_visualization = rospy.get_param('~visualize', False)
        if self.enable_visualization:
            rospy.loginfo("Visualization enabled.")
        else:
            rospy.loginfo("Visualization disabled.")

        # --- 2. MODIFIED: Load Detector (NOT pose) and Re-ID Model ---
        self.model = YOLO('yolov8n.pt') # Use standard detector
        rospy.loginfo("Loading Re-ID model (osnet_x0_25)...")
        self.reid_model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1, # Not used for feature extraction
            pretrained=True
        )
        self.reid_model.eval()
        if torch.cuda.is_available():
            self.reid_model = self.reid_model.cuda()

        # Create the image pre-processor for the Re-ID model
        self.reid_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.bridge = CvBridge()
        self.known_people = {} # Now stores {id: feature_vector}
        self.potential_people = {}
        self.next_person_id = 0
        
        self.REID_MATCH_THRESHOLD = 0.7 # Cosine similarity threshold (higher is better)
        self.CONFIDENCE_THRESHOLD = 3
        self.MIN_BBOX_AREA = 50 * 50
        self.REID_GATE_DISTANCE = 1.5
        # --- END OF MODIFICATION ---

        ### NEW: Depth Confirmation Parameters ###
        self.DEPTH_STD_DEV_THRESHOLD = 0.05 # Min std dev (meters) to be 'not flat'
        self.MIN_VALID_DEPTH_PERCENT = 0.1 # Min % of valid depth points needed

        # --- Robot Control and Navigation Setup ---
        self.cmd_pub = rospy.Publisher('/pepper/cmd_vel', Twist, queue_size=10)
        self.TURN_P_GAIN = 0.5; self.MAX_ANGULAR_SPEED = 1.2
        self.LINEAR_P_GAIN = 0.5; self.MAX_LINEAR_SPEED = 1.0
        self.DISTANCE_THRESHOLD = 0.6
        self.WAYPOINT_TIME_THRESHOLD = rospy.Duration(2)
        self.MIN_TARGET_DISTANCE = 0.9 # Personal space radius
        self.NAVIGATION_START_DISTANCE = 1.2
        self.BACKUP_DIST = 1

        # --- Look-ahead averaging setup ---
        self.navigation_target_marker_id = 9997

        # --- Running Average Setup ---
        self.VELOCITY_AVERAGE_WINDOW = 10
        self.velocity_history = collections.deque(maxlen=self.VELOCITY_AVERAGE_WINDOW)
        self.running_average_velocity = 0.0

        # --- Adaptive Look-ahead Parameters ---
        self.MIN_LOOKAHEAD = 2
        self.MAX_LOOKAHEAD = 8
        self.MIN_VELOCITY_THRESHOLD = 0.2
        self.MAX_VELOCITY_THRESHOLD = 1.0

        # --- Head Control Setup ---
        self.head_pub = rospy.Publisher('/pepper/Head_controller/command',
                                        JointTrajectory,
                                        queue_size=1)
        self.head_traj_msg = JointTrajectory()
        self.head_traj_msg.joint_names = ['HeadYaw', 'HeadPitch']
        self.head_traj_point = JointTrajectoryPoint()

        self.HEAD_YAW_P_GAIN = -0.0015
        self.HEAD_PITCH_P_GAIN = -0.0015
        self.current_head_yaw = 0.0
        self.current_head_pitch = 0.0

        self.HEAD_YAW_MIN = rospy.get_param("~head_yaw_min", -2.0857)
        self.HEAD_YAW_MAX = rospy.get_param("~head_yaw_max", 2.0857)
        self.HEAD_PITCH_MIN = rospy.get_param("~head_pitch_min", -0.7068)
        self.HEAD_PITCH_MAX = rospy.get_param("~head_pitch_max", 0.4451)

        # Head smoothing
        self.SMOOTHING_FACTOR = 0.8
        self.smoothed_head_error_x = 0.0
        self.smoothed_head_error_y = 0.0

        # --- Marker Publisher Setup ---
        self.marker_pub = rospy.Publisher('/pepper/waypoint_markers', MarkerArray, queue_size=1)

        ### NEW: Robot Pose Marker Setup ###
        self.pose_marker_pub = rospy.Publisher('/pepper/pose_marker', Marker, queue_size=1)
        self.robot_marker_id = 9998 # Unique ID for the robot pose marker

        self.marker_list = []
        self.next_marker_id = 0

        # --- State Machine and Path Queue ---
        self.robot_state = 'SEARCHING'
        self.last_target_seen_time = rospy.Time(0)
        self.LOST_TIMEOUT = rospy.Duration(1.0)
        self.SEARCH_SPEED = 0.4
        self.waypoint_queue = collections.deque()
        self.last_waypoint_log_time = rospy.Time(0)
        self.current_target_distance = None
        self.last_known_target_bbox = None # Stores the CONFIRMED bbox for control logic

        # --- TF2, Odom, and CameraInfo Setup ---
        self.tf_buffer = tf2_ros.Buffer(); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.current_pos = None; self.current_yaw = 0.0
        self.camera_info = None
        self.cam_info_sub = rospy.Subscriber("/pepper/camera/front/camera_info", CameraInfo, self.cam_info_callback)
        color_sub = message_filters.Subscriber("/pepper/camera/front/image_raw", Image)
        depth_sub = message_filters.Subscriber("/pepper/camera/depth/image_raw", Image)
        self.odom_sub = rospy.Subscriber("/pepper/odom", Odometry, self.odom_callback)
        self.joint_sub = rospy.Subscriber("/pepper/joint_states", JointState, self.joint_state_callback)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.perception_callback)

        # --- Prediction Variables ---
        self.PREDICTION_TIME_SEC = 1.5
        self.last_target_odom_point = None
        self.last_target_odom_time = None
        self.last_target_velocity_odom = (0, 0)
        self.predicted_target_point = None

        ### NEW: Time-based Waypoint Popping ###
        self.WAYPOINT_POP_INTERVAL = rospy.Duration(2.0) # Pop a waypoint every 2 seconds
        self.last_waypoint_pop_time = rospy.Time.now()

        # --- 1. NEW: Parameter for Re-ID Gating ---
        self.REID_GATE_DISTANCE = 0.8 # (meters) Only check people within this radius of the prediction
        # --- END OF NEW ---


    # --- Marker Helper Functions (unchanged) ---
    def publish_delete_all_markers(self):
        delete_all_marker = Marker()
        delete_all_marker.header.frame_id = "odom"; delete_all_marker.header.stamp = rospy.Time.now()
        delete_all_marker.ns = "waypoints"; delete_all_marker.id = 0
        delete_all_marker.action = Marker.DELETEALL
        marker_array = MarkerArray(markers=[delete_all_marker])
        self.marker_pub.publish(marker_array)
        # rospy.loginfo("Cleared all waypoint markers.") # Can be noisy

    def publish_markers(self):
        if not self.marker_list: return
        self.marker_pub.publish(MarkerArray(markers=self.marker_list))

    # --- Callbacks (cam_info, joint_state, odom) (unchanged) ---
    def cam_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("Camera info received.")
            self.camera_info = msg
            self.cam_info_sub.unregister()

    def joint_state_callback(self, msg):
        try:
            idx_yaw = msg.name.index('HeadYaw'); idx_pitch = msg.name.index('HeadPitch')
            self.current_head_yaw = msg.position[idx_yaw]; self.current_head_pitch = msg.position[idx_pitch]
        except ValueError:
            rospy.logwarn_throttle(5, "Could not find HeadYaw or HeadPitch in /pepper/joint_states")

    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.current_yaw) = tf.transformations.euler_from_quaternion(orientation_list)

    # --- 3. REPLACED: All histogram functions are replaced by this ---
    @torch.no_grad() # Disable gradient calculation for speed
    def get_reid_features(self, crops):
        """Extracts feature vectors from a batch of cropped images."""
        if not crops:
            return []
        
        # Apply the transforms to each crop
        tensors = [self.reid_transform(crop) for crop in crops]
        batch = torch.stack(tensors) # Stack into a single batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        
        # Get the feature vectors (embeddings)
        features = self.reid_model(batch)
        # Normalize features for cosine similarity
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()
    # --- END OF REPLACEMENT ---

    def deproject_pixel_to_point(self, u, v, depth):
        if self.camera_info is None:
            rospy.logwarn_throttle(5, "Waiting for camera info for accurate deprojection...")
            return None
        fx = self.camera_info.K[0]; fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]; cy = self.camera_info.K[5]
        x = (u - cx) * depth / fx; y = (v - cy) * depth / fy; z = depth
        return (x, y, z)

    # --- Marker Helper Functions (unchanged) ---
    def add_prediction_marker(self, predicted_point):
        prediction_marker_id = 9999
        marker = Marker()
        marker.header.frame_id = "odom"; marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"; marker.id = prediction_marker_id
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position = predicted_point; marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25; marker.scale.y = 0.25; marker.scale.z = 0.25
        marker.color.a = 0.7; marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0 # Blue
        marker.lifetime = rospy.Duration(5.0)
        self.marker_list.append(marker)

    def add_navigation_marker(self, nav_point_tuple):
        marker = Marker()
        marker.header.frame_id = "odom"; marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"; marker.id = self.navigation_target_marker_id
        marker.type = Marker.CUBE; marker.action = Marker.ADD
        marker.pose.position.x = nav_point_tuple[0]; marker.pose.position.y = nav_point_tuple[1]; marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.a = 0.7; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0 # Yellow
        marker.lifetime = rospy.Duration(0.2)
        self.marker_list.append(marker)

    # --- log_waypoint (unchanged) ---
    def log_waypoint(self, odom_point):
        if self.current_target_distance is None or self.current_target_distance < self.MIN_TARGET_DISTANCE:
             return # Skip logging if no distance or too close

        time_since_last_log = rospy.Time.now() - self.last_waypoint_log_time


        if (time_since_last_log > self.WAYPOINT_TIME_THRESHOLD):
            current_marker_id = self.next_marker_id
            new_waypoint = (odom_point.point.x, odom_point.point.y, current_marker_id)

            self.waypoint_queue.append(new_waypoint)
            self.last_waypoint_log_time = rospy.Time.now()
            # rospy.loginfo(f"New waypoint logged: ({new_waypoint[0]:.2f}, {new_waypoint[1]:.2f})") # Can be noisy

            marker = Marker()
            marker.header.frame_id = "odom"; marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"; marker.id = current_marker_id
            marker.type = Marker.SPHERE; marker.action = Marker.ADD
            marker.pose.position = odom_point.point; marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
            marker.color.a = 0.8; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0 # Green
            marker.lifetime = rospy.Duration()

            self.marker_list.append(marker)
            self.next_marker_id += 1

    ### NEW: Function to publish robot pose marker ###
    def publish_robot_pose_marker(self):
        """Publishes an ARROW marker representing Pepper's current pose."""
        if self.current_pos is None:
             return # Need position data

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_pose"
        marker.id = self.robot_marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD # Use ADD_MODIFY if supported, ADD works too

        # Set the pose directly from odometry
        marker.pose.position = self.current_pos # Use the Point object directly
        
        # Get orientation from current_yaw
        quat = tf.transformations.quaternion_from_euler(0, 0, self.current_yaw)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # Set the scale (arrow dimensions: length, shaft_diameter, head_diameter)
        marker.scale.x = 0.5 # Arrow length (e.g., 0.5 meters)
        marker.scale.y = 0.05 # Shaft diameter
        marker.scale.z = 0.1 # Head diameter

        # Set the color (e.g., Cyan)
        marker.color.a = 1.0 # Opaque
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.lifetime = rospy.Duration(0.2) # Lasts slightly longer than update rate

        self.pose_marker_pub.publish(marker)

    # --- perception_callback (MODIFIED: Conditional Visualization) ---
    def perception_callback(self, color_msg, depth_msg):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # Only create vis_frame if visualization is enabled
            vis_frame = cv_frame.copy() if self.enable_visualization else None
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}"); return

        # --- Run YOLO on the frame needed ---
        frame_to_process = vis_frame if self.enable_visualization else cv_frame
        # Suppress verbose YOLO output
        # We now run the pose model
        # --- THIS IS THE FIX ---
        # Changed self.model(...) to self.model.track(...)
        # The .track() method accepts 'persist=True' and provides the track IDs
        results = self.model.track(frame_to_process, persist=True, classes=[0], verbose=False)
        # --- END OF FIX ---

        target_found_this_frame = False
        confirmed_target_yolo_box_core = None
        
        # The main results object is a list of results
        result = results[0] 

        predicted_target_odom = None
        if self.robot_state != 'SEARCHING' and self.last_target_odom_point is not None:
            dt = (rospy.Time.now() - self.last_target_odom_time).to_sec()
            predicted_target_odom = (
                self.last_target_odom_point.x + self.last_target_velocity_odom[0] * dt,
                self.last_target_odom_point.y + self.last_target_velocity_odom[1] * dt
            )
            
        

        if result.boxes.id is not None:
            # --- Batch process all detections for efficiency ---
            candidates = [] # Store potential people
            crops_to_process = [] # Store image crops for batch processing

            for i, box in enumerate(result.boxes.xyxy):
                x1_c, y1_c, x2_c, y2_c = map(int, box)
                track_id_c = int(result.boxes.id[i])

                if (x2_c - x1_c) * (y2_c - y1_c) < self.MIN_BBOX_AREA: continue

                # (Gating logic is the same)
                current_detection_odom = None
                try:
                    # ... (omitted get_3d_point logic for brevity) ...
                    scale_x = depth_frame.shape[1] / cv_frame.shape[1]; scale_y = depth_frame.shape[0] / cv_frame.shape[0]
                    center_x_color = (x1_c + x2_c) / 2; center_y_color = (y1_c + y2_c) / 2
                    depth_x = int(center_x_color * scale_x); depth_y = int(center_y_color * scale_y)
                    if not (0 <= depth_y < depth_frame.shape[0] and 0 <= depth_x < depth_frame.shape[1]): continue
                    dist_mm = depth_frame[depth_y, depth_x]
                    if dist_mm == 0: continue
                    dist_m = dist_mm / 1000.0
                    point_3d = self.deproject_pixel_to_point(center_x_color, center_y_color, dist_m)
                    if point_3d is None: continue
                    cam_point = PointStamped(); cam_point.header.frame_id = depth_msg.header.frame_id; cam_point.header.stamp = depth_msg.header.stamp
                    cam_point.point.x, cam_point.point.y, cam_point.point.z = point_3d[0], point_3d[1], point_3d[2]
                    current_detection_odom = self.tf_buffer.transform(cam_point, 'odom', timeout=rospy.Duration(0.1))
                    
                    if predicted_target_odom is not None:
                        dist_to_prediction = math.dist((current_detection_odom.point.x, current_detection_odom.point.y), predicted_target_odom)
                        if dist_to_prediction > self.REID_GATE_DISTANCE:
                            continue # GATED
                            
                except Exception as e:
                    rospy.logerr_throttle(5, f"Gating Error: {e}"); continue
                
                # This person is a valid, gated candidate.
                person_crop_c = cv_frame[y1_c:y2_c, x1_c:x2_c]
                if person_crop_c.size > 0:
                    crops_to_process.append(person_crop_c)
                    candidates.append({
                        'track_id': track_id_c,
                        'bbox': (x1_c, y1_c, x2_c, y2_c)
                    })

            # --- Now, process all candidates in one batch ---
            if crops_to_process:
                feature_vectors = self.get_reid_features(crops_to_process)
                
                best_match_id = -1
                best_match_score = -1
                best_match_candidate = None

                for i, candidate in enumerate(candidates):
                    current_vector = feature_vectors[i].unsqueeze(0) # Get this person's vector
                    
                    # Compare against our *one* target (ID 0)
                    if 0 in self.known_people:
                        known_vector = self.known_people[0]
                        
                        # Use Cosine Similarity
                        score = F.cosine_similarity(current_vector, known_vector).item()
                        
                        if score > self.REID_MATCH_THRESHOLD:
                            if score > best_match_score:
                                best_match_score = score
                                best_match_id = 0
                                best_match_candidate = candidate
                    
                    # Re-ID Logic for new people
                    if best_match_id != 0:
                        track_id_c = candidate['track_id']
                        self.potential_people[track_id_c] = self.potential_people.get(track_id_c, 0) + 1
                        if self.potential_people[track_id_c] >= self.CONFIDENCE_THRESHOLD:
                            new_id = self.next_person_id
                            # We only care about ID 0, but we'll register the first person
                            if new_id == 0:
                                self.known_people[new_id] = current_vector
                                rospy.loginfo(f"====== NEW TARGET CONFIRMED! Assigned ID: {new_id} ======")
                            # To prevent re-assigning, we increment anyway
                            self.next_person_id += 1
                            if track_id_c in self.potential_people: del self.potential_people[track_id_c]
                
                if best_match_id == 0:
                    target_found_this_frame = True
                    confirmed_target_yolo_box_core = best_match_candidate['bbox']
        
        # --- Conditional Drawing ---
        if self.enable_visualization and vis_frame is not None:
            # Use the .plot() method from the results to draw skeletons
            annotated_frame = results[0].plot()
            
            # Add our custom target confirmation on top
            if confirmed_target_yolo_box_core is not None:
                 x1, y1, x2, y2 = confirmed_target_yolo_box_core
                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                 cv2.putText(annotated_frame, "TARGET (ID 0)", (x1, y1 - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            
            cv2.imshow("Pepper Perception", annotated_frame)
            cv2.waitKey(1)


        # --- Update state based on core logic results ---
        self.last_known_target_bbox = confirmed_target_yolo_box_core

        # --- Velocity, Waypoint, State Logic (uses self.last_known_target_bbox) ---
        if target_found_this_frame:
            try:
                # Calculations should use self.last_known_target_bbox if needed
                if self.last_known_target_bbox:
                    scale_x = depth_frame.shape[1] / cv_frame.shape[1] # Use original frame shape
                    scale_y = depth_frame.shape[0] / cv_frame.shape[0]

                    target_center_x_color = (self.last_known_target_bbox[0] + self.last_known_target_bbox[2]) / 2
                    target_center_y_color = (self.last_known_target_bbox[1] + self.last_known_target_bbox[3]) / 2
                    depth_coord_x = int(target_center_x_color * scale_x)
                    depth_coord_y = int(target_center_y_color * scale_y)

                    if not (0 <= depth_coord_y < depth_frame.shape[0] and 0 <= depth_coord_x < depth_frame.shape[1]):
                        return

                    distance_mm = depth_frame[depth_coord_y, depth_coord_x]
                    if distance_mm == 0: return
                    self.current_target_distance = distance_mm / 1000.0

                    point_3d = self.deproject_pixel_to_point(target_center_x_color, target_center_y_color, self.current_target_distance)
                    if point_3d is None: return

                    camera_point = PointStamped(); camera_point.header.frame_id = depth_msg.header.frame_id; camera_point.header.stamp = depth_msg.header.stamp
                    camera_point.point.x, camera_point.point.y, camera_point.point.z = point_3d[0], point_3d[1], point_3d[2]
                    odom_point = self.tf_buffer.transform(camera_point, 'odom', timeout=rospy.Duration(0.1))

                    current_time = rospy.Time.now()
                    dx = 0; dy = 0
                    if self.last_target_odom_point is not None and self.last_target_odom_time is not None:
                        dx = odom_point.point.x - self.last_target_odom_point.x; dy = odom_point.point.y - self.last_target_odom_point.y
                        dt = (current_time - self.last_target_odom_time).to_sec()
                        if dt > 0.01:
                            vx, vy = (dx / dt, dy / dt)
                            self.last_target_velocity_odom = (vx, vy)
                            self.current_target_velocity_mag = math.sqrt(vx**2 + vy**2)
                            self.velocity_history.append(self.current_target_velocity_mag)
                            if self.velocity_history: self.running_average_velocity = sum(self.velocity_history) / len(self.velocity_history)

                    self.last_target_odom_point = odom_point.point
                    self.last_target_odom_time = current_time
                    self.log_waypoint(odom_point)
                        

            except Exception as e:
                rospy.logerr(f"Error in perception 'if target_found_this_frame': {e}")

            if self.robot_state != 'TRACKING':
                rospy.loginfo("Target found! Switching to TRACKING state.")
                self.publish_delete_all_markers(); self.marker_list.clear(); self.next_marker_id = 0; self.waypoint_queue.clear()
            self.robot_state = 'TRACKING'
            self.last_target_seen_time = rospy.Time.now()
            self.predicted_target_point = None
            status_msg = Bool()
            status_msg.data = True
            self.target_status_pub.publish(status_msg)

        else:
            # Target not found OR not confirmed by Depth
            self.current_target_distance = None; self.current_target_velocity_mag = 0.0
            if self.velocity_history:
                # rospy.loginfo("Target lost/unconfirmed, resetting velocity average.")
                self.velocity_history.clear(); self.running_average_velocity = 0.0

            if self.robot_state == 'TRACKING' and (rospy.Time.now() - self.last_target_seen_time > self.LOST_TIMEOUT):
                rospy.logwarn("Target lost! Switching to NAVIGATING_PATH state.")
                if self.last_target_odom_point is not None:
                    vx, vy = self.last_target_velocity_odom
                    pred_x = self.last_target_odom_point.x + (vx * self.PREDICTION_TIME_SEC)
                    pred_y = self.last_target_odom_point.y + (vy * self.PREDICTION_TIME_SEC)
                    self.predicted_target_point = PointStamped(); self.predicted_target_point.header.frame_id = "odom"; self.predicted_target_point.header.stamp = rospy.Time.now()
                    self.predicted_target_point.point.x = pred_x; self.predicted_target_point.point.y = pred_y; self.predicted_target_point.point.z = self.last_target_odom_point.z
                    rospy.loginfo(f"Predicting target at ({pred_x:.2f}, {pred_y:.2f})")
                    self.add_prediction_marker(self.predicted_target_point.point)
                self.robot_state = 'NAVIGATING_PATH'

        # --- Conditional Display ---
        if self.enable_visualization and vis_frame is not None:
            cv2.imshow("Pepper Perception", vis_frame)
            cv2.waitKey(1)

    # --- 6. MODIFIED: control_loop now publishes markers ---
    
    def control_loop(self):
        rate = rospy.Rate(10)
        left = True
        while not rospy.is_shutdown():
            if self.current_pos is None:
                rate.sleep(); continue

            move_cmd = Twist()
            has_path = len(self.waypoint_queue) > 0
            
            # 1. Independent Waypoint Dequeueing
            if has_path:
                wp_x, wp_y, wp_id = self.waypoint_queue[0]
                distance_to_wp = math.dist((self.current_pos.x, self.current_pos.y), (wp_x, wp_y))
                if distance_to_wp < self.DISTANCE_THRESHOLD:
                    rospy.loginfo("Waypoint reached and dequeued.")
                    # We just pop. The green sphere marker will remain, which is OK.
                    # Modifying it requires storing the ID in the queue, a bigger change.
                    self.waypoint_queue.popleft()
                    has_path = len(self.waypoint_queue) > 0
            # 2. State Transition from Navigating to Searching
            if self.robot_state == 'NAVIGATING_PATH' and not has_path:
                rospy.logwarn("Path finished, but target not found. Switching to SEARCHING state.")
                self.robot_state = 'SEARCHING'
                if self.current_yaw > 0:
                    left = False
                else:
                    left = True
                status_msg = Bool()
                status_msg.data = False
                self.target_status_pub.publish(status_msg)

            # 3. Orientation Logic
            if self.robot_state == 'TRACKING' and self.last_known_target_bbox is not None:
                cam_width = self.camera_info.width if self.camera_info else 640
                target_center_x = (self.last_known_target_bbox[0] + self.last_known_target_bbox[2]) / 2
                error_x = target_center_x - (cam_width / 2)
                if abs(error_x) > 20:
                    move_cmd.angular.z = -0.008 * error_x
            elif has_path:
                target_x, target_y,_ = self.waypoint_queue[0]
                angle_to_target = math.atan2(target_y - self.current_pos.y, target_x - self.current_pos.x)
                angle_diff = angle_to_target - self.current_yaw
                if angle_diff > math.pi: angle_diff -= 2 * math.pi
                if angle_diff < -math.pi: angle_diff += 2 * math.pi
                move_cmd.angular.z = self.TURN_P_GAIN * angle_diff
            elif self.robot_state == 'SEARCHING':
                if not left:
                    move_cmd.angular.z = self.SEARCH_SPEED
                else:
                    move_cmd.angular.z = -self.SEARCH_SPEED

            # 4. Linear Movement Logic
            should_move_forward = False
            if self.robot_state == 'NAVIGATING_PATH' and has_path:
                should_move_forward = True
            elif self.robot_state == 'TRACKING' and self.current_target_distance is not None:
                if self.current_target_distance > self.NAVIGATION_START_DISTANCE:
                    should_move_forward = True
            
            if should_move_forward and has_path:
                target_x, target_y,_ = self.waypoint_queue[0]
                distance_to_wp = math.dist((self.current_pos.x, self.current_pos.y), (target_x, target_y))
                move_cmd.linear.x = self.LINEAR_P_GAIN * distance_to_wp

            # 5. Distance Maintenance
            if self.robot_state == 'TRACKING' and self.current_target_distance is not None:
                if self.current_target_distance < self.MIN_TARGET_DISTANCE:
                    rospy.logwarn(f"Target too close ({self.current_target_distance:.2f}m)! Moving back.")
                    error_dist = self.current_target_distance - self.BACKUP_DIST
                    backward_speed = self.LINEAR_P_GAIN * error_dist
                    move_cmd.linear.x = backward_speed
            
            # Clip and Publish
            move_cmd.angular.z = np.clip(move_cmd.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
            move_cmd.linear.x = np.clip(move_cmd.linear.x, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
            self.cmd_pub.publish(move_cmd)
            
            # --- NEW: Publish markers on every tick ---
            self.publish_markers()
            self.publish_robot_pose_marker()
            
            # Comprehensive logging
            log_wp_count = len(self.waypoint_queue)
            log_tgt_dist = f"{self.current_target_distance:.2f}m" if self.current_target_distance is not None else "N/A"
            dist_to_wp = None
            if has_path:
                wp_x, wp_y, wp_id = (self.waypoint_queue[0])
                dist_to_wp = math.dist((self.current_pos.x, self.current_pos.y), (wp_x,wp_y))
            else:
                dist_to_wp = None
            log_wp_dist = f"{dist_to_wp:.2f}m" if dist_to_wp is not None else "N/A"
            rospy.loginfo_throttle(1.0, f"State: {self.robot_state} | WPs: {log_wp_count} | TgtDist: {log_tgt_dist} | NextWP_Dist: {log_wp_dist}")
            
            rate.sleep()
        
        # --- NEW: Clean up markers on shutdown ---
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Shutting down, clearing markers...")
        self.publish_delete_all_markers()
        rospy.sleep(0.5)

    def run(self):
        from threading import Thread
        control_thread = Thread(target=self.control_loop)
        control_thread.start()
        rospy.spin()
        control_thread.join()

if __name__ == '__main__':
    try:
        follower = PepperPathFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
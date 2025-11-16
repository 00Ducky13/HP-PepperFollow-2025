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
from collections import deque
from threading import Thread
from std_msgs.msg import Bool
import torch
import torchreid
import torchvision.transforms as T
import torch.nn.functional as F

class PepperPathFollower:
    def __init__(self):
        rospy.init_node('pepper_path_follower', anonymous=True)
        rospy.loginfo("Starting Pepper Path Following Node...")
        self.enable_visualization = rospy.get_param('~visualize', False)

        self.target_status_pub = rospy.Publisher('/pepper/target_status', Bool, queue_size=1)

        self.model = YOLO('yolov8n.pt')
        rospy.loginfo("Loading Re-ID model (osnet_x0_25)...")
        self.reid_model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1,
            pretrained=True
        )
        self.reid_model.eval()
        if torch.cuda.is_available():
            self.reid_model = self.reid_model.cuda()

        self.reid_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.bridge = CvBridge()
        self.known_people = {}
        self.potential_people = {}
        self.next_person_id = 0
        
        self.REID_MATCH_THRESHOLD = 0.6
        self.CONFIDENCE_THRESHOLD = 3
        self.MIN_BBOX_AREA = 50 * 50
        self.REID_GATE_DISTANCE = 1.5

        self.DEPTH_STD_DEV_THRESHOLD = 0.05
        self.MIN_VALID_DEPTH_PERCENT = 0.1

        self.cmd_pub = rospy.Publisher('/pepper/cmd_vel', Twist, queue_size=10)
        self.TURN_P_GAIN = 0.8; self.MAX_ANGULAR_SPEED = 0.4
        self.LINEAR_P_GAIN = 0.5; self.MAX_LINEAR_SPEED = 0.5
        self.DISTANCE_THRESHOLD = 0.3
        self.WAYPOINT_TIME_THRESHOLD = rospy.Duration(0.2)
        self.MIN_TARGET_DISTANCE = 0.7
        self.BACKUP_DIST = 0.7
        self.NAVIGATION_START_DISTANCE = 1.0

        self.NAVIGATION_LOOKAHEAD = 3
        self.navigation_target_marker_id = 9997
        
        self.VELOCITY_AVERAGE_WINDOW = 10
        self.velocity_history = collections.deque(maxlen=self.VELOCITY_AVERAGE_WINDOW)
        self.running_average_velocity = 0.0

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

        self.SMOOTHING_FACTOR = 0.8 
        self.smoothed_head_error_x = 0.0
        self.smoothed_head_error_y = 0.0

        self.marker_pub = rospy.Publisher('/pepper/waypoint_markers', MarkerArray, queue_size=1)
        self.marker_list = []
        self.next_marker_id = 0

        self.robot_state = 'SEARCHING'
        self.last_target_seen_time = rospy.Time(0)
        self.LOST_TIMEOUT = rospy.Duration(1.0)
        self.SEARCH_SPEED = 0.3
        self.waypoint_queue = collections.deque()
        self.last_waypoint_log_time = rospy.Time(0)
        self.current_target_distance = None
        self.last_known_target_bbox = None

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

        self.PREDICTION_TIME_SEC = 1.5
        self.last_target_odom_point = None
        self.last_target_odom_time = None
        self.last_target_velocity_odom = (0, 0)
        self.predicted_target_point = None

        self.WAYPOINT_POP_INTERVAL = rospy.Duration(2.0)
        self.last_waypoint_pop_time = rospy.Time.now()

    def publish_delete_all_markers(self):
        delete_all_marker = Marker()
        delete_all_marker.header.frame_id = "odom"
        delete_all_marker.header.stamp = rospy.Time.now()
        delete_all_marker.ns = "waypoints"
        delete_all_marker.id = 0
        delete_all_marker.action = Marker.DELETEALL
        marker_array = MarkerArray(markers=[delete_all_marker])
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Cleared all waypoint markers.")
        
    def publish_markers(self):
        if not self.marker_list:
            return
        self.marker_pub.publish(MarkerArray(markers=self.marker_list))

    def cam_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("Camera info received.")
            self.camera_info = msg
            self.cam_info_sub.unregister()

    def joint_state_callback(self, msg):
        try:
            idx_yaw = msg.name.index('HeadYaw')
            idx_pitch = msg.name.index('HeadPitch')
            self.current_head_yaw = msg.position[idx_yaw]
            self.current_head_pitch = msg.position[idx_pitch]
        except ValueError:
            rospy.logwarn_throttle(5, "Could not find HeadYaw or HeadPitch in /pepper/joint_states")

    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.current_yaw) = tf.transformations.euler_from_quaternion(orientation_list)

    @torch.no_grad()
    def get_reid_features(self, crops):
        if not crops:
            return []
        
        tensors = [self.reid_transform(crop) for crop in crops]
        batch = torch.stack(tensors)
        if torch.cuda.is_available():
            batch = batch.cuda()
        
        features = self.reid_model(batch)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()
    
    def deproject_pixel_to_point(self, u, v, depth):
        if self.camera_info is None:
            rospy.logwarn_throttle(5, "Waiting for camera info for accurate deprojection...")
            return None
        fx = self.camera_info.K[0]; fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]; cy = self.camera_info.K[5]
        x = (u - cx) * depth / fx; y = (v - cy) * depth / fy; z = depth
        return (x, y, z)

    def add_prediction_marker(self, predicted_point):
        prediction_marker_id = 9999
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = prediction_marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = predicted_point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25; marker.scale.y = 0.25; marker.scale.z = 0.25
        marker.color.a = 0.7; marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0
        marker.lifetime = rospy.Duration(5.0)
        self.marker_list.append(marker)

    def add_navigation_marker(self, nav_point_tuple):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = self.navigation_target_marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = nav_point_tuple[0]
        marker.pose.position.y = nav_point_tuple[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.a = 0.7; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0
        marker.lifetime = rospy.Duration(0.2)
        self.marker_list.append(marker)

    def log_waypoint(self, odom_point):
        if self.current_target_distance < self.MIN_TARGET_DISTANCE:
            return 

        time_since_last_log = rospy.Time.now() - self.last_waypoint_log_time
        if time_since_last_log > self.WAYPOINT_TIME_THRESHOLD:
            current_marker_id = self.next_marker_id
            new_waypoint = (odom_point.point.x, odom_point.point.y, current_marker_id)
            
            self.waypoint_queue.append(new_waypoint)
            self.last_waypoint_log_time = rospy.Time.now()
            rospy.loginfo(f"New waypoint logged: ({new_waypoint[0]:.2f}, {new_waypoint[1]:.2f})")

            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = current_marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = odom_point.point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
            marker.color.a = 0.8; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
            marker.lifetime = rospy.Duration()
            
            self.marker_list.append(marker)
            self.next_marker_id += 1

    def perception_callback(self, color_msg, depth_msg):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            vis_frame = cv_frame.copy() if self.enable_visualization else None
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}"); return

        frame_to_process = vis_frame if self.enable_visualization else cv_frame
        results = self.model.track(frame_to_process, persist=True, classes=[0], verbose=False)

        target_found_this_frame = False
        confirmed_target_yolo_box_core = None
        
        result = results[0] 

        predicted_target_odom = None
        if self.robot_state != 'SEARCHING' and self.last_target_odom_point is not None:
            dt = (rospy.Time.now() - self.last_target_odom_time).to_sec()
            predicted_target_odom = (
                self.last_target_odom_point.x + self.last_target_velocity_odom[0] * dt,
                self.last_target_odom_point.y + self.last_target_velocity_odom[1] * dt
            )
  
        H, W = cv_frame.shape[:2]
        image_center_x = W / 2
        image_center_y = H / 2

        if result.boxes.id is not None:
            if 0 not in self.known_people:
                candidates = []
                for i, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(result.boxes.id[i])

                    if (x2 - x1) * (y2 - y1) < self.MIN_BBOX_AREA:
                        continue
                    
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    
                    distance = math.hypot(box_center_x - image_center_x, 
                                        box_center_y - image_center_y)
                    
                    candidates.append({
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'distance_to_center': distance,
                        'box_obj': box
                    })
                
                print(candidates[0])
                sorted_candidates = sorted(candidates, key=lambda c: c['distance_to_center'])
                
                if sorted_candidates:
                    priority_1_target = sorted_candidates[0]
                    track_id_c = priority_1_target['track_id']
                    
                    self.potential_people[track_id_c] = self.potential_people.get(track_id_c, 0) + 1
                    
                    if self.potential_people[track_id_c] >= self.CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = priority_1_target['bbox']
                        person_crop = cv_frame[y1:y2, x1:x2]
                        
                        if person_crop.size > 0:
                            feature_vector = self.get_reid_features([person_crop])[0].unsqueeze(0)
                            self.known_people[0] = feature_vector
                            self.next_person_id = 1
                            
                            rospy.loginfo(f"====== NEW TARGET (Center Priority) CONFIRMED! Assigned ID: 0 (Track ID: {track_id_c}) ======")
                            
                            target_found_this_frame = True
                            confirmed_target_yolo_box_core = priority_1_target['bbox']
                            
                            if track_id_c in self.potential_people: 
                                del self.potential_people[track_id_c]
                            
            else:
                candidates = [] 
                crops_to_process = [] 

                for i, box in enumerate(result.boxes.xyxy):
                    x1_c, y1_c, x2_c, y2_c = map(int, box)
                    track_id_c = int(result.boxes.id[i])

                    if (x2_c - x1_c) * (y2_c - y1_c) < self.MIN_BBOX_AREA: continue

                    current_detection_odom = None
                    try:
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
                                continue
                                
                    except Exception as e:
                        rospy.logerr_throttle(5, f"Gating Error: {e}"); continue
                    
                    person_crop_c = cv_frame[y1_c:y2_c, x1_c:x2_c]
                    if person_crop_c.size > 0:
                        crops_to_process.append(person_crop_c)
                        candidates.append({
                            'track_id': track_id_c,
                            'bbox': (x1_c, y1_c, x2_c, y2_c)
                        })

                if crops_to_process:
                    feature_vectors = self.get_reid_features(crops_to_process)
                    
                    best_match_score = -1
                    best_match_candidate = None
                    known_vector = self.known_people[0]

                    for i, candidate in enumerate(candidates):
                        current_vector = feature_vectors[i].unsqueeze(0)
                        
                        score = F.cosine_similarity(current_vector, known_vector).item()
                        
                        if score > self.REID_MATCH_THRESHOLD:
                            if score > best_match_score:
                                best_match_score = score
                                best_match_candidate = candidate
                    
                    if best_match_candidate is not None:
                        target_found_this_frame = True
                        confirmed_target_yolo_box_core = best_match_candidate['bbox']
        
        if self.enable_visualization and vis_frame is not None:
            annotated_frame = results[0].plot()
            
            if confirmed_target_yolo_box_core is not None:
                 x1, y1, x2, y2 = confirmed_target_yolo_box_core
                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                 cv2.putText(annotated_frame, "TARGET (ID 0)", (x1, y1 - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            
            cv2.imshow("Pepper Perception", annotated_frame)
            cv2.waitKey(1)

        self.last_known_target_bbox = confirmed_target_yolo_box_core

        if target_found_this_frame:
            try:
                if self.last_known_target_bbox:
                    scale_x = depth_frame.shape[1] / cv_frame.shape[1]
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
                self.transitionToTrack = True
                rospy.loginfo("Target found! Switching to TRACKING state.")
                self.publish_delete_all_markers(); self.marker_list.clear(); self.next_marker_id = 0; self.waypoint_queue.clear()
                self.robot_state = 'TRACKING'
                self.last_target_seen_time = rospy.Time.now()
                self.predicted_target_point = None
                status_msg = Bool()
                status_msg.data = True
                self.target_status_pub.publish(status_msg)

        else:
            self.current_target_distance = None; self.current_target_velocity_mag = 0.0
            if self.velocity_history:
                self.velocity_history.clear(); self.running_average_velocity = 0.0

            if self.robot_state == 'TRACKING' and (rospy.Time.now() - self.last_target_seen_time > self.LOST_TIMEOUT):
                rospy.logwarn("Target lost! Switching to NAVIGATING_PATH state.")
                self.transitionToTrack = False
                if self.last_target_odom_point is not None:
                    vx, vy = self.last_target_velocity_odom
                    pred_x = self.last_target_odom_point.x + (vx * self.PREDICTION_TIME_SEC)
                    pred_y = self.last_target_odom_point.y + (vy * self.PREDICTION_TIME_SEC)
                    self.predicted_target_point = PointStamped(); self.predicted_target_point.header.frame_id = "odom"; self.predicted_target_point.header.stamp = rospy.Time.now()
                    self.predicted_target_point.point.x = pred_x; self.predicted_target_point.point.y = pred_y; self.predicted_target_point.point.z = self.last_target_odom_point.z
                    rospy.loginfo(f"Predicting target at ({pred_x:.2f}, {pred_y:.2f})")
                    self.add_prediction_marker(self.predicted_target_point.point)
                self.robot_state = 'NAVIGATING_PATH'

        if self.enable_visualization and vis_frame is not None:
            cv2.imshow("Pepper Perception", vis_frame)
            cv2.waitKey(1)

    def control_loop(self):
        rate = rospy.Rate(10)
        left = True
        while not rospy.is_shutdown():
            if self.current_pos is None:
                rate.sleep(); continue

            move_cmd = Twist()
            send_head_cmd = False 
            
            has_path = len(self.waypoint_queue) > 0
            
            if has_path:
                wp_x, wp_y, wp_id = self.waypoint_queue[0]
                distance_to_wp = math.dist((self.current_pos.x, self.current_pos.y), (wp_x, wp_y))
                time_since_last_pop = rospy.Time.now() - self.last_waypoint_pop_time
                if distance_to_wp < self.DISTANCE_THRESHOLD or time_since_last_pop >= self.WAYPOINT_POP_INTERVAL:
                    rospy.loginfo(f"Waypoint {wp_id} dequeued.")
                    _, _, reached_wp_id = self.waypoint_queue.popleft()
                    for marker in self.marker_list:
                        if marker.id == reached_wp_id:
                            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.a = 0.5
                            break
                    self.last_waypoint_pop_time = rospy.Time.now()
                    self.predicted_target_point = None
                    has_path = len(self.waypoint_queue) > 0
            
            if self.robot_state == 'NAVIGATING_PATH' and not has_path:
                rospy.logwarn("Path finished. Switching to SEARCHING state.")
                self.robot_state = 'SEARCHING'
                self.predicted_target_point = None
                if self.current_yaw > 0:
                    left = False
                else:
                    left = True
                status_msg = Bool()
                status_msg.data = False
                self.target_status_pub.publish(status_msg)

            if has_path:
                orientation_target = self.waypoint_queue[0]
                
                num_to_get = min(len(self.waypoint_queue), self.NAVIGATION_LOOKAHEAD)
                points_to_average = list(self.waypoint_queue)[:num_to_get]
                
                avg_x = 0.0; avg_y = 0.0
                for wp in points_to_average:
                    avg_x += wp[0]; avg_y += wp[1]
                avg_x /= num_to_get; avg_y /= num_to_get
                navigation_target = (avg_x, avg_y)
                
                self.add_navigation_marker(navigation_target)
                
                target_x, target_y, _ = orientation_target 
                angle_to_target = math.atan2(target_y - self.current_pos.y, target_x - self.current_pos.x)
                angle_diff = angle_to_target - self.current_yaw
                if angle_diff > math.pi: angle_diff -= 2 * math.pi
                if angle_diff < -math.pi: angle_diff += 2 * math.pi
                move_cmd.angular.z = self.TURN_P_GAIN * angle_diff

                nav_x, nav_y = navigation_target
                distance_to_nav_target = math.dist((self.current_pos.x, self.current_pos.y), (nav_x, nav_y))
                move_cmd.linear.x = self.LINEAR_P_GAIN * distance_to_nav_target

            elif self.robot_state == 'SEARCHING':
                if left:
                    move_cmd.angular.z = -self.SEARCH_SPEED
                else:
                    move_cmd.angular.z = self.SEARCH_SPEED

            if self.robot_state == 'TRACKING' and self.current_target_distance is not None:
                if self.current_target_distance < self.MIN_TARGET_DISTANCE:
                    rospy.logwarn_throttle(0.5, f"Target too close ({self.current_target_distance:.2f}m)! Moving back.")
                    error_dist = self.current_target_distance - self.BACKUP_DIST 
                    backward_speed = self.LINEAR_P_GAIN * error_dist
                    move_cmd.linear.x = backward_speed

            move_cmd.angular.z = np.clip(move_cmd.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
            move_cmd.linear.x = np.clip(move_cmd.linear.x, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
            self.cmd_pub.publish(move_cmd)
            
            if send_head_cmd:
                self.head_traj_msg.header.stamp = rospy.Time.now()
                self.head_traj_msg.points = [self.head_traj_point]
                self.head_pub.publish(self.head_traj_msg)
            
            self.publish_markers()
            
            log_wp_count = len(self.waypoint_queue)
            log_tgt_dist = f"{self.current_target_distance:.2f}m" if self.current_target_distance is not None else "N/A"
            dist_to_wp = None
            if has_path:
                 dist_to_wp = math.dist((self.current_pos.x, self.current_pos.y), (self.waypoint_queue[0][0], self.waypoint_queue[0][1]))
            
            log_wp_dist = f"{dist_to_wp:.2f}m" if dist_to_wp is not None else "N/A"
            rospy.loginfo_throttle(1.0, f"State: {self.robot_state} | WPs: {log_wp_count} | TgtDist: {log_tgt_dist} | NextWP_Dist: {log_wp_dist}")
            
            rate.sleep()
        
        self.cmd_pub.publish(Twist())
        
        rospy.loginfo("Resetting head to center...")
        self.head_traj_point.positions = [0.0, 0.0]
        self.head_traj_point.time_from_start = rospy.Duration(1.0)
        self.head_traj_msg.header.stamp = rospy.Time.now()
        self.head_traj_msg.points = [self.head_traj_point]
        self.head_pub.publish(self.head_traj_msg)
        
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

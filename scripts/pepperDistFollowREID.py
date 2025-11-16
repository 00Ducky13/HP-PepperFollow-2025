#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from threading import Thread
from std_msgs.msg import Bool

import torch
import torchreid
import torchvision.transforms as T
import torch.nn.functional as F
# ---

class PepperFollower:
    def __init__(self):
        rospy.init_node('pepper_follower', anonymous=True)
        rospy.loginfo("Starting Pepper Follower Node...")

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
        
        self.REID_MATCH_THRESHOLD = 0.7 
        self.CONFIDENCE_THRESHOLD = 3
        self.MIN_BBOX_AREA = 50 * 50


        self.cmd_pub = rospy.Publisher('/pepper/cmd_vel', Twist, queue_size=10)
        self.TURN_P_GAIN = 0.005; self.MAX_ANGULAR_SPEED = 1.2; self.CENTERING_TOLERANCE = 20
        self.DESIRED_DISTANCE = 1; self.LINEAR_P_GAIN = 0.5; self.MAX_LINEAR_SPEED = 1.2; self.DISTANCE_TOLERANCE = 0.1

        self.robot_state = 'SEARCHING'
        self.last_target_seen_time = rospy.Time.now()
        self.LOST_TIMEOUT = rospy.Duration(2.0)
        self.SEARCH_SPEED = 0.3

        self.target_visible = False
        self.target_error_x = 0.0
        self.target_distance = 0.0
        self.depth_frame_nan = False

        color_sub = message_filters.Subscriber("/pepper/camera/front/image_raw", Image)
        depth_sub = message_filters.Subscriber("/pepper/camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.perception_callback)


    @torch.no_grad()
    def get_reid_features(self, crops):
        """Extracts feature vectors from a batch of cropped images."""
        if not crops:
            return []
        
        tensors = [self.reid_transform(crop) for crop in crops]
        batch = torch.stack(tensors)
        if torch.cuda.is_available():
            batch = batch.cuda()
        
        features = self.reid_model(batch)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()

    def control_loop(self):
        """
        Runs at a fixed rate, reads shared state variables, and publishes
        Twist commands.
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            move_cmd = Twist()

            if self.target_visible:
                if self.robot_state != 'TRACKING':
                    rospy.loginfo("Target found! Switching to TRACKING state.")
                self.robot_state = 'TRACKING'
                self.last_target_seen_time = rospy.Time.now()
                status_msg = Bool()
                status_msg.data = True
                self.target_status_pub.publish(status_msg)
            else:
                if self.robot_state == 'TRACKING':
                    if rospy.Time.now() - self.last_target_seen_time > self.LOST_TIMEOUT:
                        rospy.logwarn("Target lost! Starting search...")
                        self.robot_state = 'SEARCHING'
                        status_msg = Bool()
                        status_msg.data = False
                        self.target_status_pub.publish(status_msg)
            

            if self.robot_state == 'TRACKING':

                if abs(self.target_error_x) > self.CENTERING_TOLERANCE:
                    turn_speed = -self.TURN_P_GAIN * self.target_error_x
                    move_cmd.angular.z = np.clip(turn_speed, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
                
                if self.target_distance > 0:
                    error_dist = self.target_distance - self.DESIRED_DISTANCE
                    if abs(error_dist) > self.DISTANCE_TOLERANCE:
                        linear_speed = self.LINEAR_P_GAIN * error_dist
                        move_cmd.linear.x = np.clip(linear_speed, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
                elif self.depth_frame_nan:
                    rospy.logwarn_throttle(1.0, "Target visible but depth is NaN. Stopping linear motion.")
                    move_cmd.linear.x = 0.0

            elif self.robot_state == 'SEARCHING':
                if self.target_error_x > 0:
                    move_cmd.angular.z = -self.SEARCH_SPEED
                else:
                    move_cmd.angular.z = self.SEARCH_SPEED
            
            self.cmd_pub.publish(move_cmd)
            
            rospy.loginfo_throttle(1.0, f"State: {self.robot_state} | TgtVisible: {self.target_visible} | ErrX: {self.target_error_x:.2f} | Dist: {self.target_distance:.2f}m")
            
            rate.sleep()

    def perception_callback(self, color_msg, depth_msg):
        """
        Receives synced images, runs perception, and updates shared
        state variables (self.target_visible, self.target_error_x, etc.)
        """
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}"); return

        color_h, color_w, _ = cv_frame.shape
        depth_h, depth_w = depth_frame.shape
        scale_x = depth_w / color_w; scale_y = depth_h / color_h
        
        results = self.model.track(cv_frame, persist=True, classes=[0], verbose=False)

        candidates = []
        crops_to_process = []
        
        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1_c, y1_c, x2_c, y2_c = map(int, box)
                
                track_id_c = int(results[0].boxes.id[i]) 

                if (x2_c - x1_c) * (y2_c - y1_c) < self.MIN_BBOX_AREA: continue
                
                person_crop_c = cv_frame[y1_c:y2_c, x1_c:x2_c]
                if person_crop_c.size > 0:
                    crops_to_process.append(person_crop_c)
                    candidates.append({
                        'track_id': track_id_c,
                        'bbox': (x1_c, y1_c, x2_c, y2_c)
                    })

        target_bbox = None
        if crops_to_process:
            feature_vectors = self.get_reid_features(crops_to_process)
            
            best_match_id = -1
            best_match_score = -1
            best_match_candidate = None

            for i, candidate in enumerate(candidates):
                current_vector = feature_vectors[i].unsqueeze(0)
                
                if 0 in self.known_people:
                    known_vector = self.known_people[0]
                    score = F.cosine_similarity(current_vector, known_vector).item()
                    
                    if score > self.REID_MATCH_THRESHOLD and score > best_match_score:
                        best_match_score = score
                        best_match_id = 0
                        best_match_candidate = candidate
                else:
                    track_id_c = candidate['track_id']
                    self.potential_people[track_id_c] = self.potential_people.get(track_id_c, 0) + 1
                    
                    if self.potential_people[track_id_c] >= self.CONFIDENCE_THRESHOLD:
                        new_id = self.next_person_id
                        if new_id == 0:
                            self.known_people[new_id] = current_vector
                            rospy.loginfo(f"====== NEW TARGET CONFIRMED! Assigned ID: 0 ======")
                            best_match_id = 0
                            best_match_candidate = candidate
                        
                        self.next_person_id += 1
                        if track_id_c in self.potential_people: 
                            del self.potential_people[track_id_c]
            
            if best_match_id == 0 and best_match_candidate is not None:
                target_bbox = best_match_candidate['bbox']

        if target_bbox is not None:
            self.target_visible = True
            
            x1, y1, x2, y2 = target_bbox
            
            target_center_x = (x1 + x2) / 2
            image_center_x = color_w / 2
            self.target_error_x = target_center_x - image_center_x
            
            target_center_y = (y1 + y2) / 2
            depth_coord_x = int(target_center_x * scale_x)
            depth_coord_y = int(target_center_y * scale_y)
            
            if 0 <= depth_coord_y < depth_h and 0 <= depth_coord_x < depth_w:
                distance = depth_frame[depth_coord_y, depth_coord_x]
                if math.isnan(distance):
                    self.target_distance = 0.0
                    self.depth_frame_nan = True
                else:
                    self.target_distance = distance / 1000.0
                    self.depth_frame_nan = False
            else:
                self.target_distance = 0.0
                self.depth_frame_nan = True
        else:
            self.target_visible = False
            self.target_distance = 0.0
        

    def run(self):
        control_thread = Thread(target=self.control_loop)
        control_thread.daemon = True
        control_thread.start()

        rospy.spin()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("Shutting down. Sending stop command.")


if __name__ == '__main__':
    try:
        follower = PepperFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass

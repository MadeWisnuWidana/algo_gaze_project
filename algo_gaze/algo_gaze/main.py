#!/home/brone-ub/ros_venv/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import deque
import os
import time 
import math
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, DurabilityPolicy

# Import Message Types
from geometry_msgs.msg import Point 
from op3_ball_detector_msgs.msg import CircleSetStamped
from robotis_controller_msgs.srv import SetModule 

# --- Konfigurasi Visualisasi ---
PRIORITY_COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 165, 255)]
DEFAULT_COLOR = (200, 200, 200)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class FuzzyGazeNode(Node):

    def __init__(self):
        super().__init__('algo_gaze_node')
        self.get_logger().info('Fuzzy Gaze Node: AUTO-SYNC & SMOOTH START ACTIVATED.')

        # --- Load Model YOLO ---
        self.declare_parameter('yolo_model_path', 'yolo11s.pt')
        yolo_model_path = '/home/brone-ub/robotis_ws/src/algo_gaze/algo_gaze/models/yolo11s.pt'
        
        if not os.path.exists(yolo_model_path):
            self.get_logger().error(f'File model tidak ditemukan: {yolo_model_path}')
            raise FileNotFoundError()

        # --- Inisialisasi AI ---
        self.yolo_model = YOLO(yolo_model_path)
        self.holistic = mp_holistic.Holistic(
            refine_face_landmarks=True, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.bridge = CvBridge()
        self.fis_controller = self.create_fuzzy_controller()
        
        # --- Variabel Logic ---
        self.person_histories = {} 
        self.score_history = {} 
        self.alpha_score = 0.2 
        self.current_target_id = None
        self.last_target_switch_time = 0.0
        self.min_switch_delay = 1.0 
        
        # Variabel Soft Start (Ramp-Up)
        self.tracking_start_time = 0.0  
        self.ramp_duration = 1.5        

        # --- Variabel Servo ---
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.is_initialized = False # Penanda sinkronisasi posisi awal
        
        # Smoothing Input
        self.prev_norm_x = 0.0
        self.prev_norm_y = 0.0
        self.prev_norm_z = 0.0
        self.alpha_coord = 0.3 

        # --- ARAH SERVO ---
        self.pan_dir = -1 
        self.tilt_dir = 1   

        # --- Komunikasi ROS 2 ---
        self.image_subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.image_publisher = self.create_publisher(
            Image, '/gaze_model/annotated_image', qos_profile)
        
        self.center_pub_ = self.create_publisher(
            CircleSetStamped, 
            '/ball_detector_node/circle_set', 
            10)

        self.head_pub = self.create_publisher(
            JointState,
            '/robotis/head_control/set_joint_states',
            10)

        # [BARU] Subscriber untuk Sinkronisasi Posisi Awal Servo
        self.joint_sub = self.create_subscription(
            JointState,
            '/robotis/present_joint_states',
            self.joint_state_callback,
            10)

        self.set_module_client = self.create_client(SetModule, '/robotis/set_present_ctrl_modules')
        self.activate_head_module()

    # --- FUNGSI SINKRONISASI POSISI (PENTING!) ---
    def joint_state_callback(self, msg):
        """
        Membaca posisi asli robot saat ini.
        Hanya digunakan sekali di awal (atau saat idle) agar variabel internal
        sama dengan posisi fisik servo.
        """
        if not self.is_initialized:
            try:
                if 'head_pan' in msg.name and 'head_tilt' in msg.name:
                    idx_pan = msg.name.index('head_pan')
                    idx_tilt = msg.name.index('head_tilt')
                    
                    # Salin posisi asli ke variabel internal
                    self.current_pan = msg.position[idx_pan]
                    self.current_tilt = msg.position[idx_tilt]
                    
                    self.is_initialized = True # Tandai sudah sinkron
                    self.get_logger().info(f'SYNC OK: Start from Pan={self.current_pan:.2f}, Tilt={self.current_tilt:.2f}')
            except ValueError:
                pass

    def activate_head_module(self):
        if not self.set_module_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Service manager belum siap...')
            return

        req = SetModule.Request()
        req.module_name = 'head_control_module'
        self.future = self.set_module_client.call_async(req)
        self.future.add_done_callback(self.service_callback)

    def service_callback(self, future):
        try:
            res = future.result()
            if res.result:
                self.get_logger().info('SUKSES: Head Module Aktif.')
            else:
                self.get_logger().error('GAGAL: Manager menolak aktivasi.')
        except Exception as e:
            self.get_logger().error(f'Service error: {e}')

    def image_callback(self, msg):
        # Jangan proses gambar jika posisi servo belum tersinkronisasi
        if not self.is_initialized:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return
            
        frame_height, frame_width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        yolo_results = self.yolo_model.track(image_rgb, classes=0, conf=0.5, persist=True, verbose=False)
        
        detected_people = []

        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                padding = 30
                x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
                x2_pad, y2_pad = min(frame_width, x2 + padding), min(frame_height, y2 + padding)
                
                person_crop_rgb = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                if person_crop_rgb.size == 0: continue

                results = self.holistic.process(person_crop_rgb)
                cues, lip_distance = self.extract_cues(results)
                
                detected_people.append({
                    'id': int(track_id),
                    'cues': cues, 
                    'bbox_yolo': (x1, y1, x2, y2),
                    'crop_bbox': (x1_pad, y1_pad, x2_pad, y2_pad),
                    'score': 0, 
                    'results_mp': results,
                    'lip_distance': lip_distance
                })

        if detected_people:
            all_areas = [(p['bbox_yolo'][2] - p['bbox_yolo'][0]) * (p['bbox_yolo'][3] - p['bbox_yolo'][1]) for p in detected_people]
            max_area = max(all_areas) if all_areas else 1

            for p in detected_people:
                area = (p['bbox_yolo'][2] - p['bbox_yolo'][0]) * (p['bbox_yolo'][3] - p['bbox_yolo'][1])
                p['cues']['proximity'] = area / max_area
                
                x1, _, x2, _ = p['bbox_yolo']
                person_center_x = (x1 + x2) / 2
                angle_value = abs(person_center_x - (frame_width / 2)) / (frame_width / 2)
                p['cues']['angle'] = min(angle_value, 1.0)
                
                p_id = p['id']
                if p_id not in self.person_histories:
                    self.person_histories[p_id] = deque(maxlen=10)
                self.person_histories[p_id].append(p['lip_distance'])
                
                if len(self.person_histories[p_id]) == 10:
                    variance = np.var(list(self.person_histories[p_id]))
                    p['cues']['speech'] = 1 if variance > 0.00008 else 0
                else:
                    p['cues']['speech'] = 0
                
                self.fis_controller.input['proximity'] = p['cues']['proximity']
                self.fis_controller.input['speech_status'] = p['cues']['speech']
                self.fis_controller.input['pointing_gesture'] = p['cues']['pointing']
                self.fis_controller.input['waving_gesture'] = p['cues']['waving']
                self.fis_controller.input['body_orientation'] = p['cues']['body_orientation']
                self.fis_controller.input['direct_gaze'] = p['cues']['direct_gaze']
                self.fis_controller.input['angle'] = p['cues']['angle']
                
                self.fis_controller.compute()
                raw_score = self.fis_controller.output['priority']

                if p_id not in self.score_history:
                    self.score_history[p_id] = raw_score
                else:
                    self.score_history[p_id] = (self.alpha_score * raw_score) + ((1 - self.alpha_score) * self.score_history[p_id])
                p['score'] = self.score_history[p_id] 

            sorted_people = sorted(detected_people, key=lambda p: p['score'], reverse=True)
            potential_winner = sorted_people[0]
            current_time = time.time()

            # Logika Reset Ramp saat target berubah
            if self.current_target_id is None:
                self.current_target_id = potential_winner['id']
                self.last_target_switch_time = current_time
                self.tracking_start_time = current_time 
            else:
                target_still_visible = any(p['id'] == self.current_target_id for p in detected_people)
                if not target_still_visible:
                    self.current_target_id = potential_winner['id']
                    self.last_target_switch_time = current_time
                    self.tracking_start_time = current_time 
                else:
                    time_ok = (current_time - self.last_target_switch_time) > self.min_switch_delay
                    current_target_data = next((p for p in detected_people if p['id'] == self.current_target_id), None)
                    current_score = current_target_data['score'] if current_target_data else 0
                    
                    if time_ok and (potential_winner['score'] > (current_score * 1.15)):
                        self.current_target_id = potential_winner['id']
                        self.last_target_switch_time = current_time
                        self.tracking_start_time = current_time 
            
            for rank, p in enumerate(sorted_people):
                x1, y1, x2, y2 = p['bbox_yolo']
                is_target = (p['id'] == self.current_target_id)
                color = (0, 0, 255) if is_target else (PRIORITY_COLORS[rank] if rank < len(PRIORITY_COLORS) else DEFAULT_COLOR)
                
                self.draw_landmarks_with_overlay(frame, p['results_mp'], p['crop_bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{p['id']} | S:{p['score']:.2f}"
                
                if is_target:
                    label = f"TARGET | S:{p['score']:.2f}"
                    
                    face_center_x = None
                    face_center_y = None

                    if p['results_mp'].face_landmarks:
                        nose = p['results_mp'].face_landmarks.landmark[1]
                        crop_x1, crop_y1, crop_x2, crop_y2 = p['crop_bbox']
                        face_center_x = crop_x1 + (nose.x * (crop_x2 - crop_x1))
                        face_center_y = crop_y1 + (nose.y * (crop_y2 - crop_y1))
                        cv2.circle(frame, (int(face_center_x), int(face_center_y)), 8, (0, 255, 255), -1)

                    if face_center_x is None:
                        face_center_x = (x1 + x2) / 2.0
                        face_center_y = y1 + (y2 - y1) * 0.2
                        cv2.circle(frame, (int(face_center_x), int(face_center_y)), 8, (255, 0, 255), -1)

                    raw_norm_x = (face_center_x / frame_width) * 2.0 - 1.0
                    raw_norm_y = (face_center_y / frame_height) * 2.0 - 1.0
                    
                    smooth_x = (self.alpha_coord * raw_norm_x) + ((1 - self.alpha_coord) * self.prev_norm_x)
                    smooth_y = (self.alpha_coord * raw_norm_y) + ((1 - self.alpha_coord) * self.prev_norm_y)
                    
                    self.prev_norm_x, self.prev_norm_y = smooth_x, smooth_y

                    self.publish_coordinates(smooth_x, smooth_y, 0.0, msg.header)
                    self.track_face_soft_start(smooth_x, smooth_y)
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            annotated_msg.header = msg.header
            self.image_publisher.publish(annotated_msg)
        except Exception as e:
            pass

    def track_face_soft_start(self, error_x, error_y):
        error_magnitude = math.sqrt(error_x**2 + error_y**2)

        # Deadband
        DEADBAND = 0.10 
        if error_magnitude < DEADBAND:
            return 

        # Gain
        MIN_GAIN = 0.02
        MAX_GAIN = 0.12 
        base_gain = MIN_GAIN + (error_magnitude * (MAX_GAIN - MIN_GAIN))
        base_gain = min(base_gain, MAX_GAIN)

        # Ramp-Up (Soft Start)
        elapsed = time.time() - self.tracking_start_time
        if elapsed < self.ramp_duration:
            ramp_factor = 0.2 + (0.8 * (elapsed / self.ramp_duration))
        else:
            ramp_factor = 1.0

        final_gain = base_gain * ramp_factor

        # Update Posisi (Incremental dari posisi terakhir yang sudah disinkronisasi)
        self.current_pan += (error_x * final_gain * self.pan_dir)
        self.current_tilt += (error_y * final_gain * self.tilt_dir)

        # Safety Limits
        self.current_pan = max(-1.4, min(1.4, self.current_pan))
        self.current_tilt = max(-0.5, min(0.8, self.current_tilt))

        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['head_pan', 'head_tilt']
        joint_msg.position = [float(self.current_pan), float(self.current_tilt)]
        
        self.head_pub.publish(joint_msg)

    def publish_coordinates(self, x, y, z, header):
        msg = CircleSetStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id 
        circle_point = Point()
        circle_point.x, circle_point.y, circle_point.z = float(x), float(y), float(z)
        msg.circles.append(circle_point)
        self.center_pub_.publish(msg)

    # --- Helper Methods Fuzzy (Tetap Sama) ---
    def create_fuzzy_controller(self):
        proximity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'proximity')
        speech_status = ctrl.Antecedent(np.arange(0, 2, 1), 'speech_status')
        pointing_gesture = ctrl.Antecedent(np.arange(0, 2, 1), 'pointing_gesture')
        body_orientation = ctrl.Antecedent(np.arange(0, 2, 1), 'body_orientation')
        direct_gaze = ctrl.Antecedent(np.arange(0, 2, 1), 'direct_gaze')
        angle = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'angle')
        waving_gesture = ctrl.Antecedent(np.arange(0, 2, 1), 'waving_gesture')
        priority = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'priority')

        proximity['Far'] = fuzz.trimf(proximity.universe, [0, 0, 0.6])
        proximity['Close'] = fuzz.trimf(proximity.universe, [0.5, 1.0, 1.0])
        speech_status['Not_Speaking'] = fuzz.trimf(speech_status.universe, [0, 0, 1])
        speech_status['Speaking'] = fuzz.trimf(speech_status.universe, [0, 1, 1])
        pointing_gesture['Not_Pointing'] = fuzz.trimf(pointing_gesture.universe, [0, 0, 1])
        pointing_gesture['Pointing'] = fuzz.trimf(pointing_gesture.universe, [0, 1, 1])
        waving_gesture['Not_Waving'] = fuzz.trimf(waving_gesture.universe, [0, 0, 1])
        waving_gesture['Waving'] = fuzz.trimf(waving_gesture.universe, [0, 1, 1])
        body_orientation['Away'] = fuzz.trimf(body_orientation.universe, [0, 0, 1])
        body_orientation['Facing'] = fuzz.trimf(body_orientation.universe, [0, 1, 1])
        direct_gaze['Indirect'] = fuzz.trimf(direct_gaze.universe, [0, 0, 1])
        direct_gaze['Direct'] = fuzz.trimf(direct_gaze.universe, [0, 1, 1])
        angle['Center'] = fuzz.trimf(angle.universe, [0, 0, 0.3])
        angle['Mid'] = fuzz.trimf(angle.universe, [0.2, 0.5, 0.8])
        angle['Edge'] = fuzz.trimf(angle.universe, [0.7, 1.0, 1.0])
        priority['Very_Low'] = fuzz.trimf(priority.universe, [0, 0, 0.2])
        priority['Low'] = fuzz.trimf(priority.universe, [0.1, 0.3, 0.5])
        priority['Medium'] = fuzz.trimf(priority.universe, [0.4, 0.6, 0.8])
        priority['High'] = fuzz.trimf(priority.universe, [0.7, 0.85, 1.0])
        priority['Very_High'] = fuzz.trimf(priority.universe, [0.9, 1.0, 1.0])

        rules = [
            ctrl.Rule(proximity['Close'] & (body_orientation['Away'] | direct_gaze['Indirect']), priority['Low']),
            ctrl.Rule(direct_gaze['Direct'], priority['Medium']),
            ctrl.Rule(speech_status['Speaking'], priority['Very_High']),
            ctrl.Rule(speech_status['Speaking'] & proximity['Close'], priority['Very_High']),
            ctrl.Rule((pointing_gesture['Pointing'] | waving_gesture['Waving']) & proximity['Close'], priority['High']),
            ctrl.Rule(proximity['Close'] & direct_gaze['Direct'] & angle['Center'], priority['High']),
            ctrl.Rule(pointing_gesture['Pointing'] | waving_gesture['Waving'], priority['Medium']),
            ctrl.Rule(proximity['Close'] & body_orientation['Facing'], priority['Medium']),
            ctrl.Rule(direct_gaze['Direct'] | angle['Center'], priority['Low']),
            ctrl.Rule(body_orientation['Facing'], priority['Low']),
            ctrl.Rule(proximity['Far'] & (body_orientation['Away'] | angle['Edge']), priority['Very_Low']),
            ctrl.Rule(proximity['Far'] | proximity['Close'], priority['Very_Low'])
        ]
        return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def extract_cues(self, person_results):
        cues = {"speech": 0, "pointing": 0, "waving": 0, "body_orientation": 0, "direct_gaze": 0}
        if not person_results.pose_landmarks or person_results.pose_landmarks.landmark[0].visibility < 0.5:
            return cues, 0 

        pose_lm = person_results.pose_landmarks.landmark
        left_shoulder, right_shoulder = pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER], pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if left_shoulder.visibility > 0.6 and right_shoulder.visibility > 0.6 and abs(left_shoulder.y - right_shoulder.y) < 0.15:
            cues["body_orientation"] = 1
        
        lip_distance = 0
        if person_results.face_landmarks:
            face_lm = person_results.face_landmarks.landmark
            nose = face_lm[1]
            if 0.2 < nose.x < 0.8: cues["direct_gaze"] = 1
            lip_distance = abs(face_lm[13].y - face_lm[14].y)
        
        for hand_id, hand_landmarks in enumerate([person_results.left_hand_landmarks, person_results.right_hand_landmarks]):
            if hand_landmarks:
                lm = hand_landmarks.landmark
                if lm[8].y < lm[6].y and lm[12].y > lm[10].y:
                    cues["pointing"] = 1
                
                shoulder_lm = pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER] if hand_id == 0 else pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                if shoulder_lm.visibility > 0.5:
                    if (lm[0].y < shoulder_lm.y) and (lm[8].y < lm[6].y and lm[20].y < lm[18].y):
                        cues["waving"] = 1
        return cues, lip_distance

    def draw_landmarks_with_overlay(self, main_frame, mp_results, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        if x1 >= x2 or y1 >= y2: return
        frame_crop = main_frame[y1:y2, x1:x2]
        if frame_crop.size == 0: return 
        
        for landmark_type, connections in [
            ('pose_landmarks', mp_holistic.POSE_CONNECTIONS),
            ('left_hand_landmarks', mp_holistic.HAND_CONNECTIONS),
            ('right_hand_landmarks', mp_holistic.HAND_CONNECTIONS)
        ]:
            landmarks = getattr(mp_results, landmark_type)
            if landmarks and connections:
                mp_drawing.draw_landmarks(
                    image=frame_crop, landmark_list=landmarks, connections=connections,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(230, 216, 173), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

        if mp_results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame_crop, landmark_list=mp_results.face_landmarks, connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(130, 255, 130), thickness=1, circle_radius=1))
        main_frame[y1:y2, x1:x2] = frame_crop

    def destroy_node(self):
        self.holistic.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = FuzzyGazeNode() 
        rclpy.spin(node)
    except FileNotFoundError:
        print('Gagal menemukan file model YOLO. Node berhenti.')
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
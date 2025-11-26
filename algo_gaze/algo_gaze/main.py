#!/home/brone-ub/ros_venv/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image 
# <--- MODIFIKASI IMPORT --->
from geometry_msgs.msg import Point 
from op3_ball_detector_msgs.msg import CircleSetStamped
# <--- AKHIR MODIFIKASI IMPORT --->

# --- Konfigurasi (sama seperti file Anda) ---
PRIORITY_COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 165, 255)]
DEFAULT_COLOR = (200, 200, 200)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class FuzzyGazeNode(Node):

    def __init__(self):
        super().__init__('fuzzy_gaze_node')
        self.get_logger().info('Fuzzy Gaze Node_v11... Mulai dengan Smoothing & OP3 Message Type.')

        # Deklarasikan parameter untuk path model YOLO
        self.declare_parameter('yolo_model_path', 'yolo11s.pt')
        yolo_model_path = '/home/brone-ub/robotis_ws/src/algo_gaze/algo_gaze/models/yolo11s.pt'
        
        self.get_logger().info(f'Memuat model YOLO dari: {yolo_model_path}')
        if not os.path.exists(yolo_model_path):
            self.get_logger().error(f'File model YOLO tidak ditemukan di: {yolo_model_path}')
            raise FileNotFoundError()

        # --- Inisialisasi ---
        self.yolo_model = YOLO(yolo_model_path)
        self.holistic = mp_holistic.Holistic(
            refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.bridge = CvBridge()
        self.person_histories = {} # Untuk deteksi bicara
        
        # --- VARIABEL BARU UNTUK SMOOTHING ---
        # 1. Smoothing Skor
        self.score_history = {} 
        self.alpha_score = 0.2 # Faktor smoothing skor (kecil = lebih stabil)
        
        # 2. Target Locking
        self.current_target_id = None
        self.last_target_switch_time = 0.0
        self.min_switch_delay = 1.0 # Detik (Kunci target minimal 1 detik)

        # 3. Smoothing Koordinat
        self.prev_norm_x = 0.0
        self.prev_norm_y = 0.0
        self.prev_norm_z = 0.0
        self.alpha_coord = 0.15 # Faktor smoothing gerakan (kecil = gerakan servo halus)
        # --------------------------------------------------
        
        self.fis_controller = self.create_fuzzy_controller()

        # --- Subscriber & Publisher ROS ---
        self.image_subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.image_publisher = self.create_publisher(
            Image, '/gaze_model/annotated_image', qos_profile)
        
        # <--- MODIFIKASI PUBLISHER: Gunakan CircleSetStamped --->
        self.center_pub_ = self.create_publisher(
            CircleSetStamped, 
            '/ball_detector_node/circle_set', 
            10)
        self.get_logger().info('Publisher /ball_detector_node/circle_set siap dengan tipe CircleSetStamped.')
        # <--- AKHIR MODIFIKASI PUBLISHER --->


    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Gagal mengonversi gambar: {e}')
            return
            
        frame_height, frame_width, _ = frame.shape

        # --- Mulai Logika Pemrosesan ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Gunakan track() dengan persist=True agar ID stabil
        yolo_results = self.yolo_model.track(image_rgb, classes=0, conf=0.5, persist=True, verbose=False)
        
        detected_people = []

        # Pastikan ada deteksi dan ID tersedia
        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

            # Loop menggunakan zip untuk box dan ID
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                
                padding = 30
                x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
                x2_pad, y2_pad = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
                
                person_crop_rgb = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                if person_crop_rgb.size == 0: continue

                results = self.holistic.process(person_crop_rgb)
                cues, lip_distance = self.extract_cues(results)
                
                detected_people.append({
                    'id': int(track_id), # Gunakan ID dari Tracker
                    'cues': cues, 
                    'bbox_yolo': (x1, y1, x2, y2),
                    'score': 0, 
                    'results_mp': results,
                    'crop_bbox_padded': (x1_pad, y1_pad, x2_pad, y2_pad),
                    'lip_distance': lip_distance
                })

        if detected_people:
            all_areas = [(p['bbox_yolo'][2] - p['bbox_yolo'][0]) * (p['bbox_yolo'][3] - p['bbox_yolo'][1]) for p in detected_people]
            max_area = max(all_areas) if all_areas else 1

            for p in detected_people:
                # --- ALGORITMA FUZZY ---
                # 1. Kalkulasi Proximity
                area = (p['bbox_yolo'][2] - p['bbox_yolo'][0]) * (p['bbox_yolo'][3] - p['bbox_yolo'][1])
                proximity_value = area / max_area
                p['cues']['proximity'] = proximity_value
                
                # 2. Kalkulasi Angle
                x1, _, x2, _ = p['bbox_yolo']
                person_center_x = (x1 + x2) / 2
                frame_center_x = frame_width / 2
                angle_value = abs(person_center_x - frame_center_x) / frame_center_x
                angle_value = min(angle_value, 1.0)
                p['cues']['angle'] = angle_value
                
                # 3. Logika Deteksi Bicara
                p_id = p['id']
                if p_id not in self.person_histories:
                    self.person_histories[p_id] = deque(maxlen=10)
                self.person_histories[p_id].append(p['lip_distance'])
                
                if len(self.person_histories[p_id]) == 10:
                    variance = np.var(list(self.person_histories[p_id]))
                    if variance > 0.00008:
                        p['cues']['speech'] = 1
                    else:
                        p['cues']['speech'] = 0
                
                # 4. Kalkulasi Skor Fuzzy
                self.fis_controller.input['proximity'] = proximity_value
                self.fis_controller.input['speech_status'] = p['cues']['speech']
                self.fis_controller.input['pointing_gesture'] = p['cues']['pointing']
                self.fis_controller.input['waving_gesture'] = p['cues']['waving']
                self.fis_controller.input['body_orientation'] = p['cues']['body_orientation']
                self.fis_controller.input['direct_gaze'] = p['cues']['direct_gaze']
                self.fis_controller.input['angle'] = angle_value
                
                self.fis_controller.compute()
                raw_score = self.fis_controller.output['priority']

                # --- SMOOTHING SKOR (EMA) ---
                if p_id not in self.score_history:
                    self.score_history[p_id] = raw_score
                else:
                    # Rumus: value = (alpha * new) + ((1-alpha) * old)
                    self.score_history[p_id] = (self.alpha_score * raw_score) + ((1 - self.alpha_score) * self.score_history[p_id])
                
                p['score'] = self.score_history[p_id] 

            # --- LOGIKA HYSTERESIS + WAKTU (TARGET LOCKING) ---
            sorted_people = sorted(detected_people, key=lambda p: p['score'], reverse=True)
            potential_winner = sorted_people[0]
            current_time = time.time()

            if self.current_target_id is None:
                # Inisialisasi target awal
                self.current_target_id = potential_winner['id']
                self.last_target_switch_time = current_time
            else:
                # Cek apakah target lama masih ada di frame?
                target_still_visible = any(p['id'] == self.current_target_id for p in detected_people)
                
                if not target_still_visible:
                    # Target hilang -> langsung ganti
                    self.current_target_id = potential_winner['id']
                    self.last_target_switch_time = current_time
                else:
                    # Target masih ada, cek apakah layak diganti?
                    # Syarat 1: Waktu lock minimum terpenuhi
                    time_ok = (current_time - self.last_target_switch_time) > self.min_switch_delay
                    
                    # Syarat 2: Skor penantang lebih tinggi secara signifikan (Hysteresis 15%)
                    current_target_data = next((p for p in detected_people if p['id'] == self.current_target_id), None)
                    current_score = current_target_data['score'] if current_target_data else 0
                    
                    score_ok = potential_winner['score'] > (current_score * 1.15)

                    if time_ok and score_ok:
                        self.current_target_id = potential_winner['id']
                        self.last_target_switch_time = current_time
            
            # Logika Visualisasi
            for rank, p in enumerate(sorted_people):
                x1, y1, x2, y2 = p['bbox_yolo']
                
                # Ubah warna jika target
                is_target = (p['id'] == self.current_target_id)
                color = (0, 0, 255) if is_target else (PRIORITY_COLORS[rank] if rank < len(PRIORITY_COLORS) else DEFAULT_COLOR)
                
                self.draw_landmarks_with_overlay(frame, p['results_mp'], p['crop_bbox_padded'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{p['id']} | S:{p['score']:.2f}"
                
                if is_target:
                    # --- SMOOTHING KOORDINAT OUTPUT (LPF) ---
                    label = f"TARGET | Score: {p['score']:.2f}"
                    
                    # 1. Hitung pusat pixel
                    center_x_px = (x1 + x2) / 2.0
                    center_y_px = (y1 + y2) / 2.0
                    
                    # 2. Normalisasi (-1.0 s/d 1.0)
                    raw_norm_x = (center_x_px / frame_width) * 2.0 - 1.0
                    raw_norm_y = (center_y_px / frame_height) * 2.0 - 1.0
                    raw_norm_z = p['cues']['proximity']
                    
                    # 3. Terapkan Filter EMA (Smoothing)
                    smooth_x = (self.alpha_coord * raw_norm_x) + ((1 - self.alpha_coord) * self.prev_norm_x)
                    smooth_y = (self.alpha_coord * raw_norm_y) + ((1 - self.alpha_coord) * self.prev_norm_y)
                    smooth_z = (self.alpha_coord * raw_norm_z) + ((1 - self.alpha_coord) * self.prev_norm_z)
                    
                    # 4. Simpan nilai untuk frame berikutnya
                    self.prev_norm_x, self.prev_norm_y, self.prev_norm_z = smooth_x, smooth_y, smooth_z

                    # Publish nilai yang sudah halus
                    self.publish_coordinates(smooth_x, smooth_y, smooth_z, msg.header)
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # --- Selesai Logika Pemrosesan ---

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            annotated_msg.header = msg.header
            self.image_publisher.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Gagal mengonversi gambar kembali: {e}')

    # <--- MODIFIKASI FUNGSI HELPER PUBLISH (SOLUSI A) --->
    def publish_coordinates(self, x, y, z, header):
        """
        Mempersiapkan dan mem-publish pesan CircleSetStamped.
        Node OP3 biasanya menggunakan tipe ini.
        """
        # 1. Buat instance CircleSetStamped
        msg = CircleSetStamped()
        
        # 2. Isi header
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id 
        
        # 3. Buat Point (Circle Center)
        # Di message OP3 standar, 'circles' adalah list of geometry_msgs/Point
        # Kita masukkan data kita ke sana.
        circle_point = Point()
        circle_point.x = float(x)
        circle_point.y = float(y)
        circle_point.z = float(z) # Menggunakan Z (Proximity) sebagai radius/z

        # 4. Masukkan ke dalam list 'circles'
        msg.circles.append(circle_point)
        
        # 5. Publish pesan
        self.center_pub_.publish(msg)
    # <--- AKHIR MODIFIKASI FUNGSI HELPER --->

    # --- FUNGSI HELPER FUZZY (TIDAK BERUBAH) ---
    def create_fuzzy_controller(self):
        # 1. Define Antecedents (Inputs)
        proximity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'proximity')
        speech_status = ctrl.Antecedent(np.arange(0, 2, 1), 'speech_status')
        pointing_gesture = ctrl.Antecedent(np.arange(0, 2, 1), 'pointing_gesture')
        body_orientation = ctrl.Antecedent(np.arange(0, 2, 1), 'body_orientation')
        direct_gaze = ctrl.Antecedent(np.arange(0, 2, 1), 'direct_gaze')
        angle = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'angle')
        waving_gesture = ctrl.Antecedent(np.arange(0, 2, 1), 'waving_gesture')

        # 2. Define Consequent (Output)
        priority = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'priority')

        # 3. Define Membership Functions for Inputs
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

        # 4. Define Membership Functions for Output
        priority['Very_Low'] = fuzz.trimf(priority.universe, [0, 0, 0.2])
        priority['Low'] = fuzz.trimf(priority.universe, [0.1, 0.3, 0.5])
        priority['Medium'] = fuzz.trimf(priority.universe, [0.4, 0.6, 0.8])
        priority['High'] = fuzz.trimf(priority.universe, [0.7, 0.85, 1.0])
        priority['Very_High'] = fuzz.trimf(priority.universe, [0.9, 1.0, 1.0])

        # 5. Define Fuzzy Rules
        rule_penalty_close_disengaged = ctrl.Rule(proximity['Close'] & (body_orientation['Away'] | direct_gaze['Indirect']), priority['Low'])
        rule_reward_gaze = ctrl.Rule(direct_gaze['Direct'], priority['Medium'])
        rule_speech_dominant = ctrl.Rule(speech_status['Speaking'], priority['Very_High'])
        rule1 = ctrl.Rule(speech_status['Speaking'] & proximity['Close'], priority['Very_High'])
        rule3 = ctrl.Rule((pointing_gesture['Pointing'] | waving_gesture['Waving']) & proximity['Close'], priority['High'])
        rule4 = ctrl.Rule(proximity['Close'] & direct_gaze['Direct'] & angle['Center'], priority['High'])
        rule5 = ctrl.Rule(pointing_gesture['Pointing'] | waving_gesture['Waving'], priority['Medium'])
        rule6 = ctrl.Rule(proximity['Close'] & body_orientation['Facing'], priority['Medium'])
        rule7 = ctrl.Rule(direct_gaze['Direct'] | angle['Center'], priority['Low'])
        rule8 = ctrl.Rule(body_orientation['Facing'], priority['Low'])
        rule9 = ctrl.Rule(proximity['Far'] & (body_orientation['Away'] | angle['Edge']), priority['Very_Low'])
        
        rule_default = ctrl.Rule(proximity['Far'] | proximity['Close'], priority['Very_Low'])
        
        priority_ctrl_system = ctrl.ControlSystem([
            rule_penalty_close_disengaged, rule_reward_gaze ,rule_speech_dominant, rule1, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule_default
        ])
        
        return ctrl.ControlSystemSimulation(priority_ctrl_system)

    # --- FUNGSI CUES (TIDAK BERUBAH) ---
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

    # --- FUNGSI DRAWING (TIDAK BERUBAH) ---
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
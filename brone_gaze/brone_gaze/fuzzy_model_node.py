
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import skfuzzy as fuzz               # <-- DITAMBAHKAN
from skfuzzy import control as ctrl  # <-- DITAMBAHKAN
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image # Pastikan ini juga ada

# --- Konfigurasi (sama seperti file Anda) ---
PRIORITY_COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 165, 255)]
DEFAULT_COLOR = (200, 200, 200)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class FuzzyGazeNode(Node): # <-- NAMA CLASS DIGANTI

    def __init__(self):
        super().__init__('fuzzy_gaze_node') # <-- NAMA NODE DIGANTI
        self.get_logger().info('Fuzzy Gaze Node_v11... Mulai.') # <-- NAMA NODE DIGANTI

        # Deklarasikan parameter untuk path model YOLO
        self.declare_parameter('yolo_model_path', 'yolo11s.pt')
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        
        # Jika path tidak absolut, asumsikan itu relatif terhadap share dir
        if not os.path.isabs(yolo_model_path):
             pkg_share = get_package_share_directory('brone_gaze')
             yolo_model_path = os.path.join(pkg_share, 'models', os.path.basename(yolo_model_path))

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
        self.person_histories = {}
        self.current_target_id = None
        
        # --- LOGIKA FUZZY DITAMBAHKAN ---
        self.fis_controller = self.create_fuzzy_controller()
        # --- BOBOT LINEAR DIHAPUS ---

        # --- Subscriber & Publisher ROS ---
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        
        qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.image_publisher = self.create_publisher(
            Image,
            '/gaze_model/annotated_image',
            qos_profile
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Gagal mengonversi gambar: {e}')
            return
            
        # Dapatkan dimensi frame sekali saja
        frame_height, frame_width, _ = frame.shape

        # --- Mulai Logika Pemrosesan ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_results = self.yolo_model(image_rgb, classes=0, conf=0.5, verbose=False)
        
        detected_people = []

        for i, box in enumerate(yolo_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            padding = 30
            x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
            x2_pad, y2_pad = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            
            person_crop_rgb = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
            if person_crop_rgb.size == 0: continue

            results = self.holistic.process(person_crop_rgb)
            
            # Panggil extract_cues versi Kode 2 (tanpa crop_w, crop_h)
            cues, lip_distance = self.extract_cues(results)
            
            detected_people.append({
                'id': i, 'cues': cues, 'bbox_yolo': (x1, y1, x2, y2),
                'score': 0, 'results_mp': results,
                'crop_bbox_padded': (x1_pad, y1_pad, x2_pad, y2_pad),
                'lip_distance': lip_distance
            })

        if detected_people:
            all_areas = [(p['bbox_yolo'][2] - p['bbox_yolo'][0]) * (p['bbox_yolo'][3] - p['bbox_yolo'][1]) for p in detected_people]
            max_area = max(all_areas) if all_areas else 1

            for p in detected_people:
                # --- ALGORITMA FUZZY (dari Kode 2) ---
                
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
                
                # 3. Logika Deteksi Bicara (sama)
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
                
                # 4. Kalkulasi Skor Fuzzy (MENGGANTIKAN SKOR LINEAR)
                self.fis_controller.input['proximity'] = proximity_value
                self.fis_controller.input['speech_status'] = p['cues']['speech']
                self.fis_controller.input['pointing_gesture'] = p['cues']['pointing']
                self.fis_controller.input['waving_gesture'] = p['cues']['waving']
                self.fis_controller.input['body_orientation'] = p['cues']['body_orientation']
                self.fis_controller.input['direct_gaze'] = p['cues']['direct_gaze']
                self.fis_controller.input['angle'] = angle_value
                
                self.fis_controller.compute()
                p['score'] = self.fis_controller.output['priority']
                # --- AKHIR ALGORITMA FUZZY ---

            # Logika Hysteresis (Target Sticking) - (Tidak berubah)
            sorted_people = sorted(detected_people, key=lambda p: p['score'], reverse=True)
            
            if sorted_people:
                new_top_person = sorted_people[0]
                if self.current_target_id is None or self.current_target_id not in [p['id'] for p in detected_people]:
                    self.current_target_id = new_top_person['id']
                else:
                    current_target_score = 0
                    for person in detected_people:
                        if person['id'] == self.current_target_id:
                            current_target_score = person['score']
                            break
                    if new_top_person['score'] > current_target_score * 1.15:
                        self.current_target_id = new_top_person['id']
            
            # Logika Visualisasi - (Tidak berubah)
            for rank, p in enumerate(sorted_people):
                x1, y1, x2, y2 = p['bbox_yolo']
                color = PRIORITY_COLORS[rank] if rank < len(PRIORITY_COLORS) else DEFAULT_COLOR
                
                # Menggunakan fungsi drawing dari Kode 1
                self.draw_landmarks_with_overlay(frame, p['results_mp'], p['crop_bbox_padded'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                is_target = p['id'] == self.current_target_id
                label = f"P:{rank + 1} | Score: {p['score']:.2f}"
                if is_target:
                    target_rank = next((i + 1 for i, person in enumerate(sorted_people) if person['id'] == self.current_target_id), 0)
                    label = f"TARGET (P:{target_rank}) | Score: {p['score']:.2f}"
                
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

    # --- FUNGSI HELPER BARU DARI KODE 2 ---
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
        # ATURAN HADIAH: Jika seseorang menatap langsung ke kamera, beri mereka
        # setidaknya prioritas 'Medium', tidak peduli seberapa jauh mereka.
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

    # --- FUNGSI HELPER DIGANTI DENGAN VERSI KODE 2 ---
    def extract_cues(self, person_results):
        # Cues 'proximity' dan 'angle' akan dihitung di loop utama
        cues = {"speech": 0, "pointing": 0, "waving": 0, "body_orientation": 0, "direct_gaze": 0}
        
        if not person_results.pose_landmarks or person_results.pose_landmarks.landmark[0].visibility < 0.5:
            return cues, 0 # Mengembalikan jarak bibir 0 jika tidak ada landmark

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

    # --- FUNGSI DRAWING DARI KODE 1 (TIDAK BERUBAH) ---
    def draw_landmarks_with_overlay(self, main_frame, mp_results, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        if x1 >= x2 or y1 >= y2: return
        frame_crop = main_frame[y1:y2, x1:x2]
        if frame_crop.size == 0: return # Tambahan keamanan
        
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
        node = FuzzyGazeNode() # <-- Menggunakan class baru
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
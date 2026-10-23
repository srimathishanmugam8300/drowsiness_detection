# src/drowsiness_detector.py

import cv2
import mediapipe as mp
import numpy as np
import time
import threading

from utils import eye_aspect_ratio, mouth_aspect_ratio, rotationMatrixToEulerAngles


# =====================================================================
#  ALARM PLAYER (UPDATED – WORKS IN WINDOWS, LINUX, MAC, COLAB)
# =====================================================================
class AlarmPlayer:
    def __init__(self):
        self.alarm_on = False
        self.thread = None

    def _play_alarm(self):
        from playsound import playsound
        while self.alarm_on:
            try:
                playsound("alarm.mp3")
            except:
                pass
            time.sleep(0.1)

    def start(self):
        if self.alarm_on:
            return
        self.alarm_on = True
        self.thread = threading.Thread(target=self._play_alarm, daemon=True)
        self.thread.start()

    def stop(self):
        self.alarm_on = False
        if self.thread:
            self.thread.join(timeout=0.2)
            self.thread = None


# =====================================================================
#   DROWSINESS DETECTOR
# =====================================================================
class DrowsinessDetector:
    def __init__(self,
                 ear_thresh=0.20,
                 ear_consec_seconds=10.0,
                 head_pitch_thresh_deg=15.0,
                 head_consec_seconds=10.0,
                 mar_thresh=0.6,
                 mar_consec_seconds=60.0,
                 use_camera_index=0):

        self.EAR_THRESH = ear_thresh
        self.EAR_CONSEC_SECONDS = ear_consec_seconds

        self.HEAD_PITCH_THRESH = head_pitch_thresh_deg
        self.HEAD_CONSEC_SECONDS = head_consec_seconds

        self.MAR_THRESH = mar_thresh
        self.MAR_CONSEC_SECONDS = mar_consec_seconds

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_style()

        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]

        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

        self.PNP_INDICES = [1, 199, 33, 263, 61, 291]

        # timers
        self.eye_closed_start = None
        self.head_down_start = None
        self.mouth_open_start = None

        self.alarm = AlarmPlayer()

    # -------------------------------------------------------
    def _landmarks_to_xy(self, lm_list, w, h):
        pts = []
        for lm in lm_list:
            pts.append((int(lm.x * w), int(lm.y * h)))
        return pts

    def _get_2d_points_for_pnp(self, pts):
        return np.array([pts[i] for i in self.PNP_INDICES], dtype='double')

    def _get_3d_model_points(self):
        model_points = [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ]
        return np.array(model_points, dtype='double')

    def _estimate_head_pose(self, image_points, image_size):
        h, w = image_size
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double'
        )
        dist_coeffs = np.zeros((4, 1))

        model_points = self._get_3d_model_points()

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = rotationMatrixToEulerAngles(R)
        return {'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll)}

    # -------------------------------------------------------
    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        alarm_triggered = False
        info = {'ear': None, 'mar': None, 'pitch': None, 'events': []}

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            pts = self._landmarks_to_xy(face, img_w, img_h)

            # EAR
            ear_right = eye_aspect_ratio(pts, self.RIGHT_EYE)
            ear_left = eye_aspect_ratio(pts, self.LEFT_EYE)
            ear = (ear_left + ear_right) / 2
            info['ear'] = ear

            # MAR
            mar = mouth_aspect_ratio(pts, self.MOUTH_TOP, self.MOUTH_BOTTOM,
                                     self.MOUTH_LEFT, self.MOUTH_RIGHT)
            info['mar'] = mar

            # Head pose
            try:
                img_points = self._get_2d_points_for_pnp(pts)
                pose = self._estimate_head_pose(img_points, (img_h, img_w))
                if pose:
                    pitch = pose['pitch']
                    info['pitch'] = pitch
                else:
                    pitch = None
            except:
                pitch = None

            # Draw mesh
            self.drawing_utils.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.drawing_spec
            )

            now = time.time()

            # EYES CLOSED
            if ear < self.EAR_THRESH:
                if self.eye_closed_start is None:
                    self.eye_closed_start = now
                dur = now - self.eye_closed_start
                if dur >= self.EAR_CONSEC_SECONDS:
                    info['events'].append(f'EYE_CLOSED_{int(dur)}s')
            else:
                self.eye_closed_start = None

            # HEAD DOWN
            if pitch is not None and pitch > self.HEAD_PITCH_THRESH:
                if self.head_down_start is None:
                    self.head_down_start = now
                hd = now - self.head_down_start
                if hd >= self.HEAD_CONSEC_SECONDS:
                    info['events'].append(f'HEAD_DOWN_{int(hd)}s')
            else:
                self.head_down_start = None

            # YAWN (continuous)
            if mar > self.MAR_THRESH:
                if self.mouth_open_start is None:
                    self.mouth_open_start = now
                mo = now - self.mouth_open_start
                if mo >= self.MAR_CONSEC_SECONDS:
                    info['events'].append(f'MOUTH_OPEN_{int(mo)}s')
            else:
                self.mouth_open_start = None

            # Start/stop alarm
            if info['events']:
                alarm_triggered = True
                if not self.alarm.alarm_on:
                    self.alarm.start()
            else:
                if self.alarm.alarm_on:
                    self.alarm.stop()

        else:
            if self.alarm.alarm_on:
                self.alarm.stop()

        return frame, alarm_triggered, info

    # -------------------------------------------------------
    def close(self):
        self.alarm.stop()
        self.face_mesh.close()


# src/utils.py
import math
import numpy as np

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(landmarks, indices):
    """
    landmarks: list of (x,y) tuples (image coordinates)
    indices: 6 indices: [p1,p2,p3,p4,p5,p6] for one eye
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """
    Simple MAR: vertical distance between inner upper/lower lip divided by mouth width
    landmarks: list of (x,y)
    returns float
    """
    vertical = euclidean(landmarks[top_idx], landmarks[bottom_idx])
    horizontal = euclidean(landmarks[left_idx], landmarks[right_idx])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def rotationMatrixToEulerAngles(R):
    """
    Convert rotation matrix to Euler angles (in degrees).
    Returns (pitch, yaw, roll)
    Adapted for typical OpenCV solvePnP output.
    """
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # convert to degrees
    return np.degrees([y, z, x])  # pitch(y), yaw(z), roll(x)

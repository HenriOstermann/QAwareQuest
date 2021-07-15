import cv2
import mediapipe as mp
import math
import imutils

# Config
AugenAnteil: float = 0.15

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        x1: int = 80
        x2: int = 560
        y1: int = 0
        y2: int = 480
        xm: int = 320
        ym: int = 240
        xm0: int = 320
        ym0: int = 240
        dx: int = 0
        dy: int = 0
        w0: float = 0.4
        Width: float = 0.4
        angle: float = 0
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                xm0 = xm
                ym0 = ym
                w0 = Width
                h = detection.location_data.relative_keypoints[2].y - detection.location_data.relative_keypoints[1].y
                w = detection.location_data.relative_keypoints[2].x - detection.location_data.relative_keypoints[1].x
                Width = math.sqrt(h * h + w * w)
                Width = w0 + 90/100 * (Width - w0)
                Factor = max(AugenAnteil / Width, 1)
                print(AugenAnteil/Width)
                angle = math.tan(h / w)
                mp_drawing.draw_detection(image, detection)
                xm = int(detection.location_data.relative_keypoints[3].x * 640 * Factor)
                dx = xm - xm0
                xm = int(xm0 + 3 / 4 * dx)
                ym = int(detection.location_data.relative_keypoints[3].y * 480 * Factor)
                dy = ym - ym0
                ym = int(ym0 + 3 / 4 * dy)
                if dx > 50 * Factor:
                    if xm < 240 * Factor:
                        x1: int = 0
                        x2: int = 480
                    elif xm > 400 * Factor:
                        x2 = int(640 * Factor)
                        x1 = x2 - 480
                    else:
                        x1 = xm - 240
                        x2 = xm + 240
                if dy > 50 * Factor:
                    if ym < 240 * Factor:
                        y1: int = 0
                        y2: int = 480
                    elif ym > 240 * Factor:
                        y2 = int(480 * Factor)
                        y1 = y2 - 480
                    else:
                        y1 = ym - 240
                        y2 = ym + 240
        zoomed_image = cv2.resize(image, (int(640 * Factor), int(480 * Factor)))
        crop_image = zoomed_image[y1:y2, x1:x2]
        # rotated_image = imutils.rotate(crop_image, angle=angle)
        cv2.imshow('Face Detection', crop_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

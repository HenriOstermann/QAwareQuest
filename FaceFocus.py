# Import
import cv2
import mediapipe as mp
import math
import random
import imutils

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Define Functions
def average(lst):
    return sum(lst) / len(lst)


# Config
AugenAnteil: float = 0.15
smoothness: int = 25
hOut: int = 480
wOut: int = 480


# Initialise Variables
xNoseHist = [320] * smoothness
yNoseHist = [240] * smoothness
widthHist = [0.4] * smoothness
i: int = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        wIn = len(image[0])
        hIn = len(image)

        # Draw the face detection annotations on the image.
        # Initialise Variables (2)
        x1 = int((wIn - wOut) / 2)
        x2: int = wIn - x1
        y1 = int((hIn - hOut) / 2)
        y2: int = hIn - y1
        xNose = int(wIn / 2)
        yNose = int(hIn / 2)
        width: float = 0.4
        angle: float = 0

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image1 = cv2.resize(image, (wIn, hIn))

        if results.detections:
            for detection in results.detections:
                yCoordREye = detection.location_data.relative_keypoints[2].y
                yCoordLEye = detection.location_data.relative_keypoints[1].y
                hEyes = yCoordREye - yCoordLEye
                xCoordREye = detection.location_data.relative_keypoints[2].x
                xCoordLEye = detection.location_data.relative_keypoints[1].x
                wEyes = xCoordREye - xCoordLEye
                width = math.sqrt(hEyes * hEyes + wEyes * wEyes)
                widthHist[i] = width
                width = average(widthHist)
                Factor = max(AugenAnteil / width, 1)
                angle = math.tan(hEyes / wEyes)
                mp_drawing.draw_detection(image1, detection)
                xNose = int(detection.location_data.relative_keypoints[3].x * wIn * Factor)
                xNoseHist[i] = xNose
                xNose = int(average(xNoseHist))
                yNose = int(detection.location_data.relative_keypoints[3].y * hIn * Factor)
                yNoseHist[i] = yNose
                yNose = int(average(yNoseHist))

                if xNose < (wOut / 2) * Factor:
                    x1: int = 0
                    x2: int = wOut
                elif xNose > (wIn - (wOut / 2)) * Factor:
                    x2 = int(wIn * Factor)
                    x1 = x2 - wOut
                else:
                    x1 = int(xNose - (wOut / 2))
                    x2 = int(xNose + (wOut / 2))

                if yNose < (hOut / 2) * Factor:
                    y1: int = 0
                    y2: int = hOut
                elif yNose > (hIn - (hOut / 2)) * Factor:
                    y2 = int(hIn * Factor)
                    y1 = y2 - hOut
                else:
                    y1 = int(yNose - (hOut / 2))
                    y2 = int(yNose + (hOut / 2))
            if i == smoothness - 1:
                i = 0
            else:
                i += 1
        zoomed_image = cv2.resize(image, (int(wIn * Factor), int(hIn * Factor)))
        crop_image = zoomed_image[y1:y2, x1:x2]
        # rotated_image = imutils.rotate(crop_image, angle=angle)
        cv2.imshow('Focused Image', crop_image)
        f1 = random.randint(0, 255)
        f2 = random.randint(0, 255)
        f3 = random.randint(0, 255)
        cv2.rectangle(image1, (int(x1/Factor), int(y1/Factor)), (int(x2/Factor), int(y2/Factor)), (f1, f2, f3))
        cv2.imshow('Original Capture', image1)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

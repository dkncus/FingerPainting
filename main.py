import cv2 as cv
import time
import mediapipe as mp
import pyautogui
import numpy as np

# Start a video capture from the inbuilt webcam
cap = cv.VideoCapture(0)

# Hand detector model
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

frame_no = 0

past_time = 0
current_time = 0

circles = []

num_past_gestures = 11
past_gestures = []

def detect_gesture(coords, past_gestures):
    # Index finger keypoints
    index_lows, middle_lows = (coords[5, 0], coords[5, 1]), (coords[9, 0], coords[9, 1])
    index_mids, middle_mids = (coords[6, 0], coords[6, 1]), (coords[10, 0], coords[10, 1])
    index_tips, middle_tips = (coords[7, 0], coords[7, 1]), (coords[11, 0], coords[11, 1])

    # Geometric distances between points
    dist_tips = np.sqrt(np.sum(np.square(np.abs(np.array(index_tips) - np.array(middle_tips)))))
    dist_mids = np.sqrt(np.sum(np.square(np.abs(np.array(index_mids) - np.array(middle_mids)))))
    dist_lows = np.sqrt(np.sum(np.square(np.abs(np.array(index_lows) - np.array(middle_lows)))))

    # cv.line(img, index_lows, middle_lows, (255, 255, 255), thickness=2)
    # cv.line(img, index_mids, middle_mids, (255, 255, 255), thickness=2)
    # cv.line(img, index_tips, middle_tips, (255, 255, 255), thickness=2)

    # Gesture 1 - Painting
    if len(past_gestures) >= num_past_gestures:
        past_gestures.pop(0)

    if (dist_lows / dist_tips) > 0.9 and (dist_mids / dist_tips) > 0.8:
        return 1
    else:
        return 0

while True:
    # Read an image from the webcam
    success, img = cap.read()

    # Mirror the webcam's image
    img = cv.flip(img, 1)
    frame_no += 1

    # Convert color and get results marking hand keypointsd
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Create a set of coordinates of the keypoints
    coords = np.zeros(shape=(20, 2)).astype(np.int64)
    img_circles = np.zeros(shape=(img.shape[0], img.shape[1], 3))

    # Get the height, width, and # of image channels
    h, w, c = img.shape

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        # print(type(results.multi_hand_landmarks[0]))
        # print(results.multi_hand_landmarks[0].landmark[0])

        # For each hand detected in the image
        for handLms in results.multi_hand_landmarks:

            # Wireframe hand drawing
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # For each landmark in the set of landmarks
            for id, lm in enumerate(handLms.landmark):
                # Scale the landmark position to match the screen
                cx, cy, cz = int(lm.x * w), int(lm.y *h), lm.z

                # Store the coordinates
                coords[id-1] = np.array([cx, cy]).astype(np.int64)

                # # Draw a circle at the thumb, index, and middle finger tips
                if id == 4 or id == 8 or id == 12:
                    cv.circle(img, (cx,cy), 10, (0,255,0), cv.FILLED)
                    cv.circle(img_circles, (cx,cy), 10, (0,255,0), cv.FILLED)

        # Gesture Detector
        gesture = detect_gesture(coords, past_gestures)
        past_gestures.append(gesture)

        if gesture == 1:
            # Compute the average between the index and middle finger
            point = ((coords[11, 0] + coords[7, 0])//2, (coords[11, 1] + coords[7, 1])//2)
            circles.append((point))

    # Draw the computed circles on the screen
    for circle in circles:
        img = cv.circle(img, circle, 20, (255, 128, 0), cv.FILLED)

    cv.imshow("Image", img)
    cv.waitKey(1)
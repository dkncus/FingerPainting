import cv2 as cv
import time
import mediapipe as mp
import pyautogui
import numpy as np
import csv
from collections import deque
import itertools
import copy

# Gesture Tracking Models
from gesture_model import KeyPointClassifier
from gesture_model import PointHistoryClassifier

global hands, mpHands, mpDraw, hsv_icon, mask_icon, past_gestures, num_past_gestures, circles, keypoint_classifier

def interpret_frame(image):
    img_debug = image.copy()

    # Mirror the webcam's image
    img_debug = cv.flip(img_debug, 1)

    # Convert color and get results marking hand keypoints
    rgb = cv.cvtColor(img_debug, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Create a set of coordinates of the keypoints
    coords = np.zeros(shape=(20, 2)).astype(np.int64)

    # Get the height, width, and # of image channels
    h, w, c = img_debug.shape

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:

        # For each hand detected in the image
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Wireframe hand drawing
            mpDraw.draw_landmarks(img_debug, hand_landmarks, mpHands.HAND_CONNECTIONS)
            brect = calc_bounding_rect(img_debug, hand_landmarks)
            landmark_list = calc_landmark_list(img_debug, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            past_gestures.append(hand_sign_id)
            if len(past_gestures) >= num_past_gestures:
                past_gestures.pop(0)

            # Draw point marker circle
            x_1, y_1 = landmark_list[4]
            x_2, y_2 = landmark_list[8]
            x_c = int(round(((x_1 + x_2) / 2), 0))
            y_c = int(round(((y_1 + y_2) / 2), 0))
            img_debug = cv.circle(img_debug, (x_c, y_c), 9, (0, 255, 255), thickness=2)
            img_debug = cv.circle(img_debug, (x_c, y_c), 3, (0, 255, 255), thickness=1)

            # Palm sign
            if hand_sign_id == 0:
                x_1, y_1 = landmark_list[0]
                x_2, y_2 = landmark_list[5]
                x_3, y_3 = landmark_list[17]
                x_c = int(round(((x_1 + x_2 + x_3) / 3), 0))
                y_c = int(round(((y_1 + y_2 + y_3) / 3), 0))
                img_debug = cv.circle(img_debug, (x_c, y_c), 10, (0, 255, 0), cv.FILLED)

                # Paste image of color wheel at this location

                s = 50
                # Check if in image bounds
                if not (x_c - s < 0 or x_c + s > w or y_c - s < 0 or y_c + s > h):
                    color_mask = np.zeros_like(img_debug)
                    transparency_mask = np.zeros_like(img_debug)

                    color_mask[y_c - s:y_c + s, x_c - s:x_c + s, :] = hsv_icon
                    transparency_mask[y_c - s:y_c + s, x_c - s:x_c + s, :] = mask_icon
                    img_debug[transparency_mask == 1] = color_mask[transparency_mask == 1]



            # Pinch sign
            elif hand_sign_id == 3:
                img_debug = cv.circle(img_debug, (x_1, y_1), 10, (0, 255, 0), cv.FILLED)
                img_debug = cv.circle(img_debug, (x_2, y_2), 10, (0, 255, 0), cv.FILLED)
                circles.append((x_c, y_c))

            img_debug = draw_bounding_rect(use_brect=True, image=img_debug, brect=brect)

    # Draw the computed circles on the screen
    for i, circle in enumerate(circles):
        img_debug = cv.circle(img_debug, circle, 15, (255, 255, 255), cv.FILLED)

        if i < len(circles) - 1 and len(circles) > 2:
            pass
            cv.line(img_debug, circle, circles[i+1], (255, 255, 255), thickness=5)



    print(past_gestures)
    cv.imshow("Image", img_debug)
    cv.waitKey(1)




def detect_gesture(coords, past_gestures):
    # Index finger keypoints
    index_lows, middle_lows = (coords[5, 0], coords[5, 1]), (coords[9, 0], coords[9, 1])
    index_mids, middle_mids = (coords[6, 0], coords[6, 1]), (coords[10, 0], coords[10, 1])
    index_tips, middle_tips = (coords[7, 0], coords[7, 1]), (coords[11, 0], coords[11, 1])

    # Geometric distances between points
    dist_tips = np.sqrt(np.sum(np.square(np.abs(np.array(index_tips) - np.array(middle_tips)))))
    dist_mids = np.sqrt(np.sum(np.square(np.abs(np.array(index_mids) - np.array(middle_mids)))))
    dist_lows = np.sqrt(np.sum(np.square(np.abs(np.array(index_lows) - np.array(middle_lows)))))

    # cv.line(img_debug, index_lows, middle_lows, (255, 255, 255), thickness=2)
    # cv.line(img_debug, index_mids, middle_mids, (255, 255, 255), thickness=2)
    # cv.line(img_debug, index_tips, middle_tips, (255, 255, 255), thickness=2)

    # Gesture 1 - Painting
    if (dist_lows / dist_tips) > 0.9 and (dist_mids / dist_tips) > 0.8:
        return 1
    else:
        return 0


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(use_brect, image, brect):
    # Draw the image using the bounding rectangle
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def calc_landmark_list(image, landmarks):
    """
    It takes in an image and a list of landmarks, and returns a list of landmark points

    :param image: The image to be processed
    :param landmarks: The landmarks for the image
    :return: A list of lists of x,y coordinates.
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def runApp():

    # Hand detector
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # Gesture Detection Models
    keypoint_classifier = KeyPointClassifier()
    with open('gesture_model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    history_length = 16
    finger_gesture_history = deque(maxlen=history_length)

    # Start a video capture from the inbuilt webcam
    cap = cv.VideoCapture(0)

    past_time = 0
    current_time = 0
    circles = []
    num_past_gestures = 8
    past_gestures = []

    hsv_icon = cv.imread('assets/hsv_icons/hsv_icon_color.png')
    mask_icon = cv.imread('assets/hsv_icons/hsv_icon_mask.png')
    mask_icon = 1 - (mask_icon / 255).astype(np.uint8)

    # Main loop
    while True:
        # Read an image from the webcam
        success, img = cap.read()

        # If an image from the webcam comes across
        if success:
            # Interpret it
            interpret_frame(img)


if __name__ == '__main__':
    runApp()
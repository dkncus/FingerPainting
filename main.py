import cv2 as cv
import time
import mediapipe as mp
import pyautogui
import numpy as np
import csv
from collections import deque
import itertools
import copy
import math
# Gesture Tracking Models
from gesture_model import KeyPointClassifier
from gesture_model import PointHistoryClassifier

from scipy.ndimage import gaussian_filter1d

class Interpreter():

    def __init__(self):
        # Hand detector
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

        # Gesture Detection Models
        self.keypoint_classifier = KeyPointClassifier()
        with open('gesture_model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        history_length = 16
        finger_gesture_history = deque(maxlen=history_length)

        # Start a video capture from the inbuilt webcam
        self.cap = cv.VideoCapture(0)

        past_time = 0
        current_time = 0
        self.circles = []
        self.current_line_segment = []
        self.line_segments = []
        self.line_segment_colors = []

        # The list of gestures in the past
        self.past_gestures = []
        self.num_past_gestures = 6

        #
        self.drawing_line_seg = False
        self.num_line_segs = 0

        #
        self.current_color = (0, 0, 0)


        # Icons drawn on the screen when open palm is shown
        self.hsv_icon = cv.imread('assets/hsv_icons/hsv_icon_color.png')
        self.mask_icon = cv.imread('assets/hsv_icons/hsv_icon_mask.png')
        self.mask_icon = 1 - (self.mask_icon / 255).astype(np.uint8)

    def interpret_frame(self, image):
        img_debug = image.copy()

        # Mirror the webcam's image
        img_debug = cv.flip(img_debug, 1)

        # Convert color and get results marking hand keypoints
        rgb = cv.cvtColor(img_debug, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # Create a set of coordinates of the keypoints
        coords = np.zeros(shape=(20, 2)).astype(np.int64)

        # Get the height, width, and # of image channels
        h, w, c = img_debug.shape

        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            # The gesture of each hand
            hand_gesture_list = []
            hand_landmarks_list = []

            # For each hand detected in the image
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Wireframe hand drawing
                self.mpDraw.draw_landmarks(img_debug, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                # Calculate location of landmarks based on screen pos
                landmark_list = self.calc_landmark_list(img_debug, hand_landmarks)

                # Normalize the landmark list and detect hand gesture
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                hand_gesture_list.append(hand_sign_id)
                hand_landmarks_list.append(landmark_list)

                self.past_gestures.append(hand_sign_id)
                if len(self.past_gestures) >= self.num_past_gestures:
                    self.past_gestures.pop(0)

                # Drawing location guide circle
                x_1, y_1 = landmark_list[4]
                x_2, y_2 = landmark_list[8]
                x_c = int(round(((x_1 + x_2) / 2), 0))
                y_c = int(round(((y_1 + y_2) / 2), 0))
                img_debug = cv.circle(img_debug, (x_c, y_c), 9, (0, 255, 255), thickness=2)
                img_debug = cv.circle(img_debug, (x_c, y_c), 3, (0, 255, 255), thickness=1)

                img_debug = self.single_hand_gesture_handler(hand_sign_id, landmark_list, img_debug, x_c, y_c)

            # If there are 2 hands in the image
            if len(hand_gesture_list) == 2:
                img_debug = self.double_hand_gesture_handler(hand_gesture_list, hand_landmarks_list, img_debug)

        # Draw all other line segments
        for i, line_segment in enumerate(self.line_segments):
            # For each point in the set of points
            for j, point in enumerate(line_segment):
                if j < len(line_segment) - 1 and len(line_segment) > 2:
                    cv.line(img_debug, point, line_segment[j+1], self.line_segment_colors[i], thickness=10)

        # Draw current line segment
        for i, point in enumerate(self.current_line_segment):
            if i < len(self.current_line_segment) - 1 and len(self.current_line_segment) > 2:
                cv.line(img_debug, point, self.current_line_segment[i + 1], self.current_color, thickness=10)

        return img_debug

    def single_hand_gesture_handler(self, hand_sign_id, landmark_list, img_debug, x_c, y_c):
        # Get image information
        image_debug = img_debug.copy()
        h, w, c = image_debug.shape

        # Palm sign
        if hand_sign_id == 0:
            x_1, y_1 = landmark_list[0]
            x_2, y_2 = landmark_list[5]
            x_3, y_3 = landmark_list[17]
            x_hand_c = int(round(((x_1 + x_2 + x_3) / 3), 0))
            y_hand_c = int(round(((y_1 + y_2 + y_3) / 3), 0))
            image_debug = cv.circle(image_debug, (x_hand_c, y_hand_c), 10, (0, 255, 0), cv.FILLED)

            b = 135
            if not (x_hand_c - b < 0 or x_hand_c + b > w or y_hand_c - b < 0 or y_hand_c + b > h):
                image_debug = cv.circle(image_debug, (x_hand_c, y_hand_c - int(b*1.5)), color=self.current_color, radius=b//2, thickness=cv.FILLED)

            # Paste image of color wheel at this location
            s = 50

            # Check if in image bounds
            if not (x_hand_c - s < 0 or x_hand_c + s > w or y_hand_c - s < 0 or y_hand_c + s > h):
                color_mask = np.zeros_like(image_debug)
                transparency_mask = np.zeros_like(image_debug)

                color_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = self.hsv_icon
                transparency_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = self.mask_icon
                image_debug[transparency_mask == 1] = color_mask[transparency_mask == 1]

        # Detect if a line segment is being drawn
        if hand_sign_id == 3 and self.past_gestures.count(3) > self.num_past_gestures - 4:
            self.drawing_line_seg = True

        # If there is currently a line segment being drawn
        if self.drawing_line_seg:
            self.current_line_segment.append((x_c, y_c))

            if self.past_gestures.count(3) <= self.num_past_gestures - 4:
                self.current_line_segment = self.current_line_segment[:len(self.current_line_segment) - 6]
                self.line_segments.append(self.current_line_segment)
                self.current_line_segment = []
                self.line_segment_colors.append(self.current_color)
                self.num_line_segs += 1
                self.drawing_line_seg = False

        return image_debug

    def double_hand_gesture_handler(self, hand_gesture_list, landmark_list, img_debug):
        image_debug = img_debug.copy()

        # If right or left hand is pointing while other is showing palm
        if (hand_gesture_list[0] == 0 and hand_gesture_list[1] == 4) \
                or (hand_gesture_list[0] == 4 and hand_gesture_list[1] == 0):

            # Check which hand is pointing
            if hand_gesture_list[0] == 4:
                pointer_index = 0
            else:
                pointer_index = 1

            # X, Y location of the pointer point
            x_point, y_point = landmark_list[pointer_index][8]

            # X, Y location of the center of the hand
            x_hand_1, y_hand_1 = landmark_list[1-pointer_index][0]
            x_hand_2, y_hand_2 = landmark_list[1-pointer_index][5]
            x_hand_3, y_hand_3 = landmark_list[1-pointer_index][17]
            x_hand = int(round(((x_hand_1 + x_hand_2 + x_hand_3) / 3), 0))
            y_hand = int(round(((y_hand_1 + y_hand_2 + y_hand_3) / 3), 0))

            image_debug = cv.line(image_debug, color=(255, 128, 0), pt1=(x_hand, y_hand), pt2=(x_point, y_point))

            d_y = x_point - x_hand
            d_x = y_point - y_hand

            d = math.sqrt(((d_x ** 2) + (d_y ** 2)))

            if d < 50:
                theta = math.atan2(d_y, d_x)
                theta = math.degrees(theta) + 180
                theta = theta - 135

                if theta < 0:
                    theta = theta + 360

                rgb = self.hsv_to_rgb(theta/360, 1, 1)
                print(rgb)

                rgb = ( int(rgb[0] * 255),
                        int(rgb[1] * 255),
                        int(rgb[2] * 255))

                self.current_color = rgb

                print(d, theta)
        return image_debug

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h * 6.)  # XXX assume int() truncates!
        f = (h * 6.) - i
        p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
        i %= 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_bounding_rect(self, use_brect, image, brect):
        # Draw the image using the bounding rectangle
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image


    def calc_landmark_list(self, image, landmarks):
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


    def pre_process_landmark(self, landmark_list):
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

    def runApp(self):

        # Main loop
        while True:
            # Read an image from the webcam
            success, img = self.cap.read()

            # If an image from the webcam comes across
            if success:

                # Interpret it
                debug_image = self.interpret_frame(img)

                cv.imshow('debug', debug_image)
                cv.waitKey(1)


if __name__ == '__main__':
    i = Interpreter()
    i.runApp()
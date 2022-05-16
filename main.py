import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import itertools
import copy
import math

# Gesture Tracking Models
from gesture_model.model import KeyPointClassifier
from google.protobuf.json_format import MessageToDict

class Interpreter():
    def __init__(self):
        # Hand detector data
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

        # Hand Keypoint Gesture Detection Models
        self.keypoint_classifier = KeyPointClassifier()
        with open('gesture_model/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        # Drawing Mode
        self.drawing_mode = 'sketch'
        self.drawing_modes = ['sketch', 'line', 'rect', 'circle']
        self.drawing_hand = 'Left'

        # Line segment drawing data
        self.drawing_sketch = False
        self.current_sketch = []
        self.sketches = []
        self.sketch_colors = []

        # Line Drawing Data
        self.drawing_shape = False
        self.current_line = []
        self.lines = []
        self.line_colors = []

        # Rectangle Drawing Data
        self.current_rect = []
        self.rects = []
        self.rect_colors = []

        # Circle Drawing Data
        self.current_circle = []
        self.circles = []
        self.circle_colors = []

        # The list of gestures in the past
        self.past_gestures = []
        self.num_past_gestures = 10

        # Color selection parameters
        self.current_color = (0, 0, 0)
        self.icon_radius = 30
        self.black_icon_pos = (0, 0)
        self.brown_icon_pos = (0, 0)
        self.white_icon_pos = (0, 0)

        # Line Type Icons
        self.select_icon_pos = (0, 0)
        self.d_select = self.icon_radius
        self.sketch_icon_pos = (0, 0)
        self.line_icon_pos = (0, 0)
        self.rect_icon_pos = (0, 0)
        self.circle_icon_pos = (0, 0)

        # Icons drawn on the screen when open palm is shown
        self.hsv_icon = cv.imread('gui_assets/hsv_icons/hsv_icon_color.png')
        self.mask_icon = cv.imread('gui_assets/hsv_icons/hsv_icon_mask.png')
        self.mask_icon[self.mask_icon > 100] = 255
        self.mask_icon[self.mask_icon < 100] = 0
        self.mask_icon = 1 - (self.mask_icon / 255).astype(np.uint8)
        self.hsv_icon_scale = 0
        self.pallate_hand = 'Right'
        self.pallate_open = False

    def interpret_frame(self, image):
        """
        Master function - detects hands and gestures from hands, then performs actions based on gestures

        :param image: The image to be pƒmprocessed
        :return: The image with the hand keypoints and interpreted drawings drawn on it.
        """

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

        # The gesture of each hand
        hand_gesture_list = []
        hand_landmarks_list = []
        handedness_list = []

        # If there are hands detected
        if results.multi_hand_landmarks:

            # For each hand detected in the image
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate location of landmarks based on screen pos
                landmark_list = self.calc_landmark_list(img_debug, hand_landmarks)
                handedness_data = MessageToDict(handedness)
                handedness_list.append(handedness_data['classification'][0]['label'])

                # Normalize the landmark list and detect hand gesture
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                try:
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                except:
                    break
                hand_gesture_list.append(hand_sign_id)
                hand_landmarks_list.append(landmark_list)

                # Append the current gesture to the list of gestures
                self.past_gestures.append(hand_sign_id)
                if len(self.past_gestures) >= self.num_past_gestures: self.past_gestures.pop(0)

                # Drawing location guide circle
                x_1, y_1 = landmark_list[4]
                x_2, y_2 = landmark_list[8]
                x_c = int(round(((x_1 + x_2) / 2), 0))
                y_c = int(round(((y_1 + y_2) / 2), 0))
                img_debug = cv.circle(img_debug, (x_c, y_c), 12, (0, 255, 255), thickness=2)
                img_debug = cv.circle(img_debug, (x_c, y_c), 3, (0, 255, 255), thickness=1)

                # Wireframe hand drawing
                self.mpDraw.draw_landmarks(img_debug, hand_landmarks, self.mpHands.HAND_CONNECTIONS)



            # If there are 2 hands in the image
            if len(hand_gesture_list) == 2:
                img_debug = self.double_hand_gesture_handler(hand_gesture_list, hand_landmarks_list, img_debug)
                if hand_gesture_list[0] != 0 and hand_gesture_list[1] != 0:
                    self.pallate_open = False
            else:
                if hand_gesture_list[0] == 0 and handedness_list[0] != self.pallate_hand:
                    self.pallate_open = False
                    if self.pallate_hand == 'Right':
                        self.pallate_hand = 'Left'
                    elif self.pallate_hand == 'Left':
                        self.pallate_hand = 'Right'
                else:
                    self.pallate_open = False
                self.d_select = self.icon_radius * 3 + 1

            # If there is no two-handed gesture detected
            for i, landmarks in enumerate(hand_landmarks_list):
                # Detect single hand gestures
                img_debug = self.single_hand_gesture_handler(hand_gesture_list[i], landmarks, img_debug, handedness_list[i])

        # Draw the collection of shapes
        img_debug = self.draw_shapes(img_debug)

        return img_debug

    def single_hand_gesture_handler(self, hand_sign_id, landmark_list, img_debug, handedness):
        """
        Interpret the gesture being made by the hand, and take the appropriate action

        :param hand_sign_id: The id of the hand sign that was detected
        :param landmark_list: a list of tuples containing the x and y coordinates of the 21 hand landmarks
        :param img_debug: The image that will be displayed to the user
        :param x_c: x coordinate of the center of the palm
        :param y_c: y coordinate of the center of the hand
        :return: The image_debug is being returned.
        """
        # Get image information
        image_debug = img_debug.copy()
        h, w, c = image_debug.shape
        x_1, y_1 = landmark_list[4]
        x_2, y_2 = landmark_list[8]
        x_c = int(round(((x_1 + x_2) / 2), 0))
        y_c = int(round(((y_1 + y_2) / 2), 0))

        # If the hand is making a forward facing palm gesture, draw GUI
        if hand_sign_id == 0:
            if not self.pallate_open:
                self.pallate_open = True
                self.pallate_hand = handedness

            if handedness == self.pallate_hand and self.pallate_open:
                # Draw a circle with the current color at the center of the palm
                x_1, y_1 = landmark_list[0]
                x_2, y_2 = landmark_list[5]
                x_3, y_3 = landmark_list[17]
                x_hand_c = int(round(((x_1 + x_2 + x_3) / 3), 0))
                y_hand_c = int(round(((y_1 + y_2 + y_3) / 3), 0))

                image_debug = cv.circle(image_debug,
                                        (int((landmark_list[5][0] + landmark_list[17][0])/2),
                                         int((landmark_list[5][1] + landmark_list[17][1])/2)),
                                        self.hsv_icon_scale//2,
                                        self.current_color,
                                        cv.FILLED)

                x_hand_c, y_hand_c = landmark_list[0]

                image_debug = cv.circle(image_debug,
                                        (x_hand_c, y_hand_c),
                                        self.icon_radius//2,
                                        self.current_color,
                                        cv.FILLED)

                # Draw Auxilliary Colors (brown, black, white)
                # White Color
                s = self.hsv_icon_scale//2
                self.white_icon_pos = (landmark_list[0][0] + s + 10, landmark_list[0][1] + s - 15)  # landmark_list[6]
                image_debug = cv.circle(image_debug,
                                        self.white_icon_pos,
                                        color=(255, 255, 255),
                                        radius=self.hsv_icon_scale//4,
                                        thickness=cv.FILLED)
                if self.current_color == (255, 255, 255):
                    image_debug = cv.circle(image_debug,
                                            self.white_icon_pos,
                                            color=(3, 186, 252),
                                            radius=self.hsv_icon_scale // 4,
                                            thickness=4)


                # Brown Color
                self.brown_icon_pos = (landmark_list[0][0] - s - 10, landmark_list[0][1] + s - 15)  # landmark_list[7]
                image_debug = cv.circle(image_debug,
                                        self.brown_icon_pos,
                                        color=(0, 51, 102),
                                        radius=self.hsv_icon_scale//4,
                                        thickness=cv.FILLED)
                if self.current_color == (0, 51, 102):
                    image_debug = cv.circle(image_debug,
                                            self.brown_icon_pos,
                                            color=(3, 186, 252),
                                            radius=self.hsv_icon_scale // 4,
                                            thickness=4)

                # Black Color

                self.black_icon_pos = (landmark_list[0][0], landmark_list[0][1] + s + 20)  # landmark_list[8]
                image_debug = cv.circle(image_debug,
                                        self.black_icon_pos,
                                        color=(0, 0, 0),
                                        radius=self.hsv_icon_scale//4,
                                        thickness=cv.FILLED)
                if self.current_color == (0, 0, 0):
                    image_debug = cv.circle(image_debug,
                                            self.black_icon_pos,
                                            color=(3, 186, 252),
                                            radius=self.hsv_icon_scale // 4,
                                            thickness=4)


                # Distance between the index finger base and pinky base
                dist_across_hand = math.sqrt((((landmark_list[5][0] - landmark_list[17][0]) ** 2) +
                                              ((landmark_list[5][1] - landmark_list[17][1]) ** 2)))
                self.hsv_icon_scale = int(dist_across_hand)

                s = self.hsv_icon_scale

                if s % 2 != 0:
                    s += 1

                hsv_icon_scaled = self.hsv_icon.copy()
                hsv_icon_scaled = cv.resize(hsv_icon_scaled, (s, s))

                mask_icon_scaled = self.mask_icon.copy()
                mask_icon_scaled = cv.resize(mask_icon_scaled, (s, s))

                s = int(s / 2)

                # Draw the HSV Color circle transparently
                if not (x_hand_c - s < 0 or x_hand_c + s > w or y_hand_c - s < 0 or y_hand_c + s > h):
                    color_mask = np.zeros_like(image_debug)
                    transparency_mask = np.zeros_like(image_debug)

                    color_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = hsv_icon_scaled
                    transparency_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = mask_icon_scaled
                    image_debug[transparency_mask == 1] = color_mask[transparency_mask == 1]

                # Draw the selector pallate
                self.select_icon_pos = landmark_list[4]
                image_debug = cv.circle(image_debug,
                                        self.select_icon_pos,
                                        color=(50, 50, 50),
                                        radius=self.icon_radius,
                                        thickness=cv.FILLED)
                if self.drawing_mode == 'sketch':
                    image_debug = cv.putText(
                        text='S',
                        color=(255, 255, 255),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        img = image_debug,
                        fontScale = 1,
                        org = (self.select_icon_pos[0] - int(0.3 * self.icon_radius),
                               self.select_icon_pos[1] + int(0.4 * self.icon_radius)),
                        thickness=5
                    )
                elif self.drawing_mode == 'line':
                    image_debug = cv.line(image_debug,
                                          pt1=(self.select_icon_pos[0] + int(self.icon_radius * 0.5),
                                               self.select_icon_pos[1] + int(self.icon_radius * 0.5)),
                                          pt2=(self.select_icon_pos[0] - int(self.icon_radius * 0.5),
                                               self.select_icon_pos[1] - int(self.icon_radius * 0.5)),
                                          color=(255, 255, 255),
                                          thickness=5)
                elif self.drawing_mode == 'rect':
                    image_debug = cv.rectangle(image_debug,
                                               pt1=(self.select_icon_pos[0] + int(self.icon_radius * 0.5),
                                                    self.select_icon_pos[1] + int(self.icon_radius * 0.5)),
                                               pt2=(self.select_icon_pos[0] - int(self.icon_radius * 0.5),
                                                    self.select_icon_pos[1] - int(self.icon_radius * 0.5)),
                                               color=(255, 255, 255),
                                               thickness=5)
                elif self.drawing_mode == 'circle':
                    image_debug = cv.circle(image_debug,
                                            self.select_icon_pos,
                                            color=(255, 255, 255),
                                            radius=int(self.icon_radius * 0.6),
                                            thickness=5)

                if self.d_select < self.icon_radius * 3:
                    # Sketch Icon
                    self.sketch_icon_pos = (landmark_list[4][0], landmark_list[4][1] + self.icon_radius*2)
                    image_debug = cv.circle(image_debug,
                                            self.sketch_icon_pos,
                                            color=(150, 150, 150),
                                            radius=self.icon_radius,
                                            thickness=cv.FILLED)
                    image_debug = cv.putText(
                        text='S',
                        color=(25, 25, 25),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        img=image_debug,
                        fontScale=1,
                        org=(self.sketch_icon_pos[0] - int(0.3 * self.icon_radius),
                             self.sketch_icon_pos[1] + int(0.4 * self.icon_radius)),
                        thickness=5
                    )
                    if self.drawing_mode == 'sketch':
                        image_debug = cv.circle(image_debug,
                                                self.sketch_icon_pos,
                                                color=(3, 186, 252),
                                                radius=self.icon_radius,
                                                thickness=4)

                    # Line icon
                    self.line_icon_pos = (landmark_list[4][0] + self.icon_radius*2, landmark_list[4][1])
                    image_debug = cv.circle(image_debug,
                                            self.line_icon_pos,
                                            color=(150, 150, 150),
                                            radius=self.icon_radius,
                                            thickness=cv.FILLED)
                    image_debug = cv.line(image_debug,
                                               pt1=(self.line_icon_pos[0] + int(self.icon_radius * 0.5),
                                                    self.line_icon_pos[1] + int(self.icon_radius * 0.5)),
                                               pt2=(self.line_icon_pos[0] - int(self.icon_radius * 0.5),
                                                    self.line_icon_pos[1] - int(self.icon_radius * 0.5)),
                                               color=(25, 25, 25),
                                               thickness=5)

                    if self.drawing_mode == 'line':
                        image_debug = cv.circle(image_debug,
                                                self.line_icon_pos,
                                                color=(3, 186, 252),
                                                radius=self.icon_radius,
                                                thickness=4)

                    # Rect Icon
                    self.rect_icon_pos = (landmark_list[4][0] - self.icon_radius*2, landmark_list[4][1])
                    image_debug = cv.circle(image_debug,
                                            self.rect_icon_pos,
                                            color=(150, 150, 150),
                                            radius=self.icon_radius,
                                            thickness=cv.FILLED)
                    image_debug = cv.rectangle(image_debug,
                                            pt1=(self.rect_icon_pos[0] + int(self.icon_radius * 0.5), self.rect_icon_pos[1] + int(self.icon_radius * 0.5)),
                                            pt2=(self.rect_icon_pos[0] - int(self.icon_radius * 0.5), self.rect_icon_pos[1] - int(self.icon_radius * 0.5)),
                                            color=(25, 25, 25),
                                            thickness=5)
                    if self.drawing_mode == 'rect':
                        image_debug = cv.circle(image_debug,
                                                self.rect_icon_pos,
                                                color=(3, 186, 252),
                                                radius=self.icon_radius,
                                                thickness=4)

                    # Circle Icon
                    self.circle_icon_pos = (landmark_list[4][0], landmark_list[4][1] - self.icon_radius*2)
                    image_debug = cv.circle(image_debug,
                                            self.circle_icon_pos,
                                            color=(150, 150, 150),
                                            radius=self.icon_radius,
                                            thickness=cv.FILLED)
                    image_debug = cv.circle(image_debug,
                                           self.circle_icon_pos,
                                           color=(25, 25, 25),
                                           radius=int(self.icon_radius * 0.6),
                                           thickness=5)
                    if self.drawing_mode == 'circle':
                        image_debug = cv.circle(image_debug,
                                                self.circle_icon_pos,
                                                color=(3, 186, 252),
                                                radius=self.icon_radius,
                                                thickness=4)

        # Detect if a line segment is starting to be drawn
        if hand_sign_id == 3:
            if not self.drawing_sketch:
                self.drawing_hand = handedness

            if self.drawing_mode == 'sketch':
                self.drawing_sketch = True

            # If there is currently a line segment being drawn
            if self.drawing_sketch and handedness == self.drawing_hand:
                # Append the point to the set of points included in this line segment
                self.current_sketch.append((x_c, y_c))

        elif self.past_gestures.count(3) <= self.num_past_gestures - int(self.num_past_gestures * 0.8):
            self.collect_line_segment()

        return image_debug

    def double_hand_gesture_handler(self, hand_gesture_list, landmark_list, img_debug):
        """
        If the right hand is pointing and the left hand is showing a palm, then draw a line between the center of the left
        hand and the pointer finger of the right hand. If the line is less than 50 pixels away from the center of the left
        hand, then calculate the angle between the two points and convert it from HSV degrees to an RGB color. If the line is more than 50
        pixels away from the center of the left hand, then check if the line is within the radius of the black, white, or
        brown auxiliary color icons

        :param hand_gesture_list: A list of the hand gestures detected for each hand
        :param landmark_list: A list of landmarks for each hand
        :param img_debug: The image that will be displayed to the user
        :return: The image_debug is being returned.
        """
        image_debug = img_debug.copy()

        # If right or left hand is pointing while other is showing palm
        #   Open the Color Selector dialog
        if ((hand_gesture_list[0] == 0 and hand_gesture_list[1] == 4)\
                or (hand_gesture_list[0] == 4 and hand_gesture_list[1] == 0))\
                and not self.drawing_shape:
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

            x_hand, y_hand = landmark_list[1-pointer_index][0]

            # Calculate the distance between the two points
            d = math.sqrt((((x_point - x_hand) ** 2) + ((y_point - y_hand) ** 2)))


            # If the point is less than 50 pixels away from the center of the hand
            if d < self.hsv_icon_scale//2:
                # Calculate the angle between the two
                theta = math.atan2((x_point - x_hand), (y_point - y_hand))
                theta = math.degrees(theta) + 180   # Normalize to 0-360 degrees
                theta = theta - 135                 # Align to HSV circle orientation
                if theta < 0: theta = theta + 360   # Handle negative numbers

                # Convert the angle from HSV to an RGB color
                rgb = self.hsv_to_rgb(theta/360, 1, 1)
                rgb = ( int(rgb[0] * 255),
                        int(rgb[1] * 255),
                        int(rgb[2] * 255))

                # Set the current color to the selected RGB value
                self.current_color = rgb

            if d < self.hsv_icon_scale:
                # Draw a line between the center of the hand and the pointer finger
                image_debug = cv.line(image_debug, color=self.current_color, pt1=(x_hand, y_hand),
                                      pt2=(x_point, y_point), thickness=12)

            # Calculate Geometric distance to black, white, and brown pixels
            d_black = math.sqrt((((x_point - self.black_icon_pos[0]) ** 2) +
                                 ((y_point - self.black_icon_pos[1]) ** 2)))
            d_white = math.sqrt((((x_point - self.white_icon_pos[0]) ** 2) +
                                 ((y_point - self.white_icon_pos[1]) ** 2)))
            d_brown = math.sqrt((((x_point - self.brown_icon_pos[0]) ** 2) +
                                 ((y_point - self.brown_icon_pos[1]) ** 2)))

            # Select the color
            if d_black < self.icon_radius:
                self.current_color = (0, 0, 0)
            if d_white < self.icon_radius:
                self.current_color = (255, 255, 255)
            if d_brown < self.icon_radius:
                self.current_color = (0, 51, 102)

            # Check if any of the auxiliary colors are being selected
            else:
                self.d_select = math.sqrt((( (x_point - self.select_icon_pos[0]) ** 2) +
                                            ((y_point - self.select_icon_pos[1]) ** 2)))
                if self.d_select < self.icon_radius * 3:

                    d_sketch = math.sqrt((((x_point - self.sketch_icon_pos[0]) ** 2) +
                                         ((y_point - self.sketch_icon_pos[1]) ** 2)))
                    d_line = math.sqrt((((x_point - self.line_icon_pos[0]) ** 2) +
                                         ((y_point - self.line_icon_pos[1]) ** 2)))
                    d_rect = math.sqrt((((x_point - self.rect_icon_pos[0]) ** 2) +
                                         ((y_point - self.rect_icon_pos[1]) ** 2)))
                    d_circle = math.sqrt((((x_point - self.circle_icon_pos[0]) ** 2) +
                                        ((y_point - self.circle_icon_pos[1]) ** 2)))

                    text_offset = 40

                    # Change the sketch type
                    if d_sketch < self.icon_radius:
                        self.drawing_mode = 'sketch'
                        image_debug = cv.putText(image_debug, 'Sketch',
                                                 (self.select_icon_pos[0] + text_offset, int(self.select_icon_pos[1] + text_offset * 1.5)),
                                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                    if d_line < self.icon_radius:
                        self.drawing_mode = 'line'
                        image_debug = cv.putText(image_debug, 'Line',
                                                 (self.select_icon_pos[0] + text_offset, int(self.select_icon_pos[1] + text_offset * 1.5)),
                                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                    if d_rect < self.icon_radius:
                        self.drawing_mode = 'rect'
                        image_debug = cv.putText(image_debug, 'Rectangle',
                                                 (self.select_icon_pos[0] + text_offset, int(self.select_icon_pos[1] + text_offset * 1.5)),
                                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                    if d_circle < self.icon_radius:
                        self.drawing_mode = 'circle'
                        image_debug = cv.putText(image_debug, 'Circle',
                                                 (self.select_icon_pos[0] + text_offset, int(self.select_icon_pos[1] + text_offset * 1.5)),
                                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)

        # Enter Shape Creation mode if both hands are pinching
        elif (hand_gesture_list[0] == 3 and hand_gesture_list[1] == 3) or self.past_gestures.count(3) > int(0.6 * self.num_past_gestures):
            # Centerpoint between thumb and forefinger (hand 1)
            x_1, y_1 = landmark_list[0][4]
            x_2, y_2 = landmark_list[0][8]
            x_c_1 = int(round(((x_1 + x_2) / 2), 0))
            y_c_1 = int(round(((y_1 + y_2) / 2), 0))

            # Centerpoint between thumb and forefinger (hand 2)
            x_1, y_1 = landmark_list[1][4]
            x_2, y_2 = landmark_list[1][8]
            x_c_2 = int(round(((x_1 + x_2) / 2), 0))
            y_c_2 = int(round(((y_1 + y_2) / 2), 0))

            # If the drawing is in line mode
            if self.drawing_mode == 'line':
                self.drawing_shape = True

                # Set the current line
                self.current_line = [(x_c_1, y_c_1), (x_c_2, y_c_2)]

            # If the drawing is in rectangle mode
            if self.drawing_mode == 'rect':

                self.drawing_shape = True

                # Set the current rectangle
                self.current_rect = [(x_c_1, y_c_1), (x_c_2, y_c_2)]

            # If the drawing is in rectangle mode
            if self.drawing_mode == 'circle':
                self.drawing_shape = True

                circle_center = ((x_c_2 + x_c_1)//2, (y_c_2 + y_c_1)//2)

                circle_radius = int(math.sqrt((x_c_2 - x_c_1)**2 + (y_c_2 - y_c_1)**2)//2)

                # Set the current rectangle
                self.current_circle = [circle_center, circle_radius]

        # Exit shape creation mode if neither hand is pinching
        elif (((hand_gesture_list[0] == 3 and hand_gesture_list[1] != 3)
              or (hand_gesture_list[0] != 3 and hand_gesture_list[1] == 3) \
              or (hand_gesture_list[0] != 3 and hand_gesture_list[1] != 3)) \
              and self.drawing_shape):
            self.drawing_shape = False
            self.collect_shape()

        # Clear the screen if both gestures are detected as reversed hands
        elif (hand_gesture_list[0] == 1 and hand_gesture_list[1] == 1):
            if self.past_gestures.count(1) == len(self.past_gestures):
                self.clear_drawing()

        return image_debug

    def collect_shape(self):
        if self.drawing_mode == 'line':
            self.lines.append(self.current_line)
            self.line_colors.append(self.current_color)
            self.current_line = []
        if self.drawing_mode == 'rect':
            self.rects.append(self.current_rect)
            self.rect_colors.append(self.current_color)
            self.current_rect = []
        if self.drawing_mode == 'circle':
            self.circles.append(self.current_circle)
            self.circle_colors.append(self.current_color)
            self.current_circle = []

    def collect_line_segment(self):
        # Reset the collection of line segments and all associated data
        self.current_sketch = self.current_sketch[:len(self.current_sketch) - 3]
        self.sketches.append(self.current_sketch)
        self.current_sketch = []
        self.sketch_colors.append(self.current_color)
        self.drawing_sketch = False

    def draw_shapes(self, img_debug):
        """
        Draw each shape stored in the set of shapes

        :param img_debug: The image to draw on
        :return: The drawn on image
        """
        image_debug = img_debug.copy()

        # Draw all other line segments
        for i, line_segment in enumerate(self.sketches):
            line_segment = line_segment[::2]

            # For each point in the set of points
            for ii, point in enumerate(line_segment):
                if ii < len(line_segment) - 1 and len(line_segment) > 2:
                    cv.line(image_debug, point, line_segment[ii + 1], self.sketch_colors[i], thickness=10, lineType=cv.LINE_AA)
                    # cv.circle(image_debug, point, color=(0, 255, 0), radius=5, thickness=10, lineType=cv.LINE_AA)

        # Draw current line segment
        for i, point in enumerate(self.current_sketch):
            if i < len(self.current_sketch) - 1 and len(self.current_sketch) > 2:
                cv.line(image_debug, point, self.current_sketch[i + 1], self.current_color, thickness=10, lineType=cv.LINE_AA)

        # Draw the current line if there is one
        if self.current_line != []:
            image_debug = cv.line(image_debug, self.current_line[0], self.current_line[1], self.current_color, thickness=10, lineType=cv.LINE_AA)

        # Draw the set of all lines
        if self.lines != []:
            for i, line in enumerate(self.lines):
                image_debug = cv.line(image_debug, line[0], line[1], self.line_colors[i], thickness=10, lineType=cv.LINE_AA)

        # Draw the set of all rects
        if self.rects != []:
            for i, rect in enumerate(self.rects):
                image_debug = cv.rectangle(image_debug, rect[0], rect[1], self.rect_colors[i], thickness=10, lineType=cv.LINE_AA)

        # Draw the current rect if there is one
        if self.current_rect != []:
            image_debug = cv.rectangle(image_debug, self.current_rect[0], self.current_rect[1], self.current_color,
                                  thickness=10, lineType=cv.LINE_AA)

        # Draw the set of all circles
        if self.circles != []:
            for i, circle in enumerate(self.circles):
                image_debug = cv.circle(image_debug, circle[0], circle[1], self.circle_colors[i],
                                        thickness=10, lineType=cv.LINE_AA)

        # Draw the current circle if there is one
        if self.current_circle != []:
            image_debug = cv.circle(image_debug, self.current_circle[0], self.current_circle[1], self.current_color,
                                       thickness=10, lineType=cv.LINE_AA)

        return image_debug

    def clear_drawing(self):
        """
        It clears all the drawing data from the interpreter
        """
        # Sketch Drawing data
        self.drawing_sketch = False
        self.current_sketch = []
        self.sketches = []
        self.sketch_colors = []

        # Line Drawing Data
        self.drawing_shape = False
        self.current_line = []
        self.lines = []
        self.line_colors = []

        # Rectangle Drawing Data
        self.current_rect = []
        self.rects = []
        self.rect_colors = []

        # Circle Drawing Data
        self.current_circle = []
        self.circles = []
        self.circle_colors = []

    def hsv_to_rgb(self, h, s, v):
        """
        "Given a hue, saturation, and value, return the corresponding red, green, and blue values."

        The first thing it does is check if the saturation is zero. If it is, then the color is a shade of gray, and the
        red, green, and blue values are all equal to the value.

        If the saturation is not zero, then the function proceeds to calculate the red, green, and blue values.

        The first step is to calculate the hue segment, which is an integer between 0 and 5.

        The hue segment tells you which color in the rainbow you're dealing with.

        The second step is to calculate the fractional part of the hue.

        The fractional part tells you how far you are between the two colors in the hue segment.

        The third step is to calculate the RGB values

        :param h: Hue, as a number between 0 and 1
        :param s: saturation
        :param v: the value of the color (0-1)
        :return: The RGB values of the color.
        """
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
        # Capture frames from the webcam
        cap = cv.VideoCapture(0)

        # Main loop
        while True:
            # Read an image from the webcam
            success, img = cap.read()

            # If an image from the webcam comes across
            if success:

                # Interpret it
                debug_image = self.interpret_frame(img)

                cv.imshow('debug', debug_image)

                keyCode = cv.waitKey(1)
                if (keyCode & 0xFF) == ord("c"):
                    self.clear_drawing()

if __name__ == '__main__':
    i = Interpreter()
    i.runApp()
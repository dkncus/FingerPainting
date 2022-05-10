import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import itertools
import copy
import math

# Gesture Tracking Models
from model import KeyPointClassifier
from model import PointHistoryClassifier

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
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        # Start a video capture from the inbuilt webcam
        self.circles = []

        # Line segment drawing data
        self.drawing_line_seg = False
        self.current_line_segment = []
        self.line_segments = []
        self.line_segment_colors = []
        self.num_line_segs = 0

        # The list of gestures in the past
        self.past_gestures = []
        self.num_past_gestures = 6

        # Color selection parameters
        self.current_color = (0, 0, 0)
        self.aux_color_offset = 100
        self.color_icon_radius = 40

        # Icons drawn on the screen when open palm is shown
        self.hsv_icon = cv.imread('assets/hsv_icons/hsv_icon_color.png')
        self.mask_icon = cv.imread('assets/hsv_icons/hsv_icon_mask.png')
        self.mask_icon[self.mask_icon > 100] = 255
        self.mask_icon[self.mask_icon < 100] = 0
        self.mask_icon = 1 - (self.mask_icon / 255).astype(np.uint8)

    def interpret_frame(self, image):
        """
        Master function - detects hands and gestures from hands, then performs actions based on gestures

        :param image: The image to be processed
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

                # Detect single hand gestures
                img_debug = self.single_hand_gesture_handler(hand_sign_id, landmark_list, img_debug, x_c, y_c)

            # If there are 2 hands in the image
            if len(hand_gesture_list) == 2:
                img_debug = self.double_hand_gesture_handler(hand_gesture_list, hand_landmarks_list, img_debug)

        img_debug = self.draw_shapes(img_debug)

        return img_debug

    def single_hand_gesture_handler(self, hand_sign_id, landmark_list, img_debug, x_c, y_c):
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

        # If the hand is making a forward facing palm gesture
        if hand_sign_id == 0:
            # Draw a circle with the current color at the center of the palm
            x_1, y_1 = landmark_list[0]
            x_2, y_2 = landmark_list[5]
            x_3, y_3 = landmark_list[17]
            x_hand_c = int(round(((x_1 + x_2 + x_3) / 3), 0))
            y_hand_c = int(round(((y_1 + y_2 + y_3) / 3), 0))
            image_debug = cv.circle(image_debug,
                                    (x_hand_c, y_hand_c),
                                    self.color_icon_radius//2,
                                    self.current_color,
                                    cv.FILLED)

            # Draw Auxilliary Colors (brown, black, white)
            if not (y_hand_c - self.aux_color_offset< 0 or y_hand_c + self.aux_color_offset > h):
                if not (x_hand_c - self.aux_color_offset < 0 or x_hand_c + self.aux_color_offset > w):
                    # White Color
                    image_debug = cv.circle(image_debug,
                                            (x_hand_c, y_hand_c - self.aux_color_offset),
                                            color=(255, 255, 255),
                                            radius=self.color_icon_radius,
                                            thickness=cv.FILLED)

                    # Brown Color
                    image_debug = cv.circle(image_debug,
                                            (x_hand_c + self.aux_color_offset, y_hand_c - self.aux_color_offset),
                                            color=(0, 51, 102),
                                            radius=self.color_icon_radius,
                                            thickness=cv.FILLED)

                    # Black Color
                    image_debug = cv.circle(image_debug,
                                            (x_hand_c - self.aux_color_offset, y_hand_c - self.aux_color_offset),
                                            color=(0, 0, 0),
                                            radius=self.color_icon_radius,
                                            thickness=cv.FILLED)

            # Paste image of color wheel at this location
            s = 50

            # Check if in image bounds
            if not (x_hand_c - s < 0 or x_hand_c + s > w or y_hand_c - s < 0 or y_hand_c + s > h):
                color_mask = np.zeros_like(image_debug)
                transparency_mask = np.zeros_like(image_debug)

                color_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = self.hsv_icon
                transparency_mask[y_hand_c - s:y_hand_c + s, x_hand_c - s:x_hand_c + s, :] = self.mask_icon
                image_debug[transparency_mask == 1] = color_mask[transparency_mask == 1]

        # Detect if a line segment is starting to be drawn
        if hand_sign_id == 3 and self.past_gestures.count(3) > self.num_past_gestures - 4:
            self.drawing_line_seg = True

        # If there is currently a line segment being drawn
        if self.drawing_line_seg:
            # Append the point to the set of points included in this line segment
            self.current_line_segment.append((x_c, y_c))

            # If the line segment is finished being drawn
            if self.past_gestures.count(3) <= self.num_past_gestures - 4:
                # Reset the collection of line segments and all associated data
                self.current_line_segment = self.current_line_segment[:len(self.current_line_segment) - 6]
                self.line_segments.append(self.current_line_segment)
                self.current_line_segment = []
                self.line_segment_colors.append(self.current_color)
                self.num_line_segs += 1
                self.drawing_line_seg = False

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

            # Draw a line between the center of the hand and the pointer finger
            image_debug = cv.line(image_debug, color=self.current_color, pt1=(x_hand, y_hand), pt2=(x_point, y_point), thickness=12)

            # Calculate the distance between the two points
            d = math.sqrt((((x_point - x_hand) ** 2) + ((y_point - y_hand) ** 2)))

            # If the point is less than 50 pixels away from the center of the hand
            if d < 50:
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

            # Check if any of the auxiliary colors are being selected
            else:
                # Calculate Geometric distance to black, white, and brown pixels
                d_black = math.sqrt((((x_point - x_hand + self.aux_color_offset) ** 2) +
                                     ((y_point - y_hand + self.aux_color_offset) ** 2)))
                d_white = math.sqrt((((x_point - x_hand) ** 2) +
                                     ((y_point - y_hand + self.aux_color_offset) ** 2)))
                d_brown = math.sqrt((((x_point - x_hand - self.aux_color_offset) ** 2) +
                                     ((y_point - y_hand + self.aux_color_offset) ** 2)))

                # Select the color
                if d_black < self.color_icon_radius:
                    self.current_color = (0, 0, 0)
                if d_white < self.color_icon_radius:
                    self.current_color = (255, 255, 255)
                if d_brown < self.color_icon_radius:
                    self.current_color = (0, 51, 102)

        return image_debug

    def draw_shapes(self, img_debug):
        """
        Draw each shape stored in the set of shapes

        :param img_debug: The image to draw on
        :return: The drawn on image
        """
        image_debug = img_debug.copy()

        # Draw all other line segments
        for i, line_segment in enumerate(self.line_segments):
            # For each point in the set of points
            for j, point in enumerate(line_segment):

                if j < len(line_segment) - 1 and len(line_segment) > 2:
                    cv.line(image_debug, point, line_segment[j + 1], self.line_segment_colors[i], thickness=10)

        # Draw current line segment
        for i, point in enumerate(self.current_line_segment):
            if i < len(self.current_line_segment) - 1 and len(self.current_line_segment) > 2:
                cv.line(image_debug, point, self.current_line_segment[i + 1], self.current_color, thickness=10)

        return image_debug

    def clear_drawing(self):
        self.line_segment_colors = []
        self.line_segments = []
        self.drawing_line_seg = False
        self.current_line_segment = []
        self.num_line_segs = 0

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
                    self.line_segment_colors = []
                    self.line_segments = []
                    self.drawing_line_seg = False
                    self.current_line_segment = []
                    self.num_line_segs = 0

if __name__ == '__main__':
    i = Interpreter()
    i.runApp()
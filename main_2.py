import cv2 as cv
import mediapipe as mp
import numpy as np
import copy
import math
from math import sqrt

# Gesture Tracking Models
from google.protobuf.json_format import MessageToDict
from gesture_model.model import KeyPointClassifier

# Calculate the geometric distance between two points
dist = lambda pt1, pt2: float(sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))

class HandInterpreter():
    def __init__(self):
        # Hand detector data
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        self.prev_frame_time = 0

        # The landmarks associated with each hand
        self.hand_landmarks = []
        self.handedness = []
        self.hand_landmarks_screen = []

    def interpret_frame(self, image):
        """
        Master function - detects hands and gestures from hands, then performs actions based on gestures

        :param image: The image to be pƒmprocessed
        :return: The image with the hand keypoints and interpreted drawings drawn on it.
        """

        img_debug = image.copy()


        # Convert color and get results marking hand keypoints
        rgb = cv.cvtColor(img_debug, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # The gesture of each hand
        self.hand_landmarks_screen = []
        self.hand_landmarks = []
        self.handedness = []

        # If there are hands detected
        if results.multi_hand_landmarks:

            # For each hand detected in the image
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Append the raw hand landmarks data
                self.hand_landmarks.append(hand_landmarks)

                # Calculate location of landmarks based on screen pos
                landmark_list = self.calc_landmark_list(img_debug, hand_landmarks)
                self.hand_landmarks_screen.append(landmark_list)

                # Parse handedness data (right or left
                handedness_data = MessageToDict(handedness)
                self.handedness.append(handedness_data['classification'][0]['label'])

        return self.hand_landmarks, self.hand_landmarks_screen, self.handedness

    def calc_landmark_list(self, image, landmarks):
        """
        It takes in an image and a list of landmarks, and returns a list of landmark points

        :param image: The image to be processed
        :param landmarks: The landmarks for the image
        :return: A list of lists of x,y coordinates.
        """
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_points = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            # landmark_z = landmark.z
            landmark_points.append([landmark_x, landmark_y])

        landmark_points = np.array(landmark_points)

        return landmark_points

class GestureInterpreter():
    def __init__(self):
        # Classification model
        self.keypoint_classifier = KeyPointClassifier()

        # Set of gestures possible to differentiate
        self.gestures = [
            "OpenForward",
            "OpenReverse",
            "Closed",
            "Pinch",
            "Point",
            "ThumbsUp",
            "ThumbsDown",
            "FuckYou",
        ]

        # Detected gesture on the left and right hand
        self.gesture_index_right = -1
        self.gesture_index_left = -1

        # True if hand is visible in current frame, false if not
        self.hand_visible_right = False
        self.hand_visible_left = False

        # List of past hand gestures
        self.num_past_gestures = 10
        self.past_gestures_right = []
        self.past_gestures_left = []

    def update_gestures(self, hands):
        # If the right hand is visible
        if 'Right' in hands.handedness:
            # Set Hand Visibility and Index
            self.hand_visible_right = True
            hand_index = hands.handedness.index('Right')

            # Detect a gesture from landmarks
            lm = copy.deepcopy(hands.hand_landmarks_screen)
            self.gesture_index_right = self.detect_gesture_from_landmarks(lm, hand_index)

            # Add it to the list of past gestures
            self.past_gestures_right.append(self.gesture_index_right)
            if len(self.past_gestures_right) >= self.num_past_gestures: self.past_gestures_right.pop(0)
        else:
            # Unset hand visibility and set classification to error value
            self.hand_visible_right = False
            self.gesture_index_right = -1

        # If the left hand is visible
        if 'Left' in hands.handedness:
            # Set Hand Visibility and Index
            self.hand_visible_left = True
            hand_index = hands.handedness.index('Left')

            # Detect a gesture from landmarks
            lm = copy.deepcopy(hands.hand_landmarks_screen)
            self.gesture_index_left = self.detect_gesture_from_landmarks(lm, hand_index)

            # Add it to the list of past gestures
            self.past_gestures_left.append(self.gesture_index_left)
            if len(self.past_gestures_left) >= self.num_past_gestures: self.past_gestures_left.pop(0)
        else:
            # Unset hand visibility and set classification to error value
            self.hand_visible_left = False
            self.gesture_index_left = -1

    def detect_gesture_from_landmarks(self, lm, hand_index):
        # Normalize the landmarks
        base_x, base_y = lm[hand_index][0][0], lm[hand_index][0][1]
        lm[hand_index][:, 0] = lm[hand_index][:, 0] - base_x
        lm[hand_index][:, 1] = lm[hand_index][:, 1] - base_y
        normalized_landmarks = lm[hand_index] / np.amax(np.abs(lm[hand_index]))

        # Convert the array to a 1d array to pass as a tensor to gesture model
        normalized_landmarks = np.ndarray.flatten(normalized_landmarks)
        normalized_landmarks = np.ndarray.tolist(normalized_landmarks)

        # Collect the gesture made
        hand_sign_id = self.keypoint_classifier(normalized_landmarks)

        return hand_sign_id

class DrawingUserInterface():
    def __init__(self):
        # Drawing location of guide circles
        self.guide_circle_location_left = (-1, -1)
        self.guide_circle_location_right = (-1, -1)
        
        # Palette Data
        self.palette_open = False
        self.palette_hand = 'None'
        self.palette_location = (-1, -1)
        self.current_color = (0, 0, 0)
        self.icon_radius = 30

        # Shape Type Icon
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

        # Position of the pointer finger in double-hand specific gestures
        self.pointer_pos = (-1, -1)

    # Update the positions of user interface items
    def update_ui(self, hands, gestures):
        # For each set of landmarks detected
        if not gestures.hand_visible_right:
            # Update the location of the drawing guide circle
            self.guide_circle_location_right = (-1, -1)

        # For each set of landmarks detected
        if not gestures.hand_visible_left:
            # Update the location of the drawing guide circle
            self.guide_circle_location_left = (-1, -1)

        # For each hand in the set of hand landmarks
        for i, lm in enumerate(hands.hand_landmarks_screen):
            # Check if there is a right hand
            if 'Right' in hands.handedness:

                # Get the index of the right hand
                hand_index = hands.handedness.index('Right')

                # If the searched for index is the current index
                if hand_index == i:
                    self.collect_ui('Right', gestures.gesture_index_right, lm)

            # Check if there is a right hand
            if 'Left' in hands.handedness:

                # Get the index of the right hand
                hand_index = hands.handedness.index('Left')

                # If the searched for index is the current index
                if hand_index == i:
                    self.collect_ui('Left', gestures.gesture_index_left, lm)

        # If there are two hands in the frame
        if len(hands.hand_landmarks) == 2:
            self.update_selected_color(gestures, hands)
        else:
            self.pointer_pos = (-1, -1)

    # Collect user interface element positions based on the hand gesture
    def collect_ui(self, hand, gesture_index, lm):
        # Update the location of the drawing guide circle
        if hand == 'Right':
            self.guide_circle_location_right = self.get_guide_circle_location(lm)
        if hand == 'Left':
            self.guide_circle_location_left = self.get_guide_circle_location(lm)

        # If the gesture detected is a forward-facing palm
        if gesture_index == 0:
            # Update the positions of palette UI elements
            self.update_palette(lm)

            # Set the current hand the palate should display on
            self.palette_hand = hand
            self.palette_open = True

        # Otherwise, unset the palette hand and trigger the open boolean
        else:
            self.palette_hand = 'None'
            self.palette_open = False

    # Get the location of the guide circle based on hand landmarks
    def get_guide_circle_location(self, hand_landmarks):
        # Drawing location guide circle
        x_1, y_1 = hand_landmarks[4]
        x_2, y_2 = hand_landmarks[8]
        x_c = int(round(((x_1 + x_2) / 2), 0))
        y_c = int(round(((y_1 + y_2) / 2), 0))
        return (x_c, y_c)

    # Update the positions of data on the palette
    def update_palette(self, landmarks):
        # Calculate the distance across the hand in order to determine the shape of the ring
        dist_across_hand = int(dist(landmarks[5], landmarks[17]))
        self.hsv_icon_scale = int(dist_across_hand)

        # Compute the scale of the objects
        s = self.hsv_icon_scale // 2

        # Drawn location of the color palette
        x_1, y_1 = landmarks[9]
        x_2, y_2 = landmarks[13]
        x_hand_c = int(round(((x_1 + x_2) / 2), 0))
        y_hand_c = int(round(((y_1 + y_2) / 2), 0))
        self.palette_location = (x_hand_c, y_hand_c)

        # Position of the mode selector (attached to the thumb)
        self.mode_select_icon_pos = (landmarks[4][0], landmarks[4][1])

        # Get the location of the color preview
        self.color_preview_pos = (x_hand_c, y_hand_c)

    # Update the selected color based on gestures and hand positions
    def update_selected_color(self, gestures, hands):
        # If one hand is palm facing while the other is pointing
        if (gestures.gesture_index_left == 0 and gestures.gesture_index_right == 4) or \
            (gestures.gesture_index_left == 4 and gestures.gesture_index_right == 0):

            # Get the X, Y location of the pointer
            self.pointer_pos = hands.hand_landmarks_screen[0][8]
            self.pointer_pos = (self.pointer_pos[0], self.pointer_pos[1])
            self.hand_pos = hands.hand_landmarks_screen[1][0]
            self.hand_pos = (self.hand_pos[0], self.hand_pos[1])
            self.hand_pos = self.palette_location

            # Calculate the distance between the two points
            d = dist(self.pointer_pos, self.hand_pos)

            # Calculate the angle between the two
            theta = math.atan2((self.pointer_pos[0] - self.hand_pos[0]),
                               (self.pointer_pos[1] - self.hand_pos[1]))
            theta = math.degrees(theta) + 180  # Normalize to 0-360 degrees

            # If the point is less than 50 pixels away from the center of the hand
            if d < self.hsv_icon_scale * 2 + 20 and d > self.hsv_icon_scale + 20:

                # Check the gesture index
                if gestures.gesture_index_right == 0:

                    # If the HSV color was between a range of values
                    if theta > 30 and theta < 196:
                        # Align to the orientation of the color ring
                        color_theta = 2.95 * theta
                        color_theta = color_theta - 180
                        if color_theta < 0: color_theta = color_theta + 360  # Handle negative numbers

                        # Convert the angle from HSV to an RGB color
                        rgb = self.hsv_to_rgb(color_theta / 360, 1, 1)
                        rgb = (int(rgb[0] * 255),
                               int(rgb[1] * 255),
                               int(rgb[2] * 255))

                        # Set the current color to the selected RGB value
                        self.current_color = rgb

                        # If the pointer is close to the brown circle
                        if 148 < theta < 160: self.current_color = (0, 50, 100)

                        # If the pointer is close to the black circle
                        if 160 < theta < 172: self.current_color = (0, 0, 0)

                        # If the pointer is close to the white circle
                        if 172 < theta < 184: self.current_color = (100, 100, 100)

                        # If the pointer is close to the brown circle
                        if 184 < theta < 196: self.current_color = (255, 255, 255)

                    # Otherwise, make sure no lines are drawn
                    else:
                        self.pointer_pos = (-1, -1)

                # Check the gesture index
                if gestures.gesture_index_left == 0:
                    # If the HSV color was between a range of values
                    if theta > 168 and theta < 330:
                        # Align to the orientation of the color ring
                        color_theta = -3 * theta
                        color_theta += 180
                        while not (0 < color_theta < 360):
                            color_theta = color_theta + 360

                        # Convert the angle from HSV to an RGB color
                        rgb = self.hsv_to_rgb(color_theta / 360, 1, 1)
                        rgb = (int(rgb[0] * 255),
                               int(rgb[1] * 255),
                               int(rgb[2] * 255))

                        # Set the current color to the selected RGB value
                        self.current_color = rgb

                        # If the pointer is close to the brown circle
                        if 200 < theta < 210: self.current_color = (0, 50, 100)

                        # If the pointer is close to the black circle
                        if 188 < theta < 200: self.current_color = (0, 0, 0)

                        # If the pointer is close to the white circle
                        if 176 < theta < 188: self.current_color = (100, 100, 100)

                        # If the pointer is close to the brown circle
                        if 164 < theta < 176: self.current_color = (255, 255, 255)

            # Otherwise, make sure no lines are drawn
            else:
                self.pointer_pos = (-1, -1)

    # Convert HSV color to RGB color
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

    # Draw the User Interface on an input image
    def draw_ui(self, image, hands, gestures):
        img = image.copy()
        h, w, c = img.shape

        # If the right hand is within the frame
        if self.guide_circle_location_right != (-1, -1):
            img = cv.circle(img, self.guide_circle_location_right, 12, (255, 255, 255), thickness=2)
            img = cv.circle(img, self.guide_circle_location_right, 4, self.current_color, thickness=3)

        # If the left hand is within the frame
        if self.guide_circle_location_left != (-1, -1):
            img = cv.circle(img, self.guide_circle_location_left, 12, (255, 255, 255), thickness=2)
            img = cv.circle(img, self.guide_circle_location_left, 4, self.current_color, thickness=3)

        # Check if the palette is open
        if self.palette_open:
            ### Draw the color preview circle ###
            img = cv.circle(img,
                            self.color_preview_pos,
                            self.hsv_icon_scale // 2,
                            self.current_color,
                            cv.FILLED)

            ### Draw the HSV Icon ###
            s = self.hsv_icon_scale

            # If the scale is non-even, increment it by 1 pixel
            if s % 2 != 0:
                s += 1

            s *= 4

            # Resize the image and transparency masks to match scale dimensions
            hsv_icon_scaled = self.hsv_icon.copy()
            hsv_icon_scaled = cv.resize(hsv_icon_scaled, (s, s))
            mask_icon_scaled = self.mask_icon.copy()
            mask_icon_scaled = cv.resize(mask_icon_scaled, (s, s))

            # Flip the icon for the hand
            if gestures.gesture_index_left == 0:
                hsv_icon_scaled = cv.flip(hsv_icon_scaled, 1)
                mask_icon_scaled = cv.flip(mask_icon_scaled, 1)

            # Divide the scale by 2
            s = int(s / 2)
            
            # Get the location the palette should be drawn at
            x, y = self.palette_location

            # If the image will lie in bounds
            if not (x - s < 0 or x + s > w or y - s < 0 or y + s > h):
                # Create an empty color mask and transparency mask
                color_mask = np.zeros_like(img)
                transparency_mask = np.zeros_like(img)

                color_mask[y - s:y + s, x - s:x + s, :] = hsv_icon_scaled
                transparency_mask[y - s:y + s, x - s:x + s, :] = mask_icon_scaled
                img[transparency_mask == 1] = color_mask[transparency_mask == 1]
            else:
                # Make a copy of the hsv icon to modify
                hsv_icon_cropped = hsv_icon_scaled.copy()
                alpha_cropped = mask_icon_scaled.copy()

                # icon width, height, channels
                i_h, i_w, i_c = hsv_icon_cropped.shape
                x_left, x_right, y_top, y_base = x - s, x + s, y - s, y + s

                # If the paste would be out of bounds in the x range
                if ((x < s)):
                    hsv_icon_cropped = hsv_icon_cropped[:, s - x:i_w, :]
                    alpha_cropped = alpha_cropped[:, s - x:i_w, :]
                    x_left = 0
                elif ((x > (w - s))):
                    hsv_icon_cropped = hsv_icon_cropped[:, 0:i_w - ((x + s) - w), :]
                    alpha_cropped = alpha_cropped[:, 0:i_w - ((x + s) - w), :]
                    x_right = w

                # If the paste would be out of bounds in the y range
                if ((y < s)):
                    # Crop the color and alpha masks accordingly
                    hsv_icon_cropped = hsv_icon_cropped[s - y:i_h, :, :]
                    alpha_cropped = alpha_cropped[s - y:i_h, :, :]
                    y_top = 0
                elif ((y > (h - s))):
                    # Crop the color and alpha masks accordingly
                    hsv_icon_cropped = hsv_icon_cropped[0:i_h - ((y + s) - h), :, :]
                    alpha_cropped = alpha_cropped[0:i_h - ((y + s) - h), :, :]
                    y_base = h

                # Color masks
                color_mask = np.zeros_like(img)
                alpha_mask = np.zeros_like(img)

                # Apply to alpha
                color_mask[y_top:y_base, x_left:x_right, :] = hsv_icon_cropped
                alpha_mask[y_top:y_base, x_left:x_right, :] = alpha_cropped
                img[alpha_mask == 1] = color_mask[alpha_mask == 1]

        if self.pointer_pos != (-1, -1):
            img = cv.line(img, color=self.current_color, pt1=self.pointer_pos, pt2=self.hand_pos, thickness=28)

        # For each landmark in the set of hand landmarks
        for lm in hands.hand_landmarks:
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, lm, mpHands.HAND_CONNECTIONS)

        return img

class ShapeCollector():
    def __init__(self):
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

    def update(self, gestures, ):
        pass

if __name__ == '__main__':
    hands = HandInterpreter()
    gestures = GestureInterpreter()
    user_interface = DrawingUserInterface()
    shape_collector = ShapeCollector()

    # Capture frames from the webcam
    cap = cv.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    # Main loop
    while True:
        # Read an image from the webcam
        success, frame = cap.read()

        # Image on which the UI is drawn
        debug_image = frame.copy()

        # If an image from the webcam comes across
        if success:
            # Interpret it
            hand_landmarks, hand_landmarks_screen, handedness = hands.interpret_frame(frame)

            # If the number of hands detected is more than 0
            if len(hand_landmarks) > 0 and len(hand_landmarks) <=2:
                # Update hand gestures
                gestures.update_gestures(hands)

                # Update the user interface based on hand landmarks and gestures
                user_interface.update_ui(hands, gestures)
                debug_image = user_interface.draw_ui(debug_image, hands, gestures)

        debug_image = cv.flip(debug_image, 1)

        cv.imshow('debug', debug_image)
        cv.waitKey(1)
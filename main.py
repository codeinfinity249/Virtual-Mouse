import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get the screen width and height for scaling
screen_width, screen_height = pyautogui.size()

# Open a video capture stream (webcam)
cap = cv2.VideoCapture(0)

# Variables to track gestures and actions
prev_scroll_time = 0
prev_zoom_time = 0
scroll_active = False
zoom_active = False
prev_hand_position = None

# Constants for gesture sensitivity
PINCH_THRESHOLD = 40  # Threshold for detecting a pinch gesture
SCROLL_SENSITIVITY = 50  # Sensitivity for scrolling
ZOOM_THRESHOLD = 200  # Threshold distance between two index fingers for zooming

def calculate_distance(point1, point2, img_width, img_height):
    """Calculate Euclidean distance between two normalized points."""
    x1, y1 = int(point1.x * img_width), int(point1.y * img_height)
    x2, y2 = int(point2.x * img_width), int(point2.y * img_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_hand_position(hand_landmarks, img_width, img_height):
    """Get the average position of the hand."""
    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * img_width)
    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * img_height)
    return x, y

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirrored view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result = hands.process(img_rgb)  # Process the frame to detect hands

    # Get image dimensions
    img_height, img_width, _ = img.shape

    # If hands are detected
    if result.multi_hand_landmarks:
        hand_positions = []
        pinch_detected = False

        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark points for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance for pinch detection
            thumb_index_dist = calculate_distance(thumb_tip, index_tip, img_width, img_height)

            # Detect pinch for scrolling
            if thumb_index_dist < PINCH_THRESHOLD:
                pinch_detected = True
                scroll_active = True
                hand_position = get_hand_position(hand_landmarks, img_width, img_height)
                if prev_hand_position is not None:
                    y_movement = hand_position[1] - prev_hand_position[1]
                    if abs(y_movement) > SCROLL_SENSITIVITY:
                        pyautogui.scroll(-y_movement // SCROLL_SENSITIVITY)
                prev_hand_position = hand_position
            else:
                scroll_active = False

            # Store the hand position for zoom detection
            hand_positions.append(hand_landmarks)

        # Zoom gesture using the distance between two index fingers of two hands
        if len(hand_positions) == 2:
            index_tip_1 = hand_positions[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip_2 = hand_positions[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between the two index fingers
            index_finger_dist = calculate_distance(index_tip_1, index_tip_2, img_width, img_height)

            if abs(index_finger_dist - ZOOM_THRESHOLD) > ZOOM_THRESHOLD / 2:
                zoom_active = True
                if index_finger_dist > ZOOM_THRESHOLD:
                    pyautogui.hotkey('command', '+')
                else:
                    pyautogui.hotkey('command', '-')
            else:
                zoom_active = False

    # Display the frame
    cv2.imshow("AI Virtual Mouse", img)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

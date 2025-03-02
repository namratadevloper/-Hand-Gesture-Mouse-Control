import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

click_threshold = 30  # Distance threshold for clicking
right_click_threshold = 30  # Distance threshold for right-click

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get fingertip positions
            index_tip = hand_landmarks.landmark[8]  # Index finger tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
            middle_tip = hand_landmarks.landmark[12]  # Middle finger tip

            # Convert to screen coordinates
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

            screen_x, screen_y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)  # Move mouse cursor

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate distances
            thumb_index_distance = math.hypot(tx - ix, ty - iy)  # Thumb-Index for Left Click
            thumb_middle_distance = math.hypot(tx - mx, ty - my)  # Thumb-Middle for Right Click

            # Left Click (Thumb + Index Tap)
            if thumb_index_distance < click_threshold:
                pyautogui.click()
                print("Left Click")

            # Right Click (Thumb + Middle Tap)
            if thumb_middle_distance < right_click_threshold:
                pyautogui.rightClick()
                print("Right Click")

    # Display video feed
    cv2.imshow("Hand Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

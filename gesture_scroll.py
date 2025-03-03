import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start Video Capture
cap = cv2.VideoCapture(0)

# Adjust scroll speed (increase for faster scrolling)
SCROLL_SPEED = 20 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get Index and Middle Finger Tip & MCP (base joint)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Convert to pixel coordinates
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            ibx, iby = int(index_base.x * w), int(index_base.y * h)
            mbx, mby = int(middle_base.x * w), int(middle_base.y * h)

            # Draw landmarks on hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Debugging: Print finger positions
            print(f"Index Tip: ({ix}, {iy}), Base: ({ibx}, {iby})")
            print(f"Middle Tip: ({mx}, {my}), Base: ({mbx}, {mby})")

            # Check if fingers are pointing UP or DOWN
            if iy < iby and my < mby:  # Fingers are above base -> pointing UP
                print("Scrolling UP")
                pyautogui.scroll(SCROLL_SPEED)  # Scroll Up
            elif iy > iby and my > mby:  # Fingers are below base -> pointing DOWN
                print("Scrolling DOWN")
                pyautogui.scroll(-SCROLL_SPEED)  # Scroll Down

    # Display the frame
    cv2.imshow("Hand Gesture Scrolling", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

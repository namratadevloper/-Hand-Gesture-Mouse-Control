import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy

##########################
wCam, hCam = 700, 700
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Use default webcam
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size()

def fingers_up(lm_list):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # Thumb
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for id in range(1, 5):
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def find_position(image, results):
    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
    return lm_list

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    lm_list = find_position(img, results)
    
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        fingers = fingers_up(lm_list)
        
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # Moving Mode (Only Index Finger Up)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(clocX, clocY)

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        # Clicking Mode (Index and Middle Fingers Up)
        if fingers[1] == 1 and fingers[2] == 1:
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    
    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Use default webcam
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size()

def fingers_up(lm_list):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # Thumb
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for id in range(1, 5):
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def find_position(image, results):
    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
    return lm_list

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    lm_list = find_position(img, results)
    draw_landmarks(img, results)
    
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        fingers = fingers_up(lm_list)
        
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # Moving Mode (Only Index Finger Up)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        # Clicking Mode (Index and Middle Fingers Up)
        if fingers[1] == 1 and fingers[2] == 1:
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    
    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

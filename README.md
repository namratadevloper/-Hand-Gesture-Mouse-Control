## **🖐 Hand Gesture Mouse Control**  

This project allows you to control your **mouse cursor** and perform **click actions** using hand gestures and a webcam. It uses **OpenCV**, **MediaPipe**, and **PyAutoGUI** for real-time hand tracking.  

---

## **📌 Features**  
✅ Move the **cursor** using the index finger ✨  
✅ Perform **left-click** using a **thumb tap gesture** 🖱  
✅ Perform **right-click** using a specific gesture (e.g., thumb & index finger tap) 🔥  
✅ Smooth and real-time tracking 🎥  

---

## **🛠 Setup & Installation**  

### **1️⃣ Install Python** 
Ensure you have **Python 3.8.10** installed.  
Check if Python is installed:  
```bash
python --version
```
If not installed, download it from: [Python Official Website](https://www.python.org/downloads/)  

---

### **2️⃣ Install Required Libraries**  
Run the following command in the terminal or command prompt:  
```bash
pip install opencv-python mediapipe pyautogui
```

🔹 **Library Explanation**:
- `opencv-python` → Captures video from the webcam.
- `mediapipe` → Detects and tracks hand landmarks.
- `pyautogui` → Simulates mouse movements and clicks.

---

### **3️⃣ Run the Program**  
After installation, run the script:  
```bash
python hand_gesture_mouse.py
```
(Replace `hand_gesture_mouse.py` with your actual script filename)

---

## **💡 How It Works**
- **Move Cursor** → Move your **index finger** in front of the camera  
- **Left Click** → Tap **thumb and index finger** together  
- **Right Click** → Tap **thumb and middle finger** together  
- **Drag & Drop** (optional) → Hold fingers together and move  
- **Exit Program** → Press **"q"** on the keyboard  

---

## **⚠️ Troubleshooting**
🔹 **Error: `ModuleNotFoundError: No module named 'cv2'`**  
✔️ Run: `pip install opencv-python`  

🔹 **Error: `ModuleNotFoundError: No module named 'mediapipe'`**  
✔️ Run: `pip install mediapipe`  

🔹 **Mouse not moving?**  
✔️ Ensure proper **lighting conditions**  
✔️ Keep your hand **steady** in front of the camera  

🔹 **Mac/Linux Permission Issues?**  
✔️ **Mac Users**: Enable accessibility permissions for **Python** in:  
➡️ `System Preferences > Security & Privacy > Accessibility`  

✔️ **Linux Users**: Run this before executing:  
```bash
xhost +
```


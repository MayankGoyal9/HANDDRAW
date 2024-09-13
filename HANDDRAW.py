import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

canvas = None
drawing = False
prev_x, prev_y = None, None

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
current_color = colors[0]

def check_color_click(x, y):
    global current_color
    palette_x_start = frame.shape[1] - 50
    for i, color in enumerate(colors):
        if palette_x_start <= x <= frame.shape[1]:
            if 10 + i * 60 <= y <= 60 + i * 60:
                current_color = color

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.ones_like(frame) * 255
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])
            if drawing:
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    frame_with_canvas = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    for i, color in enumerate(colors):
        cv2.rectangle(frame_with_canvas, (frame.shape[1] - 50, 10 + i * 60), (frame.shape[1], 60 + i * 60), color, -1)
    cv2.rectangle(frame_with_canvas, (frame.shape[1] - 100, 400), (frame.shape[1] - 50, 450), current_color, -1)
    cv2.imshow("Whiteboard", frame_with_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = True
    elif key == ord('s'):
        drawing = False
    elif key == ord('c'):
        canvas = np.ones_like(frame) * 255

    if cv2.getWindowProperty("Whiteboard", cv2.WND_PROP_VISIBLE) < 1:
        break

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            check_color_click(x, y)

    cv2.setMouseCallback("Whiteboard", mouse_click)

cap.release()
cv2.destroyAllWindows()

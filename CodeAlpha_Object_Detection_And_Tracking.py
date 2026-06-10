import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

window_name = "YOLOv8 Object Tracking"

# Button coordinates
button_x1, button_y1 = 540, 10
button_x2, button_y2 = 630, 50

running = True

def mouse_callback(event, x, y, flags, param):
    global running

    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
            running = False

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while running:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Draw close button
    cv2.rectangle(
        annotated_frame,
        (button_x1, button_y1),
        (button_x2, button_y2),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        annotated_frame,
        "CLOSE",
        (555, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(window_name, annotated_frame)

    # Press Q to quit as backup
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("./models/delivery_vehicles/delivery_vehicles.pt")

# 0 - iphone, 1 - webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    success, img = cap.read()

    # Perform inference
    results = model.track(img, persist=True, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")

def detect_video(video_path, window_name="Detected Video"):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        results = model(frame)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box)
                label = model.model.names[int(cls)]
                if label.lower() != "player":
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Broadcast Video...")
    detect_video("videos/broadcast.mp4", "Broadcast Detection")
    print("Tacticam Video...")
    detect_video("videos/tacticam.mp4", "Tacticam Detection")

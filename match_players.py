from ultralytics import YOLO
import cv2
from utils import get_dominant_color
from scipy.spatial.distance import cdist
import numpy as np

model = YOLO("model/best.pt")

def extract_features(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    player_colors = []

    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                label = model.model.names[int(cls)]
                if label.lower() != "player":
                    continue

                x1, y1, x2, y2 = map(int, box)
                player_crop = frame[y1:y2, x1:x2]
                dominant_color = get_dominant_color(player_crop)
                player_colors.append(dominant_color)

        frame_count += 1

    cap.release()
    return np.array(player_colors)

# Step 1: Extract player dominant colors
broadcast_colors = extract_features("videos/broadcast.mp4")
tacticam_colors = extract_features("videos/tacticam.mp4")

print("Broadcast Colors Extracted:", len(broadcast_colors))
print("Tacticam Colors Extracted:", len(tacticam_colors))

# Step 2: Color Matching
distances = cdist(tacticam_colors, broadcast_colors)
mapping_indices = np.argmin(distances, axis=1)

print("\nPlayer Matching (Tacticam -> Broadcast):")
for idx, mapped_idx in enumerate(mapping_indices):
    print(f"Tacticam Player {idx} -> Broadcast Player {mapped_idx}")

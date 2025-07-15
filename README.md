# Cross-Camera Player Mapping Project 🎥

## 📁 Setup Instructions

### 1. Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 2. Folder Structure
```
project-root/
│
├── model/
│   └── best.pt  # ❗ Not included in the repository. Place manually.
│
├── videos/
│   ├── broadcast.mp4
│   └── tacticam.mp4
│
├── detect_players.py
├── match_players.py
├── utils.py
└── README.md
```

### ⚠️ Note on best.pt
- The YOLO model file `best.pt` exceeds GitHub’s 100MB limit and is **NOT included** in the repository.
- Please **manually place** your trained `best.pt` inside the `model/` directory before running the code.

---

## 🚀 How to Run

### 1. Detect players using YOLO:
```bash
python detect_players.py
```

### 2. Match players across cameras:
```bash
python match_players.py
```

---

## ✅ Features

- 🎯 **Player Detection:** Uses YOLOv8 to detect players in both broadcast and tacticam footage.
- 🟩 **Centroid Tracking:** Assigns unique IDs to players and tracks their movement.
- 🎨 **Jersey Color Extraction:** Extracts average jersey colors from players.
- 🔗 **Cross-Camera Matching:** Matches players across cameras based on color similarity.
- 📝 **Modular Code:** Easy-to-read and extendable Python scripts.

---

## 📌 Future Work
- Improve feature matching accuracy using advanced descriptors (e.g., CNN features).
- Add a live streaming inference pipeline.
- Create a dashboard for visual analysis.

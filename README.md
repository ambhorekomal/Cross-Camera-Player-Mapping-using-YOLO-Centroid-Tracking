# Cross-Camera Player Mapping Project 🎥

## Setup

1. Install dependencies:
```
pip install ultralytics opencv-python numpy
```

2. Place files:
- `best.pt` → `model/`
- `broadcast.mp4`, `tacticam.mp4` → `videos/`

3. Run detection:
```
python detect_players.py
```

4. Extract player features:
```
python match_players.py
```

---

## Features
✅ Detects players using YOLO  
✅ Draws boxes and labels  
✅ Extracts jersey colors  
✅ Next step: match players using color similarity
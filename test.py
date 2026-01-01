from ultralytics import YOLO
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load both models
print("Loading models...")
coco_model = YOLO('models/coco_yolov8n.pt')  # or yolov8n.pt
custom_model = YOLO('models/yolo12n.pt')   # Your custom model
print("✓ Models loaded successfully")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

print("\n=== CONTROLS ===")
print("ESC or 'q' - Quit")
print("'1' - Toggle COCO model (Blue)")
print("'2' - Toggle Custom model (Green)")
print("'b' - Toggle Both models")
print("================\n")

# Settings
show_coco = True
show_custom = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Create a copy for annotations
    display_frame = frame.copy()
    
    # Run COCO model and draw in BLUE
    if show_coco:
        results_coco = coco_model(frame, verbose=False)
        
        for box in results_coco[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results_coco[0].names[cls]}: {conf:.2f}"
            
            # Draw blue rectangle and label
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Run Custom model and draw in GREEN
    if show_custom:
        results_custom = custom_model(frame, verbose=False)
        
        for box in results_custom[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results_custom[0].names[cls]}: {conf:.2f}"
            
            # Draw green rectangle and label
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show status
    status_text = []
    if show_coco:
        status_text.append("COCO: ON")
    if show_custom:
        status_text.append("Custom: ON")
    
    if status_text:
        cv2.putText(display_frame, " | ".join(status_text), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Display
    cv2.imshow('Dual YOLO Detection', display_frame)
    
    # Handle keys
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC or 'q'
        break
    elif key == ord('1'):
        show_coco = not show_coco
        print(f"COCO Model: {'ON' if show_coco else 'OFF'}")
    elif key == ord('2'):
        show_custom = not show_custom
        print(f"Custom Model: {'ON' if show_custom else 'OFF'}")
    elif key == ord('b'):
        show_coco = not show_coco
        show_custom = not show_custom
        print(f"Both Models: COCO={'ON' if show_coco else 'OFF'}, Custom={'ON' if show_custom else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Camera closed")
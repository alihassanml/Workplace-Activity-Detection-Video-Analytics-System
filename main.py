from ultralytics import YOLO
import cv2
import threading
from queue import Queue
from playsound import playsound
import numpy as np
import time
from datetime import timedelta

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load both models
custom_model = YOLO('models/yolov8s.pt')  # Your custom trained model
coco_model = YOLO('models/coco_yolov8n.pt')  # Pretrained COCO model

print("Custom model classes:", custom_model.names)
print("COCO model classes:", coco_model.names)

# Map COCO classes to our main activities
COCO_TO_ACTIVITY_MAP = {
    0: 'person',    # person
    39: 'eating',   # bottle
    40: 'eating',   # wine glass
    41: 'eating',   # cup
    42: 'eating',   # fork
    43: 'eating',   # knife
    44: 'eating',   # spoon
    45: 'eating',   # bowl
    46: 'eating',   # banana
    47: 'eating',   # apple
    48: 'eating',   # sandwich
    49: 'eating',   # orange
    50: 'eating',   # broccoli
    51: 'eating',   # carrot
    52: 'eating',   # hot dog
    53: 'eating',   # pizza
    54: 'eating',   # donut
    55: 'eating',   # cake
    59: 'sleeping',  # bed
    57: 'sleeping',  # couch
    67: 'phone',     # cell phone
}

# Queues for thread communication
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)

# Tracking data
tracked_persons = {}
next_person_id = 1
ACTIVITIES = ['smoking', 'eating', 'sleeping', 'phone', 'person']

# Time limits for alerts (in seconds)
TIME_LIMITS = {
    'phone': 15,
    'smoking': None,
    'eating': None,
    'sleeping': None,
    'person': None
}

CONFIDENCE_THRESHOLDS = {
    'smoking': 0.50,
    'eating': 0.25,
    'sleeping': 0.25,
    'phone': 0.25,
    'person': 0.25
}

PLAY_SOUND_ON_START = []
PLAY_SOUND_ON_TIME_LIMIT = ['phone']

# NEW: Enhanced tracking parameters
MAX_LOST_FRAMES = 30  # Keep tracking for 30 frames even if not detected
IOU_THRESHOLD = 0.15  # Lower threshold for better tracking
CENTER_DISTANCE_THRESHOLD = 150  # Maximum pixel distance for center matching

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calculate_center_distance(box1, box2):
    """Calculate distance between centers of two boxes"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def find_best_match(detection_bbox, tracked_persons, matched_ids):
    """
    Find best matching person ID using multiple criteria:
    1. IoU overlap
    2. Center distance
    3. Temporal consistency
    """
    best_match_id = None
    best_score = 0
    
    for pid, pdata in tracked_persons.items():
        if pid in matched_ids:
            continue
        
        # Skip if person was lost too long ago
        frames_lost = pdata.get('frames_lost', 0)
        if frames_lost > MAX_LOST_FRAMES:
            continue
        
        # Calculate IoU
        iou = calculate_iou(detection_bbox, pdata['bbox'])
        
        # Calculate center distance
        center_dist = calculate_center_distance(detection_bbox, pdata['bbox'])
        
        # Normalize distance (closer = higher score)
        dist_score = max(0, 1 - (center_dist / CENTER_DISTANCE_THRESHOLD))
        
        # Combined score: weighted average of IoU and distance
        combined_score = (iou * 0.6) + (dist_score * 0.4)
        
        # Bonus for temporal consistency
        if frames_lost == 0:
            combined_score *= 1.2
        
        if combined_score > best_score and (iou > IOU_THRESHOLD or center_dist < CENTER_DISTANCE_THRESHOLD):
            best_score = combined_score
            best_match_id = pid
    
    return best_match_id

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def play_alert_sound(alert_type):
    """Play alert sound in separate thread"""
    try:
        if alert_type == "start":
            playsound('sound/drop.mp3')
        elif alert_type == "time_limit":
            playsound('sound/drop.mp3')
    except Exception as e:
        if alert_type == "start":
            print(f"🔊 ALERT: Activity started!")
        elif alert_type == "time_limit":
            print(f"⏰ TIME LIMIT ALERT: Drop sound!")

def detection_thread():
    """Run YOLO detection in separate thread with dual models and improved tracking"""
    global tracked_persons, next_person_id
    
    frame_count = 0
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Run custom model detection
            custom_results = custom_model(frame, imgsz=416, conf=0.25, verbose=False)
            custom_result = custom_results[0]
            
            # Run COCO model detection
            coco_results = coco_model(frame, imgsz=416, conf=0.25, verbose=False)
            coco_result = coco_results[0]
            
            # Extract detections by activity
            detections = {activity: [] for activity in ACTIVITIES}
            
            # Process custom model detections
            for box in custom_result.boxes:
                cls = int(box.cls)
                class_name = custom_model.names[cls]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                if class_name in ACTIVITIES:
                    required_conf = CONFIDENCE_THRESHOLDS.get(class_name, 0.25)
                    if conf >= required_conf:
                        detections[class_name].append({
                            'bbox': xyxy,
                            'conf': conf,
                            'source': 'custom',
                            'original_class': class_name
                        })
            
            # Process COCO model detections
            for box in coco_result.boxes:
                cls = int(box.cls)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                if cls in COCO_TO_ACTIVITY_MAP:
                    activity = COCO_TO_ACTIVITY_MAP[cls]
                    coco_class_name = coco_model.names[cls]
                    
                    required_conf = CONFIDENCE_THRESHOLDS.get(activity, 0.25)
                    if conf >= required_conf:
                        detections[activity].append({
                            'bbox': xyxy,
                            'conf': conf,
                            'source': 'coco',
                            'original_class': coco_class_name
                        })
            
            # Mark all existing persons as potentially lost
            for pid in tracked_persons:
                tracked_persons[pid]['frames_lost'] = tracked_persons[pid].get('frames_lost', 0) + 1
            
            # Update tracked persons
            current_frame_persons = {}
            matched_ids = set()
            
            # Process each activity detection
            for activity_name, activity_detections in detections.items():
                for detection in activity_detections:
                    det_bbox = detection['bbox']
                    det_conf = detection['conf']
                    
                    # Find best matching person using improved algorithm
                    matched_id = find_best_match(det_bbox, tracked_persons, matched_ids)
                    
                    if matched_id is None:
                        # Create new person
                        person_id = next_person_id
                        next_person_id += 1
                        
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'detection_source': detection['source'],
                            'original_class': detection['original_class'],
                            'activities': {act: {
                                'total_time': 0,
                                'start_time': None,
                                'alerted': False,
                                'time_limit_alerted': False,
                                'conf': 0
                            } for act in ACTIVITIES},
                            'last_seen': current_time,
                            'frames_lost': 0,
                            'frame_count': frame_count
                        }
                        current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                        current_frame_persons[person_id]['activities'][activity_name]['conf'] = det_conf
                    else:
                        # Update existing person
                        person_id = matched_id
                        matched_ids.add(person_id)
                        
                        old_data = tracked_persons[person_id]
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'detection_source': detection['source'],
                            'original_class': detection['original_class'],
                            'activities': {act: old_data['activities'][act].copy() for act in ACTIVITIES},
                            'last_seen': current_time,
                            'frames_lost': 0,
                            'frame_count': old_data.get('frame_count', frame_count)
                        }
                        
                        old_activity = old_data['current_activity']
                        
                        # Stop old activity timer if different
                        if old_activity != activity_name and old_data['activities'][old_activity]['start_time'] is not None:
                            elapsed = current_time - old_data['activities'][old_activity]['start_time']
                            current_frame_persons[person_id]['activities'][old_activity]['total_time'] += elapsed
                            current_frame_persons[person_id]['activities'][old_activity]['start_time'] = None
                        
                        # Start or continue activity timer
                        if current_frame_persons[person_id]['activities'][activity_name]['start_time'] is None:
                            current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                            current_frame_persons[person_id]['activities'][activity_name]['alerted'] = False
                            current_frame_persons[person_id]['activities'][activity_name]['time_limit_alerted'] = False
                        
                        # Update confidence
                        current_frame_persons[person_id]['activities'][activity_name]['conf'] = det_conf
            
            # Keep recently lost persons (within MAX_LOST_FRAMES)
            for pid, pdata in tracked_persons.items():
                if pid not in current_frame_persons and pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
                    current_frame_persons[pid] = pdata.copy()
                    current_frame_persons[pid]['activities'] = {act: pdata['activities'][act].copy() for act in ACTIVITIES}
            
            # Check time limits
            for person_id, pdata in current_frame_persons.items():
                activity = pdata['current_activity']
                activity_data = pdata['activities'][activity]
                
                if activity_data['start_time'] is not None:
                    total_time = activity_data['total_time'] + (current_time - activity_data['start_time'])
                    
                    time_limit = TIME_LIMITS.get(activity)
                    if time_limit is not None and total_time >= time_limit:
                        if not activity_data['time_limit_alerted']:
                            activity_data['time_limit_alerted'] = True
                            activity_data['exceeded_time_limit'] = True
            
            tracked_persons = current_frame_persons
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw all detections
            for person_id, pdata in tracked_persons.items():
                if pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
                    bbox = pdata['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    activity = pdata['current_activity']
                    source = pdata['detection_source']
                    original_class = pdata['original_class']
                    
                    # Color based on source
                    if source == 'custom':
                        color = (0, 255, 0)  # Green for custom model
                    else:
                        color = (255, 0, 0)  # Blue for COCO model
                    
                    # Dim color if temporarily lost
                    frames_lost = pdata.get('frames_lost', 0)
                    if frames_lost > 0:
                        alpha = max(0.3, 1 - (frames_lost / MAX_LOST_FRAMES))
                        color = tuple(int(c * alpha) for c in color)
                    
                    # Draw box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label with mapped activity and original class
                    label = f"ID:{person_id} {activity.upper()} ({original_class})"
                    if frames_lost > 0:
                        label += f" [Lost:{frames_lost}]"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if result_queue.full():
                result_queue.get()
            result_queue.put((annotated_frame, tracked_persons.copy(), current_time))

# Start detection thread
detector = threading.Thread(target=detection_thread, daemon=True)
detector.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

annotated_frame = None
frame_count = 0
display_tracked_persons = {}
current_time = time.time()

ACTIVITY_COLORS = {
    'smoking': (0, 0, 255),
    'eating': (0, 255, 0),
    'sleeping': (255, 0, 255),
    'phone': (0, 165, 255),
    'person': (128, 128, 128)
}

print("\n=== Enhanced Dual Model Activity Tracker Started ===")
print("Green boxes = Custom model detections")
print("Blue boxes = COCO model detections (mapped to activities)")
print("Tracking improvements:")
print(f"  - Max lost frames: {MAX_LOST_FRAMES}")
print(f"  - IoU threshold: {IOU_THRESHOLD}")
print(f"  - Center distance threshold: {CENTER_DISTANCE_THRESHOLD}px")
print("Press ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 2 == 0 and not frame_queue.full():
        frame_queue.put(frame.copy())
    
    if not result_queue.empty():
        annotated_frame, display_tracked_persons, current_time = result_queue.get()
        
        for person_id, pdata in display_tracked_persons.items():
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            
            if not activity_data['alerted']:
                source = pdata['detection_source']
                original_class = pdata['original_class']
                # print(f"⚠️ Person ID {person_id} started: {activity.upper()} (detected as '{original_class}' by {source} model)")
                activity_data['alerted'] = True
                
                if activity in PLAY_SOUND_ON_START:
                    print(f"🔊 Playing start sound for {activity}")
                    threading.Thread(target=play_alert_sound, args=("start",), daemon=True).start()

            if activity_data.get('exceeded_time_limit', False) and activity_data.get('time_limit_alerted', False):
                time_limit = TIME_LIMITS.get(activity)
                print(f"🚨 WARNING: Person ID {person_id} has been using {activity.upper()} for more than {time_limit} seconds!")
                
                if activity in PLAY_SOUND_ON_TIME_LIMIT:
                    print(f"🔊 Playing time limit sound for {activity}")
                    threading.Thread(target=play_alert_sound, args=("time_limit",), daemon=True).start()
                
                activity_data['exceeded_time_limit'] = False
    
    if annotated_frame is not None:
        display_frame = annotated_frame.copy()
        
        for person_id, pdata in display_tracked_persons.items():
            # Skip persons lost for too long
            if pdata.get('frames_lost', 0) > MAX_LOST_FRAMES:
                continue
            
            bbox = pdata['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            color = ACTIVITY_COLORS.get(activity, (255, 255, 255))
            
            # Dim color if temporarily lost
            frames_lost = pdata.get('frames_lost', 0)
            if frames_lost > 0:
                alpha = max(0.3, 1 - (frames_lost / MAX_LOST_FRAMES))
                color = tuple(int(c * alpha) for c in color)
            
            total_time = activity_data['total_time']
            if activity_data['start_time'] is not None:
                current_session = current_time - activity_data['start_time']
                total_time += current_session
            
            time_str = format_time(total_time)
            
            time_limit = TIME_LIMITS.get(activity)
            warning_text = ""
            if time_limit is not None:
                if total_time >= time_limit:
                    color = (0, 0, 255)
                    warning_text = "⚠️ TIME LIMIT!"
                elif total_time >= time_limit * 0.8:
                    color = (0, 165, 255)
                    remaining = int(time_limit - total_time)
                    warning_text = f"⏰ {remaining}s left"
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            source_indicator = "🟢" if pdata['detection_source'] == 'custom' else "🔵"
            info_text = [
                # f"ID: {person_id} {source_indicator}",
                f"{activity.upper()}",
                # f"({pdata['original_class']})",
                f"Time: {time_str}",
                # f"Conf: {int(activity_data.get('conf', 0) * 100)}%"
            ]
            
            if frames_lost > 0:
                info_text.append(f"Lost: {frames_lost}f")
            
            if warning_text:
                info_text.append(warning_text)
            
            y_offset = y1 - 10
            for text in reversed(info_text):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame,
                            (x1, y_offset - text_size[1] - 5),
                            (x1 + text_size[0] + 10, y_offset + 5),
                            color, -1)
                cv2.putText(display_frame, text, (x1 + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset -= (text_size[1] + 10)
            
            history_y = 30
            cv2.putText(display_frame, f"Person {person_id} Activity Log:",
                       (10, history_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            history_y += 25
            
            for act in ACTIVITIES:
                act_data = pdata['activities'][act]
                act_time = act_data['total_time']
                if pdata['current_activity'] == act and act_data['start_time'] is not None:
                    act_time += current_time - act_data['start_time']
                
                if act_time > 0:
                    time_display = format_time(act_time)
                    act_color = ACTIVITY_COLORS.get(act, (255, 255, 255))
                    
                    limit_info = ""
                    act_limit = TIME_LIMITS.get(act)
                    if act_limit is not None:
                        if act_time >= act_limit:
                            limit_info = " [EXCEEDED]"
                            act_color = (0, 0, 255)
                        else:
                            limit_info = f" [Limit: {act_limit}s]"
                    
                    cv2.putText(display_frame, f"{act}: {time_display}{limit_info}",
                               (10, history_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, act_color, 1)
                    history_y += 20
            
            history_y += 10
        
        cv2.imshow('Enhanced Dual Model Activity Tracker', display_frame)
    else:
        cv2.imshow('Enhanced Dual Model Activity Tracker', frame)
    
    if cv2.waitKey(1) == 27:
        frame_queue.put(None)
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("Final Activity Statistics:")
print("="*50)
for person_id, pdata in tracked_persons.items():
    if pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
        print(f"\nPerson ID {person_id} (Source: {pdata['detection_source']} model):")
        for activity, data in pdata['activities'].items():
            total = data['total_time']
            if pdata['current_activity'] == activity and data['start_time'] is not None:
                total += time.time() - data['start_time']
            if total > 0:
                time_str = format_time(total)
                limit = TIME_LIMITS.get(activity)
                limit_str = f" (Limit: {limit}s)" if limit else ""
                exceeded = " ⚠️ EXCEEDED" if limit and total >= limit else ""
                print(f"  {activity.capitalize()}: {time_str}{limit_str}{exceeded}")
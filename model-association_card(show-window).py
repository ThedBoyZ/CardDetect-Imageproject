import cv2
import os
import glob
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO

model = YOLO("card-yolo8.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Did't see Camera port enable...")
    exit()
    
# Global variable
results_list = []
frame_count = 0
start_enable = None

# path floder 'home'
home_folder = "card-predict"
red_percentage = 0

def clear_saved_images(folder_path):
    images = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    for image in images:
        try:
            os.remove(image)
            print(f"Deleted: {image}")
        except OSError as e:
            print(f"Error deleting {image}: {e}")

# create id don't have path 'home'
if not os.path.exists(home_folder):
    os.makedirs(home_folder)

clear_saved_images(home_folder)

def is_black_in_box(image, bbox, threshold=0.1):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cropped_region = image[y1:y2, x1:x2]
    gray_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    
    black_pixels = np.sum(gray_region < 50)
    total_pixels = gray_region.size
    black_percentage = black_pixels / total_pixels
    print("black sum : ", black_pixels)
    print("threshold : ", threshold)
    
    return black_percentage 

def is_red_suit(image, bbox):
    global red_percentage
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cropped_region = image[y1:y2, x1:x2]
    hsv_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color
    mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    red_percentage = np.mean(red_mask > 0)
    print("red percentage : ", red_percentage)
    return red_percentage

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera Feed", frame)
    
    if start_enable is None:
        print("Session is Ready !!!")
        start_enable = 1    
        
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        detected_list = [] 
        
        for _ in range(10):
            ret, frame = cap.read()
            result = model.predict(source=frame)

            predictions = result[0].boxes
            class_names = result[0].names
            
            detected_cards = []
            confidences = {}
            
            for box in predictions:
                class_idx = int(box.cls[0])
                class_name = class_names[class_idx]
                confidence = box.conf[0] 
                bbox = box.xyxy[0].tolist()

                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

                if class_name == 'Ad' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox) :
                    class_name = 'As' 
                elif class_name == 'Ah' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Ac' 
                    
                if class_name == 'Jd' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Js'
                elif class_name == 'Jh' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Jc'
                    
                if class_name == 'Qd' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Qs' 
                elif class_name == 'Qh' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Qc'  
                    
                if class_name == 'Kd' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Ks' 
                elif class_name == 'Kh' and is_black_in_box(frame, bbox) > is_red_suit(frame, bbox):
                    class_name = 'Kc' 

                confidences[class_name] = confidence
                
                # Append only the class name (your original logic)
                if class_name not in detected_cards:
                    detected_cards.append(class_name)
                
            if 'Jd' in detected_cards and 'Jh' in detected_cards:
                confidence_js = 1
                detected_cards.remove('Jd')
                detected_cards.remove('Jh')
                confidences['Js'] = confidence_js 
                detected_cards.append('Js')

            if 'Qd' in detected_cards and 'Qh' in detected_cards:
                confidence_js = 1
                detected_cards.remove('Qd')
                detected_cards.remove('Qh')
                confidences['Qs'] = confidence_js 
                detected_cards.append('Qs')
                
            if 'Kd' in detected_cards and 'Kh' in detected_cards:
                confidence_js = 1
                detected_cards.remove('Kd')
                detected_cards.remove('Kh')
                confidences['Ks'] = confidence_js 
                detected_cards.append('Ks')

            if 'Jd' in detected_cards:
                if red_percentage < 0.1:
                    confidence_js = 1
                    detected_cards.remove('Jd')
                    detected_cards.append('Js')
                    confidences['Js'] = confidence_js
                                            
            if 'Kd' in detected_cards:
                if red_percentage < 0.1:
                    confidence_js = 1
                    detected_cards.remove('Kd')
                    detected_cards.append('Ks')
                    confidences['Ks'] = confidence_js

            if 'Js' in detected_cards and 'Jc' in detected_cards:
                if confidences['Js'] > confidences['Jc']:
                    detected_cards.remove('Jc')

            if 'Qs' in detected_cards and 'Qc' in detected_cards:
                if confidences['Qs'] > confidences['Qc']:
                    detected_cards.remove('Qc')
                    
            if 'Ks' in detected_cards and 'Kc' in detected_cards:
                if confidences['Ks'] > confidences['Kc']:
                    detected_cards.remove('Kc') 
                    
            # Your original logic for priority checks
            if 'Jc' in detected_cards and 'Jh' in detected_cards:
                detected_cards.remove('Jh') 
            if 'Jc' in detected_cards and 'Jd' in detected_cards:
                detected_cards.remove('Jd') 
            if 'Js' in detected_cards and 'Jh' in detected_cards:
                detected_cards.remove('Jh') 
            if 'Js' in detected_cards and 'Jd' in detected_cards:
                detected_cards.remove('Jd') 
                
            if 'Qc' in detected_cards and 'Qh' in detected_cards:
                detected_cards.remove('Qh')  
            if 'Qc' in detected_cards and 'Qd' in detected_cards:
                detected_cards.remove('Qd')  
            if 'Qs' in detected_cards and 'Qh' in detected_cards:
                detected_cards.remove('Qh')  
            if 'Qs' in detected_cards and 'Qd' in detected_cards:
                detected_cards.remove('Qd')  
                
            if 'Kc' in detected_cards and 'Kh' in detected_cards:
                detected_cards.remove('Kh')  
            if 'Kc' in detected_cards and 'Kd' in detected_cards:
                detected_cards.remove('Kd') 
            if 'Ks' in detected_cards and 'Kh' in detected_cards:
                detected_cards.remove('Kh')  
            if 'Ks' in detected_cards and 'Kd' in detected_cards:
                detected_cards.remove('Kd') 

            detected_cards = list(set(detected_cards))
                
            for box in predictions:
                class_idx = int(box.cls[0])
                class_name = class_names[class_idx]
                confidence = confidences.get(class_name, 1.0)  
                if class_name in detected_cards: 
                    bbox = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            print(f"detected list = {detected_cards}")
            
            detected_list.extend(detected_cards)
            # Save the current frame as an image in 'home' folder
            img_filename = os.path.join(home_folder, f"screenshot_{frame_count}.jpg")
            cv2.imwrite(img_filename, frame)
            print(f"Captured and saved with bounding boxes: {img_filename}")

            frame_count += 1
            time.sleep(0.4)
        
        # conclusion most 'variable' declare in detected_list
        if detected_list:
            most_common_card = Counter(detected_list).most_common(1)[0][0]
            print(f"Most common detected card: {most_common_card}")
            print(f"Detected list: {detected_list}")
        else:
            print("No cards detected.")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

# Load the trained model
model = YOLO("runs/detect/tps_yolo_exp_h200_multi/weights/best.pt")

# let's just try with singular image first
img_path = "data/miami_fall_24_jpgs/1001.jpg"
processed_path = "temp_processed_1001.jpg"

def preprocess_for_inference(img_path, target_size=1024):
    """
    Preprocess image to exactly match the training pipeline.
    This ensures the model sees the same type of input it was trained on.
    """
    with Image.open(img_path) as img:
        print(f"Original image: {img.size}, mode: {img.mode}")
        
        # Step 1: resize with same method as training
        original_width, original_height = img.size
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # resize with same resampling method as training
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Step 2: convert to grayscale (matching training preprocessing)
        if resized_img.mode != 'L':
            resized_img = resized_img.convert('L')
        
        # Step 3: convert back to RGB for YOLO inference (3 channels expected)
        resized_img = resized_img.convert('RGB')
        
        print(f"Processed image: {resized_img.size}, mode: {resized_img.mode}")
        return resized_img

# preprocess the image to match training
processed_img = preprocess_for_inference(img_path, target_size=1024)
processed_img.save(processed_path)

# run inference with matching parameters
results = model.predict(
    source=processed_path, 
    save=True, 
    imgsz=1024,           
    conf=0.25,            
    iou=0.45,            
    show_labels=True,     
    show_conf=True,       
    save_txt=True,        
)


for r in results:
    boxes = r.boxes
    if boxes is not None:
        print(f"\nFound {len(boxes)} detections:")
        for i, box in enumerate(boxes):
            # get class names
            class_id = int(box.cls[0])
            class_names = ['Finger', 'Toe', 'Ruler']  # match our classes
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            # get confidence
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  {i+1}. {class_name}: {confidence:.3f} confidence")
            print(f"     Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
    else:
        print("\nNo detections found")

# Clean up temporary file
os.remove(processed_path)

print(f"\nResults saved to: runs/detect/predict/")
print("Check the output images to visually verify detections")
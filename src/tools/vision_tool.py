# src/tools/vision_tool.py

from langchain.tools import tool
from ultralytics import YOLO
import cv2
from typing import Dict

# --- Load Custom Model ---
# This path points to your custom-trained model file in the 'models' directory
MODEL_PATH = "models/urbanplan_yolov8.pt"
model = YOLO(MODEL_PATH) 

# --- Define Custom Class IDs ---
# These IDs must match the ones from your Roboflow training project
# 0: building, 1: green_space, 2: water_body (as an example)
BUILDING_CLASSES = [0] 
GREEN_SPACE_CLASSES = [1]

@tool
def yolo_site_analyzer(image_path: str) -> dict:
    """
    Analyzes a site plan image using a custom-trained YOLOv8 model
    to quantify the percentage of green space and building footprint.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Image not found or could not be loaded."}
            
        H, W, _ = img.shape
        total_pixel_area = H * W

        # Run inference with your custom model
        results = model(img)

        green_space_pixel_area = 0
        building_footprint_pixel_area = 0

        # Iterate through the detection results
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates and class ID
                x1, y1, x2, y2 = box.xyxy[0]
                cls_id = int(box.cls[0])
                box_pixel_area = (x2 - x1) * (y2 - y1)
                
                # Add the area to the correct category based on the class ID
                if cls_id in GREEN_SPACE_CLASSES:
                    green_space_pixel_area += box_pixel_area
                elif cls_id in BUILDING_CLASSES:
                    building_footprint_pixel_area += box_pixel_area

        # Calculate the final percentages
        green_cover_percentage = (green_space_pixel_area / total_pixel_area) * 100
        building_footprint_percentage = (building_footprint_pixel_area / total_pixel_area) * 100

        analysis_result = {
            "green_cover_percentage": round(float(green_cover_percentage), 2),
            "building_footprint_percentage": round(float(building_footprint_percentage), 2)
        }
        
        print(f"Custom Model Analysis Complete: {analysis_result}")
        return analysis_result
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"
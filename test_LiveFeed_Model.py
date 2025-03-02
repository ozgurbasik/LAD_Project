import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from typing import Tuple

def initialize_model(model_path: str) -> YOLO:
    """Initialize YOLO model from given path."""
    return YOLO(model_path)

def process_frame(
    frame: np.ndarray,
    model: YOLO,
    confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, sv.Detections]:
    """
    Process a single frame through YOLO model and return detections.
    
    Args:
        frame: Input frame as numpy array
        model: YOLO model instance
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        Tuple of annotated frame and detections object
    """
    # Get model predictions
    results = model(frame)[0]
    
    # Convert predictions to supervision Detections
    detections = sv.Detections.from_yolov8(results)
    
    # Filter by confidence
    mask = detections.confidence >= confidence_threshold
    detections = detections[mask]
    
    # Create annotator
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    # Draw detections
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for class_id, confidence 
        in zip(detections.class_id, detections.confidence)
    ]
    
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels
    )
    
    return frame, detections

def main():
    # Replace with your IPTV stream URL
    STREAM_URL = "rtsp://your_camera_ip:port/stream"
    
    # Initialize video capture
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    
    # Initialize model
    model = initialize_model("./Models/yolo11n.pt")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            annotated_frame, detections = process_frame(frame, model)
            
            # Display results
            cv2.imshow("Object Detection", annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
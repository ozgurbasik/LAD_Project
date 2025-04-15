import cv2
import time
from ultralytics import YOLO
import torch
from Metrics.FPStracker import FPSTracker
from Metrics.PowerConsumption import PowerMonitor
from Metrics.Utilization import SystemUtilizationMonitor

# Check if CUDA is available
print(torch.cuda.is_available())  # True olmalı

# Initialize trackers
fps_tracker = FPSTracker()
power_monitor = PowerMonitor(sampling_interval=0.5)
utilization_monitor = SystemUtilizationMonitor(sampling_interval=0.5)

# YOLO modelini yükleyin ve GPU'yu kullanın
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("./Models/yolo11s.pt").to(device)  # Modeli CUDA'ya taşı

# Kameradan görüntü almak için OpenCV kullanın
cap = cv2.VideoCapture(1)  # 0, varsayılan kamerayı temsil eder
print("Using device:", device)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Start monitoring in background
power_monitor.start_monitoring()
utilization_monitor.start_monitoring()
print("Monitoring started")

try:
    while True:
        # Start timing this frame
        fps_tracker.start_frame()
        
        # Kameradan bir kare alın
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break
        
        # Görüntüyü YOLO modeline verin
        results = model(frame)
        
        # Sonuçları görselleştir ve çiz
        annotated_frame = results[0].plot()
        
        # Calculate FPS for this frame
        fps = fps_tracker.end_frame()
        
        # Get current monitoring data for display
        power_data = power_monitor.get_current_power()
        utilization_data = utilization_monitor.get_current_utilization()
        
        # Draw information on frame
        #annotated_frame = fps_tracker.draw_fps_on_frame(annotated_frame, fps)
        #annotated_frame = power_monitor.draw_power_info_on_frame(annotated_frame, power_data)
        #annotated_frame = utilization_monitor.draw_utilization_on_frame(annotated_frame, utilization_data)
        
        # Görüntüyü göster
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # 'q' tuşuna basarak döngüyü durdurabilirsiniz
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop monitoring
    power_monitor.stop_monitoring()
    utilization_monitor.stop_monitoring()
    print("Monitoring stopped")
    
    # Kamerayı serbest bırakın ve pencereyi kapatın
    cap.release()
    cv2.destroyAllWindows()
    
    # Show performance graphs and statistics
    print("\n--- FPS Statistics ---")
    fps_tracker.print_fps_stats()
    fps_tracker.show_fps_graph()
    
    print("\n--- Power Consumption Statistics ---")
    power_monitor.print_power_stats()
    power_monitor.show_power_graph()
    
    print("\n--- System Utilization Statistics ---")
    utilization_monitor.print_utilization_stats()
    utilization_monitor.show_utilization_graphs()

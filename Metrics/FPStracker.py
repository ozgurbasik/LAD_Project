import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

class FPSTracker:
    def __init__(self):
        self.fps_values = []
        self.timestamps = []
        self.start_program_time = time.time()
        self.frame_start_time = None
    
    def start_frame(self):
        """Start timing for a new frame"""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """End timing for the current frame and calculate FPS"""
        if self.frame_start_time is None:
            return 0
        
        frame_time = time.time() - self.frame_start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Store FPS data
        self.fps_values.append(fps)
        self.timestamps.append(time.time() - self.start_program_time)
        
        return fps
    
    def draw_fps_on_frame(self, frame, fps):
        """Draw FPS information on the given frame"""
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def show_fps_graph(self):
        """Generate and display a graph of FPS performance over time"""
        if not self.fps_values:
            print("No FPS data to display")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.fps_values, '-b', label='FPS')
        plt.title('FPS Performance Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frames Per Second')
        plt.grid(True)
        
        # Add average FPS line
        avg_fps = np.mean(self.fps_values)
        plt.axhline(y=avg_fps, color='r', linestyle='--', label=f'Average FPS: {avg_fps:.2f}')
        
        # Add min and max FPS annotations
        min_fps = np.min(self.fps_values)
        max_fps = np.max(self.fps_values)
        plt.text(self.timestamps[-1]/2, min_fps, f'Min FPS: {min_fps:.2f}', color='red')
        plt.text(self.timestamps[-1]/2, max_fps, f'Max FPS: {max_fps:.2f}', color='green')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def print_fps_stats(self):
        """Print statistics about the collected FPS data"""
        if not self.fps_values:
            print("No FPS data available")
            return
            
        avg_fps = np.mean(self.fps_values)
        min_fps = np.min(self.fps_values)
        max_fps = np.max(self.fps_values)
        
        print(f"Program çalışma süresi: {self.timestamps[-1]:.2f} saniye")
        print(f"Ortalama FPS: {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maksimum FPS: {max_fps:.2f}")
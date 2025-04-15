import time
import subprocess
import platform
import matplotlib.pyplot as plt
import numpy as np
import psutil
import threading

class PowerMonitor:
    def __init__(self, sampling_interval=1.0):
        """
        Initialize the power monitor.
        
        Args:
            sampling_interval (float): Time between power measurements in seconds
        """
        self.sampling_interval = sampling_interval
        self.power_values = []
        self.timestamps = []
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        self.system = platform.system()
        
        # Check if we can access GPU power info
        self.has_gpu = self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if we can access GPU power information using nvidia-smi"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _get_power_usage(self):
        """Get current power usage data"""
        cpu_percent = psutil.cpu_percent()
        gpu_power = self._get_gpu_power()
        
        # For demonstration, convert CPU utilization to an estimated power value
        # This is a very rough estimate and should be calibrated for actual hardware
        estimated_cpu_power = cpu_percent * 0.5  # Assuming max CPU power of 50W at 100% utilization
        
        total_power = estimated_cpu_power + (gpu_power if gpu_power is not None else 0)
        return {
            'cpu_percent': cpu_percent,
            'estimated_cpu_power': estimated_cpu_power,
            'gpu_power': gpu_power,
            'total_power': total_power
        }
    
    def _get_gpu_power(self):
        """Get GPU power consumption in watts using nvidia-smi"""
        if not self.has_gpu:
            return None
            
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                # Parse the output to get power value
                output = result.stdout.strip()
                if output and output != "N/A":
                    return float(output)
            return None
        except:
            return None
    
    def _monitoring_loop(self):
        """Background thread for continuous power monitoring"""
        while self.monitoring:
            power_data = self._get_power_usage()
            self.power_values.append(power_data['total_power'])
            self.timestamps.append(time.time() - self.start_time)
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start the power monitoring in a background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            return True
        return False
    
    def stop_monitoring(self):
        """Stop the power monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            return True
        return False
    
    def get_current_power(self):
        """Get the current power usage (for real-time display)"""
        return self._get_power_usage()
    
    def draw_power_info_on_frame(self, frame, power_data=None):
        """Draw power consumption information on the frame"""
        if power_data is None:
            power_data = self._get_power_usage()
        
        import cv2
        y_pos = 60  # Start below FPS counter
        
        # Add CPU information
        cv2.putText(frame, f"CPU: {power_data['cpu_percent']:.1f}% ({power_data['estimated_cpu_power']:.1f}W)", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_pos += 25
        
        # Add GPU information if available
        if power_data['gpu_power'] is not None:
            cv2.putText(frame, f"GPU: {power_data['gpu_power']:.1f}W", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += 25
        
        # Add total power
        cv2.putText(frame, f"Total: {power_data['total_power']:.1f}W", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def show_power_graph(self):
        """Generate and display a graph of power consumption over time"""
        if not self.power_values:
            print("No power data to display")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.power_values, '-r', label='Power (W)')
        plt.title('Power Consumption Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Power (Watts)')
        plt.grid(True)
        
        # Add average power line
        avg_power = np.mean(self.power_values)
        plt.axhline(y=avg_power, color='b', linestyle='--', label=f'Average: {avg_power:.2f}W')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def print_power_stats(self):
        """Print statistics about power consumption"""
        if not self.power_values:
            print("No power data available")
            return
            
        avg_power = np.mean(self.power_values)
        min_power = np.min(self.power_values)
        max_power = np.max(self.power_values)
        
        print(f"Monitoring süresi: {self.timestamps[-1]:.2f} saniye")
        print(f"Ortalama güç tüketimi: {avg_power:.2f}W")
        print(f"Minimum güç tüketimi: {min_power:.2f}W")
        print(f"Maksimum güç tüketimi: {max_power:.2f}W")
        
        # Calculate energy consumption in Watt-hours
        duration_hours = self.timestamps[-1] / 3600.0  # seconds to hours
        energy_wh = avg_power * duration_hours
        print(f"Tahmini enerji tüketimi: {energy_wh:.4f} Watt-saat")
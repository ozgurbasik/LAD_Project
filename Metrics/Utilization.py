import time
import subprocess
import platform
import psutil
import threading
import matplotlib.pyplot as plt
import numpy as np
import re

class SystemUtilizationMonitor:
    def __init__(self, sampling_interval=1.0):
        """
        Initialize the system utilization monitor.
        
        Args:
            sampling_interval (float): Time between measurements in seconds
        """
        self.sampling_interval = sampling_interval
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None

        # Data storage
        self.timestamps = []
        self.cpu_percentages = []
        self.gpu_percentages = []

        # GPU monitoring
        self.system = platform.system()
        self.has_gpu = self._check_gpu_availability()
        print(self.has_gpu)
        self.gpu_utilization = 0.0  # Store last GPU usage reading

        # Start GPU monitoring in a background thread
        if self.has_gpu:
            self._gpu_thread = threading.Thread(target=self._monitor_tegrastats, daemon=True)
            self._gpu_thread.start()

    def _check_gpu_availability(self):
        """Check if GPU monitoring is available using tegrastats"""
        try:
            # Run tegrastats command and pipe the output to head -n 1
            process_tegrastats = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                  text=True)
            process_head = subprocess.Popen(['head', '-n', '1'], stdin=process_tegrastats.stdout,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Get the output of the head command (which is from tegrastats)
            output, _ = process_head.communicate()

            # Print the full output for debugging
            print("Full tegrastats working & output:" + output.strip())

            return True
        except subprocess.TimeoutExpired:
            print("tegrastats command timed out.")
        except Exception as e:
            print(f"Error running tegrastats: {e}")
        return False

    def _monitor_tegrastats(self):
        """Continuously reads tegrastats output and updates GPU utilization."""
        try:
            # Start the tegrastats process without piping to head
            process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Continuously read output lines from tegrastats
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Debug print: print each line from tegrastats
                    #print("tegrastats output:", line.strip())

                    # Parse GPU-related information using regular expressions
                    gr3d_freq_match = re.search(r'GR3D_FREQ (\d+)%', line)
                    if gr3d_freq_match:
                        # Update the GPU utilization value
                        self.gpu_utilization = float(gr3d_freq_match.group(1))
                else:
                    break

        except Exception as e:
            print(f"Error monitoring GPU utilization: {e}")

    def _get_system_utilization(self):
        """Get current system utilization metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'gpu_percent': self._get_gpu_utilization()
        }

    def _get_gpu_utilization(self):
        """Returns the last known GPU utilization value"""
        return self.gpu_utilization if self.has_gpu else 0.0

    def _monitoring_loop(self):
        """Background thread for continuous monitoring"""
        while self.monitoring:
            utilization = self._get_system_utilization()
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            self.cpu_percentages.append(utilization['cpu_percent'])
            self.gpu_percentages.append(utilization['gpu_percent'])

            time.sleep(self.sampling_interval)

    def start_monitoring(self):
        """Start the system utilization monitoring in a background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.timestamps, self.cpu_percentages, self.gpu_percentages = [], [], []
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            return True
        return False

    def stop_monitoring(self):
        """Stop the system utilization monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            return True
        return False

    def get_current_utilization(self):
        """Get the current system utilization (for real-time display)"""
        return self._get_system_utilization()

    def draw_utilization_on_frame(self, frame, utilization=None):
        """Draw system utilization information on a frame"""
        if utilization is None:
            utilization = self._get_system_utilization()
        
        import cv2
        y_pos = 135  # Start position (adjust as needed based on other info)
        
        # Draw CPU utilization
        cv2.putText(frame, f"CPU: {utilization['cpu_percent']:.1f}%", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_pos += 25
        
        # Draw GPU utilization if available
        if utilization['gpu_percent'] is not None:
            cv2.putText(frame, f"GPU: {utilization['gpu_percent']:.1f}%", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def show_utilization_graphs(self):
        """Generate and display graphs of system utilization over time."""
        if not self.timestamps:
            print("No utilization data to display")
            return

        # Determine number of subplots dynamically
        num_plots = 1 if not self.has_gpu else 2
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
        fig.suptitle('System Utilization Over Time')

        # If there's only one plot, axs will be an Axes object, not a list
        if num_plots == 1:
            axs = [axs]

        # CPU Graph
        axs[0].plot(self.timestamps, self.cpu_percentages, '-b', label='CPU Utilization')
        avg_cpu = np.mean(self.cpu_percentages)
        axs[0].axhline(avg_cpu, color='orange', linestyle='--', label=f'Avg: {avg_cpu:.2f}%')
        axs[0].set_title('CPU Utilization')
        axs[0].set_ylabel('Utilization (%)')
        axs[0].set_ylim(0, 100)
        axs[0].grid(True)
        axs[0].legend()

        # GPU Graph (Only if GPU data exists)
        if self.has_gpu and self.gpu_percentages:
            axs[1].plot(self.timestamps[:len(self.gpu_percentages)], self.gpu_percentages, '-r',
                        label='GPU Utilization')
            avg_gpu = np.mean(self.gpu_percentages)
            axs[1].axhline(avg_gpu, color='orange', linestyle='--', label=f'Avg: {avg_gpu:.2f}%')
            axs[1].set_title('GPU Utilization')
            axs[1].set_xlabel('Time (seconds)')
            axs[1].set_ylabel('Utilization (%)')
            axs[1].set_ylim(0, 100)
            axs[1].grid(True)
            axs[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def print_utilization_stats(self):
        """Print statistics about system utilization"""
        if not self.timestamps:
            print("No utilization data available")
            return
            
        # Calculate statistics
        avg_cpu = np.mean(self.cpu_percentages)
        max_cpu = np.max(self.cpu_percentages)
        
        print(f"Monitoring süresi: {self.timestamps[-1]:.2f} saniye")
        print(f"CPU Kullanımı (Ortalama): {avg_cpu:.2f}%")
        print(f"CPU Kullanımı (Maksimum): {max_cpu:.2f}%")
        
        # GPU statistics if available
        if self.gpu_percentages:
            avg_gpu = np.mean(self.gpu_percentages)
            max_gpu = np.max(self.gpu_percentages)
            print(f"GPU Kullanımı (Ortalama): {avg_gpu:.2f}%")
            print(f"GPU Kullanımı (Maksimum): {max_gpu:.2f}%")

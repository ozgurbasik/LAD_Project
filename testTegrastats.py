import subprocess
import re

def parse_gpu_info():
    """Run tegrastats | head -n 1, parse GPU-related information and print it."""
    try:
        # Run tegrastats command and pipe the output to head -n 1
        process_tegrastats = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process_head = subprocess.Popen(['head', '-n', '1'], stdin=process_tegrastats.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get the output of the head command (which is from tegrastats)
        output, _ = process_head.communicate()

        # Print the full output for debugging
        print("Full tegrastats output:")
        print(output.strip())

        # Parse GPU-related information using regular expressions
        gpu_temp_match = re.search(r'gpu@([0-9.]+)C', output)
        gr3d_freq_match = re.search(r'GR3D_FREQ (\d+)%', output)

        if gpu_temp_match:
            gpu_temp = gpu_temp_match.group(1)
            print(f"GPU Temperature: {gpu_temp}°C")
        else:
            print("GPU Temperature not found.")

        if gr3d_freq_match:
            gr3d_freq = gr3d_freq_match.group(1)
            print(f"GPU Frequency: {gr3d_freq}%")
        else:
            print("GPU Frequency not found.")

    except subprocess.TimeoutExpired:
        print("tegrastats command timed out.")
    except Exception as e:
        print(f"Error running tegrastats: {e}")

# Call the function to parse and print GPU info
parse_gpu_info()

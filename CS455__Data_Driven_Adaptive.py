import serial
import time
import numpy as np

# --- Configuration ---
PORT = 'COM16'  # Check your port in Device Manager (Windows) or /dev/tty* (Linux/Mac)
BAUD = 115200   # Change this in Device Manager if needed
TIMEOUT = 0.1

# --- Initialize Serial ---
try:
    ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=TIMEOUT)
    time.sleep(2) 
except Exception as error:
    print(f"Connection failed: {error}")
    exit()

print("\n--- DC Motor Control ---")

# --- System Variables ---
TICKS = 4128    # Counter Ticks per Half Revolution (172*24)
Ts = 0.003      # Target Sampling Time
t, theta = 0.0, 0.0
theta0 = None
dt = 0.0 # dt is calculated in the loop
ISE, ISC = 0.0, 0.0

# Control parameters
u, v = 0.0, 0.0
error, error_next = 0.0, 0.0
gamma = 0.5  # adaptive gain
epsilon = 0.5  # control gain
alpha_eta = 0.1  # learning rate for eta

# --- Start Experiment ---
print("Experiment Starts. Press Ctrl+C to stop.\n")
f = open("data.txt", "w")

try:
    start_time = time.perf_counter()
    last_sample_time = start_time
    next_sample_time = start_time + Ts # Initialize the target time
    
    while t < 20:
        # 1. Busy until the next sample time
        while time.perf_counter() < next_sample_time:
            pass
        
        # 2. Calculate actual dt
        current_time = time.perf_counter()
        dt = current_time - last_sample_time 
        last_sample_time = current_time

        # 3. Calculate absolute time since start
        t = current_time - start_time

        # 4. Schedule next loop
        next_sample_time += Ts 
        
        # 5. Read Encoder
        ser.write(bytes([3])) 
        line = ser.readline().decode('utf-8').strip()
        
        if not line:
            continue
        try:
            raw = int(line)
        except ValueError:
            continue

        # 6. Coordinate Transformation
        theta = (np.pi * raw / TICKS)
        if theta0 is None: theta0 = theta
        theta = theta - theta0
        
        # 7. Tracking Control
        ref = 0.5*np.pi * np.sin(t) + 0.5*np.pi * np.sin(t/2)
        error = error_next
        error_next = theta - ref

        # Adaptive Control Law 
        u = - gamma*error + v
        v = (1-epsilon)*v - alpha_eta*error_next

        # 8. Command Arduino
        u_sat = np.clip(u, -1.0, 1.0)
        pwm = int(255 * abs(u_sat))
        target_sel = 1 if u_sat > 0 else 2
        ser.write(bytes([target_sel, pwm]))

        # 9. Metrics & Logging
        ISE += dt * error**2
        ISC += dt * u**2
        f.write(f"{t:.4f}\t{error:.4f}\t{theta:.4f}\t{ref:.4f}\t{u:.4f}\t{dt:.4f}\n")

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        try:
            ser.write(bytes([1, 0])) 
            time.sleep(0.05)
            ser.write(bytes([4]))    
        except:
            pass
        ser.close()
    f.close()
    print(f"Results: ISE={ISE:.4f}, ISC={ISC:.4f}")
    print("Connection closed. Purrfect! =^..^=")

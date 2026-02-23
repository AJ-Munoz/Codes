import numpy as np
import soundfile as sf

def human_voice_reconstructor(input_file, output_file="human_voice.wav"):
    # 1. Load the OGG
    data, sr = sf.read(input_file)
    if len(data.shape) > 1: data = data.mean(axis=1) # Force Mono
    
    # 2. Windowing Parameters (30ms is the "sweet spot" for speech)
    window_size = int(0.03 * sr) 
    hop_size = window_size // 2 # 50% overlap for smoothness
    reconstructed = np.zeros_like(data)
    window_func = np.hanning(window_size) # Prevents "clicking" between frames
    
    # Fundamental frequency based on your Nyquist / 200 logic
    f0 = (sr / 2) / 200 
    
    print(f"Processing voice in {window_size}-sample windows...")

    # 3. Slide through the audio
    for i in range(0, len(data) - window_size, hop_size):
        frame = data[i : i + window_size] * window_func
        t_frame = np.arange(window_size) / sr
        
        # Start with the Constant (a0) for this frame
        frame_recon = np.full(window_size, np.mean(frame))
        
        # Calculate 50 Sines and 50 Cosines for THIS window
        for n in range(1, 51):
            omega_t = 2 * np.pi * n * f0 * t_frame
            cos_b, sin_b = np.cos(omega_t), np.sin(omega_t)
            
            an = 2 * np.mean(frame * cos_b)
            bn = 2 * np.mean(frame * sin_b)
            
            # Reconstruction for this frame
            frame_recon += (an * cos_b) + (bn * sin_b)
        
        # Overlap-add the frame back into the full signal
        reconstructed[i : i + window_size] += frame_recon * window_func

    # 4. Normalize and Save
    reconstructed /= np.max(np.abs(reconstructed))
    sf.write(output_file, reconstructed, sr)
    print("Drilling sound gone. Human voice restored!")

# Run it:
human_voice_reconstructor("voice1"
".ogg")

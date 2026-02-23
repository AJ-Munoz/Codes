import numpy as np
import soundfile as sf

# 1. Setup
SECRET_KEY = 42 
WIN_SIZE = 1024
input_audio = "voice1.ogg"

def get_strong_key(seed, size):
    np.random.seed(seed)
    # Magnitude mask (Multiplicative pollution)
    mag_key = np.random.uniform(0.5, .0, size) 
    # Phase mask (Rotational pollution)
    phase_key = np.random.uniform(0, 2 * np.pi, size)
    # Shuffling index (Positional pollution)
    idx = np.arange(size)
    np.random.shuffle(idx)
    return mag_key, phase_key, idx

# 2. Load
data, sr = sf.read(input_audio)
if len(data.shape) > 1: data = data.mean(axis=1)
pad = WIN_SIZE - (len(data) % WIN_SIZE)
data = np.pad(data, (0, pad))

chunks = data.reshape(-1, WIN_SIZE)
num_freqs = (WIN_SIZE // 2) + 1
mag_k, phase_k, shuffle_idx = get_strong_key(SECRET_KEY, num_freqs)

# 3. ENCRYPT
print("Polluting Spectrum...")
encrypted_audio = []
for chunk in chunks:
    spec = np.fft.rfft(chunk)
    
    # A: Scramble Magnitudes & Rotate Phases
    polluted = spec * mag_k * np.exp(1j * phase_k)
    
    # B: Shuffle the bins (The "Secret Sauce" to kill the voice)
    shuffled = polluted[shuffle_idx]
    
    encrypted_audio.append(np.fft.irfft(shuffled, n=WIN_SIZE))

sf.write("encrypted.wav", np.concatenate(encrypted_audio).astype(np.float32), sr)

# 4. DECRYPT
print("Restoring...")
noisy_data, _ = sf.read("encrypted.wav")
noisy_chunks = noisy_data.reshape(-1, WIN_SIZE)
unshuffle_idx = np.argsort(shuffle_idx) # The reverse map
decrypted_audio = []

for chunk in noisy_chunks:
    spec = np.fft.rfft(chunk)
    
    # A: Unshuffle
    unshuffled = spec[unshuffle_idx]
    
    # B: Reverse Magnitude/Phase pollution
    restored = unshuffled / (mag_k * np.exp(1j * phase_k))
    
    decrypted_audio.append(np.fft.irfft(restored, n=WIN_SIZE))

final = np.concatenate(decrypted_audio)
sf.write("decrypted_clean.wav", final.astype(np.float32), sr)
print("Done. Check 'encrypted.wav' now!")

import wave
import math
import struct

def generate_beep(filename="alarm.wav", duration=1.0, frequency=440.0, sample_rate=44100):
    """Generates a simple sine wave beep sound."""
    num_samples = int(duration * sample_rate)
    
    # Open a wave file
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: 1 channel, 2 bytes per sample, sample rate, num samples, COMPRESSION_NONE
        wav_file.setparams((1, 2, sample_rate, num_samples, "NONE", "Uncompressed"))
        
        for i in range(num_samples):
            # Generate sine wave sample
            sample = math.sin(2.0 * math.pi * frequency * i / sample_rate)
            # Scale to 16-bit integer range
            sample_int = int(sample * 32767.0)
            # Pack as 16-bit short (little-endian)
            data = struct.pack('<h', sample_int)
            wav_file.writeframesraw(data)

if __name__ == "__main__":
    generate_beep("alarm.wav", duration=0.5, frequency=1000.0) # 1000Hz beep for 0.5 seconds
    print("Generated alarm.wav")

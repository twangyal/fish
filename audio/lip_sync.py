import math
import struct

def calculate_mouth_envelope(pcm_chunk: bytes) -> float:
    """
    Calculates the RMS amplitude of a 16-bit PCM audio chunk,
    returning a normalized float between 0.0 and 1.0.
    """
    if not pcm_chunk:
        return 0.0

    # Ensure we process complete 16-bit samples (2 bytes each)
    num_samples = len(pcm_chunk) // 2
    if num_samples == 0:
        return 0.0
        
    try:
        samples = struct.unpack(f"<{num_samples}h", pcm_chunk[:num_samples*2])
    except struct.error:
        return 0.0

    sum_squares = sum(float(sample) ** 2 for sample in samples)
    rms = math.sqrt(sum_squares / num_samples)
    
    # Normalize (max value for 16-bit audio is 32768)
    # Using a slightly lower max for better responsiveness
    normalized = min(rms / 20000.0, 1.0)
    
    return normalized

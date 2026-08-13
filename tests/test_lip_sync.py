import struct
from audio.lip_sync import calculate_mouth_envelope

def test_lip_sync_silence():
    # 10 samples of silence (16-bit PCM = 2 bytes per sample, 20 bytes total)
    silence = b'\x00' * 20
    assert calculate_mouth_envelope(silence) == 0.0

def test_lip_sync_active():
    # 10 samples of 10000 amplitude (10000 in little-endian 16-bit =  ')
    samples = [10000] * 10
    active = struct.pack('<10h', *samples)
    envelope = calculate_mouth_envelope(active)
    
    # RMS of 10000 is 10000. Normalized by 20000 is 0.5.
    assert 0.49 < envelope < 0.51

def test_lip_sync_max():
    # Values over 20000 should clip to 1.0
    samples = [30000] * 10
    active = struct.pack('<10h', *samples)
    envelope = calculate_mouth_envelope(active)
    
    assert envelope == 1.0

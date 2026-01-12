import pvorca
import numpy as np
import sounddevice as sd
import constants
import queue
import threading

ACCESS_KEY = constants.ACCESS_KEY


# Initialize Orca TTS
orca = pvorca.create(
    access_key=ACCESS_KEY,
)

# Global stream variable
stream = None

def play_pcm(pcm_samples):
    """Play 16-bit PCM samples immediately."""
    # Convert to float32 in range [-1, 1] for sounddevice
    print(pcm_samples)
    audio = np.array(pcm_samples, dtype=np.int16).astype(np.float32) / 32768.0
    print(audio)
    sd.play(audio, samplerate=orca.sample_rate)
    sd.wait()

def tts_streaming_step(text_chunk):
    """Feed partial text to Orca and play immediately."""
    global stream
    if stream is None:
        stream = orca.stream_open()
    pcm = stream.synthesize(text_chunk)  # may return None if not enough text
    if pcm is not None:
        play_pcm(pcm)

def tts_flush():
    """Flush remaining text at end."""
    global stream
    pcm = stream.flush()
    if pcm is not None:
        play_pcm(pcm)
    if stream is not None:
        stream.close()
        stream = None

def open_audio_stream():
    global stream
    if stream is None:
        stream = orca.stream_open()

def add_to_audio_queue(text_chunk):
    pcm = stream.synthesize(text_chunk)
    if pcm is not None:
        audio_queue.put(pcm)

audio_queue = queue.Queue()
def audio_worker():
    while True:
            pcm = audio_queue.get()
            if pcm is None:
                break
            audio = np.array(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=orca.sample_rate)
            sd.wait()

threading.Thread(target=audio_worker, daemon=True).start()
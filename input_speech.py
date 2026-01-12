import json
import pyaudio
import numpy as np
import pvcheetah
import constants
import llm_inference

# Load vocab
"""
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
# Reverse mapping: id -> token string
id_to_token = {v: k for k, v in vocab.items()}
"""
# Initialize Cheetah
cheetah = pvcheetah.create(access_key=constants.ACCESS_KEY, endpoint_duration_sec=1,enable_automatic_punctuation=True)

# Initialize PyAudio
pa = constants.pa

def get_next_audio_frame(stream):
    """Read next audio chunk and normalize."""
    try:
        pcm = stream.read(cheetah.frame_length, exception_on_overflow=False)
    except IOError:
        # If overflow occurs, return silence
        return np.zeros(cheetah.frame_length, dtype=np.int16)

    pcm_array = np.frombuffer(pcm, dtype=np.int16)
    return pcm_array

def listen():
    stream = pa.open(
        rate=cheetah.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=cheetah.frame_length
    )
    print("Listening... Press Ctrl+C to stop.")
    last_partial = ""
    try:
        while True:
            frame = get_next_audio_frame(stream)
            partial_transcript, is_endpoint = cheetah.process(frame)
            if partial_transcript != last_partial and partial_transcript != "":
                print("Partial:", partial_transcript)
                last_partial = partial_transcript
                llm_inference.prompt_accumulation(partial_transcript)
            if is_endpoint:
                final_transcript = cheetah.flush()
                if final_transcript:
                    print("Final:", final_transcript)
                    llm_inference.generate_res1(final_transcript)
                break # reset for next segment

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Cleaning up...")
        stream.stop_stream()
        stream.close()

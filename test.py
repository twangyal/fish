import pvorca
import sounddevice as sd
import numpy as np
import picollm
import constants

pllm = picollm.create(
    access_key=constants.ACCESS_KEY,
    model_path=constants.MODEL_PATH)

orca = pvorca.create(
    access_key=constants.ACCESS_KEY,
)

def play_pcm(pcm):
    audio = np.array(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=orca.sample_rate)
    sd.wait()

def speak_llm_response(prompt: str):
    stream = orca.stream_open()

    def on_partial(text_chunk: str):
        pcm = stream.synthesize(text_chunk)
        if pcm:
            play_pcm(pcm)

    result = pllm.generate(
        prompt=prompt,
        temperature=0.6,
        stream_callback=on_partial
    )

    # Flush remaining speech
    pcm = stream.flush()
    if pcm:
        play_pcm(pcm)

    stream.close()

# Example
speak_llm_response("Explain quantum computing simply.")
orca.delete()

import pvporcupine
import pyaudio
import struct
import constants
import input_speech

ACCESS_KEY = constants.ACCESS_KEY

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=["computer"]  # default wake word
)

pa = constants.pa

stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word...")

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from(
            "h" * porcupine.frame_length, pcm
        )

        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            input_speech.listen()

except KeyboardInterrupt:
    print("Stopping...")

finally:
    print("Stopped.")
    stream.close()
    pa.terminate()
    porcupine.delete()

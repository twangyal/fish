import pyaudio
import os
from dotenv import load_dotenv
load_dotenv()

ACCESS_KEY = os.getenv("ACCESS_KEY")
MODEL_PATH = "./models/phi2-487.pllm"
pa = pyaudio.PyAudio()

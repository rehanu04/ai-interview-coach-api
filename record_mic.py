import wave, numpy as np, sounddevice as sd

FS = 16000
SECONDS = 40
OUT = r"C:\Users\rehan\Desktop\answer.wav"

print(f"Recording {SECONDS}s... Speak now.")
audio = sd.rec(int(SECONDS*FS), samplerate=FS, channels=1, dtype="float32")
sd.wait()

pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

with wave.open(OUT, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(FS)
    wf.writeframes(pcm16.tobytes())

print("Saved:", OUT)

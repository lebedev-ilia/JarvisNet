from functions import inference_model
import pyaudio
import wave
from classes import classes

def main():
  
  chunk = 1024
  FORMAT = pyaudio.paInt16
  channels = 1
  sample_rate = 16000
  record_seconds = 4
  
  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT,
                  channels=channels,
                  rate=sample_rate,
                  input=True,
                  output=True,
                  frames_per_buffer=chunk)
  
  frames = []
  
  print("Listen...")
  
  for i in range(int(sample_rate / chunk * record_seconds)):
      data = stream.read(chunk)
      frames.append(data)
      
  stream.stop_stream()
  stream.close()
  p.terminate()
  wf = wave.open('audio.wav', "wb")
  wf.setnchannels(channels)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(sample_rate)
  wf.writeframes(b"".join(frames))

  num_class = inference_model(file_path='/Users/user/Desktop/jarvis/audio.wav', tuple_audio=None, checkpoint_path='/Users/user/Desktop/jarvis/my_model_checkpoint_last.pt', num_class=3)
  
  print(num_class)
  
if __name__ == '__main__':
  main()
  
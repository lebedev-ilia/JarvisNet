import pyaudio
import wave
from random import randint
import time
import pandas as pd
import os

_METADATA = '/Users/user/Desktop/jarvis/jarvis/metadata.csv'
_FILENAME = "jarvis_dataset/data/voice_{indx}.wav"
_DATADIR = '/Users/user/Desktop/jarvis/'

def add_string(file_name: str, _class: str, metadata):
  
  indx = len(metadata['path'])

  metadata.loc[indx] = file_name, _class

def create_voice():
  
  global _FILENAME
  
  if (os.listdir(_DATADIR + _FILENAME[:11])) != []:
    indx = len((os.listdir(_DATADIR + _FILENAME[:11]))) + 1
  else: indx = 1
  
  metadata = pd.read_csv(_METADATA, index_col=0)
  
  chunk = 1024
  FORMAT = pyaudio.paInt16
  channels = 1
  sample_rate = 16000
  record_seconds = randint(3, 7)
  
  path = _FILENAME.format(indx=indx)
  
  print(f"Seconds = {record_seconds}")
    
  time.sleep(2)

  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT,
                  channels=channels,
                  rate=sample_rate,
                  input=True,
                  output=True,
                  frames_per_buffer=chunk)
  frames = []
  print("Recording...")
  for i in range(int(sample_rate / chunk * record_seconds)):
      data = stream.read(chunk)
      frames.append(data)
      
  stream.stop_stream()
  stream.close()
  p.terminate()
  wf = wave.open(path, "wb")
  wf.setnchannels(channels)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(sample_rate)
  wf.writeframes(b"".join(frames))
  wf.close()
  
  _class = input('Enter class: ')
  file_name = path[7:]
  
  add_string(file_name, _class, metadata)
  
  metadata.to_csv(_METADATA)
  
  _, num_voice = os.path.split(_FILENAME.format(indx=indx))
  
  print()
  print('------------------------')
  print(f'{num_voice} - Ok!')
  print('------------------------')
  print()
  
  indx += 1
    
while 1:
  create_voice()


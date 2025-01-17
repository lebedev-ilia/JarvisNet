import pyttsx3
import time
from g4f.client import Client

client = Client()

response = client.images.generate(
  model='flux',
  prompt='доберман',
  response_format='url',
)

img_url = response.data[0].url


# engine = pyttsx3.init()

# engine.say(response.choices[0].message.content)

# engine.runAndWait()
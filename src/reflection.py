# A script to build an agent capable to self-reflect.
#
# The goal is to get Gemini to *consistently* return correctly
# extracted information from the SAFO booking system.
from vertexai.generative_models import GenerativeModel, Part
import google.generativeai as genai

image_path = "screenshots/79.jpg" # an image showing available/booked tennis courts

image_file = genai.upload_file(path=image_path)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
response = model.generate_content([prompt, image_file], request_options={"timeout": 120})
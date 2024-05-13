from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to(device)

# prepare image and text prompt, using the appropriate prompt template
url = "/home/wl2927/data/accident/test_33.jpg"
image = Image.open(url)
# prompt = "[INST] <image>\nDescribe the image with details.[/INST]\n"
# prompt = "[INST] <image>\nDescribe the image with details. If there is an accident, describe it without using the word \"accident\".[/INST]\n"
prompt = "[INST] <image>\nIf there is an accident, describe the accident without using the word \"accident\" and focusing on the environment of the road, like the distance of the items. Otherwise, describe the image with details. [/INST]\n"

inputs = processor(prompt, image, return_tensors="pt").to(device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000)

print(processor.decode(output[0], skip_special_tokens=True))

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from pathlib import Path
import pandas as pd

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")
prompt = "[INST] <image>\nDescribe the image with details. [/INST]"
# prompt = "[INST] <image>\nIf there is an accident, describe the accident without using the word \"accident\". Otherwise, describe the image with details. [/INST]"
# prompt = "[INST] <image>\nThere is an accident or there is an accident about happening in the image. Describe the image focusing on the road condition, distance between cars and other things which might be involved in an accident and whether the cars or people obey the rules. Don't use the word \"accident\". [/INST]\n"

# prepare image and text prompt, using the appropriate prompt template
def generate(url, prompt = prompt, processor = processor, model = model):
    # url = "/home/wl2927/data/train/Accident/test_29.jpg"
    # global prompt, processor, model
    image = Image.open(url)

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=500)

    res = processor.decode(output[0], skip_special_tokens=True).split('[/INST]')[1]
    res = res.replace("\n", "")
    return res

cato = ["accident", "nonaccident"]
for it in cato:
    prefix = f'/home/wl2927/data/{it}/'
    images = []
    pathlist = Path(prefix).rglob('*')
    for path in pathlist:
        images.append(str(path.resolve()))
    total = len(images)
    res_path= f'/home/wl2927/description/{it}_0.csv'
    count = 0
    for url in images:
        res = generate(url)
        name = url.split('/')[-1]
        res = res.replace('"','')
        df = pd.DataFrame({
            'image':[name],
            'description':[res]
        })
        df.to_csv(res_path, sep='$', mode='a', header=False, index=False)
        count += 1
        print(f"{count}/{total}: {name}")


from torch.utils.data import Dataset, DataLoader
import clip
import torch,torchvision
from PIL import Image, ImageFile
import torch.nn as nn
import shutil
import numpy as np
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

import os
import sys
import json
sys.path.append(os.getcwd())

import contextlib
import io

ASSETS_PATH = "assets"

from vae import prepare_image, encode, decode_latents

def jpeg_compressibility(images):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    pil_images = [Image.fromarray(image) for image in images]

    sizes = []
    with contextlib.ExitStack() as stack:
        buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / 1000)  # Size in kilobytes
    
    return -np.array(sizes)

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            # print(filename)
            path = os.path.join(folder, filename)
            images.append(path)
    return images

# Set up the device for GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load images from the folder
folder = './classifier_free/train'
images = load_images_from_folder(folder)
print("Number of images:", len(images))

### Generate CLIP Embeddings
# model, preprocess = clip.load("ViT-L/14", device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip.to(device)
clip.requires_grad_(False)
clip.eval()

eval_model = MLPDiff().to(device)
eval_model.requires_grad_(False)
eval_model.eval()
s = torch.load(os.path.join(ASSETS_PATH,"sac+logos+ava1-l14-linearMSE.pth"), map_location=device)   # load the model you trained previously or the model available in this repo
eval_model.load_state_dict(s)

x = []

strings = []
failed_images = []
# encoded_images = []
c = 0

data_list = []

for img_path in tqdm(images, desc="Processing images"):
    try:
        # image_input = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        raw_img = Image.open(img_path)  # classifier-free/train/2_fox.png
        
        caption = img_path.split('/')[-1].split('_')[-1].split('.')[0]
        # print("Caption:", caption)

        inputs = processor(images=raw_img, return_tensors="pt")
        with torch.no_grad():

            # Get CLIP embeddings
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = clip.get_image_features(**inputs)
            embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
            
            real_aes_score = eval_model(embeddings).to(device)
        
        aes_score = f"{real_aes_score.item():.2f}"
        
        # get the compressibility score
        raw_img = raw_img.convert('RGB')
        transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])
        img_tensor = transform(raw_img)
                
        real_comp_score = jpeg_compressibility(img_tensor.unsqueeze(0))
        comp_score = f"{real_comp_score.item():.2f}"
        
        
        file_name = img_path.split('/')[-1]
        
        # Create a new dictionary for each row
        data_dict = {
            "file_name": file_name,
            "text": caption,
            'aes_score': aes_score,
            'comp_score': comp_score,
        }
        
        data_list.append(data_dict)

        
    except Exception as e:
        c -= 1
        print(f"Error processing image {img_path}: {e}")
        failed_images.append(img_path)
        continue

jsonl_file_path = './classifier_free/train/metadata.jsonl'
with open(jsonl_file_path, mode='w', encoding='utf-8') as json_file:
    for json_object in data_list:
        json_file.write(json.dumps(json_object) + "\n")

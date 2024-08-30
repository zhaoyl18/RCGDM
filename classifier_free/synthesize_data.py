import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline

# Load the prompts from the file
with open('assets/simple_animals.txt', 'r') as file:
    animal_prompts = [line.strip() for line in file.readlines()]

# Initialize the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    safety_checker=None
)
pipeline = pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=True)

# Set the random seed for reproducibility
generator = torch.Generator(device='cuda').manual_seed(42)
generator_cpu = torch.Generator(device='cpu').manual_seed(42)

# Create a directory to save the images
output_dir = Path("./classifier_free/train_new")
output_dir.mkdir(exist_ok=True)

def sample_prompt(generator, prompts):
    index = torch.randint(0, len(prompts), (1,), generator=generator).item()
    return prompts[index]

# Generate images
for i in tqdm(range(10240), desc="Generating images"):
    prompt = sample_prompt(generator_cpu, animal_prompts)  # Uniformly sample a prompt
    with torch.autocast("cuda"):
        image = pipeline(prompt, num_inference_steps=50, generator=generator).images[0]
    image.save(output_dir / f"{i}_{prompt}.png", format="PNG")

print(f"Generated 10240 images in {output_dir}.")

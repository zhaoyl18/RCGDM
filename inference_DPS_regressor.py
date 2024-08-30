
from sd_pipeline import DPS_continuous_SDPipeline
from diffusers import DDIMScheduler
import torch
import numpy as np
import random
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import AVACompressibilityDataset, AVACLIPDataset
from vae import encode
import os
from aesthetic_scorer import SinusoidalTimeMLP, MLPDiff
import wandb
import argparse
from tqdm import tqdm
import datetime
from compressibility_scorer import CompressibilityScorerDiff, jpeg_compressibility
from aesthetic_scorer import AestheticScorerDiff


def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", type=str, default='compressibility')
    parser.add_argument("--guidance", type=float, default=10)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--val_bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
device= args.device
save_file = True

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    shape = (args.num_images//args.bs, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

run_name = f"{args.reward}_guidance={args.guidance}"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project=f"DPS-continuous-{args.reward}", name=run_name,config=args)


sd_model = DPS_continuous_SDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)

# switch to DDIM scheduler
sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(50, device=device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()

if args.reward == 'compressibility':
    scorer = CompressibilityScorerDiff(dtype=torch.float32).to(device)
elif args.reward == 'aesthetic':
    scorer = AestheticScorerDiff(dtype=torch.float32).to(device)
else:
    raise ValueError("Invalid reward")

scorer.requires_grad_(False)
scorer.eval()

sd_model.setup_scorer(scorer)
# sd_model.set_target(args.target)
sd_model.set_reward(args.reward)
sd_model.set_guidance(args.guidance)

### introducing evaluation prompts
import prompts as prompts_file
eval_prompt_fn = getattr(prompts_file, 'eval_aesthetic_animals')


image = []
eval_prompt_list = []
KL_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images"):
    wandb.log(
        {"inner_iter": i}
    )
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    eval_prompts, _ = zip(
        *[eval_prompt_fn() for _ in range(args.bs)]
    )
    eval_prompts = list(eval_prompts)
    eval_prompt_list.extend(eval_prompts)
    
    image_, kl_loss = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    image.extend(image_)
    KL_list.append(kl_loss)

KL_entropy = torch.mean(torch.stack(KL_list))

assert len(image) == len(eval_prompt_list)

###### evaluation and metric #####
if args.reward == 'compressibility':
    gt_dataset= AVACompressibilityDataset(image)
elif args.reward == 'aesthetic':
    from importlib import resources
    ASSETS_PATH = resources.files("assets")
    eval_model = MLPDiff().to(device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device, weights_only=True)
    eval_model.load_state_dict(s)
    gt_dataset= AVACLIPDataset(image)    
    
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

with torch.no_grad():
    eval_rewards = []
    
    for inputs in gt_dataloader:
        inputs = inputs.to(device)

        if args.reward == 'compressibility':
            jpeg_compressibility_scores = jpeg_compressibility(inputs)
            scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)
        
        elif args.reward == 'aesthetic':
            scores = eval_model(inputs)
            scores = scores.squeeze(1)
        
        eval_rewards.extend(scores.tolist())

    eval_rewards = torch.tensor(eval_rewards)

    print("KL-entropy: ", KL_entropy)
    print(f"eval_{args.reward}_rewards_mean", torch.mean(eval_rewards))

    
    wandb.log({
        f"eval_{args.reward}_rewards_mean": torch.mean(eval_rewards),
    })
    wandb.log({"KL-entropy": KL_entropy })

if save_file:
    images = []
    log_dir = os.path.join(args.out_dir, "eval_vis")
    os.makedirs(log_dir, exist_ok=True)
    np.save(f"{args.out_dir}/scores.npy", eval_rewards)

    # Function to save array to a text file with commas
    def save_array_to_text_file(array, file_path):
        with open(file_path, 'w') as file:
            array_str = ','.join(map(str, array.tolist()))
            file.write(array_str + ',')

    # Save the arrays to text files
    save_array_to_text_file(eval_rewards, f"{args.out_dir}/eval_rewards.txt")
    print("Arrays have been saved to text files.")
    
    for idx, im in enumerate(image):
        prompt = eval_prompt_list[idx]
        reward = eval_rewards[idx]
        
        im.save(f"{log_dir}/{idx:03d}_{prompt}_score={reward:2f}.png")
        
        pil = im.resize((256, 256))

        images.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f}"))

    wandb.log(
        {"images": images}
    )
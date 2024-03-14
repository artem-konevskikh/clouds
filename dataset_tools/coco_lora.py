import os
import tqdm

import cv2
import numpy as np

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid

from pycocotools.coco import COCO


device = "mps"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
cloud_lora_id = "joachimsallstrom/aether-cloud-lora-for-sdxl"

controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
).to(device)
pipe.enable_attention_slicing()
pipe.load_lora_weights(cloud_lora_id)
pipe.fuse_lora()


def get_masked_image(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def get_canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def generate_image(
    prompt: str,
    canny_image,
    controlnet_conditioning_scale: float = 0.5,
    num_inference_steps: int = 40,
    guidance_scale: float = 0,
    eta: float = 0.3,
):
    image = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]
    return image


def process_coco_dataset(coco_json_path, img_dir, labels_dir):
    pass


for subs in ["train", "val"]:
    print(f"Processing {subs} data...")
    coco_json_path = f"annotations/instances_{subs}2017.json"
    output_dir = f"data-filtered/{subs}"
    img_dir = f"{output_dir}/images"
    labels_dir = f"{output_dir}/labels"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    process_coco_dataset(coco_json_path, img_dir, labels_dir)

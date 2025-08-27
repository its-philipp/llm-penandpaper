import os
import time
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
IMAGES_DIR = "generated_images"
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

def detect_device():
    """Detect the best available device for image generation."""
    # Check for AMD ROCm (AMD GPU support)
    if hasattr(torch, 'hip') and torch.hip.is_available():
        return "cuda", torch.float16  # AMD GPUs use CUDA API via ROCm
    
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        return "cuda", torch.float16
    
    # Check for Apple Metal
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # MPS requires float32
    
    # Fallback to CPU
    return "cpu", torch.float32

# Device detection
DEVICE, TORCH_DTYPE = detect_device()
logging.info(f"Selected device: {DEVICE} with dtype: {TORCH_DTYPE}")

# Global pipeline variable
pipeline = None

def initialize_pipeline():
    """Initialize the Stable Diffusion pipeline with device-specific optimizations."""
    global pipeline
    if pipeline is not None:
        return True

    logging.info(f"Loading Stable Diffusion model '{DEFAULT_MODEL}' onto {DEVICE}...")

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(DEVICE)
        pipeline.enable_attention_slicing()
        
        # Device-specific optimizations
        if DEVICE == "mps":
            # MPS-specific fixes
            if hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention(False)
            pipeline.generator = torch.Generator(device="cpu")
        elif DEVICE == "cuda":
            # CUDA/AMD optimizations
            if hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    logging.info("xformers not available, using standard attention")

        logging.info(f"Model loaded successfully on {DEVICE}!")
        return True

    except Exception as e:
        logging.error(f"Error loading model on {DEVICE}: {e}")
        pipeline = None
        return False

def summarize_and_clean_prompt(description):
    """Clean and enhance the prompt with style keywords."""
    style_prompt = "fantasy artwork, digital painting, epic, atmospheric, detailed, cinematic lighting"
    return f"{style_prompt}, {description}"

def _fix_nan_values(image):
    """Fix NaN values in the image tensor before saving."""
    if image is None:
        return image
    
    # Convert to numpy array if it's a PIL image
    if hasattr(image, 'convert'):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Check for NaN values and replace them
    if np.isnan(img_array).any():
        logging.warning("Warning: NaN values detected in image, fixing...")
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to PIL if needed
        if hasattr(image, 'convert'):
            image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

def generate_scene_image(scene_description, width=512, height=512):
    """Generate an image using the initialized Stable Diffusion pipeline."""
    if not initialize_pipeline():
        logging.warning("Failed to initialize the model. Generating a fallback image.")
        return generate_simple_fallback_image(scene_description)

    prompt = summarize_and_clean_prompt(scene_description)
    negative_prompt = "blurry, low quality, deformed, bad anatomy, watermark, signature, text, nsfw"

    logging.info(f"Generating image with prompt: \"{prompt}\"")

    try:
        # Device-specific generator setup
        if DEVICE == "mps":
            generator = torch.Generator(device="cpu")
        else:
            generator = torch.Generator(device=DEVICE)
        
        generator.manual_seed(int(time.time()))

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=20,
                guidance_scale=7.0,
                generator=generator
            )
            
            image = result.images[0]
            image = _fix_nan_values(image)

        # Validate and save image
        if not _validate_image(image):
            logging.warning("Generated image appears invalid, but attempting to save anyway...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scene_{timestamp}_{DEVICE}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        os.makedirs(IMAGES_DIR, exist_ok=True)
        image.save(filepath)
        logging.info(f"Image saved to: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"An error occurred during image generation on {DEVICE}: {e}")
        logging.warning("Generating a fallback image.")
        return generate_simple_fallback_image(scene_description)
    finally:
        # Clear cache based on device
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

def generate_simple_fallback_image(scene_description):
    """Generate a simple placeholder image when the full model fails."""
    width, height = 512, 512
    
    # Create gradient background
    image = Image.new('RGB', (width, height), color='#1a1a2e')
    draw = ImageDraw.Draw(image)
    
    # Create gradient effect
    for y in range(height):
        r = int(26 + (y / height) * 30)
        g = int(26 + (y / height) * 20)
        b = int(46 + (y / height) * 40)
        color = (r, g, b)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Add decorative border
    draw.rectangle([10, 10, width-10, height-10], outline='#ffd700', width=3)
    draw.rectangle([20, 20, width-20, height-20], outline='#8b4513', width=2)
    
    # Add title
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    title = "Fantasy RPG Scene"
    title_bbox = draw.textbbox((0, 0), title, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    
    # Title with shadow
    draw.text((title_x+1, 51), title, fill='#000000', font=font_large)
    draw.text((title_x, 50), title, fill='#ffd700', font=font_large)
    
    # Add scene description
    desc = f"Scene: {scene_description[:100]}..."
    lines = textwrap.wrap(desc, width=40)
    y_pos = 100
    
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font_small)
        line_width = line_bbox[2] - line_bbox[0]
        x = (width - line_width) // 2
        draw.text((x+1, y_pos+1), line, fill='#000000', font=font_small)
        draw.text((x, y_pos), line, fill='#ffffff', font=font_small)
        y_pos += 20
    
    # Add decorative elements
    # Moon
    draw.ellipse([50, 350, 100, 400], fill='#ffd700', outline='#8b4513', width=2)
    draw.ellipse([60, 360, 90, 390], fill='#fff8dc', outline='#ffd700', width=1)
    
    # Castle silhouette
    draw.rectangle([350, 250, 400, 400], fill='#2f2f2f', outline='#000000', width=2)
    draw.rectangle([360, 230, 390, 250], fill='#2f2f2f', outline='#000000', width=2)
    draw.rectangle([370, 210, 380, 230], fill='#2f2f2f', outline='#000000', width=2)
    
    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scene_fallback_{timestamp}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    image.save(filepath)
    logging.info(f"Fallback image created: {filepath}")
    return filepath

def _validate_image(image):
    """Validate image quality and basic properties."""
    if image is None:
        return False
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Check for completely black or white images
        if img_array.shape[0] == 0 or img_array.shape[1] == 0:
            return False
        
        # Check if image is mostly one color (very low variance)
        variance = np.var(img_array)
        if variance < 100:
            return False
        
        # Check for reasonable brightness
        mean_brightness = np.mean(img_array)
        if mean_brightness < 10 or mean_brightness > 245:
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False

if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    test_description = "As dawn breaks over the Grand Midsummer Festival in Valecross, you find yourself standing atop a bustling platform overlooking the vibrant city. The cobblestone streets below teem with revelers, their laughter and joyful cries carried on the wind. To your left, the grandeur of King Alaric's castle stands proudly, its towering spires piercing the morning sky. On your right, the lush gardens of Queen Aelinthra's palace unfurl like a verdant carpet adorned with exotic flowers and delicate fountains."
    image_path = generate_scene_image(test_description)
    if image_path:
        logging.info(f"\nTest finished. Image path: {image_path}")
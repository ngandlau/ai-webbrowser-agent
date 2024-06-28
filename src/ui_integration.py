import base64
from datetime import datetime
import io
import os
import re
from PIL import Image
import anthropic
import base64
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import json
from pathlib import Path
import pdb
import re
import time
from typing import Annotated, Any, Callable, Literal
from litellm import completion
import os
from dotenv import load_dotenv
from playwright.sync_api import Page, sync_playwright
from termcolor import colored

from src.utils import convert_function_to_openai_tool, create_user_message, encode_image, make_screenshot

# Replace with your actual API key
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_description(text):
    match = re.search(r'DESCRIPTION:\s*(.*)', text)
    return match.group(1).strip() if match else None

def extract_result(text):
    match = re.search(r'(?<=RESULT: )\d+', text)
    return int(match.group()) if match else None

def prompt_claude_with_images(images, prompt, max_tokens=250):
    print(f"""Prompt: 
          {prompt}
          """)
    try:
        image_contents = []
        for image in images:
            # Convert the image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            image_contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_str
                }
            })

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        *image_contents
                    ]
                }
            ]
        )
    except anthropic.APIError as e:
        print(f"API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
    print(f"Response content: {message.content[0].text}")
    return message.content[0].text

def add_rectangles_and_numbers_to_image(masks, coordinates):
    if len(masks) == 0:
        return
    
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for i, mask in enumerate(sorted_masks, 1):
        bbox = mask['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                         fill=False, edgecolor='red', linewidth=0.2)
        ax.add_patch(rect)
        
        # Add number using pre-calculated positions
        cx, cy = coordinates[i]['x'], coordinates[i]['y']
        ax.text(cx, cy, str(i), color='red', fontsize=10, 
                ha='center', va='center')

def draw_rectangles_and_save_image(image_path, masks, coordinates):
    image = Image.open(image_path)

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    add_rectangles_and_numbers_to_image(masks, coordinates)
    plt.axis('off')

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ressources/output/segmented_image_boxes_{current_time}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    return filename


def ask_for_target_coordinates_for_segmented_image(segmented_image_path, task, coordinates):
    start_prompt = f"""Describe the image in a few short sentences. Your goal is to solve the following task: {task}. 
    Where do you have to click next to solve the task? Explain your reasoning. 
    Return a description of the location you have to click next. End with: DESCRIPTION: <description>."""

    print("Start iterations:")
    start_image = Image.open(segmented_image_path)
    start_message = prompt_claude_with_images([start_image], start_prompt, max_tokens=600)
    print(start_message)
    target_location_description = extract_description(start_message)

    print("First iteration:")
    prompt = f"""You want to navigate to the following location on the image(s): {target_location_description} 
    The provided image(s) are overlayed with a grid. Describe briefly in 2 sentences what you see on the image. 
    Identify the grid cell and grid cell number that corresponds to the target location. 
    Explain what cell you choose and why. Return the number on the corresponding grid cell 
    at the end of your reasoning in the format: RESULT: <number>. 
    If you are not sure, return the number 0."""

    first_message = prompt_claude_with_images([start_image], prompt)
    first_result = extract_result(first_message)
    print(first_result)
    start_image.show()
    return coordinates[first_result]

def segment_image(image_path):
    sam = sam_model_registry["vit_h"](checkpoint="ressources/sam_vit_h_4b8939.pth")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize SAM
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=15,  # Higher: more detail but slower; Lower: faster but may miss small objects
        pred_iou_thresh=0.8,  # Higher: better quality masks but fewer; Lower: more masks but lower quality
        stability_score_thresh=0.5,  # Higher: more stable masks but fewer; Lower: more masks but less stable
        crop_n_layers=0,  # More layers help with large images; 0 for no cropping
        crop_n_points_downscale_factor=10,  # Higher: faster for crops but less detail; Lower: more detailed crops
        min_mask_region_area=130,  # Higher: removes small segments; Lower: keeps small details but may add noise
    )
    masks = mask_generator.generate(image)
    return masks

def calculate_number_positions(anns):
    if len(anns) == 0:
        return {}
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    positions = {}
    
    for i, ann in enumerate(sorted_anns, 1):
        bbox = ann['bbox']
        random_divisor_x = random.uniform(1, 6)
        random_divisor_y = random.uniform(3, 6)
        cx = int(bbox[0] + bbox[2] / random_divisor_x)
        cy = int(bbox[1] + bbox[3] / random_divisor_y)
        positions[i] = {'x': cx, 'y': cy}
    
    return positions

def find_target_coordinates_for_image(image_path, task):
    print("Segmenting image (this may take a while)...")
    masks = segment_image(image_path)
    print("Drawing segments and numbers...")
    coordinates = calculate_number_positions(masks)
    segmented_image_path = draw_rectangles_and_save_image(image_path, masks, coordinates)
    print("Asking for target coordinates...")
    task = "Book the field P2 at 17:00pm."
    result = ask_for_target_coordinates_for_segmented_image(segmented_image_path, task, coordinates)
    return result



# SCREENSHOT_DIR = "screenshots"
# PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
# VIMIUM_PATH = "vimium"  # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

# with sync_playwright() as p:
#     browser = p.chromium.launch_persistent_context(
#         user_data_dir=PLAYWRIGHT_USER_DATA_DIRECTORY,
#         headless=False,
#         args=[
#             f"--disable-extensions-except={VIMIUM_PATH}",
#             f"--load-extension={VIMIUM_PATH}",
#         ],
#         viewport={"width": 760, "height": 800},
#         screen={"width": 760, "height": 800},
#     )

#     # navigate to booking site
#     page = browser.new_page()
#     # page.goto("https://safo.ebusy.de")
#     page.goto("https://safo.ebusy.de/lite-module/407")
#     time.sleep(2)


#     # make a screenshot
#     page.keyboard.press("Escape")
#     page.keyboard.press("f")
#     screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
#     screenshot_base64 = encode_image(screenshot_path)
    
#     coordinates = find_target_coordinates_for_image(screenshot_path.as_posix(), "Book the field P2 at 17:00pm.")
#     print(coordinates)
#     # {'x': 25, 'y': 598}
#     page.mouse.click(coordinates['x'], coordinates['y'])
#     time.sleep(3)
#     page.keyboard.press("f")
    
    
    

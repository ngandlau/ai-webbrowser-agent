

import base64
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Literal
from playwright.sync_api import Page
from PIL import Image
import re
from openai.types.chat import ChatCompletion
import requests
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.generativeai as genai


def get_next_screenshot_number(screenshot_dir: Path) -> str:
    """Gets the next screenshot number, given 00.jpg, 01.jpg, 02.jpg, ..."""
    screenshots = [Path(file).stem for file in os.listdir(screenshot_dir)]
    if not screenshots:
        return "00"
    try:
        last = max(screenshots, key=int)
        next_number = int(last) + 1
    except ValueError:
        raise ValueError(f"Make sure that there are only images in {screenshot_dir} named like 00.jpg, 01.jpg, ...")
    return f"{next_number:02d}"

def make_screenshot(page: Page, screenshot_dir: Path) -> Path:
    """Makes a screenshot of the current page and stores it in SCREENSHOT_DIRECTORY."""
    img_path = Path(screenshot_dir, get_next_screenshot_number(screenshot_dir=screenshot_dir) + ".jpg")
    page.screenshot(path=img_path)
    print(f"saved screenshot to {img_path}.")
    return img_path

def compress_image(image_path: str | Path) -> None:
    with Image.open(image_path) as img:
        max_size = (900, 635) # corresponds to "low res mode" as described in the OpenAI docs: https://platform.openai.com/docs/guides/vision
        img.thumbnail(max_size, Image.LANCZOS)
        img.save(image_path, "JPEG")
 
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def create_user_message(prompt: str | None, image_paths: list[Path] | None) -> dict:
    content = []
    if prompt: content += [{"type": "text", "text": prompt}]
    if image_paths: content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}} for img_path in image_paths]
    return {"role": "user", "content": content}

def create_payload(user_message: dict, max_tokens: int = 300) -> dict:
    return {
        "model": "gpt-4o",
        "messages": [user_message],
        # "max_tokens": max_tokens
    }

def get_openai_response(
    api_key: str,
    payload: dict,
) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_openai_response_text(openai_response: dict | ChatCompletion) -> str:
    try: 
        if isinstance(openai_response, ChatCompletion):
            return openai_response.choices[0].message.content
        else:
            return openai_response["choices"][0]["message"]["content"]
    except:
        raise ValueError("Could not extract response text from OpenAI response", openai_response)

def parse_actor_response(text: str) -> tuple[str, str]:
    found_answer_actions: list = re.findall(r'ANSWER\("[a-zA-Z]+\)', text)
    found_click_actions: list = re.findall(r'CLICK\("[a-zA-Z]+"\)', text)
    found_type_actions: list = re.findall(r'INPUT\("[a-zA-z0-9]+"\)', text)
    found_scroll_actions: list = re.findall(r'SCROLL\("[a-zA-z0-9]+"\)', text)
    found_analyze_table_actions: list = re.findall(r'ANALYZE_TABLE\("([^"]*)"\)', text)

    if len(found_answer_actions) > 1: raise ValueError(f"Found multiple ANSWER actions in response!")
    if len(found_click_actions) > 1: raise ValueError(f"Found multiple CLICK actions in response!")
    if len(found_type_actions) > 1: raise ValueError(f"Found multiple TYPE actions in response!")
    if len(found_scroll_actions) > 1: raise ValueError(f"Found multiple SCROLL actions in response!")
    if len(found_analyze_table_actions) > 1: raise ValueError(f"Found multiple ANALYZE_TABLE actions in response!")
    if not found_answer_actions and not found_click_actions and not found_type_actions and not found_scroll_actions and not found_analyze_table_actions: raise ValueError("No Action found in response!")

    if found_answer_actions:
        answer = found_answer_actions[0].split('\"')[1]
        return ("ANSWER", answer)
    elif found_scroll_actions:
        direction = found_scroll_actions[0].split('\"')[1]
        return ("SCROLL", direction)
    elif found_click_actions:
        letters = found_click_actions[0].split('\"')[1]
        return ("CLICK", letters)
    elif found_type_actions:
        text = found_type_actions[0].split('\"')[1]
        return ("INPUT", text)
    elif found_analyze_table_actions:
        match = re.search(r'ANALYZE_TABLE\("([^"]*)"\)', text)
        if match:
            extracted_text = match.group(1)
            return ("ANALYZE_TABLE", extracted_text)
    else:
        raise ValueError("No Action found in response!")

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_filename = datetime.now().strftime('logs/application_%Y%m%d_%H%M%S.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    return logger

def get_scroll_info(page) -> dict[str, int]:
    scroll_info = page.evaluate("""
        () => {
            const body = document.body;
            const scrollHeight = body.scrollHeight;
            const clientHeight = body.clientHeight;
            return {
                scrollable: scrollHeight > clientHeight,
                scrollAmount: scrollHeight - clientHeight
            };
        }
    """)
    return {
        "is_page_scrollable": scroll_info["scrollable"],
        "scroll_amount_px": scroll_info["scrollAmount"],
        "scroll_amount_n_times_viewport": scroll_info["scrollAmount"] // page.viewport_size["height"],
    }

ToolArgument = dict["name": str, "type": str]

@dataclass
class Tool:
    name: str
    args: list[ToolArgument]
    description: str
    examples: list[str]
    def __str__(self): return f"{self.name}({', '.join([f"{arg["name"]}: {arg["type"]}" for arg in self.args])}): {self.description}. Examples: {"; ".join(self.examples)}."

# tool = Tool(
#     name="ANSWER",
#     args=[{"name": "description", "type": "string"}],
#     description="Use this action if you think you can provide a good answer to the task",
#     examples=['ANSWER("The answer to the task")', 'ANSWER("The answer to the universe is 42")'],
# )

def get_gpt_observer_response(prompt: str, image_path: Path) -> str:
    message = create_user_message(prompt=prompt, image_paths=[image_path])
    payload = create_payload(user_message=message)
    response = get_openai_response(os.getenv("OPENAI_API_KEY"), payload)
    response_text = get_openai_response_text(response)
    return response_text

def get_gemini_observer_response(prompt: str, image_path: Path) -> str:
    image_file = genai.upload_file(path=image_path)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content([prompt, image_file], request_options={"timeout": 120})
    try:
        response_text = response.text
    except ValueError as msg:
        raise ValueError(f"{msg}\n {response.candidates.safety_ratings}")
    return response_text

def get_gpt_actor_response(prompt: str, image_path: Path) -> str:
    user_message = create_user_message(prompt=prompt, image_paths=[image_path])
    payload = create_payload(user_message=user_message)
    response = get_openai_response(os.getenv("OPENAI_API_KEY"), payload)
    response_text = get_openai_response_text(response)
    return response_text

def log_response(logger, agent_type: str, prompt: str, response_text: str) -> None:
    logger.info(f"\n====== {agent_type} ======\n == PROMPT ==\n{prompt}\n== RESPONSE ==\n{response_text}\n")
    logger.info("-"*80)
    print(f"\n====== {agent_type} ======\n == PROMPT ==\n{prompt}\n== RESPONSE ==\n{response_text}\n")
    print("-"*80)
import base64
import json
import logging
import os
import pdb
import re
import typing
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Union,
    _AnnotatedAlias,
    _LiteralGenericAlias,
    get_type_hints,
)

import google.generativeai as genai
import requests
import vertexai
import yaml
from openai.types.chat import ChatCompletion
from PIL import Image
from playwright.sync_api import Page
from termcolor import colored
from vertexai.generative_models import GenerativeModel, Part


def get_next_screenshot_number(screenshot_dir: Path) -> str:
    """Gets the next screenshot number, given 00.jpg, 01.jpg, 02.jpg, ..."""
    screenshots = [Path(file).stem for file in os.listdir(screenshot_dir)]
    if not screenshots:
        return "00"
    try:
        last = max(screenshots, key=int)
        next_number = int(last) + 1
    except ValueError:
        raise ValueError(
            f"Make sure that there are only images in {screenshot_dir} named like 00.jpg, 01.jpg, ..."
        )
    return f"{next_number:02d}"


def make_screenshot(page: Page, screenshot_dir: Path) -> Path:
    """Makes a screenshot of the current page and stores it in SCREENSHOT_DIRECTORY."""
    img_path = Path(
        screenshot_dir,
        get_next_screenshot_number(screenshot_dir=screenshot_dir) + ".jpg",
    )
    page.screenshot(path=img_path)
    print(colored(f"\n<< screenshot saved to {img_path} >>\n", color="light_grey"))
    return img_path


def compress_image(image_path: str | Path) -> None:
    with Image.open(image_path) as img:
        max_size = (
            900,
            635,
        )  # corresponds to "low res mode" as described in the OpenAI docs: https://platform.openai.com/docs/guides/vision
        img.thumbnail(max_size, Image.LANCZOS)
        img.save(image_path, "JPEG")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_user_message(prompt: str | None, image_paths: list[Path] | None) -> dict:
    content = []
    if prompt:
        content += [{"type": "text", "text": prompt}]
    if image_paths:
        content += [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"},
            }
            for img_path in image_paths
        ]
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
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
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
    found_parse_table_data_actions: list = re.findall(r'PARSE_TABLE_DATA\("([^"]*)"\)', text)

    if len(found_answer_actions) > 1:
        raise ValueError(f"Found multiple ANSWER actions in response!")
    if len(found_click_actions) > 1:
        raise ValueError(f"Found multiple CLICK actions in response!")
    if len(found_type_actions) > 1:
        raise ValueError(f"Found multiple TYPE actions in response!")
    if len(found_scroll_actions) > 1:
        raise ValueError(f"Found multiple SCROLL actions in response!")
    if len(found_parse_table_data_actions) > 1:
        raise ValueError(f"Found multiple PARSE_TABLE_DATA actions in response!")
    if (
        not found_answer_actions
        and not found_click_actions
        and not found_type_actions
        and not found_scroll_actions
        and not found_parse_table_data_actions
    ):
        raise ValueError("No Action found in response!")

    if found_answer_actions:
        answer = found_answer_actions[0].split('"')[1]
        return ("ANSWER", answer)
    elif found_scroll_actions:
        direction = found_scroll_actions[0].split('"')[1]
        return ("SCROLL", direction)
    elif found_click_actions:
        letters = found_click_actions[0].split('"')[1]
        return ("CLICK", letters)
    elif found_type_actions:
        text = found_type_actions[0].split('"')[1]
        return ("INPUT", text)
    elif found_parse_table_data_actions:
        match = re.search(r'PARSE_TABLE_DATA\("([^"]*)"\)', text)
        if match:
            extracted_text = match.group(1)
            return ("PARSE_TABLE_DATA", extracted_text)
    else:
        raise ValueError("No Action found in response!")


def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_filename = datetime.now().strftime("logs/application_%Y%m%d_%H%M%S.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    return logger


def get_scroll_info(page) -> dict[str, int]:
    scroll_info = page.evaluate(
        """
        () => {
            const body = document.body;
            const scrollHeight = body.scrollHeight;
            const clientHeight = body.clientHeight;
            return {
                scrollable: scrollHeight > clientHeight,
                scrollAmount: scrollHeight - clientHeight
            };
        }
    """
    )
    return {
        "is_page_scrollable": scroll_info["scrollable"],
        "scroll_amount_px": scroll_info["scrollAmount"],
        "scroll_amount_n_times_viewport": scroll_info["scrollAmount"]
        // page.viewport_size["height"],
    }


ToolArgument = dict["name":str, "type":str]


@dataclass
class Tool:
    name: str
    args: list[ToolArgument]
    description: str
    examples: list[str]

    def __str__(self):
        return f"{self.name}({', '.join([f"{arg["name"]}: {arg["type"]}" for arg in self.args])}): {self.description}. Examples: {"; ".join(self.examples)}."


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
        return response_text
    except ValueError as msg:
        raise ValueError(f"{msg}\n {response.candidates.safety_ratings}")


def get_gpt_actor_response(prompt: str, image_path: Path) -> str:
    user_message = create_user_message(prompt=prompt, image_paths=[image_path])
    payload = create_payload(user_message=user_message)
    response = get_openai_response(os.getenv("OPENAI_API_KEY"), payload)
    response_text = get_openai_response_text(response)
    return response_text


def log_response(logger, agent_type: str, prompt: str, response_text: str) -> None:
    logger.info(
        f"\n====== {agent_type} ======\n == PROMPT ==\n{prompt}\n== RESPONSE ==\n{response_text}\n"
    )
    logger.info("-" * 80)
    print(
        f"\n====== {agent_type} ======\n == PROMPT ==\n{prompt}\n== RESPONSE ==\n{response_text}\n"
    )
    print("-" * 80)


def get_prompts(path: str | Path) -> dict:
    """Opens and loads 'prompts.yaml' file"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return DotDict(data)


class DotDict(dict):
    """
    If you wrap a Python dictionary with this class, e.g. DotDict(my_dict), it
    gives the dictionary the ability to be accessed via dot-notation, e.g.
    my_dict.my_var instead of my_dict["myvar"]
    """

    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value


class OpenAIDataTypes(Enum):
    """OpenAI uses https://json-schema.org/understanding-json-schema/reference/type"""
    str = "string"
    int = "integer"
    float = "number"
    bool = "boolean"
    list = "array"
    NoneType = "null"
    object = "object"

    def get_mapping():
        return {dtype.name: dtype.value for dtype in OpenAIDataTypes}
    

def convert_function_to_openai_tool_json_format(func: Callable) -> str:
    """Converts a Python function into an OpenAI-formatted Assistant tool."""
    properties = {} # the `properties` field for OpenAI formatted tools
    required = []

    type_hints = get_type_hints(func, include_extras=True)
    for param, hint in type_hints.items():
        # check if the type hint of the argument has a standard type (int, str, float)
        # without any Annotation[] or Optional[] hint.
        if isinstance(hint, type):
            dtype = OpenAIDataTypes.get_mapping()[hint.__name__]
            description = ""
            required.append(param)
            properties[param] = {"type": dtype, "description": description}
        # check if the type hint of the argument has an Optional[] hint.
        elif hint.__name__ == "Optional":
            dtype = OpenAIDataTypes.get_mapping()[hint.__args__[0].__name__]
            description = ""
            properties[param] = {"type": dtype, "description": description}
        # check if the type hint of the argument has an Annotated[] hint.
        elif isinstance(hint, _AnnotatedAlias):

            # get the description of the argument, if there is any
            if hint.__metadata__: # the argument has a description
                description = hint.__metadata__[0]
                # If the description contains "IGNORE", we don't want 
                # the parameter to be included in the tool description
                # when passed to the LLM
                if "IGNORE" in description:
                    continue
            else: # the argument does not have a description
                description = ""

            # get the type of the argument, which could be wrapped inside an Optional[] hint
            if hint.__origin__.__name__ == "Optional":
                dtype = OpenAIDataTypes.get_mapping()[hint.__origin__.__args__[0].__name__]
            else:
                dtype = OpenAIDataTypes.get_mapping()[hint.__origin__.__name__]
                required.append(param)

            properties[param] = {"type": dtype, "description": description}
        elif isinstance(hint, _LiteralGenericAlias):
            values = hint.__args__
            # raise if not all values in side values have same type
            if not all(isinstance(value, type(values[0])) for value in values):
                raise ValueError(f"All values in {values} must have the same type.")
            dtype = OpenAIDataTypes.get_mapping()[type(values[0]).__name__]
            description = ""
            required.append(param)
            properties[param] = {"type": dtype, "enum": values, "description": description}
        else:
            raise ValueError(f"Unsupported type hint: {hint}")

    tool = {
        "name": func.__name__,
        "description": func.__doc__ if func.__doc__ else "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return tool
    # return json.dumps({"tools": [tool]}, indent=4)


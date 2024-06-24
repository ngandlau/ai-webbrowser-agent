import json
from pathlib import Path
import pdb
import time
from typing import Annotated, Any, Callable, Literal
from litellm import completion
import os
from dotenv import load_dotenv
from playwright.sync_api import Page, sync_playwright
from termcolor import colored

from src.utils import convert_function_to_openai_tool, encode_image, make_screenshot

## set ENV variables
load_dotenv()

# general setup
SCREENSHOT_DIR = "screenshots"
PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
VIMIUM_PATH = "vimium"  # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

# Agent Tools
def scroll(
    page: Annotated[Page, "IGNORE"],
    scroll_direction: Literal["up", "down"]
):
    """Use this function to scroll up or down a webpage."""
    page.keyboard.press("PageDown" if scroll_direction == "down" else "PageUp")
    return f"Scrolled {scroll_direction} on the webpage. Waiting for the website to respond, which can take a while..."

def click(
    page: Annotated[Page, "IGNORE"],
    ui_element_id: Annotated[
        str, "The unique identifier of a UI element consisting of 1-2 letters"
    ],
):
    """Use this function to click on an UI element on a webpage. UI elements that can be clicked are marked by small yellow boxes with letters inside. The letters uniquely identify a UI element."""
    page.keyboard.press(ui_element_id)
    return f"Clicked on the UI element with ID '{ui_element_id}'. Waiting for the website to respond, which can take a while..."

name_to_function_map: dict[str, Callable] = {
    scroll.__name__: scroll,
    click.__name__: click,
}
tools = [
    {"function": convert_function_to_openai_tool(func), "type": "function"}
    for func in name_to_function_map.values()
]

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=PLAYWRIGHT_USER_DATA_DIRECTORY,
        headless=False,
        args=[
            f"--disable-extensions-except={VIMIUM_PATH}",
            f"--load-extension={VIMIUM_PATH}",
        ],
        viewport={"width": 700, "height": 800},
        screen={"width": 700, "height": 800},
    )

    # navigate to booking site
    page = browser.new_page()
    page.goto("https://safo.ebusy.de")
    time.sleep(1)

    page.keyboard.press("f")
    screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    screenshot_base64 = encode_image(screenshot_path)

    messages = []

    system_msg = "You help the user navigate a website based on the screenshot he provides. The user is looking to answer a question he has by navigating the website. You can call tools to navigate the website. Always respond providing your thinking step-by-step."
    messages.append({"role": "system", "content": system_msg})
    print(colored(f"\nSystem:\n{system_msg}", color="red"))

    user_msg = "Which outside tennis courts are free for 1 hour today between 17:00 and 19:00?"
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + screenshot_base64}},
            ]
        }
    )
    print(colored(f"\nHuman:\n{user_msg}", color="cyan"))

    max_recursions = 5
    for i in range(max_recursions):
        response = completion(
            model="claude-3-5-sonnet-20240620",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        print(colored(f"\nAI:\n{response['choices'][0]['message']['content']}", color="magenta"))

        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(response.choices[0].message.tool_calls[0].function.arguments, str)

        messages.append(response.choices[0].message.model_dump())  # Add assistant tool invokes

        if response['choices'][0]['finish_reason'] == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            print(colored(f"\nTool Calls:\n{tool_calls}", color="yellow"))
            assert len(tool_calls) == 1, "More than one tool_calls. Currently not implemented."
            for tool_call in tool_calls:
                # get tool information
                tool_name: str = tool_call.function.name
                tool_args: dict = json.loads(tool_call.function.arguments)
                tool_func: Callable = name_to_function_map[tool_name]

                # execute tool
                if tool_name == "click":
                    tool_output = click(page=page, ui_element_id=tool_args["ui_element_id"])
                elif tool_name == "scroll":
                    tool_output = scroll(page=page, scroll_direction=tool_args["scroll_direction"])
                else:
                    raise ValueError(f"Unknown tool name: {tool_name}. Available tools: {name_to_function_map.keys()}")

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output,
                })
            # continue the response
            response = completion(
                model="claude-3-5-sonnet-20240620",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            print(colored(f"\nAI:\n{response['choices'][0]['message']['content']}", color="magenta"))

        # next screenshot
        page.keyboard.press("f")
        screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
        screenshot_base64 = encode_image(screenshot_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Next screenshot"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + screenshot_base64}},
                ]
            }
        )


            



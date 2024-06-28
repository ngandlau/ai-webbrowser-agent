import json
from pathlib import Path
import pdb
import re
import time
from typing import Annotated, Any, Callable, Literal, Optional
from litellm import completion
import os
from dotenv import load_dotenv
from playwright.sync_api import Page, sync_playwright
from termcolor import colored
from src.ui_integration import find_target_coordinates_for_image

from src.utils import convert_function_to_openai_tool, create_user_message, encode_image, make_screenshot

## set ENV variables
load_dotenv()

# general setup
SCREENSHOT_DIR = "screenshots"
PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
VIMIUM_PATH = "vimium"  # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

# custom typing hints
LLMAnswer = str
Base64Img = Any

class LLM:
    GPT_4o = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    GEMINI_1_5_FLASH = "gemini/gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini/gemini-1.5-pro"


# Agent Tools
def scroll(
    page: Annotated[Page, "IGNORE"],
    scroll_direction: Literal["up", "down"]
) -> str:
    """
Use this function to scroll up or down a webpage. Scrolling up or down is \
useful if you think there might be relevant information further up or down \
the webpage. For example, scroll(scroll_direction='down') scrolls down the \
webpage."""
    scroll_length = 600 if scroll_direction == "down" else -600
    page.evaluate(f"window.scrollBy(0, {scroll_length})")
    return (
        f"Scrolled {scroll_direction} on the webpage."
        "Waiting for the website to respond, which can take a while..."
    )

def click(
    page: Annotated[Page, "IGNORE"],
    ui_element_id: Annotated[Optional[str], "If you want to click on an UI element annotated with a small yellow box you also need to provide the corresponding letter."],
    screenshot_path: Annotated[Path, "IGNORE"],
    task: Annotated[str, "IGNORE"],
    is_ui_element_annotated_with_small_yellow_box: Annotated[bool, "If the UI element is annotated with a small yellow box, set this to True. If the UI element is not annotated with a small yellow box, set this to False."],
) -> str:
    """Use this function to click on an UI element on a webpage. UI elements \
that can be clicked are marked by small yellow boxes with letters inside. \
The letters uniquely identify a UI element."""
    if is_ui_element_annotated_with_small_yellow_box:
        page.keyboard.press(ui_element_id)
    else: 
        coordinates = find_target_coordinates_for_image(screenshot_path.as_posix(), task)
        page.mouse.click(coordinates['x'], coordinates['y'])

    if is_ui_element_annotated_with_small_yellow_box:
        return (
            f"Clicked on the UI element with ID '{ui_element_id}'. "
            "Waiting for the website to respond, which can take a while..."
        )
    else:
        return (
            f"Clicked on the UI element with coordinates {coordinates['x']}, {coordinates['y']}. "
            "Waiting for the website to respond, which can take a while..."
        )


def type_text(
    page: Annotated[Page, "IGNORE"],
    text: Annotated[str, "The text you want to insert into the text field on the webpage"],
    ui_element_id: Annotated[str, "Provide the letter of the text field element that you want to click on first."],
) -> str:
    """Use this function to type text into a text field on a webpage."""
    page.keyboard.press(ui_element_id)
    time.sleep(1)
    page.keyboard.type(text)
    return "Typed the text into the text field on the webpage."


def extract_information_from_table(
    screenshot: Annotated[Base64Img, "IGNORE"],
    task_description: Annotated[str, "IGNORE"],
    model: Annotated[str, "IGNORE"],
    temperature: Annotated[float, "IGNORE"],
) -> LLMAnswer: 
    """Use this function if you want to reliably extract information from a table or a booking schedule that you see in an image."""
    prompt = (
        f"The user wants to solve the following task: {task_description}."
        " There might be relevant information in the table in the image that I provided you."
        " First, describe the structure of the table and the type of cell contents there are."
        " Second, extract the data from each cell in the table and provide the table in a markdown format. "
        " Try to assign each cell's value to a cell content type."
        " If a cell looks empty or you are unsure about the cell's content, write 'NOT AVAILABLE' in the cell."
    )
    messages = [create_user_message(prompt=prompt, images_base64=[screenshot])]
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    response_text = response.choices[0].message.content
    return response_text

name_to_function_map: dict[str, Callable] = {
    scroll.__name__: scroll,
    click.__name__: click,
    # extract_information_from_table.__name__: extract_information_from_table,
    type_text.__name__: type_text,
}
tools = [
    {"function": convert_function_to_openai_tool(func), "type": "function"}
    for func in name_to_function_map.values()
]
print(colored(f"\nAVAILABLE TOOLS:{"".join(["\n* " + func_name for func_name in name_to_function_map.keys()])}", color="green"))

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=PLAYWRIGHT_USER_DATA_DIRECTORY,
        headless=False,
        args=[
            f"--disable-extensions-except={VIMIUM_PATH}",
            f"--load-extension={VIMIUM_PATH}",
        ],
        viewport={"width": 760, "height": 800},
        screen={"width": 760, "height": 800},
    )

    # navigate to booking site
    page = browser.new_page()
    # page.goto("https://safo.ebusy.de")
    page.goto("https://safo.ebusy.de/lite-module/407")
    time.sleep(2)

    # make a screenshot
    page.keyboard.press("Escape")
    page.keyboard.press("f")
    screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)

    screenshot_base64 = encode_image(screenshot_path)

    messages = []

    system_msg = """\
    You are an assistant that helps the user check the availability of bookable tennis courts on a website. 

    The user will ask you to book a tennis court at certain time slot. \
    For example, he might ask you to book court P2 at 17:00pm.
    
    To book a court, you must navigate to the right page on the website. \
    You can navigate the website by calling tools, for example you can click, scroll or type. \
    Refer to the UI elements by using those letters.
    """

#     system_msg = """\
# You are an assistant that helps the user check the availability of bookable tennis courts on a website. 

# The user will ask you to check for available tennis courts at certain time slots. \
# For example, he might ask whether there are any courts free between 8:00 and 10:00.

# To check for available courts, you must navigate to the right page on the website. \
# You can navigate the website by calling tools, for example you can click, scroll or type. \
# You can only click on ui elements that are marked by small yellow boxes with letters inside. \
# Refer to the UI elements by using those letters.

# The page you need to navigate to is the "Freiplätze" page. You can find the button \
# "Freiplätze" in the navigation bar.

# The tennis club has 13 courts. When you see the booking page, you need to check \
# every court. As the user only provides you a small part of the website, you might  \
# need to scroll up or down to see all time slots. \

# In the booking table, courts that are bookable are marked as "BUCHEN". All other courts are NOT bookable. \
# If you find any courts that are bookable, provide the court number along with the bookable time slots in the following format:

# <ANSWER>
# Court 1:
# - 12:00-12:30

# Court 2:
# None

# Court 3:
# - 13:00-13:30
# - 13:30-14:00
# </ANSWER>

# Here, "None" means that there are no available courts for the requested time slot. \
# Make sure that if you see 3 courts, you provide the information for all 3 courts.  \
# """
# You are an assistant that helps the user solve a task by navigating a website \
# and searching for relevant information on the website. The user will provide \
# you screenshots of the website, and you must help him navigate the website \
# until you find the information needed to answer the user's task. 

# If the provided screenshot of the website has small yellow boxes on top of UI \
# elements, then you must provide an explanation of what you think the UI element
# is for and what kind of page it likely navigates to. \
# The yellow boxes have small letters written inside of them that uniquely
# identify a UI element. Refer to the UI elements by using those letters. \
# For example: 'ee (Home): Likely navigates to the home page of the website'.

# You can call tools to navigate the website, for example to click, scroll or type.

# Every time you respond to the user, provide your step-by-step thought process. \
# When you think you can provide an answer based on the information gathered, 
# give your final answer to the user by writing it inside <ANSWER></ANSWER> tags."""

    messages.append({"role": "system", "content": system_msg})
    print(colored(f"\nSystem:\n{system_msg}", color="red"))

    task_description = """Book the field P2 at 17:00pm. Use the name: Nils Gandlau."""

    print(colored(f"\nHuman:\n{task_description}", color="cyan"))
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_description},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + screenshot_base64}},
            ]
        }
    )

    max_recursions = 5
    for i in range(max_recursions):
        response = completion(
            model=LLM.CLAUDE_3_5_SONNET,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        response_text = response.choices[0].message.content
        print(colored(f"\nAI:\n{response_text}", color="magenta"))

        # check if the LLM has called tools. If so, we need to invoke them
        print(f"\n<< finish_reason: {response.choices[0].finish_reason} >>")
        print(f"\n<< tool_calls: {response.choices[0].message.tool_calls} >>")
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
                    tool_args["ui_element_id"] = None if not "ui_element_id" in tool_args.keys() else tool_args["ui_element_id"]
                    tool_output = click(page=page,
                                        ui_element_id=tool_args["ui_element_id"],
                                        screenshot_path=screenshot_path,
                                        task=task_description,
                                        is_ui_element_annotated_with_small_yellow_box=tool_args["is_ui_element_annotated_with_small_yellow_box"])
                    print(f"\n<< key press: {tool_args['ui_element_id']} >>")
                elif tool_name == "scroll":
                    tool_output = scroll(page=page, scroll_direction=tool_args["scroll_direction"])
                    print(f"\n<< scroll {tool_args['scroll_direction']} >>")
                elif tool_name == "extract_information_from_table":
                    tool_output = extract_information_from_table(
                        screenshot=screenshot_base64,
                        task_description=task_description,
                        model=LLM.CLAUDE_3_5_SONNET,
                        temperature=0.3,
                    )
                    print(f"\n<< extract information from table >>\n")
                    print(tool_output)
                    print(f"\n<< extract information from table >>\n")
                elif tool_name == "type_text":
                    tool_output = type_text(
                        page=page,
                        text=tool_args["text"],
                        ui_element_id=tool_args["ui_element_id"],
                    )
                    print(f"\n<< type text >>\n")
                else:
                    raise ValueError(f"Unknown tool name: {tool_name}. Available tools: {name_to_function_map.keys()}")

                # add tool output to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output,
                })

            # Let the LLM finish his answer after the tool call
            response = completion(
                model=LLM.CLAUDE_3_5_SONNET,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            response_text = response.choices[0].message.content
            print(colored(f"\nAI:\n{response_text}", color="magenta"))

        # check if the LLM has finished
        if response_text is not None:
            if "<ANSWER>" in response_text and "</ANSWER>" in response_text:
                print(colored(f"<< ANSWER PROVIDED >>", color="green"))
                pattern = r'<ANSWER>([\s\S]*?)</ANSWER>'
                match = re.search(pattern, response_text)
                answer = match.group(1)
                print(colored(f"\nFINAL ANSWER:\n{answer}", color="green"))
                break

        # create the next screenshot after navigating
        time.sleep(3)
        page.keyboard.press("Escape")
        page.keyboard.press("f")
        screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)

        screenshot_base64 = encode_image(screenshot_path)

        # give the LLM the next screenshot
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the next screenshot."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + screenshot_base64}},
                ]
            }
        )


            



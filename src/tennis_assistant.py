import json
import os
import time
from pathlib import Path
from typing import Callable, Literal

import google.generativeai as genai
import vertexai
from annotated_docs.json_schema import as_json_schema
from dotenv import load_dotenv

## models
from openai import OpenAI
from playwright.sync_api import Page, sync_playwright
from termcolor import colored
from vertexai.generative_models import GenerativeModel, Part

from utils import *

logger = setup_logger()
DEBUG_OBSERVER = False
DEBUG_ACTOR = False

# general setup
SCREENSHOT_DIR = "screenshots"
PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
VIMIUM_PATH = "vimium"  # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

# OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
genai.configure(api_key=GOOGLE_API_KEY)
vertexai.init(project=PROJECT_ID, location="europe-central2")


################################################################
## For streaming
from openai import AssistantEventHandler
from typing_extensions import override


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(colored(f"\nAssistant:\n", color="red"), end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(colored(f"\nAssistant > {tool_call.type}\n", color="yellow"), flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)


################################################################


def get_screenshot_description_from_gemini(screenshot_path: str | Path) -> str:
    prompt = "First, describe what you see on the webpage. UI elements that can be clicked are marked by small yellow boxes with letters inside. Extract the letters inside each yellow box and describe what the corresponding UI element might lead to. Answer in the following format:\nDescription of website:\n...\nDescription of UI Elements:\n* <letter> (<name of the button>): Likely navigates to ..."
    answer = get_gemini_observer_response(prompt=prompt, image_path=screenshot_path)
    print(colored(f"\n== Gemini==\nPrompt:{prompt}\n\nAnswer:\n{answer}", color="cyan"))
    return answer


# Agent Tools
def scroll(page: Annotated[Page, "IGNORE"], scroll_direction: Literal["up", "down"]):
    """Use this function to scroll up or down a webpage."""
    page.keyboard.press("PageDown" if tool_args["direction"] == "down" else "PageUp")
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


def extract_data_from_table(
    screenshot_path: Annotated[Path, "IGNORE"],
    table_description: Annotated[str, "A description of the table to extract data from."],
):
    """Use this function to scrape and extract structured data, for example a table, from an image with great reliability."""
    prompt = """Extract the data from the table in the image into a JSON format. \
    There might be several tables in the image, but the table you should scrape \
    data from can be described as follows: {table_description}"""
    response_text = get_gemini_observer_response(prompt=prompt, image_path=screenshot_path)
    return response_text

def scroll(scroll_direction: Literal["up", "down"]):
    """Use this tool to scroll DOWN or UP the website."""
    return f"Scrolled {scroll_direction} on the webpage. Waiting for the website to respond, which can take a while..."

def click(ui_element_id: Annotated[str, "The ID that consists of 1-2 letters which uniquely identifies a UI element."]):
    """Use this tool to click on an UI element."""
    return f"Clicked on the UI element with ID '{ui_element_id}'. Waiting for the website to respond, which can take a while..."

def scroll(direction: Literal["up", "down"]):
    """Use this to scroll up or down a webpage."""
    pass

def click(ui_element_id: str):
    """Use this to click on an UI element on a webpage. UI elements that can be clicked are marked by small yellow boxes with letters inside. The letters uniquely identify a UI element."""
    pass

name_to_function_map: dict[str, Callable] = {
    scroll.__name__: scroll,
    click.__name__: click,
    # extract_data_from_table.__name__: extract_data_from_table,
}
openai_formatted_tools = [
    # {"function": convert_function_to_openai_tool_json_format(func), "type": "function"}
    {"function": as_json_schema(func), "type": "function"}
    for func in name_to_function_map.values()
]
print(openai_formatted_tools)

# Initialize assistant
instructions = """\
You are a helpful assistant that helps the user navigate a website in order to solve the user's task. \

The user will provide you a screenshot of the current webpage and a description of what he sees in the screenshot. \
You must decide whether the user's task is solvable given the current webpage, or whether the user should navigate the webpage (e.g. by scrolling, clicking, or typing something into a form). \
You can call a tool to navigate the website. \
If you call a tool to navigate the website, wait for the user to provide you the next screenshot along with the description he sees.

Always think step-by-step before responding to the user. \
Always provide your thoughts and reasoning behind what you plan to do.\
You must always respond in the following format:

Task: Rephrase the user's task.
Thought: Describe your thoughts and reasoning behind what you plan to do.
Action: Describe in detail the plan you want to do.
Observation: If you called a function, describe the output of the function. 
Final Answer (optional): Only provide the final answer if you have found it.
"""
assistant = client.beta.assistants.create(
    instructions=instructions,
    model="gpt-4o",
    tools=openai_formatted_tools,
    temperature=0.0,
)
thread = client.beta.threads.create()


# Main Loop
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

    client.beta.threads.create()
    num_recursions = 5
    for i in range(num_recursions):
        # make a screenshot
        page.keyboard.press("f")
        screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
        screenshot_file = client.files.create(file=open(screenshot_path, "rb"), purpose="vision")

        # ask Gemini for a description of the screenshot
        screenshot_description = get_screenshot_description_from_gemini(
            screenshot_path=screenshot_path
        )

        # Ask GPT to navigate or answer, given the task description, the website description, and the screenshot
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=[
                {
                    "type": "text",
                    "text": f"I want to find out whether there are any tennis courts free between 17:00 and 19:00. I provide you a screenshot of the webpage I am currently seeing. I see the following in it:\n\n{screenshot_description}",
                },
                {"type": "image_file", "image_file": {"file_id": screenshot_file.id}},
            ],
        )
        print(colored(f"User:\n{message.content[0].text.value}", color="magenta"))

        with client.beta.threads.runs.stream(
            thread_id=thread.id, assistant_id=assistant.id, event_handler=EventHandler()
        ) as stream:
            print("here?")
            stream.until_done()
            run = stream.get_final_run()

        # Call tools and format tool outputs
        if run.status == "requires_action":
            tool_outputs = [] # stores formatted function outputs that will be passed back to the LLM
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                # get tool information
                tool_name: str = tool.function.name
                tool_args: dict = json.loads(tool.function.arguments)
                tool_func: Callable = name_to_function_map[tool_name]

                # execute tool
                if tool_name == "click":
                    tool_output = click(page=page, ui_element_id=tool_args["ui_element_id"])
                elif tool_name == "scroll":
                    tool_output = scroll(page=page, scroll_direction=tool_args["scroll_direction"])
                elif tool_name == "extract_data_from_table":
                    tool_output = extract_data_from_table(
                        screenshot_path=screenshot_description,
                        table_description=tool_args["table_description"],
                    )
                else:
                    raise ValueError(
                        f"Unknown tool name: {tool_name}. Available tools: {name_to_function_map.keys()}"
                    )

                # pass the LLM-friendly formatted tool output back to the LLM
                tool_outputs.append({"tool_call_id": tool.id, "output": tool_output})

            # Submit tool outputs such that the LLM can continue its answer
            with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
                run = stream.get_final_run()

        # Check if the final answer was provided in the last message
        messages = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        last_message = list(messages)[0]
        last_message_text = last_message.content[0].text.value
        if run.status == "completed" and "final answer" in last_message_text.lower():
            print(colored("\nFinal answer provided in last message!", color="green"))
            break  # stop the recursion

    # screenshot of the final webpage view
    make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)

    browser.close()

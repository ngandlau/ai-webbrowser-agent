import os
import time
import json
from pathlib import Path
from typing import Callable, Literal
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

from annotated_docs.json_schema import as_json_schema
from utils import *

## models
from openai import OpenAI
from termcolor import colored
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.generativeai as genai


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
from typing_extensions import override
from openai import AssistantEventHandler

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
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)
################################################################


def get_screenshot_description_from_gemini(screenshot_path):
    prompt = "First, describe what you see on the webpage. UI elements that can be clicked are marked by small yellow boxes with letters inside. Extract the letters inside each yellow box and describe what the corresponding UI element might lead to. Answer in the following format:\nDescription of website:\n...\nDescription of UI Elements:\n* <letter> (<name of the button>): Likely navigates to ..."
    answer: str = get_gemini_observer_response(prompt=prompt, image_path=screenshot_path)
    return answer

# Agent Tools
def scroll(direction: Literal["up", "down"]):
    """Use this to scroll up or down a webpage."""
    pass
def click(ui_element_id: str):
    """Use this to click on an UI element on a webpage. UI elements that can be clicked are marked by small yellow boxes with letters inside. The letters uniquely identify a UI element."""
    pass
name_to_function_map: dict[str, Callable] = {"scroll": scroll, "click": click}
openai_formatted_tools = [
    {"function": as_json_schema(func), "type": "function"}
    for func in name_to_function_map.values()
]

# Initialize assistant 
instructions = """\
You are a helpful assistant that helps the user navigate thourgh a website in order to solve the user's task. \

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

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    screenshot_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    screenshot_file = client.files.create(file=open(screenshot_path, "rb"), purpose="vision") # upload image to OpenAI. This is required to pass images to Assistants
    
    # get the description of the screenshot from Gemini
    # screenshot_description = get_screenshot_description_from_gemini(screenshot_path=screenshot_path)
    screenshot_description = """\
Description of website:
The website is a landing page for the Sportclub SAFO Frankfurt e.V.. The website \
features a blue header bar with a logo and several navigation links. The main \
content of the page is a welcome message that introduces the club and provides \
information about their tennis and hockey offerings.

Description of UI Elements:
* A (Camps): Likely navigates to a page about the club's camps.
* E (Events & Kurse): Likely navigates to a page about upcoming events and courses offered by the club.
* F (Freiplätze): Likely navigates to a page about the club's outdoor facilities.
* K (Guthaben & Gutscheine): Likely navigates to a page about credits and vouchers that can be used at the club.
* L (Login): Likely navigates to a login page.
* S (Registrieren): Likely navigates to a registration page.
* W (WIR ERFAHREN): Likely navigates to a page that provides more information about the club's offerings.
"""
    print(colored(f"GEMINI: {screenshot_description}", color="cyan"))

    client.beta.threads.create()


    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=[
            {"type": "text", "text": f"I want to find out whether there are any tennis courts free between 17:00 and 19:00. I provide you a screenshot of the webpage I am currently seeing. I see the following in it:\n\n{screenshot_description}"},
            {"type": "image_file", "image_file": {"file_id": screenshot_file.id}},
        ],
    )

    run = None
    with client.beta.threads.runs.stream( thread_id=thread.id, assistant_id=assistant.id, event_handler=EventHandler()) as stream:
       stream.until_done()
       run = stream.get_final_run()

    # Call tools and format tool outputs
    if run.status == "requires_action":
        tool_outputs = [] # stores formatted function outputs that will be passed back to the LLM 
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            # get tool information
            tool_name = tool.function.name
            tool_args = json.loads(tool.function.arguments)
            tool_func = name_to_function_map[tool_name]
            
            # execute tool
            if tool_name == "click": 
                page.keyboard.press(tool_args["ui_element_id"])
                tool_output = f"Clicked on the UI element with ID {tool_args['ui_element_id']}. Waiting for the website to respond, which can take a while..."
            elif tool_name == "scroll":
                page.keyboard.press("PageDown" if tool_args["direction"] == "down" else "PageUp")
                tool_output = f"Scrolled {tool_args['direction']} on the webpage. Waiting for the website to respond, which can take a while..."
            else: # unkown tool_name
               raise ValueError(f"Unknown tool name: {tool_name}")
            
            # format tool output for the LLM
            tool_outputs.append({ "tool_call_id": tool.id, "output": tool_output})

    # Submit tool outputs back to LLM 
    with client.beta.threads.runs.submit_tool_outputs_stream(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs, event_handler=EventHandler()) as stream:
        stream.until_done()
        run = stream.get_final_run()

    browser.close()

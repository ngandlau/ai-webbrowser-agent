import base64
from datetime import datetime
from pathlib import Path
from typing import Any
from playwright.sync_api import sync_playwright, ViewportSize, Page
import os
import time
from dotenv import load_dotenv

## models
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.generativeai as genai


from prompts import get_gemini_observer_prompt, get_observer_prompt, get_actor_prompt, answer_tool, click_tool, input_tool, scroll_tool, parse_table_data_tool
from utils import * 
import history

logger = setup_logger()
DEBUG_OBSERVER = False 
DEBUG_ACTOR = False

# general setup
SCREENSHOT_DIR = "screenshots"
PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
VIMIUM_PATH = "vimium" # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

# OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
genai.configure(api_key=GOOGLE_API_KEY)
vertexai.init(project=PROJECT_ID, location="europe-central2")

# user task
task_description = """Find and name outside tennis courts that are free for 1 hour today between 17:00 and 19:00. The process is described as follows:
The video demonstrates the online booking process for a tennis court at the Sportclub SAFO Frankfurt e.V. 

1. **Navigate to the "Freiplätze" (Available Courts) page:** The user starts on the homepage of the Sportclub SAFO Frankfurt e.V. and clicks on the "Freiplätze" tab in the top navigation bar.
2. **View available time slots:** The "Freiplätze" page displays a calendar with available time slots for each tennis court (P1, P2, P3). The current date is selected, showing the available time slots for that day.
3. **Select a time slot:** The user scrolls down to view the available time slots and clicks on the desired time slot, which is 18:00-18:30 on Court P1. 
"""
default_observer = "gpt"

# tools for the actor
actor_tools = [
    scroll_tool,
    click_tool,
    input_tool,
    parse_table_data_tool,
    answer_tool,
]

## MAIN APP LOOP
with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=PLAYWRIGHT_USER_DATA_DIRECTORY,
        headless=False,
        args=[
            f"--disable-extensions-except={VIMIUM_PATH}",
            f"--load-extension={VIMIUM_PATH}"
        ],
        viewport={"width": 700, "height": 800},
        screen={"width": 700, "height": 800},
    )

    # navigate to booking site 
    page = browser.new_page()
    page.goto("https://safo.ebusy.de")
    time.sleep(1)
    
    logger.info("########## ROUND 1 ##########")
    print("########## ROUND 1 ##########")

    last_observer = "gpt"

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    image_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    scroll_info = get_scroll_info(page=page)

    #### OBSERVER
    # pass the screenshot over to the observer LLM, who describes what he sees
    if not last_observer == "gemini": 
        if not DEBUG_OBSERVER:
            prompt = get_observer_prompt()
            response_text = get_gpt_observer_response(prompt=prompt, image_path=image_path)
            log_response(logger, agent_type="GPT OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
        if DEBUG_OBSERVER:
            response_text = history.observer_response_text_2
            log_response(logger, agent_type="SIMULATED OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"

    #### ACTOR
    # pass the screenshot and the observer's observations to the actor LLM
    response_text += f"""\n\nWebsite view:\nThe screenshot of the website does not show the full website. More information might be contained on the webpage when you scroll down or scroll up. You can scroll down by {scroll_info["scroll_amount_px"]} pixels."""
    actor_prompt = get_actor_prompt(
        website_description=response_text,
        task_description=task_description,
        tools=actor_tools,
    )
    if not DEBUG_ACTOR:
        response_text = get_gpt_actor_response(prompt=actor_prompt, image_path=image_path)
        log_response(logger, agent_type="GPT ACTOR", prompt=actor_prompt, response_text=response_text)
    if DEBUG_ACTOR: 
        response_text = history.actor_response_text_1
        log_response(logger, agent_type="SIMULATED ACTOR", prompt=actor_prompt, response_text=response_text)

    #### ACTION
    # parse the action from the actor's response
    action_type, action = parse_actor_response(response_text)
    print(f"Action type: {action_type}, Action: {action}")
    # execute the action
    if action_type == answer_tool.name:
        print(f"ANSWER: {action}")
        browser.close()
    if action_type == click_tool.name:
        page.keyboard.press(action)
    if action_type == scroll_tool.name:
        page.keyboard.press("Escape") # untoggle vimium
        if action == "down": 
            page.evaluate("window.scrollBy(0, 600)")
        if action == "up": 
            page.evaluate("window.scrollBy(0, -600)")
    if action_type == input_tool.name:
        page.keyboard.type(action)
    if action_type == parse_table_data_tool.name:
        prompt = get_gemini_observer_prompt(instructions=action)
        response_text = get_gemini_observer_response(prompt=prompt, image_path=image_path)
        log_response(logger, agent_type="GEMINI OBSERVER", prompt=prompt, response_text=response_text)
        last_observer = "gemini"
    time.sleep(3) 

    logger.info("########## ROUND 2 ##########")
    print("########## ROUND 2 ##########")

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    image_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    scroll_info = get_scroll_info(page=page)

    #### OBSERVER
    # pass the screenshot over to the observer LLM, who describes what he sees
    if not last_observer == "gemini": 
        if not DEBUG_OBSERVER:
            prompt = get_observer_prompt()
            response_text = get_gpt_observer_response(prompt=prompt, image_path=image_path)
            log_response(logger, agent_type="GPT OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
        if DEBUG_OBSERVER:
            response_text = history.observer_response_text_2
            log_response(logger, agent_type="SIMULATED OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
    
    #### ACTOR
    # pass the screenshot and the observer's observations to the actor LLM
    response_text += f"""\n\nWebsite view:\nThe screenshot of the website does not show the full website. More information might be contained on the webpage when you scroll down or scroll up. You can scroll down by {scroll_info["scroll_amount_px"]} pixels."""
    actor_prompt = get_actor_prompt(
        website_description=response_text,
        task_description=task_description,
        tools=actor_tools,
    )
    if not DEBUG_ACTOR:
        response_text = get_gpt_actor_response(prompt=actor_prompt, image_path=image_path)
        log_response(logger, agent_type="GPT ACTOR", prompt=actor_prompt, response_text=response_text)
    if DEBUG_ACTOR: 
        response_text = history.actor_response_text_1
        log_response(logger, agent_type="SIMULATED ACTOR", prompt=actor_prompt, response_text=response_text)

    #### ACTION
    # parse the action from the actor's response
    action_type, action = parse_actor_response(response_text)
    print(f"Action type: {action_type}, Action: {action}")
    # execute the action
    if action_type == answer_tool.name:
        print(f"ANSWER: {action}")
        browser.close()
    if action_type == click_tool.name:
        page.keyboard.press(action)
    if action_type == scroll_tool.name:
        page.keyboard.press("Escape") # untoggle vimium
        if action == "down": 
            page.evaluate("window.scrollBy(0, 600)")
        if action == "up": 
            page.evaluate("window.scrollBy(0, -600)")
    if action_type == input_tool.name:
        page.keyboard.type(action)
    if action_type == parse_table_data_tool.name:
        prompt = get_gemini_observer_prompt(instructions=action)
        response_text = get_gemini_observer_response(prompt=prompt, image_path=image_path)
        log_response(logger, agent_type="GEMINI OBSERVER", prompt=prompt, response_text=response_text)
        last_observer = "gemini"
    time.sleep(3)  


    logger.info("########## ROUND 3 ##########")
    print("########## ROUND 3 ##########")

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    image_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    scroll_info = get_scroll_info(page=page)

    #### OBSERVER
    # pass the screenshot over to the observer LLM, who describes what he sees
    if not last_observer == "gemini": 
        if not DEBUG_OBSERVER:
            prompt = get_observer_prompt()
            response_text = get_gpt_observer_response(prompt=prompt, image_path=image_path)
            log_response(logger, agent_type="GPT OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
        if DEBUG_OBSERVER:
            response_text = history.observer_response_text_2
            log_response(logger, agent_type="SIMULATED OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
    
    #### ACTOR
    # pass the screenshot and the observer's observations to the actor LLM
    response_text += f"""\n\nWebsite view:\nThe screenshot of the website does not show the full website. More information might be contained on the webpage when you scroll down or scroll up. You can scroll down by {scroll_info["scroll_amount_px"]} pixels. Only scroll if you have not all infromation that you need to complete the task."""
    actor_prompt = get_actor_prompt(
        website_description=response_text,
        task_description=task_description,
        tools=actor_tools,
    )
    if not DEBUG_ACTOR:
        response_text = get_gpt_actor_response(prompt=actor_prompt, image_path=image_path)
        log_response(logger, agent_type="GPT ACTOR", prompt=actor_prompt, response_text=response_text)
    if DEBUG_ACTOR: 
        response_text = history.actor_response_text_1
        log_response(logger, agent_type="SIMULATED ACTOR", prompt=actor_prompt, response_text=response_text)

    #### ACTION
    # parse the action from the actor's response
    action_type, action = parse_actor_response(response_text)
    print(f"Action type: {action_type}, Action: {action}")
    # execute the action
    if action_type == answer_tool.name:
        print(f"ANSWER: {action}")
        browser.close()
    if action_type == click_tool.name:
        page.keyboard.press(action)
    if action_type == scroll_tool.name:
        page.keyboard.press("Escape") # untoggle vimium
        if action == "down": 
            page.evaluate("window.scrollBy(0, 600)")
        if action == "up": 
            page.evaluate("window.scrollBy(0, -600)")
    if action_type == input_tool.name:
        page.keyboard.type(action)
    if action_type == parse_table_data_tool.name:
        prompt = get_gemini_observer_prompt(instructions=action)
        response_text = get_gemini_observer_response(prompt=prompt, image_path=image_path)
        log_response(logger, agent_type="GEMINI OBSERVER", prompt=prompt, response_text=response_text)
        last_observer = "gemini"
    time.sleep(3)  

    logger.info("########## ROUND 4 ##########")
    print("########## ROUND 4 ##########")

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    image_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    scroll_info = get_scroll_info(page=page)

    #### OBSERVER
    # pass the screenshot over to the observer LLM, who describes what he sees
    if not last_observer == "gemini": 
        if not DEBUG_OBSERVER:
            prompt = get_observer_prompt()
            response_text = get_gpt_observer_response(prompt=prompt, image_path=image_path)
            log_response(logger, agent_type="GPT OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
        if DEBUG_OBSERVER:
            response_text = history.observer_response_text_2
            log_response(logger, agent_type="SIMULATED OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
    
    #### ACTOR
    # pass the screenshot and the observer's observations to the actor LLM
    response_text += f"""\n\nWebsite view:\nThe screenshot of the website does not show the full website. More information might be contained on the webpage when you scroll down or scroll up. You can scroll down by {scroll_info["scroll_amount_px"]} pixels."""
    actor_prompt = get_actor_prompt(
        website_description=response_text,
        task_description=task_description,
        tools=actor_tools,
    )
    if not DEBUG_ACTOR:
        response_text = get_gpt_actor_response(prompt=actor_prompt, image_path=image_path)
        log_response(logger, agent_type="GPT ACTOR", prompt=actor_prompt, response_text=response_text)
    if DEBUG_ACTOR: 
        response_text = history.actor_response_text_1
        log_response(logger, agent_type="SIMULATED ACTOR", prompt=actor_prompt, response_text=response_text)

    #### ACTION
    # parse the action from the actor's response
    action_type, action = parse_actor_response(response_text)
    print(f"Action type: {action_type}, Action: {action}")
    # execute the action
    if action_type == answer_tool.name:
        print(f"ANSWER: {action}")
        browser.close()
    if action_type == click_tool.name:
        page.keyboard.press(action)
    if action_type == scroll_tool.name:
        page.keyboard.press("Escape") # untoggle vimium
        if action == "down": 
            page.evaluate("window.scrollBy(0, 600)")
        if action == "up": 
            page.evaluate("window.scrollBy(0, -600)")
    if action_type == input_tool.name:
        page.keyboard.type(action)
    if action_type == parse_table_data_tool.name:
        prompt = get_gemini_observer_prompt(instructions=action)
        response_text = get_gemini_observer_response(prompt=prompt, image_path=image_path)
        log_response(logger, agent_type="GEMINI OBSERVER", prompt=prompt, response_text=response_text)
        last_observer = "gemini"
    time.sleep(3)  


    logger.info("########## ROUND 5 ##########")
    print("########## ROUND 5 ##########")

    # make a screenshot
    page.keyboard.press("f")
    time.sleep(1)
    image_path = make_screenshot(page=page, screenshot_dir=SCREENSHOT_DIR)
    scroll_info = get_scroll_info(page=page)

    #### OBSERVER
    # pass the screenshot over to the observer LLM, who describes what he sees
    if not last_observer == "gemini": 
        if not DEBUG_OBSERVER:
            prompt = get_observer_prompt()
            response_text = get_gpt_observer_response(prompt=prompt, image_path=image_path)
            log_response(logger, agent_type="GPT OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
        if DEBUG_OBSERVER:
            response_text = history.observer_response_text_2
            log_response(logger, agent_type="SIMULATED OBSERVER", prompt=prompt, response_text=response_text)
            last_observer = "gpt"
    
    #### ACTOR
    # pass the screenshot and the observer's observations to the actor LLM
    response_text += f"""\n\nWebsite view:\nThe screenshot of the website does not show the full website. More information might be contained on the webpage when you scroll down or scroll up. You can scroll down by {scroll_info["scroll_amount_px"]} pixels."""
    actor_prompt = get_actor_prompt(
        website_description=response_text,
        task_description=task_description,
        tools=actor_tools,
    )
    if not DEBUG_ACTOR:
        response_text = get_gpt_actor_response(prompt=actor_prompt, image_path=image_path)
        log_response(logger, agent_type="GPT ACTOR", prompt=actor_prompt, response_text=response_text)
    if DEBUG_ACTOR: 
        response_text = history.actor_response_text_1
        log_response(logger, agent_type="SIMULATED ACTOR", prompt=actor_prompt, response_text=response_text)

    #### ACTION
    # parse the action from the actor's response
    action_type, action = parse_actor_response(response_text)
    print(f"Action type: {action_type}, Action: {action}")
    # execute the action
    if action_type == answer_tool.name:
        print(f"ANSWER: {action}")
        browser.close()
    if action_type == click_tool.name:
        page.keyboard.press(action)
    if action_type == scroll_tool.name:
        page.keyboard.press("Escape") # untoggle vimium
        if action == "down": 
            page.evaluate("window.scrollBy(0, 600)")
        if action == "up": 
            page.evaluate("window.scrollBy(0, -600)")
    if action_type == input_tool.name:
        page.keyboard.type(action)
    if action_type == parse_table_data_tool.name:
        prompt = get_gemini_observer_prompt(instructions=action)
        response_text = get_gemini_observer_response(prompt=prompt, image_path=image_path)
        log_response(logger, agent_type="GEMINI OBSERVER", prompt=prompt, response_text=response_text)
        last_observer = "gemini"
    time.sleep(3)  

    input()
    browser.close()
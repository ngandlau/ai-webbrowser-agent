from utils import Tool

answer_tool = Tool(
    name="ANSWER",
    args=[{"name": "answer", "type": "string"}],
    description="Use this action if you think you can provide a detailed answer to the task. Write the answer within quotation marks.",
    examples=['ANSWER("The answer to the task.")'],
)
click_tool = Tool(
    name="CLICK",
    args=[{"name": "letters", "type": "string"}],
    description="Use this action to navigate to a different page by clicking on a UI element. Specify the letters corresponding to the element.",
    examples=['CLICK("a")', 'CLICK("yy")'],
)
input_tool = Tool(
    name="INPUT",
    args=[{"name": "text", "type": "string"}],
    description="Use this action to type text into e.g. a form or search bar. Specify the text you want to type.",
    examples=['INPUT("my_username")', 'INPUT("my_password")'],
)
scroll_tool = Tool(
    name="SCROLL",
    args=[{"name": "direction", "type": "string"}],
    description="Use this action to scroll the current webpage up or down to find relevant information",
    examples=['SCROLL("down")', 'SCROLL("up")'],
)
analyze_table_tool = Tool(
    name="ANALYZE_TABLE",
    args=[{"name": "description", "type": "string"}],
    description="Use this action to analyze structured data such as tables, timetables, or grids. Describe the data and information you seek.",
    examples=['ANALYZE_TABLE("Extract the price information from the table that contains information about the product.")'],
)

def get_actor_prompt(
    website_description: str,
    task_description: str,
    tools: list[Tool],
) -> str: 
    return f"""\
You are an assistant that helps a user to solve a task. The task provided by the user is the following:
{task_description}

You can choose from one of the following actions to progress with the task. \ 
Here are the names and descriptions of the actions you can take:

{"\n".join([str(tool) for tool in tools])}

As input, you are given an image of the current webpage, \
a description of the webpage, \
a description of all clickable UI elements on the webpage, \
and information about whether you can scroll down further on the webpage.

{website_description}

You need to respond in the following format:

Thought: Your reasoning behind the action you are taking.
Action: Your chosen action, should be one of {", ".join([tool.name for tool in tools])}. Only provide the action.
"""

def get_observer_prompt() -> str:
    return f"""I give you a screenshot of a part of a webpage. \
Your first task is to describe what you see on the webpage in bullet points. \

Your second task is to describe each clickable UI element and make a guess about what page it likely navigates to. \
In the screenshot, yellow boxes are placed on top of clickable UI elements. \
And each yellow box contains one or two letters that uniquely identify the UI element.\

Answer in the following format:

Description of the webpage:
* ...
Description of clickable UI elements:
* "<letter>" (Home): Likely navigates to the home page of the website.
* ...
"""

def get_gemini_observer_prompt(instructions: str):
    return f"""\
You are given an image of a webpage.\
The user wants to extract information from the webpage that is relevant to solving the following task:
Task: {instructions}\

First, describe and summarize what is in the image. \
If there is structured data, such as tables, timetables, or grids, provide a brief summary about what information they contain.\

Second, think about whether there is structured data (for example, tables or timetables) that contain relevant information \
for the user's task. If so, extract the data in a structured markdown format.\
"""

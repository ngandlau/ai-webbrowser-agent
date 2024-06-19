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
parse_table_data_tool = Tool(
    name="PARSE_TABLE_DATA",
    args=[{"name": "description", "type": "string"}],
    description="Everytime you are confronted by data in a table or timetable, use this action to parse structured data such as tables, timetables, or grids. Describe the data and information you seek. You response needs to start with: This is the parsed data in text format",
    examples=['PARSE_TABLE_DATA("Extract the price information from the table that contains information about the product.")'],
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
potentially structured data in a text format, \
and information about whether you can scroll down further on the webpage. \
If you got any information that was parsed from structured data, make sure to use it to complete the task.

{website_description}

You need to respond in the following format:

Thought: Your reasoning behind the action you are taking.
Action: If another action is necessary, you can chose one of the following actions: {", ".join([tool.name for tool in tools])}. Only provide the action.
Answer: If you found the answer to the task, provide the answer in the following format: This is the answer to the task: <answer>. Else provide the reason why you could not find the answer.
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
    return f"""Extract the data from the table in the image. Respond in a readable markdown format."""

import json
from litellm import completion
import pytest
from src.utils import create_assistant_message, create_user_message, encode_image, extract_json, extract_and_fix_json_llm_call, fix_json_regex, get_gemini_observer_response

screenshot_paths = {
    "800x800": "tests/data/booking_800x800.jpeg",
}

screenshot_base64 = {
    name: encode_image(path)
    for name, path in screenshot_paths.items()
}

actual_bookings_800x800 = {
  "Platz 1": {
    "16:30-17:00": "booked",
    "17:00-17:30": "booked",
    "17:30-18:00": "booked",
    "18:00-18:30": "booked",
    "18:30-19:00": "booked",
    "19:00-19:30": "booked",
    "19:30-20:00": "booked",
    "20:00-20:30": "booked",
    "20:30-21:00": "booked",
    "21:00-21:30": "booked",
    "21:30-22:00": "free"
  },
  "Platz 2": {
    "16:30-17:00": "booked",
    "17:00-17:30": "booked",
    "17:30-18:00": "free",
    "18:00-18:30": "booked",
    "18:30-19:00": "booked",
    "19:00-19:30": "booked",
    "19:30-20:00": "booked",
    "20:00-20:30": "booked",
    "20:30-21:00": "booked",
    "21:00-21:30": "booked",
    "21:30-22:00": "free"
  },
  "Platz 3": {
    "16:30-17:00": "booked",
    "17:00-17:30": "booked",
    "17:30-18:00": "booked",
    "18:00-18:30": "booked",
    "18:30-19:00": "booked",
    "19:00-19:30": "booked",
    "19:30-20:00": "booked",
    "20:00-20:30": "booked",
    "20:30-21:00": "booked",
    "21:00-21:30": "booked",
    "21:30-22:00": "free"
  }
}

expected_output = {
    "Platz 1": ["21:30-22:00"],
    "Platz 2": ["17:30-18:00", "21:30-22:00"],
    "Platz 3": ["21:30-22:00"]
}

prompt = """\
Here is a screenshot of a table. \
The table contains information about the booking status of tennis courts. \
In which time slots are the courts free and bookable? \
Give me all free and bookable time slots for each court. \
"""
# Respond in JSON format. \
# The JSON should be formatted as follows:

# {{"Platz 1": ["16:30-17:00", ...], "Platz 2": ["13:30-14:00", ...], "Platz 3": []}}

# Where [] indicates that the court is not bookable in any time slot.\
# """

user_message = create_user_message(
    prompt=prompt,
    images_base64=[screenshot_base64["800x800"]],
)
messages = [user_message]

@pytest.mark.skip()
def test_google_gemini_1_5_flash():
    response_text = get_gemini_observer_response(
        prompt=prompt,
        image_path=screenshot_paths["800x800"],
    )
    print(response_text)
    # response_json = extract_and_fix_json_llm_call(json_str=response_text)
    # assert response_json == expected_output 

# @pytest.mark.skip()
def test_gpt_4o():
    messages = []
    user_message = create_user_message(
        prompt="What do you see in the image? If you see a table, describe it in detail.",
        images_base64=[screenshot_base64["800x800"]],
    )
    messages.append(user_message)

    response = completion(
        model="gpt-4o",
        messages=messages,
    )
    response_text = response.choices[0].message.content
    messages.append(create_assistant_message(response_text))
    print(response_text)

    user_message = "Based on the table and your previous response, what time slots are free and bookable for each court?"
    messages.append(user_message)

    response = completion(
        model="gpt-4o",
        messages=messages,
    )
    response_text = response.choices[0].message.content
    messages.append(create_assistant_message(response_text))
    print(response_text)

    user_message = 'Please repeat your answer but respond in JSON format. The format should look like this:\n{{"Platz 1": ["16:30-17:00", ...], "Platz 2": ["13:30-14:00", ...], "Platz 3": []}}.'
    messages.append(user_message)

    response = completion(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"}
    )


    response_text = response.choices[0].message.content
    response_json = extract_and_fix_json_llm_call(json_str=response_text)
    assert response_json == expected_output

@pytest.mark.skip()
def test_claude_3_5_sonnet():
    response = completion(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
    )
    response_text = response.choices[0].message.content
    print(response_text)
    # response_json = extract_and_fix_json_llm_call(json_str=response_text)
    # assert response_json == expected_output

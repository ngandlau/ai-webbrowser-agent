import os
from playwright.sync_api import sync_playwright, ViewportSize, Page


PLAYWRIGHT_USER_DATA_DIRECTORY = os.path.expanduser("~/playwright_user_data")
VIMIUM_PATH = "vimium" # vimium must be downloaded via google-extension-downloader and unzipped into this project's directory

def is_scrollable(page):
    return page.evaluate("""
        () => {
            const body = document.body;
            return body.scrollHeight > body.clientHeight;
        }
    """)

def get_scroll_info(page):
    return page.evaluate("""
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

def calculate_scroll_times(scroll_amount, viewport_height):
    scroll_times = scroll_amount // viewport_height
    return scroll_times

with sync_playwright() as p:
    browser = None 
    try:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=PLAYWRIGHT_USER_DATA_DIRECTORY,
            headless=False,
            args=[
                f"--disable-extensions-except={VIMIUM_PATH}",
                f"--load-extension={VIMIUM_PATH}"
            ],
            viewport={"width": 1200, "height": 1000},
            screen={"width": 1200, "height": 1000},
        )
        page = browser.new_page()

        # navigate to a website
        page.goto("https://safo.ebusy.de")
        page.screenshot(path="screenshots/fullpage.jpg", full_page=True)

        scroll_info = get_scroll_info(page)
        scrollable = scroll_info['scrollable']
        scroll_amount = scroll_info['scrollAmount']
        scroll_times = scroll_amount // 1000

        print(f"Is the page scrollable? {'Yes' if scrollable else 'No'}")
        if scrollable:
            print(f"The page is scrollable by {scroll_amount} pixels.")
            print(f"You can scroll down {scroll_times} times if you scroll by the height of the viewport each time.")

    finally:
        if browser:
            browser.close()
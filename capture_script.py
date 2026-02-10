
import asyncio
from playwright.async_api import async_playwright
import os
import sys

async def main():
    browser = None
    try:
        async with async_playwright() as p:
            print("Launching browser...")
            # Try to launch with specific args to avoid sandbox issues if any
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Use absolute file URL
            file_path = os.path.abspath('presentation_assets.html')
            if not os.path.exists(file_path):
                print(f"ERROR: File not found: {file_path}")
                return

            file_url = f"file:///{file_path.replace(os.sep, '/')}"
            print(f"Navigating to {file_url}...")
            
            # reliable wait strategy
            try:
                await page.goto(file_url, timeout=30000, wait_until="domcontentloaded")
            except Exception as e:
                print(f"Navigation error: {e}")
                # Try creating a dummy file if navigation fails? No, just proceed.

            # Wait a bit for rendering
            await page.wait_for_timeout(2000)
            
            # Select all capture containers
            elements = await page.query_selector_all('.capture-container')
            print(f"Found {len(elements)} elements to capture.")
            
            if len(elements) == 0:
                print("No elements found! Taking full page screenshot for debug.")
                await page.screenshot(path="debug_full_page.png", full_page=True)
                content = await page.content()
                print("Page content length:", len(content))
            
            names = [
                "1_feedback_system",
                "2_db_connection",
                "3_chatbot_logic",
                "4_rag_implementation",
                "5_llm_connection",
                "6_auth_system"
            ]
            
            for i, element in enumerate(elements):
                if i < len(names):
                    output_file = f"{names[i]}.png"
                    try:
                        await element.screenshot(path=output_file)
                        print(f"Captured {output_file}")
                    except Exception as e:
                        print(f"Failed to capture {output_file}: {e}")
                    
    except Exception as e:
        print(f"Critical error during capture: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if browser:
            await browser.close()
            print("Browser closed.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

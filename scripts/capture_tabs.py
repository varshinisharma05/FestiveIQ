import subprocess
import time
import json
import urllib.request
import asyncio
import websockets
import base64
from pathlib import Path

Path("screenshots").mkdir(exist_ok=True)

tabs_config = [
    (0, "overview_tab.png"),
    (1, "festive_planner_tab.png"),
    (2, "mnow_picker_tab.png"),
    (3, "pricing_simulator_tab.png"),
    (4, "analyst_tab.png"),
    (5, "brief_tab.png"),
]

proc = subprocess.Popen([
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "--headless=new",
    "--remote-debugging-port=9226",
    "--window-size=1440,1100",
    r"--user-data-dir=C:\Users\91929\AppData\Local\Temp\cdp_profile_tabs",
    "http://localhost:8501"
])

time.sleep(3)
try:
    tabs = json.loads(urllib.request.urlopen("http://localhost:9226/json").read())
    ws_url = tabs[0]["webSocketDebuggerUrl"]

    async def capture_all():
        async with websockets.connect(ws_url) as ws:
            await asyncio.sleep(6)  # wait for initial page load
            
            # Print detected tabs
            await ws.send(json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": "Array.from(document.querySelectorAll('div[role=\"tab\"]')).map(b => b.innerText.trim())",
                    "returnByValue": True
                }
            }))
            tab_names = json.loads(await ws.recv())["result"]["result"]["value"]
            print("Detected tabs:", tab_names)

            for idx, filename in tabs_config:
                click_js = f"""
                (function() {{
                    const tabs = document.querySelectorAll('div[role="tab"]');
                    if (tabs[{idx}]) {{
                        tabs[{idx}].scrollIntoView();
                        tabs[{idx}].click();
                        tabs[{idx}].dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true, view: window }}));
                        return tabs[{idx}].innerText.trim();
                    }}
                    return 'not found';
                }})()
                """
                await ws.send(json.dumps({
                    "id": 100 + idx,
                    "method": "Runtime.evaluate",
                    "params": {"expression": click_js, "returnByValue": True}
                }))
                clicked = json.loads(await ws.recv())["result"]["result"]["value"]
                print(f"Clicked tab {idx} ({clicked})")
                
                # Wait for tab content and any charts to render
                await asyncio.sleep(3.5)
                
                await ws.send(json.dumps({"id": 200 + idx, "method": "Page.captureScreenshot"}))
                res = json.loads(await ws.recv())
                img_data = base64.b64decode(res["result"]["data"])
                out_path = Path("screenshots") / filename
                with open(out_path, "wb") as f:
                    f.write(img_data)
                print(f"Saved {filename}: {len(img_data)} bytes")

    asyncio.run(capture_all())
finally:
    proc.terminate()

import asyncio
import websockets
from aiohttp import web
import socket
import logging
import json
import os
import aiohttp

# =====================================================================
# GLOBALS
# =====================================================================

csi_clients = set()
OPTIONS_FILE = "settings/options.json"

# ------------------ Notification Endpoints ------------------
# Active: ntfy topic
NTFY_TOPIC = "csi-activity-label"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# Backup (kept, but commented)
# CLOUDFLARE_URL = "https://csi-api.cipher0001.workers.dev/send?text="


# =====================================================================
# DETECT LAN IP
# =====================================================================

def get_local_ip():
    """Return the LAN IP of the computer."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# =====================================================================
# LOAD / FALLBACK OPTIONS
# =====================================================================

def load_saved_options():
    """Load saved dashboard settings or fallback to defaults."""
    os.makedirs("settings", exist_ok=True)

    if not os.path.exists(OPTIONS_FILE):
        return {
            "filters": [
                {"name": "Bending", "checked": True},
                {"name": "Fall Down", "checked": True},
                {"name": "Jumping", "checked": True},
                {"name": "Lying Down", "checked": True},
                {"name": "Sit Down", "checked": True},
                {"name": "Standing", "checked": True},
                {"name": "Walking", "checked": True},
            ],
            "notifications": "on"
        }

    try:
        with open(OPTIONS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"filters": [], "notifications": "off"}


# =====================================================================
# HUMAN-READABLE LABEL TEXT
# =====================================================================

def readable_label(label):
    clean = label.replace("[TEST] ", "")

    mapping = {
        "Bending": "Person is bending",
        "Fall Down": "⚠ PERSON FELL!",
        "Jumping": "Person is jumping",
        "Lying Down": "Person is lying down",
        "Sit Down": "Person is sitting down",
        "Standing": "Person is standing",
        "Walking": "Person is walking",
    }

    return mapping.get(clean, "Person activity: " + clean)


# =====================================================================
# SEND NOTIFICATION TO NTFY (ACTIVE)
# =====================================================================

async def send_api_notification(text: str):
    #print(f"[NTFY] Sending → {text}")

    headers = {
        "User-Agent": "CSI-Activity-Notifier/1.0",
        "Content-Type": "text/plain",
        "Title": "CSI Activity Alert",
        # "Priority": "high"   # optional
    }

    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                NTFY_URL,
                data=text,
                headers=headers,
                timeout=5
            )

            body = await resp.text()
            #print("[NTFY] Status:", resp.status)
            #print("[NTFY] Body:", body)

            if resp.status not in (200, 204):
                print("⚠ ntfy returned error:", resp.status, body)

    except Exception as e:
        print("⚠ ntfy send failed:", e)


# =====================================================================
# SAFE WS BROADCAST
# =====================================================================

async def csi_broadcast(payload: str):
    clients_copy = list(csi_clients)
    dead = []

    for ws in clients_copy:
        try:
            await ws.send(payload)
        except:
            dead.append(ws)

    for ws in dead:
        csi_clients.discard(ws)


# =====================================================================
# MAIN ENTRY: SEND LABEL + NOTIFY USING SAVED JSON
# =====================================================================

async def send_label(label: str):

    # =========================================================
    # 1) ALWAYS try pushing to dashboard (even if no client)
    # =========================================================
    payload = json.dumps({"image": "", "label": label})
    await csi_broadcast(payload)   # <- this is safe even with 0 clients

    # =========================================================
    # 2) SERVER-SIDE NOTIFICATIONS (HTML NOT REQUIRED)
    # =========================================================
    options = load_saved_options()

    if options.get("notifications", "on") != "on":
        return

    allowed = [
        x["name"]
        for x in options.get("filters", [])
        if x.get("checked", False)
    ]

    clean = label.replace("[TEST] ", "")

    if clean not in allowed:
        return

    text = readable_label(label)
    await send_api_notification(text)


# =====================================================================
# START CSI SERVER
# =====================================================================

async def start_csi_server():

    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    logging.getLogger("websockets.client").setLevel(logging.CRITICAL)

    LAN_IP = get_local_ip()

    # ---------------------------------------------------------
    # WEBSOCKET HANDLER
    # ---------------------------------------------------------
    async def ws_handler(websocket):
        print(f"Client connected: {websocket.remote_address}")
        csi_clients.add(websocket)

        try:
            async for _ in websocket:
                pass
        except:
            pass
        finally:
            print(f"Client disconnected: {websocket.remote_address}")
            csi_clients.discard(websocket)

    await websockets.serve(ws_handler, "0.0.0.0", 9000)
    print(f"🟢 WebSocket server ready on ws://{LAN_IP}:9000")

    # ---------------------------------------------------------
    # HTTP LOG VIEWER
    # ---------------------------------------------------------
    async def handle_logs(request):
        cors = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }

        if request.method == "OPTIONS":
            return web.Response(text="", headers=cors)

        try:
            with open("logs/activity_log.txt", "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            text = ""

        return web.Response(text=text, headers=cors)

    app = web.Application()
    app.router.add_route("GET", "/logs", handle_logs)
    app.router.add_route("OPTIONS", "/logs", handle_logs)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", 8001)
    await site.start()

    print(f"🟢 HTTP log server ready at http://{LAN_IP}:8001/logs")

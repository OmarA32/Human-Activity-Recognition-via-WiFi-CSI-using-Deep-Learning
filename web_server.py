from http.server import SimpleHTTPRequestHandler, HTTPServer
import json, os, socket

PORT = 8080
SETTINGS_FILE = "settings/options.json"

# Ensure folder exists
os.makedirs("settings", exist_ok=True)


# =====================================================
#   LAN IP DETECTION 
# =====================================================
def get_local_ip():
    """Automatically detect your LAN / hotspot IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return "127.0.0.1"
    finally:
        s.close()


# =====================================================
#   REQUEST HANDLER
# =====================================================
class Handler(SimpleHTTPRequestHandler):

    # Silence server logs (clean console)
    def log_message(self, *args):
        return

    # ---------- POST (Saving options) ----------
    def do_POST(self):
        if self.path == "/save-options":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode("utf-8")

            try:
                data = json.loads(body)
                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._send_json({"status": "ok"})
            except Exception as e:
                print("SAVE ERROR:", e)
                self._send_json({"status": "error"})
        else:
            self.send_error(404)

    # ---------- GET (Loading options or serving files) ----------
    def do_GET(self):
        if self.path == "/load-options":
            if not os.path.exists(SETTINGS_FILE):
                self._send_json({"status": "empty"})
                return

            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._send_json({"status": "ok", "data": data})
        else:
            super().do_GET()

    # Utility send JSON responses
    def _send_json(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(payload))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


# =====================================================
#   SERVER START
# =====================================================
def run():
    LAN_IP = get_local_ip()

    print("=========================================")
    print("🌐 Dashboard available:")
    print(f"   ▶ PC:      http://127.0.0.1:{PORT}")
    print(f"   ▶ Network: http://{LAN_IP}:{PORT}")
    print("=========================================")
    print("Press CTRL+C to stop.\n")

    with HTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    run()

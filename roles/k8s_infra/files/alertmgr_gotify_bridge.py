#!/usr/bin/env python3
"""Alertmanager → Gotify webhook bridge.

Receives Alertmanager webhook POSTs and forwards each alert as a Gotify
push notification. Configuration via environment variables:

  GOTIFY_URL    Base URL of the Gotify server (e.g. https://gotify.example.com)
  GOTIFY_TOKEN  Application token for posting messages
"""

import json
import os
import sys

import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

GOTIFY_URL = os.environ.get("GOTIFY_URL", "https://gotify.example.com")
GOTIFY_TOKEN = os.environ.get("GOTIFY_TOKEN", "")


def pp(msg: object) -> None:
    """Print a diagnostic message to stderr.

    Args:
        msg: The message to print. Accepts any object; converted via str().
    """
    print(msg, file=sys.stderr)


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Alertmanager → Gotify bridge.

    Handles health-check GET requests and Alertmanager webhook POST
    requests, forwarding each alert as a Gotify push notification.
    """

    def do_GET(self) -> None:
        """Handle GET requests for liveness and readiness probes.

        Responds with 200 OK and body ``OK`` for ``/healthz`` and
        ``/readyz``. All other paths return 404.
        """
        if self.path in ("/healthz", "/readyz"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests on ``/webhook``.

        Parses the Alertmanager JSON payload, iterates over the ``alerts``
        list, and POSTs each alert as a Gotify message. Fires with priority
        8 for ``firing`` alerts and 5 for resolved alerts. Returns 200 on
        success, 404 for unknown paths, and 500 if JSON parsing or the
        Gotify request raises an exception.
        """
        if self.path == "/webhook":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)

                status = data.get("status", "unknown")
                alerts = data.get("alerts", [])

                for alert in alerts:
                    annotations = alert.get("annotations", {})
                    labels = alert.get("labels", {})

                    title = f"{'🔥' if status == 'firing' else '✅'} {labels.get('alertname', 'Alert')}"
                    message = (
                        f"{annotations.get('summary', 'No summary')}\n"
                        f"{annotations.get('description', '')}\n"
                        f"Status: {status}"
                    )

                    resp = requests.post(
                        f"{GOTIFY_URL}/message",
                        params={"token": GOTIFY_TOKEN},
                        json={
                            "title": title,
                            "message": message,
                            "priority": 8 if status == "firing" else 5,
                        },
                    )

                    if resp.status_code != 200:
                        pp(f"Failed to send to Gotify: {resp.status_code} {resp.text}")

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")

            except Exception as e:
                pp(f"Error processing webhook: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    pp("Alertmanager-Gotify bridge __main__ invoked")
    server = HTTPServer(("0.0.0.0", 8080), WebhookHandler)
    pp("Alertmanager-Gotify bridge listening on :8080")
    server.serve_forever()

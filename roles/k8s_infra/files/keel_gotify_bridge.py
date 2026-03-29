#!/usr/bin/env python3
"""Keel → Gotify webhook bridge.

Receives Keel webhook POSTs and forwards each notification as a Gotify
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
    """Print a message to stderr.

    Args:
        msg: The message to print. Accepts any object; converted via ``str()``.
    """
    print(msg, file=sys.stderr)


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler that bridges Keel webhooks to Gotify notifications.

    Responds to health-check GET requests on ``/healthz`` and ``/readyz``, and
    processes Keel webhook POST requests on ``/webhook`` by forwarding them as
    Gotify push notifications.
    """

    def do_GET(self) -> None:
        """Handle GET requests.

        Returns HTTP 200 with body ``OK`` for ``/healthz`` and ``/readyz``
        health-check paths. All other paths return HTTP 404.
        """
        if self.path in ("/healthz", "/readyz"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests.

        Accepts a JSON payload on ``/webhook`` with the Keel notification
        format ``{"name": "...", "message": "...", "createdAt": "..."}``,
        constructs a Gotify message from it, and POSTs it to the Gotify server
        configured via ``GOTIFY_URL`` and ``GOTIFY_TOKEN``. Returns HTTP 200 on
        success, HTTP 500 if the payload cannot be parsed or forwarded, and
        HTTP 404 for any other path.
        """
        if self.path == "/webhook":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)

                # Keel payload: {"name": "...", "message": "...", "createdAt": "..."}
                name = data.get("name", "Keel")
                message = data.get("message", "No message")
                created_at = data.get("createdAt", "")

                title = f"⚙️ Keel: {name}"
                body_text = f"{message}\n{created_at}" if created_at else message

                resp = requests.post(
                    f"{GOTIFY_URL}/message",
                    params={"token": GOTIFY_TOKEN},
                    json={
                        "title": title,
                        "message": body_text,
                        "priority": 5,
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
    pp("Keel-Gotify bridge __main__ invoked")
    server = HTTPServer(("0.0.0.0", 8080), WebhookHandler)
    pp("Keel-Gotify bridge listening on :8080")
    server.serve_forever()

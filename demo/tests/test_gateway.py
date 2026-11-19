import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage_demo.gateway import create_app


class GatewayTests(unittest.TestCase):
    def test_static_page_health_and_backend_failure(self):
        cfg = {
            "worker_port": 19310,
            "worker_count": 2,
            "frontend_model_name": "Sage-Interaction",
            "backend_model_name": "test-model",
            "backend_base_url": "https://example.invalid/v1",
        }
        available = True

        def respond(request):
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "test-model"}]}
                    if available
                    else {"data": []},
                )
            index = request.url.port - 19310
            return httpx.Response(
                200, json={"gpu": index, "ready": True, "busy": index == 1}
            )

        original = httpx.AsyncClient

        def client(**kwargs):
            return original(transport=httpx.MockTransport(respond), **kwargs)

        with (
            patch.dict(os.environ, {"SAGE_BACKEND_API_KEY": "unit-test-only"}),
            patch("sage_demo.gateway.httpx.AsyncClient", client),
        ):
            with TestClient(create_app(cfg)) as browser:
                self.assertEqual(browser.get("/").status_code, 200)
                self.assertEqual(browser.get("/static/app.js").status_code, 200)
                response = browser.get("/api/health")
                self.assertEqual(response.json()["available_sessions"], 1)
                self.assertTrue(response.json()["ready"])
                self.assertNotIn("unit-test-only", response.text)
                self.assertNotIn("example.invalid", response.text)
                available = False
                self.assertFalse(browser.get("/api/health").json()["ready"])


if __name__ == "__main__":
    unittest.main()

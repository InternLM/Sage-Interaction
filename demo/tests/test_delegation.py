import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage_demo import protocol


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = json.loads((protocol.PROJECT / "config.example.json").read_text())
        self.cfg["backend_prompt"] = str(protocol.PROJECT / self.cfg["backend_prompt"])

    async def test_timer_does_not_block_dialogue_or_call_backend(self):
        gate = asyncio.Event()
        sleep = asyncio.sleep
        delays = []

        async def wait(delay):
            delays.append(delay)
            await gate.wait()

        with patch("asyncio.sleep", wait):
            task = asyncio.create_task(
                protocol.ask_backend(
                    None, self.cfg, "timer", "Remind me to stretch in 60 seconds"
                )
            )
            await sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(delays, [60.0])
            gate.set()
            result = await task
        self.assertEqual(result["payload"]["event"], "time_up")
        self.assertEqual(result["payload"]["task_id"], "timer")
        self.assertIn("<|im_start|>tool", result["event"])

    async def test_session_cancellation_cancels_pending_reminder(self):
        task = asyncio.create_task(
            protocol.ask_backend(None, self.cfg, "timer", "Remind me in 60 seconds")
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_reminder_limit_does_not_schedule_or_request_inference(self):
        result = await protocol.ask_backend(
            None, self.cfg, "timer", "Remind me after 30 minutes"
        )
        self.assertEqual(result["error"], "reminder_too_far")

    def test_relative_time_parser_leaves_ordinary_questions_unchanged(self):
        self.assertEqual(
            protocol.parse_timer_request("TimeSpeak task: stretch after two minutes"),
            120,
        )
        self.assertEqual(protocol.parse_timer_request("Remind me in half a minute"), 30)
        self.assertIsNone(
            protocol.parse_timer_request(
                "What is the speed of light in meters per second?"
            )
        )

    async def test_backend_authentication_stays_in_headers(self):
        def respond(request):
            self.assertEqual(request.headers["Authorization"], "Bearer unit-test-only")
            self.assertNotIn(b"unit-test-only", request.content)
            self.assertEqual(
                json.loads(request.content)["response_format"]["type"], "json_schema"
            )
            return httpx.Response(
                200,
                json={
                    "model": self.cfg["backend_model_name"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"content":"The answer is 42."}'},
                        }
                    ],
                },
            )

        with patch.dict(os.environ, {"SAGE_BACKEND_API_KEY": "unit-test-only"}):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as client:
                result = await protocol.ask_backend(
                    client, self.cfg, "question", "Calculate the answer"
                )
        self.assertEqual(result["payload"]["content"], "The answer is 42.")
        self.assertNotIn("unit-test-only", json.dumps(result))


if __name__ == "__main__":
    unittest.main()

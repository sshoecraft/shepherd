#!/usr/bin/env python3
"""
API Server Integration Tests

Tests the OpenAI-compatible API server endpoints.
These tests start a real shepherd server with --apiserver and test against it.

Usage:
    PROVIDER=<provider_name> python3 test_api_server.py

Test IDs (from docs/testing.md):
    API-001: POST /v1/chat/completions - valid request
    API-002: POST with streaming - SSE events
    API-003: POST with tools - tool_calls in response
    API-004: POST invalid JSON - 400 error
    API-005: GET /v1/models - model list
    API-006: GET /health - healthy status
    API-007: Tool result submission
    API-008: Context limit handling
    API-009: Prefix caching - multiple sequential requests (CRITICAL: catches KV cache bug)
"""

import os
import sys
import time
import json
import signal
import subprocess
import unittest
import requests
from typing import Optional

# Configuration
DEFAULT_PORT = 18080
DEFAULT_HOST = "localhost"
SHEPHERD_BINARY = os.environ.get("SHEPHERD_BINARY", "./build/shepherd")
PROVIDER = os.environ.get("PROVIDER", "")
MODEL = os.environ.get("MODEL", "")
API_KEY = os.environ.get("API_KEY", "test-key")  # For auth if enabled

# Try to use OpenAI client if available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed, some tests will use raw HTTP")


class ShepherdServer:
    """Context manager for starting/stopping shepherd server"""

    def __init__(self, provider: str, port: int = DEFAULT_PORT, extra_args: list = None):
        self.provider = provider
        self.port = port
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{DEFAULT_HOST}:{port}"

    def __enter__(self):
        if not self.provider:
            raise ValueError("PROVIDER environment variable must be set")

        cmd = [
            SHEPHERD_BINARY,
            "--provider", self.provider,
            "--apiserver",
            "--port", str(self.port),
        ] + self.extra_args

        print(f"Starting server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait for server to be ready
        max_wait = 30
        for i in range(max_wait):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print(f"Server ready after {i+1} seconds")
                    return self
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        self._cleanup()
        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def _cleanup(self):
        if self.process:
            # Try graceful shutdown first
            try:
                shutdown_response = requests.post(
                    f"{self.base_url}/shutdown",
                    timeout=5
                )
            except:
                pass

            # Give it a moment
            time.sleep(1)

            # Force kill if still running
            if self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=5)
                except:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

            self.process = None


class TestAPIServer(unittest.TestCase):
    """API Server Integration Tests"""

    server: Optional[ShepherdServer] = None
    base_url: str = ""
    client = None  # OpenAI client if available

    @classmethod
    def setUpClass(cls):
        """Start the server once for all tests"""
        if not PROVIDER:
            raise unittest.SkipTest("PROVIDER environment variable not set")

        cls.server = ShepherdServer(PROVIDER)
        cls.server.__enter__()
        cls.base_url = cls.server.base_url

        if HAS_OPENAI:
            cls.client = OpenAI(
                base_url=f"{cls.base_url}/v1",
                api_key=API_KEY
            )

    @classmethod
    def tearDownClass(cls):
        """Stop the server after all tests"""
        if cls.server:
            cls.server.__exit__(None, None, None)

    # =========================================================================
    # API-001: POST /v1/chat/completions - valid request
    # =========================================================================

    def test_API001_chat_completions_valid(self):
        """Test valid chat completion request"""
        payload = {
            "model": MODEL or "default",
            "messages": [
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            "max_tokens": 50
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}: {response.text}")

        data = response.json()
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)
        self.assertIn("message", data["choices"][0])
        self.assertIn("content", data["choices"][0]["message"])
        self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0,
                       "Response content should not be empty")

    def test_API001_chat_completions_with_openai_client(self):
        """Test chat completion using OpenAI client"""
        if not HAS_OPENAI:
            self.skipTest("OpenAI client not available")

        response = self.client.chat.completions.create(
            model=MODEL or "default",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=50
        )

        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertTrue(len(response.choices[0].message.content) > 0)

    # =========================================================================
    # API-002: POST with streaming - SSE events
    # =========================================================================

    def test_API002_streaming(self):
        """Test streaming chat completion"""
        payload = {
            "model": MODEL or "default",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            "max_tokens": 100,
            "stream": True
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )

        self.assertEqual(response.status_code, 200)

        # Collect SSE events
        events = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str != "[DONE]":
                        events.append(json.loads(data_str))

        self.assertGreater(len(events), 0, "Should receive SSE events")

        # Check event structure
        for event in events:
            self.assertIn("choices", event)
            if event["choices"]:
                self.assertIn("delta", event["choices"][0])

    def test_API002_streaming_with_openai_client(self):
        """Test streaming using OpenAI client"""
        if not HAS_OPENAI:
            self.skipTest("OpenAI client not available")

        stream = self.client.chat.completions.create(
            model=MODEL or "default",
            messages=[{"role": "user", "content": "Say 'stream test'."}],
            max_tokens=50,
            stream=True
        )

        chunks = list(stream)
        self.assertGreater(len(chunks), 0, "Should receive streaming chunks")

    # =========================================================================
    # API-003: POST with tools - tool_calls in response
    # =========================================================================

    def test_API003_tools_request(self):
        """Test chat completion with tools"""
        payload = {
            "model": MODEL or "default",
            "messages": [
                {"role": "user", "content": "What is 2 + 2? Use the calculator tool."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Perform arithmetic calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "The math expression to evaluate"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Response should have choices
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)

        # Model may or may not use the tool - just verify the request succeeded
        # Tool calls would be in choices[0].message.tool_calls if used

    # =========================================================================
    # API-004: POST invalid JSON - 400 error
    # =========================================================================

    def test_API004_invalid_json(self):
        """Test invalid JSON returns 400"""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(response.status_code, 400,
                        f"Invalid JSON should return 400, got {response.status_code}")

    def test_API004_missing_messages(self):
        """Test missing messages field returns 400"""
        payload = {
            "model": MODEL or "default"
            # Missing 'messages' field
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(response.status_code, 400,
                        "Missing messages should return 400")

    # =========================================================================
    # API-005: GET /v1/models - model list
    # =========================================================================

    def test_API005_list_models(self):
        """Test GET /v1/models returns model list"""
        response = requests.get(f"{self.base_url}/v1/models")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        self.assertGreater(len(data["data"]), 0, "Should have at least one model")

        # Check model object structure
        model = data["data"][0]
        self.assertIn("id", model)
        self.assertIn("object", model)
        self.assertEqual(model["object"], "model")

    # =========================================================================
    # API-006: GET /health - healthy status
    # =========================================================================

    def test_API006_health_check(self):
        """Test GET /health returns healthy status"""
        response = requests.get(f"{self.base_url}/health")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    # =========================================================================
    # API-007: Tool result submission
    # =========================================================================

    def test_API007_tool_result(self):
        """Test submitting tool results in conversation"""
        # First, get a response that might use tools
        messages = [
            {"role": "user", "content": "What is 5 + 3?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "5 + 3"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "8"
            }
        ]

        payload = {
            "model": MODEL or "default",
            "messages": messages,
            "max_tokens": 100
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Should accept tool results without error
        self.assertEqual(response.status_code, 200,
                        f"Tool result submission failed: {response.text}")

    # =========================================================================
    # API-008: Context limit handling
    # =========================================================================

    def test_API008_context_limit(self):
        """Test handling of large context"""
        # Create a message with substantial content
        long_content = "This is a test message. " * 100  # ~2500 chars

        payload = {
            "model": MODEL or "default",
            "messages": [
                {"role": "user", "content": long_content + " Summarize this in one word."}
            ],
            "max_tokens": 50
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Should handle large context (either success or appropriate error)
        self.assertIn(response.status_code, [200, 400],
                     f"Unexpected status: {response.status_code}")

    # =========================================================================
    # API-009: Prefix caching - multiple sequential requests
    # CRITICAL: This test would have caught the KV cache sync bug!
    # =========================================================================

    def test_API009_sequential_requests_no_alternating_failures(self):
        """
        Test multiple sequential requests don't have alternating failures.

        This is the CRITICAL test that would have caught the KV cache sync bug
        where channel-based models (like GPT-OSS) had alternating success/failure
        due to KV cache not being properly synced after generation.
        """
        results = []
        error_count = 0

        # Make 5 sequential requests with the same prompt
        for i in range(5):
            payload = {
                "model": MODEL or "default",
                "messages": [
                    {"role": "user", "content": "Say 'test response' and nothing else."}
                ],
                "max_tokens": 20  # Small max_tokens like the bug scenario
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            result = {
                "iteration": i + 1,
                "status_code": response.status_code,
                "success": False,
                "content": None,
                "error": None
            }

            if response.status_code == 200:
                try:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    result["content"] = content
                    result["success"] = len(content.strip()) > 0
                except:
                    result["error"] = "Failed to parse response"
            else:
                result["error"] = response.text[:200]
                error_count += 1

            results.append(result)
            print(f"  Request {i+1}: success={result['success']}, content={result['content'][:50] if result['content'] else 'N/A'}")

        # Check for alternating pattern (the bug symptom)
        successes = [r["success"] for r in results]

        # If we have alternating True/False/True/False pattern, that's the bug
        is_alternating = all(successes[i] != successes[i+1] for i in range(len(successes)-1))

        self.assertFalse(is_alternating and len(set(successes)) == 2,
                        f"DETECTED ALTERNATING SUCCESS/FAILURE PATTERN! "
                        f"This indicates the KV cache sync bug. Results: {successes}")

        # All requests should succeed (no errors)
        self.assertEqual(error_count, 0,
                        f"Had {error_count} errors in {len(results)} requests")

        # All responses should have non-empty content
        empty_responses = sum(1 for r in results if not r["success"])
        self.assertEqual(empty_responses, 0,
                        f"Had {empty_responses} empty responses out of {len(results)}")

    def test_API009_with_openai_client(self):
        """Test sequential requests using OpenAI client"""
        if not HAS_OPENAI:
            self.skipTest("OpenAI client not available")

        results = []

        for i in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL or "default",
                    messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
                    max_tokens=10
                )
                content = response.choices[0].message.content or ""
                results.append({"success": len(content.strip()) > 0, "content": content})
            except Exception as e:
                results.append({"success": False, "error": str(e)})

        successes = [r["success"] for r in results]

        # Check for alternating pattern
        is_alternating = all(successes[i] != successes[i+1] for i in range(len(successes)-1))
        self.assertFalse(is_alternating and len(set(successes)) == 2,
                        "Detected alternating success/failure pattern!")

        # All should succeed
        self.assertTrue(all(successes),
                       f"Not all requests succeeded: {results}")


if __name__ == "__main__":
    if not PROVIDER:
        print("ERROR: PROVIDER environment variable must be set")
        print("Usage: PROVIDER=<provider_name> python3 test_api_server.py")
        print("Example: PROVIDER=openai python3 test_api_server.py")
        sys.exit(1)

    print(f"Running API Server tests with PROVIDER={PROVIDER}")
    unittest.main(verbosity=2)

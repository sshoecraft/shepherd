#!/usr/bin/env python3
"""
Comprehensive Eviction Test Suite for Shepherd
Tests all eviction scenarios across multiple context sizes
"""

import openai
import json
import time
import sys
import argparse
import requests
from typing import List, Dict, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# ============================================================================
# Context Size Profiles
# ============================================================================

CONTEXT_PROFILES = {
    "micro": {
        "size": 512,
        "description": "Ultra-small for rapid eviction testing",
    },
    "tiny": {
        "size": 4096,  # Changed from 2048 to 4096
        "description": "Small context for fast boundary testing",
    },
    "small": {
        "size": 8192,
        "description": "Moderate context for pattern testing",
    },
    "medium": {
        "size": 16384,  # Changed from 32768 to 16384
        "description": "Medium context for testing",
    },
    "large": {
        "size": 32768,  # Changed from 131072 to 32768
        "description": "Large context for stress testing",
    }
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestResult:
    id: str
    name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_ms: float
    context_size: int
    server_context_size: int
    error: Optional[str] = None
    metrics: Dict[str, Any] = None
    verification: Dict[str, bool] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.verification is None:
            self.verification = {}

@dataclass
class EvictionMetrics:
    message_count_before: int = 0
    message_count_after: int = 0
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    utilization_before: float = 0.0
    utilization_after: float = 0.0
    eviction_occurred: bool = False
    messages_evicted: int = 0
    tokens_freed: int = 0
    request_latency_ms: float = 0.0
    server_error: bool = False
    status_code: Optional[int] = None

# ============================================================================
# Test Suite
# ============================================================================

class EvictionTestSuite:
    def __init__(self, api_base: str, model: str,
                 server_context_size: Optional[int] = None,
                 test_mode: str = "standard",
                 verbose: bool = False):

        self.api_base = api_base
        self.model = model
        self.verbose = verbose
        self.test_mode = test_mode
        self.results = []

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key="dummy", base_url=api_base)

        # Detect server's actual context size
        self.server_context_size = server_context_size or self._detect_server_context()

        # Current test context (can be overridden per test)
        self.current_test_context = self.server_context_size

        print(f"\n{'='*70}")
        print(f"Shepherd Eviction Test Suite")
        print(f"{'='*70}")
        print(f"Target: {api_base}")
        print(f"Model: {model}")
        print(f"Server Context Size: {self.server_context_size:,} tokens")
        print(f"Test Mode: {test_mode}")
        print(f"{'='*70}\n")

    def _detect_server_context(self) -> int:
        """Query server to get actual context window size"""
        try:
            # Try OpenAI-compatible models endpoint
            resp = requests.get(
                f"{self.api_base}/models/{self.model}",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if 'context_window' in data:
                    return data['context_window']
                if 'max_model_len' in data:
                    return data['max_model_len']
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not detect server context size: {e}")

        # Default fallback
        return 128000

    def estimate_tokens(self, text: str) -> int:
        """Conservative token estimation (4 chars ≈ 1 token)"""
        return int(len(text) / 4.0 + 0.5)

    def create_message(self, role: str, size_tokens: int) -> Dict:
        """Generate message of specific token size"""
        # Account for role overhead (~10 tokens)
        target_chars = max(1, (size_tokens - 10) * 4)
        content = "X" * int(target_chars)
        return {"role": role, "content": content}

    def measure_context_state(self, messages: List[Dict]) -> Dict:
        """Calculate context metrics"""
        total_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in messages)
        return {
            "message_count": len(messages),
            "estimated_tokens": total_tokens,
            "utilization": total_tokens / self.current_test_context if self.current_test_context > 0 else 0.0
        }

    def set_test_context(self, size: Union[int, str, Callable]):
        """Override context size for testing"""
        if isinstance(size, int):
            self.current_test_context = size
        elif isinstance(size, str):
            if size in ["use_server_default", "match_server"]:
                self.current_test_context = self.server_context_size
            elif size in CONTEXT_PROFILES:
                self.current_test_context = CONTEXT_PROFILES[size]["size"]
            else:
                raise ValueError(f"Unknown profile: {size}")
        elif callable(size):
            self.current_test_context = size(self.server_context_size)
        else:
            raise TypeError(f"Invalid context size type: {type(size)}")

    def fill_to_capacity(self, messages: List[Dict], target_tokens: int, pair_size: int = 100) -> List[Dict]:
        """Fill messages list to target token count"""
        current_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in messages)

        while current_tokens < target_tokens:
            remaining = target_tokens - current_tokens
            msg_size = min(pair_size, remaining)

            messages.append(self.create_message("user", msg_size))
            current_tokens += msg_size

            if current_tokens < target_tokens:
                remaining = target_tokens - current_tokens
                msg_size = min(pair_size, remaining)
                messages.append(self.create_message("assistant", msg_size))
                current_tokens += msg_size

        return messages

    def send_request(self, messages: List[Dict], max_tokens: int = 150) -> tuple[Optional[Any], EvictionMetrics]:
        """Send chat completion request and track metrics"""
        metrics = EvictionMetrics()

        # Measure before
        state_before = self.measure_context_state(messages)
        metrics.message_count_before = state_before["message_count"]
        metrics.estimated_tokens_before = state_before["estimated_tokens"]
        metrics.utilization_before = state_before["utilization"]

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            metrics.request_latency_ms = (time.time() - start_time) * 1000
            metrics.status_code = 200

            # Add assistant response to messages for state tracking
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": content})

            # Measure after
            state_after = self.measure_context_state(messages)
            metrics.message_count_after = state_after["message_count"]
            metrics.estimated_tokens_after = state_after["estimated_tokens"]
            metrics.utilization_after = state_after["utilization"]

            # Detect eviction (messages disappeared)
            if metrics.message_count_after < metrics.message_count_before:
                metrics.eviction_occurred = True
                metrics.messages_evicted = metrics.message_count_before - metrics.message_count_after
                metrics.tokens_freed = metrics.estimated_tokens_before - metrics.estimated_tokens_after

            return response, metrics

        except Exception as e:
            metrics.request_latency_ms = (time.time() - start_time) * 1000
            metrics.server_error = True

            # Try to extract status code from error
            error_str = str(e)
            if "400" in error_str:
                metrics.status_code = 400
            elif "500" in error_str:
                metrics.status_code = 500

            return None, metrics

    def run_test(self, test_id: str, test_name: str, test_func: Callable, context_size: Union[int, str]) -> TestResult:
        """Run a single test"""
        self.set_test_context(context_size)

        client_manages_eviction = self.current_test_context < self.server_context_size

        print(f"\n[RUNNING] Test {test_id}: {test_name}")
        print(f"  Context: {self.current_test_context:,} tokens (server: {self.server_context_size:,})")
        if client_manages_eviction:
            print(f"  Mode: CLIENT-SIDE EVICTION")

        result = TestResult(
            id=test_id,
            name=test_name,
            status="RUNNING",
            duration_ms=0,
            context_size=self.current_test_context,
            server_context_size=self.server_context_size
        )

        start_time = time.time()

        try:
            test_func(result)
            result.status = "PASS" if not result.error else "FAIL"

        except Exception as e:
            result.status = "ERROR"
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                result.error += f"\n{traceback.format_exc()}"

        result.duration_ms = (time.time() - start_time) * 1000

        # Print result
        status_symbol = "✓" if result.status == "PASS" else "✗"
        print(f"  [{status_symbol} {result.status}] {result.duration_ms:.0f}ms")

        if result.error and result.status != "PASS":
            print(f"  Error: {result.error}")

        self.results.append(result)
        return result

    # ========================================================================
    # CATEGORY 1: BOUNDARY CONDITION TESTS
    # ========================================================================

    def test_1_1_exact_capacity(self, result: TestResult):
        """Fill context to exactly max capacity - should NOT trigger eviction"""
        messages = [self.create_message("system", 50)]

        # Fill to 99.8% of capacity
        target_tokens = int(self.current_test_context * 0.998)
        messages = self.fill_to_capacity(messages, target_tokens)

        response, metrics = self.send_request(messages, max_tokens=20)

        result.metrics = asdict(metrics)
        result.verification = {
            "no_eviction": not metrics.eviction_occurred,
            "under_limit": metrics.utilization_before < 1.0,
            "response_success": response is not None
        }

        if metrics.eviction_occurred:
            result.error = "Unexpected eviction at 99.8% capacity"

    def test_1_2_one_token_over(self, result: TestResult):
        """Exceed by small amount - should evict minimal messages"""
        messages = [self.create_message("system", 50)]

        # Fill to capacity - 8 tokens
        target_tokens = self.current_test_context - 58  # 50 system + 8 buffer
        messages = self.fill_to_capacity(messages, target_tokens, pair_size=100)

        # Add message that exceeds by ~12 tokens
        messages.append(self.create_message("user", 20))

        response, metrics = self.send_request(messages, max_tokens=20)

        result.metrics = asdict(metrics)
        result.verification = {
            "eviction_occurred": metrics.eviction_occurred,
            "small_eviction": metrics.messages_evicted <= 4 if metrics.eviction_occurred else True
        }

    def test_1_3_massive_overage(self, result: TestResult):
        """Add huge message requiring eviction of majority of context"""
        messages = [self.create_message("system", 50)]

        # Fill to 50%
        target_tokens = int(self.current_test_context * 0.50)
        messages = self.fill_to_capacity(messages, target_tokens, pair_size=100)

        # Add message that's 60% of context
        huge_size = int(self.current_test_context * 0.60)
        messages.append(self.create_message("user", huge_size))

        response, metrics = self.send_request(messages, max_tokens=100)

        result.metrics = asdict(metrics)
        result.verification = {
            "eviction_occurred": metrics.eviction_occurred,
            "large_eviction": metrics.messages_evicted >= 10 if metrics.eviction_occurred else True,
            "response_success": response is not None
        }

    def test_1_4_cannot_evict(self, result: TestResult):
        """System + User fill context - cannot evict protected messages"""
        system_size = int(self.current_test_context * 0.45)
        user_size = int(self.current_test_context * 0.60)  # Total 105%

        messages = [
            self.create_message("system", system_size),
            self.create_message("user", user_size)
        ]

        response, metrics = self.send_request(messages, max_tokens=100)

        result.metrics = asdict(metrics)
        result.verification = {
            "error_occurred": metrics.server_error or response is None,
            "context_error": metrics.status_code == 400 if metrics.status_code else False
        }

        if not metrics.server_error and response is not None:
            result.error = "Expected error when protected messages exceed context"

    # ========================================================================
    # CATEGORY 2: MESSAGE SIZE VARIATION TESTS
    # ========================================================================

    def test_2_1_tiny_messages(self, result: TestResult):
        """Fill context with very small messages"""
        messages = [self.create_message("system", 50)]

        eviction_count = 0
        for i in range(500):  # Many tiny messages
            messages.append({"role": "user", "content": "Hi"})

            response, metrics = self.send_request(messages, max_tokens=5)

            if metrics.eviction_occurred:
                eviction_count += 1

            if metrics.server_error:
                break

            if i >= 100 and eviction_count == 0:  # Should have evicted by now
                break

        result.metrics = {"eviction_count": eviction_count, "iterations": i + 1}
        result.verification = {
            "evictions_occurred": eviction_count > 0
        }

        if eviction_count == 0:
            result.error = "No evictions after 100+ tiny messages"

    def test_2_2_mixed_sizes(self, result: TestResult):
        """Random distribution of message sizes"""
        import random
        messages = [self.create_message("system", 100)]

        sizes = [10, 50, 100, 200, 500]
        eviction_count = 0

        for i in range(50):
            size = random.choice(sizes)
            messages.append(self.create_message("user", size))

            response, metrics = self.send_request(messages, max_tokens=size)

            if metrics.eviction_occurred:
                eviction_count += 1

            if metrics.server_error:
                break

        result.metrics = {"eviction_count": eviction_count, "iterations": i + 1}
        result.verification = {
            "evictions_occurred": eviction_count > 0
        }

    # ========================================================================
    # CATEGORY 7: CLIENT vs SERVER CONTEXT SIZE MISMATCH
    # ========================================================================

    def test_7_1_detect_server_context(self, result: TestResult):
        """Detect server's default context size"""
        detected_size = self._detect_server_context()

        result.metrics = {
            "detected_context_size": detected_size
        }
        result.verification = {
            "size_detected": detected_size > 0,
            "reasonable_size": 512 <= detected_size <= 1000000
        }

        print(f"  Detected server context: {detected_size:,} tokens")

    def test_7_2_client_smaller_than_server(self, result: TestResult):
        """Client context < server context - client should handle eviction"""
        # Note: current_test_context is already set by run_test() before calling this function
        # Verify that client context is smaller than server
        if self.current_test_context >= self.server_context_size:
            result.error = f"Test setup error: client context ({self.current_test_context}) >= server context ({self.server_context_size})"
            return

        messages = [self.create_message("system", 50)]

        # Fill to client capacity
        target_tokens = int(self.current_test_context * 0.80)
        messages = self.fill_to_capacity(messages, target_tokens)

        eviction_count = 0

        # Add many more messages to trigger multiple evictions
        for i in range(20):
            messages.append(self.create_message("user", 150))
            response, metrics = self.send_request(messages, max_tokens=150)

            if metrics.eviction_occurred:
                eviction_count += 1

            # Should never get server error since client manages eviction
            if metrics.server_error:
                result.error = f"Server error occurred when client should handle eviction"
                break

        result.metrics = {"eviction_count": eviction_count}
        result.verification = {
            "client_evicted": eviction_count > 0,
            "no_server_errors": not any(r.metrics.get("server_error") for r in self.results if r.id == result.id)
        }

    def test_7_3_client_larger_than_server(self, result: TestResult):
        """Client context > server context - server should reject"""
        # Set client context much larger than server
        self.current_test_context = self.server_context_size * 2

        messages = [self.create_message("system", 100)]

        # Fill beyond server capacity
        target_tokens = int(self.server_context_size * 1.1)
        messages = self.fill_to_capacity(messages, target_tokens)

        response, metrics = self.send_request(messages, max_tokens=100)

        result.metrics = asdict(metrics)
        result.verification = {
            "server_rejected": metrics.server_error,
            "status_400": metrics.status_code == 400 if metrics.status_code else False
        }

        if not metrics.server_error:
            result.error = "Expected server to reject oversized context"

    # ========================================================================
    # Test Runner
    # ========================================================================

    def run_all_tests(self):
        """Run all tests based on test mode"""

        # Category 1: Boundary Conditions - Test sequence: 4k, 8k, 16k, 32k, server default
        if self.test_mode in ["fast", "standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 1: Boundary Condition Tests (4K context)")
            print(f"{'='*70}")

            self.run_test("1.1", "Context Exactly at Limit",
                         self.test_1_1_exact_capacity, "tiny")
            self.run_test("1.2", "One Token Over Limit",
                         self.test_1_2_one_token_over, "tiny")
            self.run_test("1.3", "Massively Over Limit (50% overage)",
                         self.test_1_3_massive_overage, "tiny")
            self.run_test("1.4", "System + User Fill Context (Cannot Evict)",
                         self.test_1_4_cannot_evict, "tiny")

            print(f"\n{'='*70}")
            print("Category 1: Boundary Condition Tests (8K context)")
            print(f"{'='*70}")

            self.run_test("1.1b", "Context Exactly at Limit",
                         self.test_1_1_exact_capacity, "small")
            self.run_test("1.2b", "One Token Over Limit",
                         self.test_1_2_one_token_over, "small")
            self.run_test("1.3b", "Massively Over Limit (50% overage)",
                         self.test_1_3_massive_overage, "small")
            self.run_test("1.4b", "System + User Fill Context (Cannot Evict)",
                         self.test_1_4_cannot_evict, "small")

            print(f"\n{'='*70}")
            print("Category 1: Boundary Condition Tests (16K context)")
            print(f"{'='*70}")

            self.run_test("1.1c", "Context Exactly at Limit",
                         self.test_1_1_exact_capacity, "medium")
            self.run_test("1.2c", "One Token Over Limit",
                         self.test_1_2_one_token_over, "medium")
            self.run_test("1.3c", "Massively Over Limit (50% overage)",
                         self.test_1_3_massive_overage, "medium")
            self.run_test("1.4c", "System + User Fill Context (Cannot Evict)",
                         self.test_1_4_cannot_evict, "medium")

            print(f"\n{'='*70}")
            print("Category 1: Boundary Condition Tests (32K context)")
            print(f"{'='*70}")

            self.run_test("1.1d", "Context Exactly at Limit",
                         self.test_1_1_exact_capacity, "large")
            self.run_test("1.2d", "One Token Over Limit",
                         self.test_1_2_one_token_over, "large")
            self.run_test("1.3d", "Massively Over Limit (50% overage)",
                         self.test_1_3_massive_overage, "large")
            self.run_test("1.4d", "System + User Fill Context (Cannot Evict)",
                         self.test_1_4_cannot_evict, "large")

            print(f"\n{'='*70}")
            print("Category 1: Boundary Condition Tests (Server Default)")
            print(f"{'='*70}")

            self.run_test("1.1e", "Context Exactly at Limit",
                         self.test_1_1_exact_capacity, "use_server_default")
            self.run_test("1.2e", "One Token Over Limit",
                         self.test_1_2_one_token_over, "use_server_default")
            self.run_test("1.3e", "Massively Over Limit (50% overage)",
                         self.test_1_3_massive_overage, "use_server_default")
            self.run_test("1.4e", "System + User Fill Context (Cannot Evict)",
                         self.test_1_4_cannot_evict, "use_server_default")

        # Category 2: Message Size Variations - Test sequence: 4k, 8k, 16k, 32k, server default
        if self.test_mode in ["fast", "standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 2: Message Size Variation Tests (4K context)")
            print(f"{'='*70}")

            self.run_test("2.1", "Tiny Messages (< 10 tokens)",
                         self.test_2_1_tiny_messages, "tiny")
            self.run_test("2.2", "Mixed Size Distribution",
                         self.test_2_2_mixed_sizes, "tiny")

            print(f"\n{'='*70}")
            print("Category 2: Message Size Variation Tests (8K context)")
            print(f"{'='*70}")

            self.run_test("2.1b", "Tiny Messages (< 10 tokens)",
                         self.test_2_1_tiny_messages, "small")
            self.run_test("2.2b", "Mixed Size Distribution",
                         self.test_2_2_mixed_sizes, "small")

            print(f"\n{'='*70}")
            print("Category 2: Message Size Variation Tests (16K context)")
            print(f"{'='*70}")

            self.run_test("2.1c", "Tiny Messages (< 10 tokens)",
                         self.test_2_1_tiny_messages, "medium")
            self.run_test("2.2c", "Mixed Size Distribution",
                         self.test_2_2_mixed_sizes, "medium")

            print(f"\n{'='*70}")
            print("Category 2: Message Size Variation Tests (32K context)")
            print(f"{'='*70}")

            self.run_test("2.1d", "Tiny Messages (< 10 tokens)",
                         self.test_2_1_tiny_messages, "large")
            self.run_test("2.2d", "Mixed Size Distribution",
                         self.test_2_2_mixed_sizes, "large")

            print(f"\n{'='*70}")
            print("Category 2: Message Size Variation Tests (Server Default)")
            print(f"{'='*70}")

            self.run_test("2.1e", "Tiny Messages (< 10 tokens)",
                         self.test_2_1_tiny_messages, "use_server_default")
            self.run_test("2.2e", "Mixed Size Distribution",
                         self.test_2_2_mixed_sizes, "use_server_default")

        # Category 7: Client vs Server Mismatch
        if self.test_mode in ["fast", "standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 7: Client vs Server Context Size Mismatch")
            print(f"{'='*70}")

            self.run_test("7.1", "Detect Server Context Size",
                         self.test_7_1_detect_server_context, "use_server_default")

            print(f"\n{'='*70}")
            print("Category 7: Client Smaller Tests - 4K, 8K, 16K, 32K")
            print(f"{'='*70}")

            self.run_test("7.2a", "Client Context < Server (Auto-Eviction) - 4K",
                         self.test_7_2_client_smaller_than_server, "tiny")
            self.run_test("7.2b", "Client Context < Server (Auto-Eviction) - 8K",
                         self.test_7_2_client_smaller_than_server, "small")
            self.run_test("7.2c", "Client Context < Server (Auto-Eviction) - 16K",
                         self.test_7_2_client_smaller_than_server, "medium")
            self.run_test("7.2d", "Client Context < Server (Auto-Eviction) - 32K",
                         self.test_7_2_client_smaller_than_server, "large")

            print(f"\n{'='*70}")
            print("Category 7: Client Larger Tests")
            print(f"{'='*70}")

            self.run_test("7.3", "Client Context > Server (Server Rejects)",
                         self.test_7_3_client_larger_than_server,
                         lambda s: s * 2)

    # ========================================================================
    # Reporting
    # ========================================================================

    def print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        errors = sum(1 for r in self.results if r.status == "ERROR")

        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")

        total_evictions = sum(
            r.metrics.get("eviction_count", 0) if isinstance(r.metrics, dict)
            else r.metrics.get("eviction_occurred", 0)
            for r in self.results
        )
        print(f"\nTotal Evictions Detected: {total_evictions}")

        avg_duration = sum(r.duration_ms for r in self.results) / total if total > 0 else 0
        print(f"Average Test Duration: {avg_duration:.0f}ms")

        if failed > 0 or errors > 0:
            print(f"\nFailed/Error Tests:")
            for r in self.results:
                if r.status in ["FAIL", "ERROR"]:
                    print(f"  - {r.id}: {r.name}")
                    if r.error:
                        print(f"    {r.error}")

        print(f"\n{'='*70}\n")

        return passed == total

    def save_report(self, filename: str):
        """Save detailed JSON report"""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "api_base": self.api_base,
                "model": self.model,
                "server_context_size": self.server_context_size,
                "test_mode": self.test_mode,
                "duration_seconds": sum(r.duration_ms for r in self.results) / 1000
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "PASS"),
                "failed": sum(1 for r in self.results if r.status == "FAIL"),
                "errors": sum(1 for r in self.results if r.status == "ERROR"),
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {filename}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Eviction Test Suite for Shepherd"
    )
    parser.add_argument("--api-base",
                       default="http://192.168.1.166:8000/v1",
                       help="API base URL")
    parser.add_argument("--model",
                       default="gpt-4",
                       help="Model name")
    parser.add_argument("--mode",
                       choices=["fast", "standard", "full"],
                       default="standard",
                       help="Test mode")
    parser.add_argument("--context-size",
                       type=int,
                       help="Override server context size")
    parser.add_argument("--output",
                       help="Output JSON report file")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Load config if available
    config_path = None
    try:
        import os
        config_path = os.path.expanduser("~/.shepherd/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                if not args.api_base and config.get("api_base"):
                    args.api_base = config["api_base"]
                print(f"Loaded config from {config_path}")
    except Exception as e:
        if args.verbose:
            print(f"Could not load config: {e}")

    # Create test suite
    suite = EvictionTestSuite(
        api_base=args.api_base,
        model=args.model,
        server_context_size=args.context_size,
        test_mode=args.mode,
        verbose=args.verbose
    )

    # Run tests
    try:
        suite.run_all_tests()

        # Print summary
        all_passed = suite.print_summary()

        # Save report
        if args.output:
            suite.save_report(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suite.save_report(f"eviction_test_report_{timestamp}.json")

        # Exit code
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        suite.print_summary()
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

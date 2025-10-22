#!/usr/bin/env python3
"""
Unified Shepherd CLIENT Eviction Test Suite

Tests the shepherd binary's client-side eviction and RAG archival by:
1. Spawning shepherd subprocesses with different context sizes
2. Sending messages via stdin/stdout (not HTTP)
3. Verifying evicted messages are archived to RAG database
4. Testing various eviction scenarios

This tests the SHEPHERD CLIENT, not the server API.
"""

import subprocess
import os
import sys
import time
import json
import sqlite3
import threading
import queue
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil
import signal
import functools

print = functools.partial(print, flush=True)

# ============================================================================
# Configuration
# ============================================================================

SHEPHERD_BINARY = "/home/steve/bin/shepherd"
DEFAULT_CONFIG = os.path.expanduser("~/.shepherd/config.json")
DEFAULT_RAG_DB = os.path.expanduser("~/.shepherd/memory.db")

# Test context sizes
CONTEXT_SIZES = {
    "tiny": 4096,
    "small": 8192,
    "medium": 16384,
    "large": 32768,
    "server": 0,  # 0 = use server's context size (for testing server limit overflow)
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestResult:
    id: str
    name: str
    status: str  # PASS, FAIL, ERROR
    duration_ms: float
    context_size: int
    error: Optional[str] = None
    metrics: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

# ============================================================================
# Shepherd Process Manager
# ============================================================================

class ShepherdProcess:
    """Manages a shepherd subprocess with stdin/stdout communication"""

    def __init__(self, context_size: int, memory_db: str, verbose: bool = False):
        self.context_size = context_size
        self.memory_db = memory_db
        self.verbose = verbose
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdout_thread = None
        self.stderr_thread = None
        self.current_tokens = 0
        self.max_tokens = context_size
        self.all_stdout_lines = []
        self.all_stderr_lines = []

    def _read_stream(self, stream, q, line_list):
        """Read stream in background thread"""
        try:
            for line in iter(stream.readline, ''):
                if line:
                    line = line.rstrip()
                    q.put(line)
                    line_list.append(line)
        except:
            pass

    def start(self) -> bool:
        """Start shepherd process with custom context size and memory DB"""

        # Build command using command-line arguments (ALWAYS use --debug for token tracking)
        cmd = [
            SHEPHERD_BINARY,
            "--debug=5",
        ]

        # Only add context-size if not 0 (0 means use server's context size)
        if self.context_size > 0:
            cmd.extend(["--context-size", str(self.context_size)])

        cmd.extend(["--memory-db", self.memory_db])

        if self.verbose:
            print(f"  Starting: {' '.join(cmd)}")
            if self.context_size > 0:
                print(f"  Context: {self.context_size:,} tokens")
            else:
                print(f"  Context: Using server's context size (auto-detect)")
            print(f"  Memory DB: {self.memory_db}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid  # Create new process group
            )

            # Start background threads to read stdout/stderr
            self.stdout_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stdout, self.stdout_queue, self.all_stdout_lines),
                daemon=True
            )
            self.stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stderr, self.stderr_queue, self.all_stderr_lines),
                daemon=True
            )
            self.stdout_thread.start()
            self.stderr_thread.start()

            # Wait for startup
            time.sleep(3)

            # Check if process started
            if self.process.poll() is not None:
                print(f"  ERROR: Process exited (code {self.process.returncode})")

                # Drain queues to get error messages
                stderr_lines = []
                try:
                    while True:
                        stderr_lines.append(self.stderr_queue.get_nowait())
                except queue.Empty:
                    pass

                if stderr_lines:
                    print(f"  Stderr output:")
                    for line in stderr_lines[:20]:
                        print(f"    {line}")

                return False

            # Drain initial output
            time.sleep(1)
            self._drain_queues()

            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def send_message(self, message: str, timeout: float = 30.0) -> Tuple[str, List[str]]:
        """Send message and get response"""
        if not self.process or self.process.poll() is not None:
            return "", ["ERROR: Process not running"]

        try:
            # Send message
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()

            # Collect response
            response_lines = []
            stderr_lines = []
            start_time = time.time()
            last_output_time = time.time()

            while time.time() - start_time < timeout:
                # Check stdout
                got_output = False
                try:
                    while True:
                        line = self.stdout_queue.get_nowait()
                        response_lines.append(line)
                        got_output = True
                        last_output_time = time.time()
                except queue.Empty:
                    pass

                # Check stderr - parse for token state
                try:
                    while True:
                        line = self.stderr_queue.get_nowait()
                        stderr_lines.append(line)
                        got_output = True

                        # Parse: [DEBUG] Estimated context tokens: 930/4096
                        if "Estimated context tokens:" in line:
                            try:
                                parts = line.split("Estimated context tokens:")[1].strip()
                                current, total = parts.split("/")
                                self.current_tokens = int(current)
                                self.max_tokens = int(total)
                            except:
                                pass
                except queue.Empty:
                    pass

                # If we got output and nothing for 2 seconds, assume response complete
                if response_lines and (time.time() - last_output_time > 2.0):
                    break

                time.sleep(0.1)

            response = '\n'.join(response_lines)
            return response, stderr_lines

        except Exception as e:
            return "", [f"ERROR: {e}"]

    def _drain_queues(self):
        """Drain queues without blocking"""
        try:
            while True:
                self.stdout_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.stderr_queue.get_nowait()
        except queue.Empty:
            pass



    def stop(self):
        """Stop shepherd process"""
        if self.process and self.process.poll() is None:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass

    def cleanup(self):
        """Clean up resources"""
        self.stop()

# ============================================================================
# RAG Inspector
# ============================================================================

class RAGInspector:
    """Inspect RAG database to verify eviction"""

    def __init__(self, db_path: str = DEFAULT_RAG_DB):
        self.db_path = db_path

    def clear_database(self):
        """Clear all conversations from RAG"""
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM conversations")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"WARNING: Could not clear RAG database: {e}")

    def count_conversations(self) -> int:
        """Count archived conversations"""
        if not os.path.exists(self.db_path):
            return 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations"""
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, user_message, assistant_response
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "user": row[2][:100],  # Truncate for display
                    "assistant": row[3][:100]
                })

            conn.close()
            return results
        except:
            return []

    def search_messages(self, text: str) -> List[Dict]:
        """Search for messages containing text"""
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_message, assistant_response
                FROM conversations
                WHERE user_message LIKE ? OR assistant_response LIKE ?
                LIMIT 10
            """, (f'%{text}%', f'%{text}%'))

            results = [{"user": r[0], "assistant": r[1]} for r in cursor.fetchall()]
            conn.close()
            return results
        except:
            return []

# ============================================================================
# Test Suite
# ============================================================================

class UnifiedEvictionTestSuite:
    """Unified test suite for shepherd client eviction"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.temp_dir = tempfile.mkdtemp(prefix="shepherd_test_")

        print(f"\n{'='*70}")
        print("Unified Shepherd CLIENT Eviction Test Suite")
        print(f"{'='*70}")
        print(f"Binary: {SHEPHERD_BINARY}")
        print(f"Config: {DEFAULT_CONFIG}")
        print(f"Test temp dir: {self.temp_dir}")
        print(f"{'='*70}\n")

    def run_test(self, test_id: str, test_name: str, test_func, context_size: int) -> TestResult:
        """Run a single test"""

        print(f"\n[{test_id}] {test_name}")
        if context_size > 0:
            print(f"  Context: {context_size:,} tokens")
        else:
            print(f"  Context: Server default (auto-detect)")

        result = TestResult(
            id=test_id,
            name=test_name,
            status="RUNNING",
            duration_ms=0,
            context_size=context_size
        )

        # Create unique RAG database for this test
        memory_db = os.path.join(self.temp_dir, f"test_{test_id.replace('.', '_')}.db")
        rag = RAGInspector(memory_db)
        rag_before = 0  # Always starts empty since each test gets its own DB

        shepherd = None
        start_time = time.time()

        try:
            # Create and start shepherd with isolated memory DB
            shepherd = ShepherdProcess(context_size, memory_db, verbose=self.verbose)

            if not shepherd.start():
                result.status = "ERROR"
                result.error = "Failed to start shepherd"
                return result

            # Run test
            test_func(result, shepherd)

            result.status = "PASS" if not result.error else "FAIL"

        except Exception as e:
            result.status = "ERROR"
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                import traceback
                result.error += f"\n{traceback.format_exc()}"

        finally:
            if shepherd:
                # Get all logs
                all_log_lines = shepherd.all_stdout_lines + shepherd.all_stderr_lines

                # Count eviction-related log messages
                eviction_logs = sum(1 for line in all_log_lines
                                   if any(keyword in line.lower()
                                         for keyword in ['evict', 'archiv', 'rag']))

                result.metrics["eviction_logs"] = eviction_logs

                if self.verbose and shepherd.all_stderr_lines:
                    print(f"  Stderr ({len(shepherd.all_stderr_lines)} lines):")
                    for line in shepherd.all_stderr_lines[-10:]:
                        print(f"    {line}")

                shepherd.cleanup()

            # Check RAG after test
            rag_after = rag.count_conversations()
            result.metrics["rag_before"] = rag_before
            result.metrics["rag_after"] = rag_after
            result.metrics["rag_added"] = rag_after - rag_before
            result.metrics["memory_db"] = memory_db

            # Set error if expected eviction didn't happen
            # (only for tests that expect eviction)
            if result.id in ["1.1", "1.2", "1.3", "2.2", "3.1", "5.1"] and rag_after == 0:
                result.error = f"Expected eviction but RAG is empty (sent {result.metrics.get('messages_sent', 'unknown')} messages)"

            # Check for unexpected eviction in test 2.1 (near capacity)
            if result.id == "2.1" and rag_after > 2:
                result.error = f"Unexpected early eviction ({rag_after} conversations) at 90% capacity"

            # For test 1.2, check if markers were found in RAG
            if result.id == "1.2" and result.metrics.get("test_needs_rag_check"):
                found = 0
                for marker in ["TESTMARKER_ALPHA", "TESTMARKER_BETA", "TESTMARKER_GAMMA"]:
                    results = rag.search_messages(marker)
                    if results:
                        found += 1
                result.metrics["markers_found"] = found
                if found == 0 and rag_after > 0:
                    result.error = "Eviction occurred but markers not found in RAG"

            # For test 4.1, check RAG content structure
            if result.id == "4.1" and result.metrics.get("test_needs_rag_content_check"):
                conversations = rag.get_recent_conversations(limit=10)
                valid_conversations = sum(1 for c in conversations
                                         if c.get("user") and c.get("assistant"))
                result.metrics.update({
                    "conversations_retrieved": len(conversations),
                    "valid_conversations": valid_conversations
                })
                if len(conversations) == 0:
                    result.error = "No conversations in RAG"
                elif valid_conversations == 0:
                    result.error = "RAG conversations missing user or assistant messages"

        result.duration_ms = (time.time() - start_time) * 1000

        # Print result
        status_symbol = "✓" if result.status == "PASS" else "✗"
        print(f"  [{status_symbol}] {result.status} ({result.duration_ms:.0f}ms)")

        if result.metrics:
            for key, value in result.metrics.items():
                print(f"    {key}: {value}")

        if result.error:
            print(f"    Error: {result.error}")

        self.results.append(result)
        return result

    # ========================================================================
    # CATEGORY 1: BASIC EVICTION TESTS
    # ========================================================================

    def test_fill_and_evict(self, result: TestResult, shepherd: ShepherdProcess):
        """Fill context beyond capacity and verify eviction to RAG"""

        messages_sent = 0
        responses_received = 0
        eviction_count = 0

        # Phase 1: Fill to 85% capacity
        print(f"    Phase 1: Filling to 85% of {shepherd.context_size} tokens...")
        target_tokens = int(shepherd.context_size * 0.85)

        while shepherd.current_tokens < target_tokens and messages_sent < 100:
            msg = f"Tell me about topic {messages_sent}. Give me a detailed response."
            response, stderr = shepherd.send_message(msg, timeout=15)

            if response:
                responses_received += 1

            messages_sent += 1

            # Check for eviction in stdout and stderr
            all_output = stderr + response.split('\n')
            for line in all_output:
                if "Successfully evicted" in line or "KV cache full" in line:
                    eviction_count += 1

            if self.verbose and messages_sent % 10 == 0:
                print(f"    Sent {messages_sent} messages, tokens: {shepherd.current_tokens}/{shepherd.max_tokens}")

        print(f"    Phase 1 complete: {shepherd.current_tokens}/{shepherd.max_tokens} tokens after {messages_sent} messages")

        # Phase 2: Send one large message to trigger eviction
        print(f"    Phase 2: Sending large message to exceed capacity...")
        large_msg = "Write a very detailed essay about artificial intelligence, covering history, current state, and future prospects. Make it at least 500 words."
        response, stderr = shepherd.send_message(large_msg, timeout=30)

        if response:
            responses_received += 1
        messages_sent += 1

        # Check for eviction
        all_output = stderr + response.split('\n')
        for line in all_output:
            if "Successfully evicted" in line or "KV cache full" in line:
                eviction_count += 1
                print(f"    EVICTION DETECTED: {line}")

        print(f"    Phase 2 complete: {shepherd.current_tokens}/{shepherd.max_tokens} tokens")

        result.metrics.update({
            "messages_sent": messages_sent,
            "responses_received": responses_received,
            "eviction_count_from_logs": eviction_count,
            "final_token_state": f"{shepherd.current_tokens}/{shepherd.max_tokens}"
        })

        # Note: rag_added will be checked in the finally block after test completes

    def test_distinctive_messages(self, result: TestResult, shepherd: ShepherdProcess):
        """Send distinctive messages and verify they're archived in RAG"""

        # Send distinctive markers
        markers = [
            "TESTMARKER_ALPHA: The first marker message",
            "TESTMARKER_BETA: The second marker message",
            "TESTMARKER_GAMMA: The third marker message"
        ]

        for marker in markers:
            shepherd.send_message(marker, timeout=10)

        # Fill context to force eviction
        for i in range(40):
            shepherd.send_message(
                f"Tell me about subject {i}. Give a comprehensive answer with at least 200 words.",
                timeout=15
            )

        result.metrics.update({
            "markers_sent": len(markers),
            "test_needs_rag_check": True  # Flag for post-test RAG verification
        })

    def test_rapid_eviction(self, result: TestResult, shepherd: ShepherdProcess):
        """Test rapid eviction with small context"""

        messages_sent = 0

        for i in range(50):
            msg = f"Question {i}: Brief answer please, at least 50 words."
            response, _ = shepherd.send_message(msg, timeout=10)

            if response:
                messages_sent += 1

        result.metrics.update({
            "messages_sent": messages_sent
        })

        # Note: RAG check happens in finally block

    # ========================================================================
    # CATEGORY 2: BOUNDARY TESTS
    # ========================================================================

    def test_near_capacity(self, result: TestResult, shepherd: ShepherdProcess):
        """Fill to ~90% capacity - should not evict yet"""

        # Send messages to fill to 90%
        num_messages = int((shepherd.context_size * 0.90) / 200)

        for i in range(num_messages):
            msg = f"Message {i}: Please respond with about 100 words."
            shepherd.send_message(msg, timeout=15)

        result.metrics["messages_sent"] = num_messages

        # Note: RAG check happens in finally block - test 2.1 expects NO eviction

    def test_over_capacity(self, result: TestResult, shepherd: ShepherdProcess):
        """Push slightly over capacity - should trigger eviction"""

        # Fill to 95% then add more
        num_messages = int((shepherd.context_size * 0.95) / 200) + 15

        for i in range(num_messages):
            msg = f"Query {i}: Detailed response please, at least 150 words."
            shepherd.send_message(msg, timeout=15)

        result.metrics["messages_sent"] = num_messages

        # Note: RAG check happens in finally block

    # ========================================================================
    # CATEGORY 3: TINY MESSAGE TESTS
    # ========================================================================

    def test_many_tiny_messages(self, result: TestResult, shepherd: ShepherdProcess):
        """Send many tiny messages - should still trigger eviction"""

        messages_sent = 0

        for i in range(100):
            msg = f"Q{i}: Answer briefly."
            response, _ = shepherd.send_message(msg, timeout=10)

            if response:
                messages_sent += 1

        result.metrics["messages_sent"] = messages_sent

        # Note: RAG check happens in finally block

    # ========================================================================
    # CATEGORY 4: RAG RETRIEVAL TESTS
    # ========================================================================

    def test_rag_content(self, result: TestResult, shepherd: ShepherdProcess):
        """Verify RAG content is properly structured"""

        # Send some messages
        for i in range(20):
            shepherd.send_message(f"Tell me about topic {i} in detail.", timeout=15)

        result.metrics["test_needs_rag_content_check"] = True  # Flag for post-test verification

    # ========================================================================
    # CATEGORY 5: SERVER LIMIT OVERFLOW TESTS (client context >= server context)
    # ========================================================================

    def test_server_limit_overflow(self, result: TestResult, shepherd: ShepherdProcess):
        """Test with context-size 0 (use server's limit) and exceed it to trigger 400/413 errors"""

        messages_sent = 0
        responses_received = 0
        eviction_detected = False

        # Server has 98K context - quickly fill it by reading large files using the read tool
        print(f"    Phase 1: Filling context by reading large files with read tool...")

        # Read multiple large source files to fill context
        # Each "read X" command will invoke the read tool and add file contents to context
        # Server has 98K context - need ~400K chars to exceed it
        source_files = [
            "/home/steve/src/shepherd/backends/api_backend.cpp",
            "/home/steve/src/shepherd/backends/openai.cpp",
            "/home/steve/src/shepherd/backends/anthropic.cpp",
            "/home/steve/src/shepherd/backends/gemini.cpp",
            "/home/steve/src/shepherd/backends/grok.cpp",
            "/home/steve/src/shepherd/backends/ollama.cpp",
            "/home/steve/src/shepherd/backends/llamacpp.cpp",  # This is the big one
            "/home/steve/src/shepherd/llama.cpp",
            "/home/steve/src/shepherd/main.cpp",
            "/home/steve/src/shepherd/config.cpp",
            "/home/steve/src/shepherd/http_client.cpp",
            "/home/steve/src/shepherd/context_manager.cpp",
            "/home/steve/src/shepherd/backend_manager.cpp",
            "/home/steve/src/shepherd/tools/read_tool.cpp",
            "/home/steve/src/shepherd/tools/write_tool.cpp",
        ]

        for i, file_path in enumerate(source_files):
            # Use simple "read <file>" command to invoke the read tool
            msg = f"read {file_path}"
            response, stderr = shepherd.send_message(msg, timeout=60)

            if response:
                responses_received += 1

            messages_sent += 1

            # Check for eviction or context overflow messages in BOTH stdout and stderr
            # (LOG_INFO messages go to stdout, LOG_ERROR messages go to stderr)
            for line in response.split('\n'):
                if "Context overflow" in line or "evict" in line.lower() or "archiv" in line.lower():
                    print(f"    EVICTION/OVERFLOW DETECTED in stdout (file {i+1}): {line}")
                    eviction_detected = True

            for line in stderr:
                if "Context overflow" in line or "evict" in line.lower() or "archiv" in line.lower():
                    print(f"    EVICTION/OVERFLOW DETECTED in stderr (file {i+1}): {line}")
                    eviction_detected = True

            print(f"    Read file {i+1}/{len(source_files)}: {file_path}")
            if self.verbose:
                print(f"      Response length: {len(response)} chars")

        # Phase 2: Send more reads to definitely trigger overflow
        print(f"    Phase 2: Sending more large file reads to trigger overflow...")
        overflow_files = [
            "/home/steve/src/shepherd/tools/bash_tool.cpp",
            "/home/steve/src/shepherd/tools/web_search_tool.cpp",
            "/home/steve/src/shepherd/server.cpp",
        ]

        for file_path in overflow_files:
            msg = f"read {file_path}"
            response, stderr = shepherd.send_message(msg, timeout=60)

            if response:
                responses_received += 1
            messages_sent += 1

            # Check for eviction in BOTH stdout and stderr
            for line in response.split('\n'):
                if "Context overflow" in line or "evict" in line.lower() or "archiv" in line.lower():
                    print(f"    EVICTION/OVERFLOW DETECTED in stdout (overflow file): {line}")
                    eviction_detected = True

            for line in stderr:
                if "Context overflow" in line or "evict" in line.lower() or "archiv" in line.lower():
                    print(f"    EVICTION/OVERFLOW DETECTED in stderr (overflow file): {line}")
                    eviction_detected = True

        result.metrics.update({
            "messages_sent": messages_sent,
            "responses_received": responses_received,
            "eviction_detected": eviction_detected,
            "final_token_state": f"{shepherd.current_tokens}/{shepherd.max_tokens}"
        })

    # ========================================================================
    # Test Runner
    # ========================================================================

    def run_all_tests(self, mode: str = "standard", specific_test: str = None):
        """Run all tests or a specific test"""

        # Define all available tests
        all_tests = {
            "1.1": ("Fill and evict (8K)", self.test_fill_and_evict, CONTEXT_SIZES["small"]),
            "1.2": ("Distinctive messages (16K)", self.test_distinctive_messages, CONTEXT_SIZES["medium"]),
            "1.3": ("Rapid eviction (4K)", self.test_rapid_eviction, CONTEXT_SIZES["tiny"]),
            "2.1": ("Near capacity (8K)", self.test_near_capacity, CONTEXT_SIZES["small"]),
            "2.2": ("Over capacity (8K)", self.test_over_capacity, CONTEXT_SIZES["small"]),
            "3.1": ("Many tiny messages (4K)", self.test_many_tiny_messages, CONTEXT_SIZES["tiny"]),
            "4.1": ("RAG content structure (16K)", self.test_rag_content, CONTEXT_SIZES["medium"]),
            "5.1": ("Server limit overflow (context-size 0)", self.test_server_limit_overflow, CONTEXT_SIZES["server"]),
        }

        # If specific test requested, run only that test
        if specific_test:
            if specific_test not in all_tests:
                print(f"ERROR: Test '{specific_test}' not found")
                print(f"Available tests: {', '.join(all_tests.keys())}")
                return

            print(f"\nRunning specific test: {specific_test}\n")
            name, func, context_size = all_tests[specific_test]
            self.run_test(specific_test, name, func, context_size)
            return

        # Otherwise run tests based on mode
        print("\nEach test uses an isolated RAG database\n")

        # Category 1: Basic Eviction
        print(f"\n{'='*70}")
        print("Category 1: Basic Eviction Tests")
        print(f"{'='*70}")

        if mode in ["fast", "standard", "full"]:
            self.run_test("1.1", "Fill and evict (8K)",
                         self.test_fill_and_evict, CONTEXT_SIZES["small"])
            self.run_test("1.2", "Distinctive messages (16K)",
                         self.test_distinctive_messages, CONTEXT_SIZES["medium"])

        if mode in ["full"]:
            self.run_test("1.3", "Rapid eviction (4K)",
                         self.test_rapid_eviction, CONTEXT_SIZES["tiny"])

        # Category 2: Boundary Tests
        print(f"\n{'='*70}")
        print("Category 2: Boundary Tests")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("2.1", "Near capacity (8K)",
                         self.test_near_capacity, CONTEXT_SIZES["small"])
            self.run_test("2.2", "Over capacity (8K)",
                         self.test_over_capacity, CONTEXT_SIZES["small"])

        # Category 3: Tiny Messages
        print(f"\n{'='*70}")
        print("Category 3: Tiny Message Tests")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("3.1", "Many tiny messages (4K)",
                         self.test_many_tiny_messages, CONTEXT_SIZES["tiny"])

        # Category 4: RAG Content
        print(f"\n{'='*70}")
        print("Category 4: RAG Content Verification")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("4.1", "RAG content structure (16K)",
                         self.test_rag_content, CONTEXT_SIZES["medium"])

        # Category 5: Server Limit Overflow (client context >= server context)
        print(f"\n{'='*70}")
        print("Category 5: Server Limit Overflow (400/413 Error Handling)")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("5.1", "Server limit overflow (context-size 0)",
                         self.test_server_limit_overflow, CONTEXT_SIZES["server"])

        # Multi-context size tests
        if mode == "full":
            print(f"\n{'='*70}")
            print("Category 6: Multi-Context Tests")
            print(f"{'='*70}")

            for name, size in CONTEXT_SIZES.items():
                if size > 0:  # Skip "server" context size for multi-context tests
                    self.run_test(f"6.{name}", f"Basic eviction ({name})",
                                 self.test_fill_and_evict, size)

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
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ✓")
        print(f"Failed:       {failed} ✗")
        print(f"Errors:       {errors} ⚠")

        total_rag = sum(r.metrics.get("rag_added", 0) for r in self.results)
        print(f"\nTotal Conversations Archived to RAG: {total_rag}")

        avg_duration = sum(r.duration_ms for r in self.results) / total if total > 0 else 0
        print(f"Average Test Duration: {avg_duration:.0f}ms")

        if failed > 0 or errors > 0:
            print(f"\nFailed/Error Tests:")
            for r in self.results:
                if r.status in ["FAIL", "ERROR"]:
                    print(f"  [{r.id}] {r.name}")
                    if r.error:
                        print(f"      {r.error}")

        print(f"\n{'='*70}\n")

        return passed == total

    def save_report(self, filename: str):
        """Save JSON report"""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "binary": SHEPHERD_BINARY,
                "config": DEFAULT_CONFIG,
                "rag_db": DEFAULT_RAG_DB,
                "duration_seconds": sum(r.duration_ms for r in self.results) / 1000
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "PASS"),
                "failed": sum(1 for r in self.results if r.status == "FAIL"),
                "errors": sum(1 for r in self.results if r.status == "ERROR"),
                "total_rag_archived": sum(r.metrics.get("rag_added", 0) for r in self.results)
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved: {filename}")

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Shepherd CLIENT eviction test suite"
    )
    parser.add_argument("--mode",
                       choices=["fast", "standard", "full"],
                       default="standard",
                       help="Test mode: fast (2 tests), standard (7 tests), full (all tests)")
    parser.add_argument("--test",
                       help="Run only a specific test by ID (e.g., '5.1' for server limit overflow)")
    parser.add_argument("--output",
                       help="Output JSON report file")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Check binary exists
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        print("Please build shepherd or run from the correct directory")
        sys.exit(1)

    # Check config exists
    if not os.path.exists(DEFAULT_CONFIG):
        print(f"ERROR: Config not found: {DEFAULT_CONFIG}")
        print("Please create a shepherd config file")
        sys.exit(1)

    # Create test suite
    suite = UnifiedEvictionTestSuite(verbose=args.verbose)

    try:
        # Run tests
        suite.run_all_tests(mode=args.mode, specific_test=args.test)

        # Print summary
        all_passed = suite.print_summary()

        # Save report
        if args.output:
            suite.save_report(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suite.save_report(f"unified_eviction_report_{timestamp}.json")

        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        suite.print_summary()
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

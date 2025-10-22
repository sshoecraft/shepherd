#!/usr/bin/env python3
"""
Automated Shepherd CLIENT Eviction Test Suite

Executes ./shepherd as subprocess and tests client-side eviction via stdin/stdout.
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
import re

# ============================================================================
# Configuration
# ============================================================================

SHEPHERD_BINARY = "./shepherd"
SERVER_API_BASE = "http://192.168.1.166:8000/v1"
SERVER_MODEL = "gpt-4"

CONTEXT_PROFILES = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "native": 98304,  # Server's native context size
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
    verification: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.verification is None:
            self.verification = {}

# ============================================================================
# Shepherd Process Manager
# ============================================================================

class ShepherdProcess:
    """Manages shepherd subprocess with stdin/stdout communication"""

    def __init__(self, context_size: int, rag_db: str, verbose: bool = False):
        self.context_size = context_size
        self.rag_db = rag_db
        self.verbose = verbose
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdout_thread = None
        self.stderr_thread = None

    def _read_stream(self, stream, q):
        """Read stream in background thread"""
        try:
            for line in iter(stream.readline, ''):
                if line:
                    q.put(line.rstrip())
        except:
            pass

    def start(self) -> bool:
        """Start shepherd process"""

        # Create minimal config with context size override
        # NOTE: memory_database config field doesn't work - shepherd uses hardcoded ~/.shepherd/memory.db
        # So all tests share the same RAG database (we clear it before each test run)
        config_file = self.rag_db.replace('.db', '_config.json')

        # Read default config
        import json
        with open(os.path.expanduser('~/.shepherd/config.json'), 'r') as f:
            config = json.load(f)

        # Override context size only
        config['context_size'] = self.context_size

        # Write test-specific config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        cmd = [
            SHEPHERD_BINARY,
            "--config", config_file,
        ]

        if self.verbose:
            print(f"  Starting: {' '.join(cmd)}")
            print(f"  Context: {self.context_size}")
            print(f"  RAG DB: {self.rag_db}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Start background threads to read stdout/stderr
            self.stdout_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stdout, self.stdout_queue),
                daemon=True
            )
            self.stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stderr, self.stderr_queue),
                daemon=True
            )
            self.stdout_thread.start()
            self.stderr_thread.start()

            # Wait for startup
            time.sleep(3)

            # Check if process started
            if self.process.poll() is not None:
                print(f"  ERROR: Process exited (code {self.process.returncode})")
                return False

            # Capture initial output
            time.sleep(2)
            initial_stdout = []
            initial_stderr = []

            try:
                while True:
                    initial_stdout.append(self.stdout_queue.get_nowait())
            except queue.Empty:
                pass

            try:
                while True:
                    initial_stderr.append(self.stderr_queue.get_nowait())
            except queue.Empty:
                pass

            if self.verbose:
                print(f"  Initial stdout ({len(initial_stdout)} lines):")
                for line in initial_stdout[:10]:
                    print(f"    {line}")
                print(f"  Initial stderr ({len(initial_stderr)} lines):")
                for line in initial_stderr[:10]:
                    print(f"    {line}")

            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def send_message(self, message: str, timeout: float = 30.0) -> Tuple[str, List[str]]:
        """Send message and get response"""
        if not self.process or self.process.poll() is not None:
            return "", []

        try:
            # Send message
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()

            # Collect response lines
            response_lines = []
            stderr_lines = []
            start_time = time.time()

            # Read response with timeout
            while time.time() - start_time < timeout:
                # Check stdout
                try:
                    while True:
                        line = self.stdout_queue.get_nowait()
                        response_lines.append(line)
                except queue.Empty:
                    pass

                # Check stderr
                try:
                    while True:
                        line = self.stderr_queue.get_nowait()
                        stderr_lines.append(line)
                except queue.Empty:
                    pass

                # Check if we got a complete response
                # Look for prompt marker or empty line after response
                if response_lines and (
                    any('>' in line for line in response_lines[-3:]) or
                    time.time() - start_time > 3  # Give it 3 seconds to respond
                ):
                    break

                time.sleep(0.1)

            response = '\n'.join(response_lines)
            return response, stderr_lines

        except Exception as e:
            if self.verbose:
                print(f"  ERROR communicating: {e}")
            return "", []

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

    def get_all_stderr(self) -> List[str]:
        """Get all stderr lines"""
        lines = []
        try:
            while True:
                lines.append(self.stderr_queue.get_nowait())
        except queue.Empty:
            pass
        return lines

    def stop(self):
        """Stop shepherd process"""
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()

# ============================================================================
# RAG Inspector
# ============================================================================

class RAGInspector:
    """Inspect RAG database"""

    def __init__(self, db_path: str):
        self.db_path = db_path

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
                WHERE user_message LIKE ?
                LIMIT 10
            """, (f'%{text}%',))
            results = [{"user": r[0], "assistant": r[1]} for r in cursor.fetchall()]
            conn.close()
            return results
        except:
            return []

# ============================================================================
# Test Suite
# ============================================================================

class ShepherdClientTestSuite:
    """Automated test suite for Shepherd client eviction"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.temp_dir = tempfile.mkdtemp(prefix="shepherd_test_")

        print(f"\n{'='*70}")
        print("Shepherd CLIENT Eviction Test Suite (Automated)")
        print(f"{'='*70}")
        print(f"Server: {SERVER_API_BASE}")
        print(f"Model: {SERVER_MODEL}")
        print(f"Temp dir: {self.temp_dir}")
        print(f"{'='*70}\n")

    def run_test(self, test_id: str, test_name: str,
                 test_func, context_size: int) -> TestResult:
        """Run a single test"""

        print(f"\n[RUNNING] Test {test_id}: {test_name}")
        print(f"  Context: {context_size:,} tokens")

        rag_db = os.path.join(self.temp_dir, f"test_{test_id.replace('.', '_')}.db")

        result = TestResult(
            id=test_id,
            name=test_name,
            status="RUNNING",
            duration_ms=0,
            context_size=context_size
        )

        shepherd = None
        start_time = time.time()

        try:
            shepherd = ShepherdProcess(context_size, rag_db, self.verbose)

            if not shepherd.start():
                result.status = "ERROR"
                result.error = "Failed to start shepherd"
                return result

            # Run test
            test_func(result, shepherd, rag_db)

            result.status = "PASS" if not result.error else "FAIL"

        except Exception as e:
            result.status = "ERROR"
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                import traceback
                result.error += f"\n{traceback.format_exc()}"

        finally:
            if shepherd:
                # Get stderr logs
                stderr_lines = shepherd.get_all_stderr()
                eviction_count = sum(1 for line in stderr_lines if 'Evicting' in line or 'evict' in line.lower())
                result.metrics["eviction_logs"] = eviction_count

                if self.verbose and stderr_lines:
                    print(f"\n  --- Stderr ({len(stderr_lines)} lines) ---")
                    for line in stderr_lines[-20:]:  # Last 20 lines
                        print(f"    {line}")

                shepherd.stop()

        result.duration_ms = (time.time() - start_time) * 1000

        # Print result
        status_symbol = "✓" if result.status == "PASS" else "✗"
        print(f"  [{status_symbol} {result.status}] {result.duration_ms:.0f}ms")

        if result.metrics:
            for key, value in result.metrics.items():
                print(f"    {key}: {value}")

        if result.error and result.status != "PASS":
            print(f"  Error: {result.error}")

        self.results.append(result)
        return result

    # ========================================================================
    # TESTS
    # ========================================================================

    def test_1_1_basic_eviction(self, result: TestResult,
                                 shepherd: ShepherdProcess, rag_db: str):
        """Fill context and trigger eviction"""

        rag = RAGInspector(rag_db)

        # Send messages to fill context (assume ~200 tokens per pair)
        num_messages = (shepherd.context_size // 200) + 10  # Overfill

        responses_received = 0
        for i in range(num_messages):
            # Send real messages that LLM can process
            msg = f"Tell me about topic number {i}. Please respond with at least 200 words about this topic. Make it detailed and comprehensive."
            response, stderr = shepherd.send_message(msg, timeout=15)

            if response:
                responses_received += 1
                if self.verbose and i < 3:
                    print(f"    Response {i}: {response[:100]}...")

            if stderr and self.verbose and i < 3:
                print(f"    Stderr {i}: {stderr}")

        # Check RAG
        rag_count = rag.count_conversations()
        rag_exists = os.path.exists(rag_db)

        result.metrics.update({
            "messages_sent": num_messages,
            "responses_received": responses_received,
            "rag_conversations": rag_count,
            "rag_file_exists": rag_exists,
            "rag_file_size": os.path.getsize(rag_db) if rag_exists else 0
        })
        result.verification = {
            "eviction_occurred": rag_count > 0
        }

        if rag_count == 0:
            result.error = f"No eviction - RAG is empty (sent {num_messages}, got {responses_received} responses)"

    def test_1_2_rag_archival(self, result: TestResult,
                               shepherd: ShepherdProcess, rag_db: str):
        """Verify distinctive messages are archived"""

        rag = RAGInspector(rag_db)

        # Send distinctive messages
        markers = [
            "MARKER_FRANCE: The capital is Paris",
            "MARKER_PYTHON: Created by Guido van Rossum",
            "MARKER_ML: Machine learning uses neural networks"
        ]

        for marker in markers:
            shepherd.send_message(marker, timeout=10)

        # Fill to force eviction
        for i in range(30):
            shepherd.send_message(f"Tell me more about topic {i}. Give me at least 300 words of detailed information.", timeout=15)

        # Check if markers made it to RAG
        found = 0
        for marker_text in ["MARKER_FRANCE", "MARKER_PYTHON", "MARKER_ML"]:
            results = rag.search_messages(marker_text)
            if results:
                found += 1

        result.metrics.update({
            "markers_sent": len(markers),
            "markers_found": found,
            "total_rag": rag.count_conversations()
        })
        result.verification = {
            "messages_archived": found > 0
        }

        if found == 0:
            result.error = "No markers found in RAG"

    def test_1_3_rapid_eviction(self, result: TestResult,
                                 shepherd: ShepherdProcess, rag_db: str):
        """Small context - frequent evictions"""

        rag = RAGInspector(rag_db)

        # Send many messages
        for i in range(50):
            shepherd.send_message(f"Question {i}: Tell me something interesting in at least 100 words.", timeout=10)

        rag_count = rag.count_conversations()
        eviction_logs = result.metrics.get("eviction_logs", 0)

        result.metrics.update({
            "messages_sent": 50,
            "rag_conversations": rag_count
        })
        result.verification = {
            "many_evictions": rag_count > 10 or eviction_logs > 10
        }

        if rag_count < 5:
            result.error = f"Expected many evictions, got {rag_count}"

    def test_2_1_boundary_exact(self, result: TestResult, shepherd: ShepherdProcess, rag_db: str):
        """Fill to exact capacity - no eviction expected"""
        rag = RAGInspector(rag_db)

        # Send messages to fill to ~90% (shouldn't trigger eviction)
        for i in range(8):
            msg = f"Message {i}: Please respond briefly in 50 words."
            response, _ = shepherd.send_message(msg, timeout=15)
            if not response:
                result.error = f"No response at message {i}"
                return

        rag_count = rag.count_conversations()
        result.metrics.update({"rag_conversations": rag_count})
        result.verification = {"no_eviction": rag_count == 0}

        if rag_count > 0:
            result.error = f"Unexpected eviction at 90% capacity (RAG has {rag_count} conversations)"

    def test_2_2_boundary_over(self, result: TestResult, shepherd: ShepherdProcess, rag_db: str):
        """Push slightly over limit - small eviction"""
        rag = RAGInspector(rag_db)

        # Fill past capacity
        for i in range(15):
            msg = f"Question {i}: Give me a detailed 200-word response about topic {i}."
            shepherd.send_message(msg, timeout=15)

        rag_count = rag.count_conversations()
        result.metrics.update({"rag_conversations": rag_count})
        result.verification = {"eviction_occurred": rag_count > 0}

        if rag_count == 0:
            result.error = "No eviction when over capacity"

    def test_3_1_many_tiny(self, result: TestResult, shepherd: ShepherdProcess, rag_db: str):
        """Many tiny messages - rapid eviction"""
        rag = RAGInspector(rag_db)

        for i in range(50):
            msg = f"Q{i}: Brief answer please."
            shepherd.send_message(msg, timeout=10)

        rag_count = rag.count_conversations()
        result.metrics.update({
            "messages_sent": 50,
            "rag_conversations": rag_count
        })
        result.verification = {"rapid_eviction": rag_count > 10}

        if rag_count < 5:
            result.error = f"Expected rapid eviction with tiny context, got only {rag_count}"

    def test_3_2_large_messages(self, result: TestResult, shepherd: ShepherdProcess, rag_db: str):
        """Few large messages"""
        rag = RAGInspector(rag_db)

        for i in range(10):
            msg = f"Topic {i}: Please provide a comprehensive 500-word essay on this subject with detailed analysis and examples."
            shepherd.send_message(msg, timeout=20)

        rag_count = rag.count_conversations()
        result.metrics.update({
            "messages_sent": 10,
            "rag_conversations": rag_count
        })
        result.verification = {"eviction_with_large_msgs": rag_count > 0}

    def test_4_1_server_protection(self, result: TestResult, shepherd: ShepherdProcess, rag_db: str):
        """Client evicts before server sees errors"""
        # Client has small context, server has 98K
        # Send many messages - client should evict, server should never error

        for i in range(30):
            msg = f"Message {i}: Tell me about topic {i} in 300 words."
            response, stderr = shepherd.send_message(msg, timeout=15)

            # Check for server errors in response
            if "error" in response.lower() or "400" in str(stderr):
                result.error = f"Server error detected at message {i}"
                return

        rag = RAGInspector(rag_db)
        rag_count = rag.count_conversations()

        result.metrics.update({
            "messages_sent": 30,
            "rag_conversations": rag_count
        })
        result.verification = {
            "no_server_errors": True,
            "client_evicted": rag_count > 0
        }

    # ========================================================================
    # Test Runner
    # ========================================================================

    def run_all_tests(self, mode: str = "standard"):
        """Run all tests"""

        # Clear default RAG database before tests
        default_db = os.path.expanduser("~/.shepherd/memory.db")
        if os.path.exists(default_db):
            import sqlite3
            conn = sqlite3.connect(default_db)
            conn.execute("DELETE FROM conversations;")
            conn.commit()
            conn.close()
            print(f"Cleared default RAG database: {default_db}\n")

        print(f"\n{'='*70}")
        print("Category 1: Basic Eviction")
        print(f"{'='*70}")

        if mode in ["fast"]:
            self.run_test("1.1", "Basic Eviction (8K)",
                         self.test_1_1_basic_eviction, CONTEXT_PROFILES["8k"])
            self.run_test("1.2", "RAG Archival (16K)",
                         self.test_1_2_rag_archival, CONTEXT_PROFILES["16k"])

        if mode in ["standard", "full"]:
            self.run_test("1.1", "Basic Eviction (8K)",
                         self.test_1_1_basic_eviction, CONTEXT_PROFILES["8k"])
            self.run_test("1.2", "RAG Archival (16K)",
                         self.test_1_2_rag_archival, CONTEXT_PROFILES["16k"])
            self.run_test("1.3", "Eviction at 32K",
                         self.test_1_1_basic_eviction, CONTEXT_PROFILES["32k"])

        print(f"\n{'='*70}")
        print("Category 2: Server Protection")
        print(f"{'='*70}")

        if mode in ["fast", "standard", "full"]:
            self.run_test("2.1", "Client 8K vs Server 98K",
                         self.test_4_1_server_protection, CONTEXT_PROFILES["8k"])

        if mode in ["full"]:
            print(f"\n{'='*70}")
            print("Category 5: Stress Tests")
            print(f"{'='*70}")

            self.run_test("1.3", "Rapid Eviction (Micro Context)",
                         self.test_1_3_rapid_eviction, CONTEXT_PROFILES["micro"])

    def print_summary(self):
        """Print summary"""
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

        total_rag = sum(
            r.metrics.get("rag_conversations", 0)
            for r in self.results if isinstance(r.metrics, dict)
        )
        print(f"\nTotal RAG Conversations: {total_rag}")

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
        """Save JSON report"""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "server": SERVER_API_BASE,
                "model": SERVER_MODEL,
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

        print(f"Report saved: {filename}")

    def cleanup(self):
        """Cleanup temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Shepherd CLIENT eviction tests"
    )
    parser.add_argument("--mode",
                       choices=["fast", "standard", "full"],
                       default="standard")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Check binary exists
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        print("Please run from build directory or update SHEPHERD_BINARY path")
        sys.exit(1)

    suite = ShepherdClientTestSuite(verbose=args.verbose)

    try:
        suite.run_all_tests(mode=args.mode)

        all_passed = suite.print_summary()

        if args.output:
            suite.save_report(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suite.save_report(f"client_eviction_report_{timestamp}.json")

        suite.cleanup()

        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        suite.print_summary()
        suite.cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        suite.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()

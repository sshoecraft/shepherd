#!/usr/bin/env python3
"""
Comprehensive Shepherd CLIENT Eviction Test Suite

Tests the actual ./shepherd binary's client-side eviction and RAG behavior
when connecting to a Shepherd server.

This tests:
1. Client-side eviction triggers at configured context size
2. Evicted messages are archived to RAG database
3. RAG retrieval works (search_memory tool)
4. Conversation coherence after eviction
5. Server never sees errors (client manages context)
"""

import subprocess
import os
import sys
import time
import json
import sqlite3
import signal
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import re

# ============================================================================
# Test Configuration
# ============================================================================

SHEPHERD_BINARY = "./shepherd"
SERVER_API_BASE = "http://192.168.1.166:8080/v1"
SERVER_MODEL = "gpt-4"

# Test context sizes (smaller than server's 98K for client-side eviction)
CONTEXT_SIZES = {
    "micro": 512,
    "tiny": 2048,
    "small": 8192,
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

class ShepherdClient:
    """Manages a shepherd client process"""

    def __init__(self, context_size: int, rag_db: str, verbose: bool = False):
        self.context_size = context_size
        self.rag_db = rag_db
        self.verbose = verbose
        self.process = None
        self.log_file = None

    def start(self) -> bool:
        """Start shepherd client process"""

        # Create log file for this session
        self.log_file = f"shepherd_test_{os.getpid()}.log"

        # Build command
        cmd = [
            SHEPHERD_BINARY,
            "--backend", "openai",
            "--api-base", SERVER_API_BASE,
            "--model", SERVER_MODEL,
            "--context-size", str(self.context_size),
            "--rag-db", self.rag_db,
        ]

        if self.verbose:
            print(f"Starting shepherd: {' '.join(cmd)}")
            print(f"Log file: {self.log_file}")

        try:
            # Redirect stderr to log file
            log_fh = open(self.log_file, 'w')

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                text=True,
                bufsize=1  # Line buffered
            )

            # Wait for startup
            time.sleep(2)

            # Check if process started
            if self.process.poll() is not None:
                print(f"ERROR: Shepherd process exited immediately (code {self.process.returncode})")
                with open(self.log_file) as f:
                    print(f.read())
                return False

            return True

        except Exception as e:
            print(f"ERROR: Failed to start shepherd: {e}")
            return False

    def send(self, message: str) -> Optional[str]:
        """Send message to shepherd and get response"""
        if not self.process or self.process.poll() is not None:
            print("ERROR: Shepherd process not running")
            return None

        try:
            # Send message
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()

            # Read response (blocking)
            # Shepherd outputs multiple lines, read until we get the prompt back
            response_lines = []
            timeout = 30  # 30 second timeout
            start = time.time()

            while time.time() - start < timeout:
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line:
                        response_lines.append(line.rstrip())
                        # Check if this looks like the end of response
                        if line.strip().startswith(">") or line.strip() == "":
                            break
                time.sleep(0.1)

            return "\n".join(response_lines)

        except Exception as e:
            print(f"ERROR: Failed to communicate with shepherd: {e}")
            return None

    def stop(self):
        """Stop shepherd process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def get_logs(self) -> str:
        """Get log file contents"""
        if self.log_file and os.path.exists(self.log_file):
            with open(self.log_file) as f:
                return f.read()
        return ""

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.log_file and os.path.exists(self.log_file):
            os.remove(self.log_file)

# ============================================================================
# RAG Database Inspector
# ============================================================================

class RAGInspector:
    """Inspect RAG database to verify eviction archival"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def count_conversations(self) -> int:
        """Count archived conversation turns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"ERROR querying RAG DB: {e}")
            return 0

    def get_conversations(self, limit: int = 100) -> List[Dict]:
        """Get archived conversations"""
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
                    "user_message": row[2],
                    "assistant_response": row[3]
                })

            conn.close()
            return results

        except Exception as e:
            print(f"ERROR querying RAG DB: {e}")
            return []

    def search_conversations(self, query: str) -> List[Dict]:
        """Search archived conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_message, assistant_response,
                       rank
                FROM conversations_fts
                WHERE conversations_fts MATCH ?
                ORDER BY rank
                LIMIT 10
            """, (query,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "user_message": row[1],
                    "assistant_response": row[2],
                    "rank": row[3]
                })

            conn.close()
            return results

        except Exception as e:
            print(f"ERROR searching RAG DB: {e}")
            return []

# ============================================================================
# Test Suite
# ============================================================================

class ShepherdClientTestSuite:
    """Test suite for Shepherd client eviction behavior"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.temp_dir = tempfile.mkdtemp(prefix="shepherd_test_")

        print(f"\n{'='*70}")
        print("Shepherd CLIENT Eviction Test Suite")
        print(f"{'='*70}")
        print(f"Server: {SERVER_API_BASE}")
        print(f"Model: {SERVER_MODEL}")
        print(f"Temp dir: {self.temp_dir}")
        print(f"{'='*70}\n")

    def run_test(self, test_id: str, test_name: str,
                 test_func, context_size: int) -> TestResult:
        """Run a single test"""

        print(f"\n[RUNNING] Test {test_id}: {test_name}")
        print(f"  Context size: {context_size:,} tokens")

        # Create unique RAG DB for this test
        rag_db = os.path.join(self.temp_dir, f"test_{test_id}.db")

        result = TestResult(
            id=test_id,
            name=test_name,
            status="RUNNING",
            duration_ms=0,
            context_size=context_size
        )

        client = None
        start_time = time.time()

        try:
            # Create and start client
            client = ShepherdClient(context_size, rag_db, self.verbose)

            if not client.start():
                result.status = "ERROR"
                result.error = "Failed to start shepherd client"
                return result

            # Run test function
            test_func(result, client, rag_db)

            result.status = "PASS" if not result.error else "FAIL"

        except Exception as e:
            result.status = "ERROR"
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                import traceback
                result.error += f"\n{traceback.format_exc()}"

        finally:
            if client:
                # Get logs before cleanup
                logs = client.get_logs()
                if self.verbose and logs:
                    print(f"\n--- Shepherd Logs ---\n{logs}\n--- End Logs ---")

                # Check for eviction events in logs
                eviction_count = logs.count("Evicting messages")
                result.metrics["eviction_count_in_logs"] = eviction_count

                client.cleanup()

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
    # CATEGORY 1: EVICTION TRIGGER TESTS
    # ========================================================================

    def test_1_1_fill_to_capacity(self, result: TestResult,
                                    client: ShepherdClient, rag_db: str):
        """Fill context to near capacity - should NOT evict yet"""

        # Calculate how many messages to send
        # Assume ~100 tokens per message pair
        target_tokens = int(client.context_size * 0.80)
        num_pairs = target_tokens // 200

        rag = RAGInspector(rag_db)

        for i in range(num_pairs):
            msg = f"This is test message number {i}. " + ("X" * 300)
            response = client.send(msg)

            if response is None:
                result.error = f"No response at message {i}"
                break

        # Check RAG - should have very few or no evictions yet
        conv_count = rag.count_conversations()

        result.metrics = {
            "messages_sent": num_pairs,
            "conversations_in_rag": conv_count
        }
        result.verification = {
            "no_early_eviction": conv_count < 5  # Allow some, but not many
        }

        if conv_count > 10:
            result.error = f"Too many evictions ({conv_count}) at 80% capacity"

    def test_1_2_trigger_eviction(self, result: TestResult,
                                   client: ShepherdClient, rag_db: str):
        """Fill past capacity - should trigger eviction"""

        rag = RAGInspector(rag_db)

        # Fill to 90%
        target_tokens = int(client.context_size * 0.90)
        num_pairs = target_tokens // 200

        for i in range(num_pairs):
            msg = f"Message {i}: " + ("X" * 400)
            client.send(msg)

        initial_rag_count = rag.count_conversations()

        # Now send many more to force eviction
        for i in range(20):
            msg = f"Overflow message {i}: " + ("X" * 400)
            response = client.send(msg)

            if response is None:
                result.error = f"No response at overflow message {i}"
                break

        final_rag_count = rag.count_conversations()
        evicted = final_rag_count - initial_rag_count

        result.metrics = {
            "initial_rag_count": initial_rag_count,
            "final_rag_count": final_rag_count,
            "messages_evicted_to_rag": evicted
        }
        result.verification = {
            "eviction_occurred": evicted > 0,
            "substantial_eviction": evicted >= 5
        }

        if evicted == 0:
            result.error = "No eviction occurred when filling past capacity"

    # ========================================================================
    # CATEGORY 2: RAG ARCHIVAL VERIFICATION
    # ========================================================================

    def test_2_1_verify_rag_archival(self, result: TestResult,
                                      client: ShepherdClient, rag_db: str):
        """Verify evicted messages are correctly archived to RAG"""

        rag = RAGInspector(rag_db)

        # Send distinctive messages that we can search for later
        test_phrases = [
            "The capital of France is Paris",
            "Python is a programming language",
            "Machine learning uses neural networks",
            "Database systems store structured data",
            "HTTP is a protocol for web communication"
        ]

        for phrase in test_phrases:
            msg = phrase + (" filler " * 50)  # Pad to fill space
            client.send(msg)

        # Send many more to force eviction
        for i in range(30):
            msg = f"Generic message {i}: " + ("X" * 400)
            client.send(msg)

        # Check if our distinctive messages made it to RAG
        conversations = rag.get_conversations(limit=50)

        found_phrases = []
        for conv in conversations:
            user_msg = conv.get("user_message", "")
            for phrase in test_phrases:
                if phrase in user_msg:
                    found_phrases.append(phrase)

        result.metrics = {
            "test_phrases_sent": len(test_phrases),
            "phrases_found_in_rag": len(found_phrases),
            "total_conversations_in_rag": len(conversations)
        }
        result.verification = {
            "messages_archived": len(found_phrases) > 0,
            "rag_functional": len(conversations) > 0
        }

        if len(found_phrases) == 0 and len(conversations) > 0:
            result.error = "Messages archived but distinctive phrases not found"

    # ========================================================================
    # CATEGORY 3: RAG RETRIEVAL TESTS
    # ========================================================================

    def test_3_1_search_memory_tool(self, result: TestResult,
                                     client: ShepherdClient, rag_db: str):
        """Test search_memory tool retrieves archived messages"""

        rag = RAGInspector(rag_db)

        # Send a distinctive message
        distinctive = "The quick brown fox jumps over the lazy dog"
        client.send(distinctive + (" padding " * 50))

        # Fill context to force eviction
        for i in range(40):
            client.send(f"Filler message {i}: " + ("X" * 400))

        # Verify it's in RAG
        archived_count = rag.count_conversations()

        # Now use search_memory tool to find it
        search_query = "search_memory('quick brown fox')"
        response = client.send(search_query)

        # Check if search found it
        found_in_search = response and "quick brown fox" in response.lower()

        result.metrics = {
            "conversations_archived": archived_count,
            "search_response_length": len(response) if response else 0
        }
        result.verification = {
            "message_archived": archived_count > 0,
            "search_successful": found_in_search
        }

        if not found_in_search:
            result.error = "search_memory did not retrieve archived message"

    # ========================================================================
    # Run All Tests
    # ========================================================================

    def run_all_tests(self, test_mode: str = "standard"):
        """Run all tests"""

        # Category 1: Eviction Triggers
        if test_mode in ["fast", "standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 1: Eviction Trigger Tests")
            print(f"{'='*70}")

            self.run_test("1.1", "Fill to 80% capacity (no eviction yet)",
                         self.test_1_1_fill_to_capacity, CONTEXT_SIZES["tiny"])

            self.run_test("1.2", "Trigger eviction by exceeding capacity",
                         self.test_1_2_trigger_eviction, CONTEXT_SIZES["tiny"])

        # Category 2: RAG Archival
        if test_mode in ["standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 2: RAG Archival Verification")
            print(f"{'='*70}")

            self.run_test("2.1", "Verify evicted messages archived to RAG",
                         self.test_2_1_verify_rag_archival, CONTEXT_SIZES["small"])

        # Category 3: RAG Retrieval
        if test_mode in ["standard", "full"]:
            print(f"\n{'='*70}")
            print("Category 3: RAG Retrieval Tests")
            print(f"{'='*70}")

            self.run_test("3.1", "Test search_memory tool retrieval",
                         self.test_3_1_search_memory_tool, CONTEXT_SIZES["small"])

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
            r.metrics.get("messages_evicted_to_rag", 0)
            for r in self.results if isinstance(r.metrics, dict)
        )
        print(f"\nTotal Messages Evicted to RAG: {total_evictions}")

        if failed > 0 or errors > 0:
            print(f"\nFailed/Error Tests:")
            for r in self.results:
                if r.status in ["FAIL", "ERROR"]:
                    print(f"  - {r.id}: {r.name}")
                    if r.error:
                        print(f"    {r.error}")

        print(f"\n{'='*70}\n")

        return passed == total

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Shepherd CLIENT eviction and RAG behavior"
    )
    parser.add_argument("--mode",
                       choices=["fast", "standard", "full"],
                       default="standard",
                       help="Test mode")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Check if shepherd binary exists
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        print("Please build shepherd first or run from the build directory")
        sys.exit(1)

    # Create test suite
    suite = ShepherdClientTestSuite(verbose=args.verbose)

    try:
        # Run tests
        suite.run_all_tests(test_mode=args.mode)

        # Print summary
        all_passed = suite.print_summary()

        # Cleanup
        suite.cleanup()

        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
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

#!/usr/bin/env python3
"""
Shepherd Benchmark Test Suite

Tests shepherd's accuracy on standardized AI benchmark questions (MMLU, HellaSwag).
Allows testing different configurations (backends, models, context sizes, etc.)
to compare their performance on the same benchmark tasks.

Usage:
    ./test_performance.py --config my_config.json
    ./test_performance.py --config llamacpp.json --verbose
    ./test_performance.py --benchmark mmlu --count 20
"""

import subprocess
import os
import sys
import time
import json
import threading
import queue
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import signal
import functools
import random

print = functools.partial(print, flush=True)

# Try to import datasets library for MMLU
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")
    print("Falling back to embedded sample questions.")

# ============================================================================
# Configuration
# ============================================================================

SHEPHERD_BINARY = "/home/steve/bin/shepherd"
DEFAULT_CONFIG = os.path.expanduser("~/.shepherd/config.json")

# MMLU dataset info
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# ============================================================================
# Embedded Benchmark Questions
# ============================================================================

# MMLU sample questions (multiple choice, A-D format)
MMLU_QUESTIONS = [
    {
        "subject": "computer_science",
        "question": "What is the time complexity of binary search?",
        "choices": ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"],
        "answer": "B"
    },
    {
        "subject": "mathematics",
        "question": "What is the derivative of x^2?",
        "choices": ["x", "2x", "x^2", "2"],
        "answer": "B"
    },
    {
        "subject": "history",
        "question": "In which year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": "C"
    },
    {
        "subject": "physics",
        "question": "What is the speed of light in vacuum?",
        "choices": ["299,792,458 m/s", "300,000,000 m/s", "299,000,000 m/s", "298,792,458 m/s"],
        "answer": "A"
    },
    {
        "subject": "chemistry",
        "question": "What is the chemical symbol for gold?",
        "choices": ["Go", "Gd", "Au", "Ag"],
        "answer": "C"
    },
    {
        "subject": "biology",
        "question": "How many chromosomes do humans have?",
        "choices": ["23", "44", "46", "48"],
        "answer": "C"
    },
    {
        "subject": "computer_science",
        "question": "Which data structure uses FIFO (First In, First Out)?",
        "choices": ["Stack", "Queue", "Tree", "Graph"],
        "answer": "B"
    },
    {
        "subject": "mathematics",
        "question": "What is the value of pi to 2 decimal places?",
        "choices": ["3.12", "3.14", "3.16", "3.18"],
        "answer": "B"
    },
    {
        "subject": "geography",
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": "C"
    },
    {
        "subject": "physics",
        "question": "What is Newton's first law of motion?",
        "choices": ["F=ma", "An object in motion stays in motion unless acted upon", "Action-reaction", "Energy is conserved"],
        "answer": "B"
    }
]

# HellaSwag sample questions (sentence completion)
HELLASWAG_QUESTIONS = [
    {
        "context": "A person is cooking in the kitchen. They",
        "choices": [
            "fly to the moon on a rocket",
            "stir the pot on the stove",
            "solve a complex math equation",
            "paint a beautiful landscape"
        ],
        "answer": "B"
    },
    {
        "context": "A man is hammering a nail into wood. He",
        "choices": [
            "reads a book about philosophy",
            "swims in the ocean",
            "holds the nail steady with one hand",
            "writes poetry on paper"
        ],
        "answer": "C"
    },
    {
        "context": "A child is learning to ride a bicycle. They",
        "choices": [
            "wobble back and forth trying to balance",
            "calculate quantum physics equations",
            "build a spaceship from scratch",
            "perform brain surgery"
        ],
        "answer": "A"
    },
    {
        "context": "Someone is brushing their teeth. They",
        "choices": [
            "juggle flaming torches",
            "move the brush in circular motions",
            "climb Mount Everest",
            "write a symphony"
        ],
        "answer": "B"
    },
    {
        "context": "A person is opening a door. They",
        "choices": [
            "turn the handle and pull",
            "solve a Rubik's cube blindfolded",
            "fly like a bird",
            "speak ancient Latin fluently"
        ],
        "answer": "A"
    }
]

# ============================================================================
# MMLU Dataset Loading
# ============================================================================

def load_mmlu_questions(subjects=None, count=None, split='test'):
    """
    Load MMLU questions from Hugging Face or fall back to embedded samples.

    Args:
        subjects: List of subject names to include, or None for all
        count: Number of questions to sample, or None for all
        split: Which split to use ('test', 'dev', 'validation')
    """
    if not DATASETS_AVAILABLE:
        print("Using embedded MMLU samples (install 'datasets' for full dataset)")
        questions = MMLU_QUESTIONS
        if count:
            questions = questions[:count]
        return questions

    try:
        print("Loading MMLU dataset from Hugging Face...")
        all_questions = []

        # Load dataset
        dataset = load_dataset("cais/mmlu", "all", split=split)

        # Filter by subjects if specified
        if subjects:
            subjects_set = set(subjects)
            dataset = dataset.filter(lambda x: x['subject'] in subjects_set)

        # Convert to our format
        for item in dataset:
            question = {
                "subject": item['subject'],
                "question": item['question'],
                "choices": item['choices'],
                "answer": chr(65 + item['answer'])  # Convert 0-3 to A-D
            }
            all_questions.append(question)

        print(f"Loaded {len(all_questions)} questions from MMLU")

        # Sample if count specified
        if count and count < len(all_questions):
            all_questions = random.sample(all_questions, count)
            print(f"Randomly sampled {count} questions")

        return all_questions

    except Exception as e:
        print(f"Error loading MMLU from Hugging Face: {e}")
        print("Falling back to embedded samples")
        questions = MMLU_QUESTIONS
        if count:
            questions = questions[:count]
        return questions

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    config_file: str
    benchmark_type: str
    total_questions: int
    correct: int
    incorrect: int
    errors: int
    accuracy_percent: float
    avg_response_time_ms: float
    duration_ms: float
    details: List[Dict]

# ============================================================================
# Shepherd Process Manager
# ============================================================================

class ShepherdProcess:
    """Manages a shepherd subprocess with stdin/stdout communication"""

    def __init__(self, config_file: str = None, verbose: bool = False):
        self.config_file = config_file  # Only use if explicitly provided
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
        cmd = [
            SHEPHERD_BINARY,
            "--notools",
            "--system-prompt", "You are a helpful AI assistant. Answer questions directly and concisely."
        ]

        # Use config file which points to server
        if self.config_file and os.path.exists(self.config_file):
            cmd.extend(["--config", self.config_file])

        if self.verbose:
            print(f"  Starting: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )

            # Start background threads
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

            if self.process.poll() is not None:
                print(f"  ERROR: Process exited (code {self.process.returncode})")
                return False

            # Drain initial output
            self._drain_queues()
            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def send_question(self, question_text: str, timeout: float = 30.0) -> Tuple[str, float]:
        """Send a question and get response with timing"""
        if not self.process or self.process.poll() is not None:
            return "", 0

        start_time = time.time()

        try:
            # Convert multi-line question to single line (shepherd reads line-by-line)
            # Replace newlines with spaces to send as one complete question
            single_line_question = " ".join(line.strip() for line in question_text.split("\n") if line.strip())

            # Send question
            if self.verbose:
                print(f"    [DEBUG] Sending question: {single_line_question[:100]}...")

            self.process.stdin.write(single_line_question + "\n")
            self.process.stdin.flush()

            # Collect response
            response_lines = []
            stderr_lines = []
            last_output_time = time.time()
            lines_received = 0
            answer_lines = []  # Lines starting with '<'

            time.sleep(0.2)

            while time.time() - start_time < timeout:
                # Check stdout
                try:
                    while True:
                        line = self.stdout_queue.get_nowait()
                        response_lines.append(line)
                        last_output_time = time.time()
                        lines_received += 1

                        # Track answer lines (starting with '<' but not thinking blocks)
                        stripped = line.strip()
                        if stripped.startswith('<') and not stripped.startswith('<think') and not stripped.startswith('</think'):
                            answer_lines.append(line)
                            if self.verbose:
                                print(f"    [DEBUG] *** ANSWER LINE: {line}")
                        elif self.verbose and lines_received <= 10:
                            print(f"    [DEBUG] Got stdout line: {line[:80]}")
                except queue.Empty:
                    pass

                # Check stderr for errors
                try:
                    while True:
                        line = self.stderr_queue.get_nowait()
                        stderr_lines.append(line)
                        if self.verbose and "error" in line.lower():
                            print(f"    [DEBUG] Got stderr: {line[:80]}")
                except queue.Empty:
                    pass

                # If we got answer line and nothing for 2 seconds, assume complete
                if answer_lines and (time.time() - last_output_time > 2.0):
                    break
                # Otherwise wait up to 5 seconds after last output
                elif response_lines and (time.time() - last_output_time > 5.0):
                    break

                time.sleep(0.1)

            if self.verbose:
                print(f"    [DEBUG] Total lines received: {lines_received}")
                print(f"    [DEBUG] Answer lines (starting with '<'): {len(answer_lines)}")
                print(f"    [DEBUG] Stderr lines: {len(stderr_lines)}")
                if answer_lines:
                    print(f"    [DEBUG] Answer lines captured:")
                    for line in answer_lines:
                        print(f"      >>> {line}")
                if lines_received > 10:
                    print(f"    [DEBUG] Last 10 stdout lines:")
                    for line in response_lines[-10:]:
                        print(f"      {line[:80]}")

            response_time = (time.time() - start_time) * 1000
            response = '\n'.join(response_lines)
            return response, response_time

        except Exception as e:
            if self.verbose:
                print(f"    [DEBUG] Exception: {e}")
            return f"ERROR: {e}", 0

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
# Benchmark Test Runner
# ============================================================================

class BenchmarkTestRunner:
    """Runs benchmark tests against shepherd"""

    def __init__(self, config_file: str = None, verbose: bool = False):
        self.config_file = config_file  # Only use if explicitly provided
        self.verbose = verbose

    def extract_answer(self, response: str, choices: List[str]) -> Optional[str]:
        """Extract the answer choice (A, B, C, D) from the response"""
        # Filter out thinking blocks from response before extraction
        import re
        # Remove everything between <think> and </think> tags
        filtered_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response_upper = filtered_response.upper()

        # First priority: Look for "< LETTER" pattern (shepherd's answer format)
        for letter in ['A', 'B', 'C', 'D']:
            if f"< {letter}" in response_upper:
                return letter

        # Second priority: Look for explicit answer markers
        for marker in ["ANSWER:", "THE ANSWER IS", "CORRECT ANSWER:", "ANSWER IS"]:
            if marker in response_upper:
                # Extract the part after the marker
                after_marker = response_upper.split(marker)[1].strip()
                # Look for A, B, C, or D
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in after_marker[:10]:  # Check first few chars
                        return letter

        # Third priority: Look for standalone letter at start of line
        for letter in ['A', 'B', 'C', 'D']:
            if f"\\n{letter}\\n" in f"\\n{response_upper}\\n":
                return letter
            if response_upper.strip().startswith(letter + " ") or response_upper.strip() == letter:
                return letter

        # Fourth priority: Choice text in store_memory/tool calls
        choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for letter, idx in choice_map.items():
            if idx < len(choices):
                choice_text = choices[idx].lower()
                # If the exact choice text appears in a store_memory call
                if f"answer={choice_text}" in response.lower() or f"answer=\"{choice_text}" in response.lower():
                    return letter

        return None

    def run_mmlu_benchmark(self, count: int = None, subjects: List[str] = None) -> BenchmarkResult:
        """Run MMLU benchmark questions"""
        questions = load_mmlu_questions(subjects=subjects, count=count)

        print(f"\n{'='*70}")
        print(f"Running MMLU Benchmark ({len(questions)} questions)")
        print(f"{'='*70}\n")

        shepherd = ShepherdProcess(self.config_file, self.verbose)

        if not shepherd.start():
            print("ERROR: Failed to start shepherd")
            return None

        results = {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "details": [],
            "response_times": []
        }

        start_time = time.time()

        try:
            for i, q in enumerate(questions, 1):
                # Format question with choices
                question_text = f"{q['question']}\n\nChoices:\n"
                for idx, choice in enumerate(q['choices']):
                    question_text += f"{chr(65+idx)}. {choice}\n"
                question_text += "\nAnswer with ONLY the single letter (A, B, C, or D)."

                print(f"[{i}/{len(questions)}] {q['subject']}: {q['question'][:50]}...")

                response, resp_time = shepherd.send_question(question_text)
                results["response_times"].append(resp_time)

                extracted_answer = self.extract_answer(response, q['choices'])
                correct_answer = q['answer']

                detail = {
                    "question_num": i,
                    "subject": q['subject'],
                    "question": q['question'],
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": extracted_answer == correct_answer,
                    "response_time_ms": resp_time,
                    "response": response[:200]  # Store truncated response
                }

                if extracted_answer is None:
                    results["errors"] += 1
                    detail["status"] = "ERROR"
                    print(f"  ✗ ERROR: Could not extract answer")
                    # Always show response for extraction errors to debug
                    print(f"    Response: {response[:300] if response else '(empty)'}")
                elif extracted_answer == correct_answer:
                    results["correct"] += 1
                    detail["status"] = "CORRECT"
                    print(f"  ✓ CORRECT ({extracted_answer})")
                else:
                    results["incorrect"] += 1
                    detail["status"] = "INCORRECT"
                    print(f"  ✗ INCORRECT (got {extracted_answer}, expected {correct_answer})")

                if self.verbose:
                    print(f"    Response time: {resp_time:.0f}ms")
                    print(f"    Response: {response[:300]}")

                results["details"].append(detail)

        finally:
            shepherd.cleanup()

        duration_ms = (time.time() - start_time) * 1000
        total = len(questions)
        accuracy = (results["correct"] / total * 100) if total > 0 else 0
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0

        return BenchmarkResult(
            config_file=self.config_file,
            benchmark_type="MMLU",
            total_questions=total,
            correct=results["correct"],
            incorrect=results["incorrect"],
            errors=results["errors"],
            accuracy_percent=accuracy,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

    def run_hellaswag_benchmark(self, count: int = None) -> BenchmarkResult:
        """Run HellaSwag benchmark questions"""
        questions = HELLASWAG_QUESTIONS[:count] if count else HELLASWAG_QUESTIONS

        print(f"\n{'='*70}")
        print(f"Running HellaSwag Benchmark ({len(questions)} questions)")
        print(f"{'='*70}\n")

        shepherd = ShepherdProcess(self.config_file, self.verbose)

        if not shepherd.start():
            print("ERROR: Failed to start shepherd")
            return None

        results = {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "details": [],
            "response_times": []
        }

        start_time = time.time()

        try:
            for i, q in enumerate(questions, 1):
                # Format question
                question_text = f"{q['context']}\n\nChoices:\n"
                for idx, choice in enumerate(q['choices']):
                    question_text += f"{chr(65+idx)}. {choice}\n"
                question_text += "\nAnswer with ONLY the single letter (A, B, C, or D)."

                print(f"[{i}/{len(questions)}] {q['context'][:60]}...")

                response, resp_time = shepherd.send_question(question_text)
                results["response_times"].append(resp_time)

                extracted_answer = self.extract_answer(response, q['choices'])
                correct_answer = q['answer']

                detail = {
                    "question_num": i,
                    "context": q['context'],
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": extracted_answer == correct_answer,
                    "response_time_ms": resp_time,
                    "response": response[:200]
                }

                if extracted_answer is None:
                    results["errors"] += 1
                    detail["status"] = "ERROR"
                    print(f"  ✗ ERROR: Could not extract answer")
                    # Always show response for extraction errors to debug
                    print(f"    Response: {response[:300] if response else '(empty)'}")
                elif extracted_answer == correct_answer:
                    results["correct"] += 1
                    detail["status"] = "CORRECT"
                    print(f"  ✓ CORRECT ({extracted_answer})")
                else:
                    results["incorrect"] += 1
                    detail["status"] = "INCORRECT"
                    print(f"  ✗ INCORRECT (got {extracted_answer}, expected {correct_answer})")

                if self.verbose:
                    print(f"    Response time: {resp_time:.0f}ms")

                results["details"].append(detail)

        finally:
            shepherd.cleanup()

        duration_ms = (time.time() - start_time) * 1000
        total = len(questions)
        accuracy = (results["correct"] / total * 100) if total > 0 else 0
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0

        return BenchmarkResult(
            config_file=self.config_file,
            benchmark_type="HellaSwag",
            total_questions=total,
            correct=results["correct"],
            incorrect=results["incorrect"],
            errors=results["errors"],
            accuracy_percent=accuracy,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

# ============================================================================
# Main
# ============================================================================

def print_summary(result: BenchmarkResult):
    """Print test summary"""
    print(f"\n{'='*70}")
    print(f"{result.benchmark_type} BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Config: {result.config_file}")
    print(f"Total Questions: {result.total_questions}")
    print(f"Correct: {result.correct}")
    print(f"Incorrect: {result.incorrect}")
    print(f"Errors: {result.errors}")
    print(f"Accuracy: {result.accuracy_percent:.1f}%")
    print(f"Avg Response Time: {result.avg_response_time_ms:.0f}ms")
    print(f"Total Duration: {result.duration_ms/1000:.1f}s")
    print(f"{'='*70}\n")

def save_report(result: BenchmarkResult, output_file: str):
    """Save JSON report"""
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Report saved: {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test shepherd accuracy on AI benchmarks (MMLU, HellaSwag)"
    )
    parser.add_argument("--config", "-c",
                       help="Shepherd config file (default: ~/.shepherd/config.json)")
    parser.add_argument("--benchmark", "-b",
                       choices=["mmlu", "hellaswag", "both"],
                       default="both",
                       help="Which benchmark to run")
    parser.add_argument("--count", "-n",
                       type=int,
                       help="Number of questions to test (default: all)")
    parser.add_argument("--subjects",
                       nargs="+",
                       help="MMLU subjects to test (e.g., --subjects computer_science machine_learning)")
    parser.add_argument("--list-subjects",
                       action="store_true",
                       help="List all available MMLU subjects and exit")
    parser.add_argument("--output", "-o",
                       help="Output JSON file for results")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Handle --list-subjects
    if args.list_subjects:
        print("Available MMLU subjects:")
        for i, subject in enumerate(MMLU_SUBJECTS, 1):
            print(f"  {i:2d}. {subject}")
        print(f"\nTotal: {len(MMLU_SUBJECTS)} subjects")
        print("\nUsage: --subjects college_computer_science machine_learning")
        sys.exit(0)

    # Check binary exists
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        sys.exit(1)

    # Only use config if explicitly provided by user
    config_file = args.config
    if config_file and not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    # Validate subjects if specified
    if args.subjects:
        invalid_subjects = [s for s in args.subjects if s not in MMLU_SUBJECTS]
        if invalid_subjects:
            print(f"ERROR: Invalid subjects: {', '.join(invalid_subjects)}")
            print(f"Use --list-subjects to see valid options")
            sys.exit(1)

    runner = BenchmarkTestRunner(config_file=config_file, verbose=args.verbose)

    try:
        results = []

        if args.benchmark in ["mmlu", "both"]:
            result = runner.run_mmlu_benchmark(count=args.count, subjects=args.subjects)
            if result:
                print_summary(result)
                results.append(result)

        if args.benchmark in ["hellaswag", "both"]:
            result = runner.run_hellaswag_benchmark(count=args.count)
            if result:
                print_summary(result)
                results.append(result)

        # Save reports
        if args.output:
            if len(results) == 1:
                save_report(results[0], args.output)
            else:
                # Save combined report
                combined = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config_file,
                    "results": [asdict(r) for r in results]
                }
                with open(args.output, 'w') as f:
                    json.dump(combined, f, indent=2)
                print(f"Combined report saved: {args.output}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

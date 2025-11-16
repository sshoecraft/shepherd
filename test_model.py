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
import tempfile
import shutil

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
#CODEX_BINARY = "/usr/local/bin/open-codex"
SHEPHERD_SAFETY_WRAPPER = "/home/steve/src/shepherd/swebench_safety_wrapper.sh"
#DEFAULT_CONFIG = os.path.expanduser("~/.shepherd/config.json")
DEFAULT_CONFIG = ""

# SWE-bench configuration
SWEBENCH_TIMEOUT_SECONDS = 600  # 10 minutes per task
SWEBENCH_REPOS = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
}

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

# HellaSwag sample questions - TODO: Remove these and download from HF properly
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
    Load MMLU questions from Hugging Face.

    Args:
        subjects: List of subject names to include, or None for all
        count: Number of questions to sample, or None for all
        split: Which split to use ('test', 'dev', 'validation')
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    print("Loading MMLU dataset from Hugging Face...")

    # Load dataset (returns DatasetDict with splits)
    dataset_dict = load_dataset("cais/mmlu", "all")

    # Get the requested split
    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]

    # Convert to our format
    all_questions = []
    for item in dataset:
        question = {
            "subject": item['subject'],
            "question": item['question'],
            "choices": item['choices'],
            "answer": chr(65 + item['answer'])  # Convert 0-3 to A-D
        }
        all_questions.append(question)

    print(f"Loaded {len(all_questions)} questions from MMLU")

    # Filter by subjects if specified
    if subjects:
        subjects_set = set(subjects)
        all_questions = [q for q in all_questions if q['subject'] in subjects_set]
        print(f"Filtered to {len(all_questions)} questions for subjects: {', '.join(subjects)}")

    # Sample if count specified
    if count and count < len(all_questions):
        all_questions = random.sample(all_questions, count)
        print(f"Randomly sampled {count} questions")

    return all_questions

# ============================================================================
# SWE-bench Dataset Loading
# ============================================================================

def load_swebench_tasks(count=None, difficulty=None, repos=None, split='test'):
    """
    Load SWE-bench Verified tasks from Hugging Face.

    Args:
        count: Number of tasks to sample, or None for all
        difficulty: Filter by difficulty ('easy', 'medium', 'hard', 'very hard')
        repos: List of repo names to include (e.g., ['django/django', 'flask/flask'])
        split: Which split to use ('test' is the only one available)
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    print("Loading SWE-bench Verified dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)

    # Convert to our format
    all_tasks = []
    for item in dataset:
        task = {
            "instance_id": item['instance_id'],
            "repo": item['repo'],
            "base_commit": item['base_commit'],
            "problem_statement": item['problem_statement'],
            "hints_text": item.get('hints_text', ''),
            "patch": item['patch'],
            "test_patch": item['test_patch'],
            "FAIL_TO_PASS": item['FAIL_TO_PASS'],
            "PASS_TO_PASS": item['PASS_TO_PASS'],
            "difficulty": item.get('difficulty', 'unknown'),
            "version": item.get('version', ''),
            "environment_setup_commit": item.get('environment_setup_commit', '')
        }
        all_tasks.append(task)

    print(f"Loaded {len(all_tasks)} tasks from SWE-bench Verified")

    # Filter by difficulty if specified
    if difficulty:
        all_tasks = [t for t in all_tasks if t['difficulty'] == difficulty]
        print(f"Filtered to {len(all_tasks)} tasks with difficulty: {difficulty}")

    # Filter by repos if specified
    if repos:
        repos_set = set(repos)
        all_tasks = [t for t in all_tasks if t['repo'] in repos_set]
        print(f"Filtered to {len(all_tasks)} tasks for repos: {', '.join(repos)}")

    # Sample if count specified
    if count and count < len(all_tasks):
        all_tasks = random.sample(all_tasks, count)
        print(f"Randomly sampled {count} tasks")

    return all_tasks

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

    def _read_char_with_timeout(self, timeout_ms: int) -> tuple[str, bool]:
        """
        Read one character from stdout with timeout.
        Returns: (char, timed_out) where char is the character read (or empty if timeout)
        and timed_out is True if we hit the timeout.
        """
        import select
        import os

        if not self.process or self.process.poll() is not None:
            return "", True

        fd = self.process.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_ms / 1000.0)

        if not ready:
            return "", True  # Timeout

        try:
            data = os.read(fd, 1)
            if not data:
                return "", True
            return data.decode('utf-8', errors='replace'), False
        except Exception as e:
            import sys
            print(f"[read error: {e}]", file=sys.stderr)
            return "", True

    def start(self) -> bool:
        """Start shepherd process"""
        cmd = [
            SHEPHERD_BINARY,
            "--notools",
            "--system-prompt", "You are taking a test - ***CRITICAL:*** DO NOT respond with any reasons or explanations of any kind - just answer the questions to the best of your ability.  Part of your grade is based on if you are able to answer the question as directed."
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
                bufsize=0,
                preexec_fn=os.setsid
            )

            if self.verbose:
                print(f"  Process started, ready for questions")
            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def send_question(self, question_text: str) -> Tuple[str, float]:
        """Send a question and get response with timing"""
        if not self.process or self.process.poll() is not None:
            return "", 0

        start_time = time.time()

        try:
            # Convert multi-line question to single line
            single_line_question = " ".join(line.strip() for line in question_text.split("\n") if line.strip())

            # Send question
            if self.verbose:
                print(f"    ===== SENDING TO SHEPHERD =====")
                print(f"    {single_line_question}")
                print(f"    ===============================")

            # Write to stdin
            self.process.stdin.write((single_line_question + "\n").encode('utf-8'))
            self.process.stdin.flush()

            # Read characters until 1000ms timeout (after last character received)
            # First, wait up to 60 seconds for the FIRST character (shepherd thinking time)
            # Then, keep reading with 1000ms timeout between characters
            buffer = ""
            last_ch = ""

            # Wait for first character (up to 60 seconds)
            ch, timed_out = self._read_char_with_timeout(60000)
            if timed_out:
                # No response at all
                response_time = (time.time() - start_time) * 1000
                return "", response_time

            buffer += ch
            if self.verbose:
                print(repr(ch), end='', flush=True)

            # Now keep reading with 1000ms timeout until silence
            while True:
                ch, timed_out = self._read_char_with_timeout(1000)

                if timed_out:
                    # Timeout - response is complete
                    break

                buffer += ch
                last_ch = ch
                if self.verbose:
                    print(repr(ch), end='', flush=True)

            # Record time (subtracting the final 1000ms timeout)
            response_time = (time.time() - start_time) * 1000 - 1000

            if self.verbose:
                print(f"    ===== RESPONSE FROM SHEPHERD =====")
                print(f"    {buffer}")
                print(f"    ===================================")
                print(f"    Last 2 chars: {repr(buffer[-2:])}")

            return buffer, response_time

        except Exception as e:
            if self.verbose:
                print(f"    [DEBUG] Exception: {e}")
            return f"ERROR: {e}", 0


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
            if f"\n{letter}\n" in f"\n{response_upper}\n":
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

    def run_swebench_benchmark(self, count: int = None, difficulty: str = None, repos: List[str] = None) -> BenchmarkResult:
        """Run SWE-bench Verified benchmark tasks"""
        tasks = load_swebench_tasks(count=count, difficulty=difficulty, repos=repos)

        print(f"\n{'='*70}")
        print(f"Running SWE-bench Verified Benchmark ({len(tasks)} tasks)")
        print(f"{'='*70}\n")

        results = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "details": [],
            "task_times": []
        }

        start_time = time.time()

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task['instance_id']}")
            print(f"  Repo: {task['repo']}")
            print(f"  Difficulty: {task['difficulty']}")
            if self.verbose:
                print(f"\n  FULL PROBLEM STATEMENT:")
                print(f"  {'-'*70}")
                print(f"  {task['problem_statement']}")
                print(f"  {'-'*70}\n")
            else:
                print(f"  Problem: {task['problem_statement'][:100]}...")

            task_start = time.time()

            # Run task in isolated environment
            passed, error_msg = self._run_isolated_task(task)
            task_time = (time.time() - task_start) * 1000
            results["task_times"].append(task_time)

            detail = {
                "task_num": i,
                "instance_id": task['instance_id'],
                "repo": task['repo'],
                "difficulty": task['difficulty'],
                "passed": passed,
                "error": error_msg,
                "time_ms": task_time
            }

            if error_msg:
                results["errors"] += 1
                detail["status"] = "ERROR"
                print(f"  ✗ ERROR: {error_msg}")
            elif passed:
                results["passed"] += 1
                detail["status"] = "PASSED"
                print(f"  ✓ PASSED")
            else:
                results["failed"] += 1
                detail["status"] = "FAILED"
                print(f"  ✗ FAILED: Tests did not pass")

            if self.verbose:
                print(f"    Task time: {task_time/1000:.1f}s")

            results["details"].append(detail)

        duration_ms = (time.time() - start_time) * 1000
        total = len(tasks)
        success_rate = (results["passed"] / total * 100) if total > 0 else 0
        avg_time = sum(results["task_times"]) / len(results["task_times"]) if results["task_times"] else 0

        return BenchmarkResult(
            config_file=self.config_file,
            benchmark_type="SWE-bench",
            total_questions=total,
            correct=results["passed"],
            incorrect=results["failed"],
            errors=results["errors"],
            accuracy_percent=success_rate,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

    def _run_isolated_task(self, task: Dict) -> Tuple[bool, Optional[str]]:
        """
        Run a single SWE-bench task in an isolated temporary directory.

        Returns:
            (passed, error_msg): True if tests passed, error message if failed
        """
        work_dir = None

        try:
            # Create isolated temporary directory
            work_dir = tempfile.mkdtemp(prefix=f"swebench_{task['instance_id']}_")
            if self.verbose:
                print(f"    Work dir: {work_dir}")

            # Get repository URL
            repo_name = task['repo']
            if repo_name not in SWEBENCH_REPOS:
                return False, f"Unknown repository: {repo_name}"

            repo_url = SWEBENCH_REPOS[repo_name]
            repo_dir = os.path.join(work_dir, repo_name.split('/')[-1])

            # Clone repository
            if self.verbose:
                print(f"    Cloning {repo_url}...")

            clone_result = subprocess.run(
                ["git", "clone", repo_url, repo_dir],
                capture_output=True,
                text=True,
                timeout=300
            )

            if clone_result.returncode != 0:
                return False, f"Clone failed: {clone_result.stderr[:200]}"

            # Checkout base commit
            if self.verbose:
                print(f"    Checking out commit {task['base_commit'][:8]}...")

            checkout_result = subprocess.run(
                ["git", "checkout", task['base_commit']],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if checkout_result.returncode != 0:
                return False, f"Checkout failed: {checkout_result.stderr[:200]}"

            # Apply test patch to set up test environment
            if task['test_patch']:
                if self.verbose:
                    print(f"    Applying test patch...")

                test_patch_file = os.path.join(work_dir, "test.patch")
                with open(test_patch_file, 'w') as f:
                    f.write(task['test_patch'])

                patch_result = subprocess.run(
                    ["git", "apply", test_patch_file],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if patch_result.returncode != 0:
                    if self.verbose:
                        print(f"    Warning: Test patch failed: {patch_result.stderr[:100]}")

            # Start Shepherd in the repo directory with safety wrapper
            if self.verbose:
                print(f"    Starting Shepherd with safety restrictions...")

            shepherd_base_cmd = [
                SHEPHERD_BINARY,
                "--system-prompt",
                f"You are an expert software engineer. Fix the following issue in the {repo_name} codebase:\n\n{task['problem_statement']}\n\nUse the available tools to read files, make changes, and test your solution. Stay focused on fixing this specific issue."
            ]

            if self.config_file and os.path.exists(self.config_file):
                shepherd_base_cmd.extend(["--config", self.config_file])

            # Wrap with safety script
            shepherd_cmd = [SHEPHERD_SAFETY_WRAPPER, repo_dir] + shepherd_base_cmd

            # Set up restricted environment
            env = os.environ.copy()
            env['HOME'] = work_dir  # Prevent access to real home directory
            env['SHEPHERD_WORKSPACE'] = repo_dir  # Hint at working directory
            # Remove potentially dangerous env vars
            for key in ['SSH_AUTH_SOCK', 'SSH_AGENT_PID', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']:
                env.pop(key, None)

            # Run Shepherd with timeout and restricted environment
            try:
                if self.verbose:
                    # Real-time output mode - don't capture, let it print directly
                    shepherd_proc = subprocess.Popen(
                        shepherd_cmd,
                        stdin=subprocess.PIPE,
                        text=True,
                        cwd=repo_dir,  # CRITICAL: Start in repo dir, not home
                        env=env,  # Use restricted environment
                        preexec_fn=os.setsid
                    )

                    # Send problem and wait
                    shepherd_proc.stdin.write(f"{task['problem_statement']}\n\nPlease fix this issue.\nexit\n")
                    shepherd_proc.stdin.flush()
                    shepherd_proc.wait(timeout=SWEBENCH_TIMEOUT_SECONDS)
                    stdout = ""
                    stderr = ""
                else:
                    # Capture mode
                    shepherd_proc = subprocess.Popen(
                        shepherd_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=repo_dir,  # CRITICAL: Start in repo dir, not home
                        env=env,  # Use restricted environment
                        preexec_fn=os.setsid
                    )

                    # Send "exit" after timeout or when done
                    stdout, stderr = shepherd_proc.communicate(
                        input=f"{task['problem_statement']}\n\nPlease fix this issue.\nexit\n",
                        timeout=SWEBENCH_TIMEOUT_SECONDS
                    )

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(shepherd_proc.pid), signal.SIGKILL)
                except:
                    pass
                return False, "Timeout: Task took longer than 10 minutes"

            # Run tests to validate solution
            if self.verbose:
                print(f"    Running tests...")

            # Parse FAIL_TO_PASS tests
            tests_to_check = self._parse_test_list(task['FAIL_TO_PASS'])

            if not tests_to_check:
                return False, "No tests specified to validate"

            # Find python3 executable (use system python, not virtualenv)
            python_exe = shutil.which('python3') or shutil.which('python') or '/usr/bin/python3'

            # Run pytest on the specified tests
            test_cmd = [python_exe, "-m", "pytest", "-xvs"] + tests_to_check

            test_result = subprocess.run(
                test_cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy()  # Use full environment for test execution
            )

            # Check if tests passed
            passed = test_result.returncode == 0

            if self.verbose and not passed:
                print(f"    Test output: {test_result.stdout[-500:]}")

            return passed, None

        except Exception as e:
            return False, f"Exception: {str(e)[:200]}"

        finally:
            # Cleanup: Remove temporary directory
            if work_dir and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    if self.verbose:
                        print(f"    Cleaned up {work_dir}")
                except Exception as e:
                    print(f"    Warning: Failed to cleanup {work_dir}: {e}")

    def _parse_test_list(self, test_string: str) -> List[str]:
        """Parse test specification string into list of test paths"""
        if not test_string:
            return []

        # Test strings are JSON arrays like: '["test_file.py::test_name"]'
        try:
            import json
            tests = json.loads(test_string)
            return tests if isinstance(tests, list) else [str(tests)]
        except:
            # Fallback: split by newlines/commas
            return [t.strip() for t in test_string.replace(',', '\n').split('\n') if t.strip()]

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
        description="Test shepherd accuracy on AI benchmarks (MMLU, HellaSwag, SWE-bench)"
    )
    parser.add_argument("--config", "-c",
                       help="Shepherd config file (default: ~/.shepherd/config.json)")
    parser.add_argument("--benchmark", "-b",
                       choices=["mmlu", "hellaswag", "swebench", "all"],
                       default="all",
                       help="Which benchmark to run")
    parser.add_argument("--count", "-n",
                       type=int,
                       help="Number of questions/tasks to test (default: all)")
    parser.add_argument("--subjects",
                       nargs="+",
                       help="MMLU subjects to test (e.g., --subjects computer_science machine_learning)")
    parser.add_argument("--list-subjects",
                       action="store_true",
                       help="List all available MMLU subjects and exit")
    parser.add_argument("--difficulty",
                       choices=["easy", "medium", "hard", "very hard"],
                       help="SWE-bench difficulty filter")
    parser.add_argument("--repos",
                       nargs="+",
                       help="SWE-bench repositories to test (e.g., --repos django/django flask/flask)")
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

        if args.benchmark in ["mmlu", "all"]:
            result = runner.run_mmlu_benchmark(count=args.count, subjects=args.subjects)
            if result:
                print_summary(result)
                results.append(result)

        if args.benchmark in ["hellaswag", "all"]:
            result = runner.run_hellaswag_benchmark(count=args.count)
            if result:
                print_summary(result)
                results.append(result)

        if args.benchmark in ["swebench", "all"]:
            result = runner.run_swebench_benchmark(
                count=args.count,
                difficulty=args.difficulty,
                repos=args.repos
            )
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

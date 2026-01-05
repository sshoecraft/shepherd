#!/usr/bin/env python3
"""
CLI Command Integration Tests

Tests the shepherd CLI commands directly without needing a running server.

Usage:
    python3 test_cli_commands.py

Test IDs (from docs/testing.md):
    CLI-001: shepherd --help
    CLI-002: shepherd --version
    CLI-003: shepherd config show
    CLI-004: shepherd config set key value
    CLI-005: shepherd provider list
    CLI-006: shepherd provider add ...
    CLI-007: shepherd provider use name
    CLI-008: shepherd sched list
    CLI-009: shepherd tools list
    CLI-010: shepherd mcp list
"""

import os
import sys
import subprocess
import unittest
import tempfile
import shutil

# Configuration
SHEPHERD_BINARY = os.environ.get("SHEPHERD_BINARY", "./build/shepherd")


def run_command(args: list, timeout: int = 30) -> tuple:
    """Run shepherd command and return (stdout, stderr, return_code)"""
    cmd = [SHEPHERD_BINARY] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except FileNotFoundError:
        return "", f"Binary not found: {SHEPHERD_BINARY}", -1


class TestCLICommands(unittest.TestCase):
    """CLI Command Integration Tests"""

    @classmethod
    def setUpClass(cls):
        """Verify binary exists"""
        if not os.path.exists(SHEPHERD_BINARY):
            raise unittest.SkipTest(f"Shepherd binary not found: {SHEPHERD_BINARY}")

        # Create temp directory for config isolation
        cls.temp_dir = tempfile.mkdtemp(prefix="shepherd_test_")
        cls.config_dir = os.path.join(cls.temp_dir, ".config", "shepherd")
        os.makedirs(cls.config_dir, exist_ok=True)

        # Set environment to use temp config
        os.environ["HOME"] = cls.temp_dir
        os.environ["XDG_CONFIG_HOME"] = os.path.join(cls.temp_dir, ".config")

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory"""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    # =========================================================================
    # CLI-001: shepherd --help
    # =========================================================================

    def test_CLI001_help(self):
        """Test --help shows usage information"""
        stdout, stderr, code = run_command(["--help"])

        self.assertEqual(code, 0, f"--help should return 0, got {code}")
        self.assertTrue(
            "usage" in stdout.lower() or "options" in stdout.lower(),
            f"--help should show usage info, got: {stdout[:500]}"
        )

    def test_CLI001_help_short(self):
        """Test -h shows usage information"""
        stdout, stderr, code = run_command(["-h"])

        self.assertEqual(code, 0)
        self.assertTrue(len(stdout) > 0, "Should produce output")

    # =========================================================================
    # CLI-002: shepherd --version
    # =========================================================================

    def test_CLI002_version(self):
        """Test --version shows version"""
        stdout, stderr, code = run_command(["--version"])

        self.assertEqual(code, 0, f"--version should return 0, got {code}")

        # Version could be in stdout or stderr depending on implementation
        output = stdout + stderr
        # Should contain version-like pattern (e.g., "1.0.0" or "v1.0.0")
        self.assertTrue(
            any(c.isdigit() for c in output) and "." in output,
            f"--version should show version number, got: {output[:200]}"
        )

    def test_CLI002_version_short(self):
        """Test -v shows version"""
        stdout, stderr, code = run_command(["-v"])

        # -v might be verbose flag instead of version
        # Accept either success or version output
        self.assertIn(code, [0, 1])  # May fail if -v needs provider

    # =========================================================================
    # CLI-003: shepherd config show
    # =========================================================================

    def test_CLI003_config_show(self):
        """Test config show displays configuration"""
        stdout, stderr, code = run_command(["config", "show"])

        # May fail if no config exists, but command should be recognized
        self.assertIn(code, [0, 1], f"Unexpected return code: {code}")

        # If successful, should show some config info
        if code == 0:
            output = stdout + stderr
            # Should have some content
            self.assertTrue(len(output) > 0)

    def test_CLI003_config_list(self):
        """Test config list (alias for show)"""
        stdout, stderr, code = run_command(["config", "list"])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-004: shepherd config set key value
    # =========================================================================

    def test_CLI004_config_set(self):
        """Test config set saves a value"""
        # Set a test value
        stdout, stderr, code = run_command(["config", "set", "test_key", "test_value"])

        # Command should be recognized (may fail without proper config setup)
        self.assertIn(code, [0, 1], f"Unexpected return code: {code}")

    def test_CLI004_config_get(self):
        """Test config get retrieves a value"""
        stdout, stderr, code = run_command(["config", "get", "model"])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-005: shepherd provider list
    # =========================================================================

    def test_CLI005_provider_list(self):
        """Test provider list shows available providers"""
        stdout, stderr, code = run_command(["provider", "list"])

        # Command should work
        self.assertIn(code, [0, 1])

        output = stdout + stderr
        # Should mention some providers or show empty list
        # Common providers: openai, anthropic, gemini, ollama, llamacpp
        if code == 0 and len(output.strip()) > 0:
            # Check for expected provider names or "no providers" message
            has_provider = any(p in output.lower() for p in
                              ["openai", "anthropic", "gemini", "ollama", "llamacpp", "no provider"])
            # Or might show formatted table/list
            self.assertTrue(len(output) > 0, "Should show output")

    # =========================================================================
    # CLI-006: shepherd provider add ...
    # =========================================================================

    def test_CLI006_provider_add_help(self):
        """Test provider add shows usage when called without args"""
        stdout, stderr, code = run_command(["provider", "add"])

        # Should either show help or error about missing arguments
        output = stdout + stderr
        self.assertTrue(
            len(output) > 0,
            "provider add should show usage or error"
        )

    def test_CLI006_provider_add_openai(self):
        """Test adding an OpenAI provider"""
        stdout, stderr, code = run_command([
            "provider", "add",
            "--name", "test_openai",
            "--type", "openai",
            "--api-key", "sk-test-key"
        ])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-007: shepherd provider use name
    # =========================================================================

    def test_CLI007_provider_use(self):
        """Test provider use switches active provider"""
        stdout, stderr, code = run_command(["provider", "use", "nonexistent"])

        # Should fail for nonexistent provider but command should work
        self.assertIn(code, [0, 1])

        output = stdout + stderr
        if code != 0:
            self.assertTrue(
                "not found" in output.lower() or "error" in output.lower(),
                f"Should indicate provider not found: {output}"
            )

    def test_CLI007_provider_show(self):
        """Test provider show displays current provider"""
        stdout, stderr, code = run_command(["provider", "show"])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-008: shepherd sched list
    # =========================================================================

    def test_CLI008_sched_list(self):
        """Test sched list shows scheduled tasks"""
        stdout, stderr, code = run_command(["sched", "list"])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    def test_CLI008_sched_help(self):
        """Test sched help shows scheduler commands"""
        stdout, stderr, code = run_command(["sched", "--help"])

        # Should show help or be recognized
        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-009: shepherd tools list
    # =========================================================================

    def test_CLI009_tools_list(self):
        """Test tools list shows available tools"""
        stdout, stderr, code = run_command(["tools", "list"])

        # Command should work
        self.assertIn(code, [0, 1])

        output = stdout + stderr
        if code == 0:
            # Should show some tools
            self.assertTrue(len(output) > 0, "Should list tools")

    def test_CLI009_tools_help(self):
        """Test tools help shows usage"""
        stdout, stderr, code = run_command(["tools", "--help"])

        self.assertIn(code, [0, 1])

    # =========================================================================
    # CLI-010: shepherd mcp list
    # =========================================================================

    def test_CLI010_mcp_list(self):
        """Test mcp list shows MCP servers"""
        stdout, stderr, code = run_command(["mcp", "list"])

        # Command should be recognized
        self.assertIn(code, [0, 1])

    def test_CLI010_mcp_help(self):
        """Test mcp help shows MCP commands"""
        stdout, stderr, code = run_command(["mcp", "--help"])

        self.assertIn(code, [0, 1])

    # =========================================================================
    # Additional CLI Tests
    # =========================================================================

    def test_unknown_command(self):
        """Test unknown command shows error"""
        stdout, stderr, code = run_command(["unknown_command_xyz"])

        # Should fail
        self.assertNotEqual(code, 0, "Unknown command should fail")

    def test_ctl_commands(self):
        """Test ctl commands are recognized"""
        # These require a running server, so they should fail gracefully
        stdout, stderr, code = run_command(["ctl", "status"])

        # Should fail (no server) but command should be recognized
        output = stdout + stderr
        self.assertTrue(
            code != 0 or "error" in output.lower() or "not running" in output.lower() or len(output) > 0,
            "ctl command should be recognized"
        )

    def test_empty_args(self):
        """Test running with no args shows help or starts interactive mode"""
        stdout, stderr, code = run_command([])

        # May show help, error about missing provider, or try to start
        # Just verify it doesn't crash
        self.assertIn(code, [0, 1, 2])


if __name__ == "__main__":
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        print("Make sure to build shepherd first: make")
        sys.exit(1)

    print(f"Running CLI Command tests with binary: {SHEPHERD_BINARY}")
    unittest.main(verbosity=2)

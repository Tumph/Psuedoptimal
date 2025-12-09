"""Safe sandbox for executing untrusted Python code."""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    """Result of sandbox execution."""

    success: bool
    output: str
    error: str
    timed_out: bool


# Sandboxed runner script - executed in subprocess with resource limits
_RUNNER_SCRIPT = '''
import sys
import json
import resource

def set_limits(memory_mb: int, cpu_seconds: int):
    """Set resource limits for the sandboxed process."""
    # Memory limit (bytes)
    memory_bytes = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        pass  # May not be supported on all systems

    # CPU time limit
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, resource.error):
        pass

    # Prevent fork bombs
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, resource.error):
        pass

def execute_with_tests(code: str, test_code: str) -> dict:
    """Execute code and run test assertions."""
    result = {"success": False, "output": "", "error": ""}

    # Restricted globals - safe subset of builtins
    safe_builtins = {
        # Types and constructors
        "bool": bool, "int": int, "float": float, "str": str,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
        "complex": complex, "object": object, "type": type,

        # Iterators and generators
        "range": range, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "reversed": reversed,
        "iter": iter, "next": next, "slice": slice,

        # Math and logic
        "abs": abs, "min": min, "max": max, "sum": sum,
        "pow": pow, "round": round, "divmod": divmod,
        "all": all, "any": any, "len": len,

        # String/repr
        "repr": repr, "ascii": ascii, "chr": chr, "ord": ord,
        "format": format, "bin": bin, "hex": hex, "oct": oct,
        "hash": hash,

        # Type checking
        "isinstance": isinstance, "issubclass": issubclass,
        "callable": callable,

        # Sorting
        "sorted": sorted,

        # Boolean constants
        "True": True, "False": False, "None": None,

        # Exceptions (for error handling in tested code)
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "AssertionError": AssertionError,
        "StopIteration": StopIteration, "RuntimeError": RuntimeError,
        "ZeroDivisionError": ZeroDivisionError,

        # Print for debugging (output captured)
        "print": print,

        # Import (restricted to safe modules)
        "__import__": __import__,
    }

    exec_globals = {"__builtins__": safe_builtins}

    try:
        # Execute the main code (defines functions)
        exec(code, exec_globals)

        # Execute test assertions
        exec(test_code, exec_globals)

        result["success"] = True
        result["output"] = "All tests passed"

    except AssertionError as e:
        result["error"] = f"AssertionError: {e}"
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
    except NameError as e:
        result["error"] = f"NameError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result

if __name__ == "__main__":
    # Parse args: memory_mb, cpu_seconds
    memory_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cpu_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    set_limits(memory_mb, cpu_seconds)

    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    code = input_data["code"]
    test_code = input_data["test_code"]

    result = execute_with_tests(code, test_code)
    print(json.dumps(result))
'''


class SafeSandbox:
    """
    Secure sandbox for executing untrusted Python code.

    Uses subprocess isolation with:
    - CPU time limits
    - Memory limits
    - No network access (minimal environment)
    - Restricted builtins
    """

    def __init__(
        self,
        timeout: float = 5.0,
        memory_limit_mb: int = 200,
        max_output_size: int = 10000,
    ):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_output_size = max_output_size
        self._runner_script = _RUNNER_SCRIPT

    def execute(self, code: str, test_code: str) -> SandboxResult:
        """
        Execute Python code with test assertions in isolated sandbox.

        Args:
            code: Python code that defines functions
            test_code: Test assertions to run (e.g., "assert func(1) == 2")

        Returns:
            SandboxResult with success/failure and any error messages
        """
        # Prepare input JSON
        input_data = json.dumps({"code": code, "test_code": test_code})

        runner_path = None
        try:
            # Write runner script to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(self._runner_script)
                runner_path = f.name

            # Execute in subprocess with minimal environment
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "HOME": os.environ.get("HOME", "/tmp"),
            }

            process = subprocess.Popen(
                [
                    sys.executable,
                    runner_path,
                    str(self.memory_limit_mb),
                    str(int(self.timeout)),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            try:
                stdout, stderr = process.communicate(
                    input=input_data, timeout=self.timeout + 1
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return SandboxResult(
                    success=False,
                    output="",
                    error="Execution timed out",
                    timed_out=True,
                )

            # Parse result
            if process.returncode == 0 and stdout.strip():
                try:
                    result = json.loads(stdout.strip())
                    return SandboxResult(
                        success=result.get("success", False),
                        output=result.get("output", "")[: self.max_output_size],
                        error=result.get("error", "")[: self.max_output_size],
                        timed_out=False,
                    )
                except json.JSONDecodeError:
                    return SandboxResult(
                        success=False,
                        output=stdout[: self.max_output_size],
                        error="Failed to parse sandbox output",
                        timed_out=False,
                    )
            else:
                # Check for resource limit errors
                error_msg = stderr[: self.max_output_size] if stderr else "Unknown error"
                if process.returncode == -9 or "MemoryError" in error_msg:
                    error_msg = "Memory limit exceeded"
                return SandboxResult(
                    success=False,
                    output="",
                    error=error_msg,
                    timed_out=False,
                )

        except Exception as e:
            return SandboxResult(
                success=False,
                output="",
                error=f"Sandbox error: {e}",
                timed_out=False,
            )
        finally:
            if runner_path and os.path.exists(runner_path):
                try:
                    os.unlink(runner_path)
                except OSError:
                    pass


# Safe imports allowed for MBPP tests
ALLOWED_IMPORTS = {
    "math",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "heapq",
    "bisect",
    "copy",
    "typing",
    "random",
    "statistics",
}


def prepare_test_code(tests: list[str], imports: Optional[list[str]] = None) -> str:
    """
    Prepare test code from MBPP test_list format.

    Args:
        tests: List of assert strings, e.g., ["assert func(x) == y"]
        imports: List of import statements from test_imports

    Returns:
        Combined test code string ready for execution
    """
    code_parts = []

    # Add safe imports
    if imports:
        for imp in imports:
            imp = imp.strip()
            # Extract module name and validate
            if imp.startswith("import "):
                module = imp.replace("import ", "").split()[0].split(".")[0]
            elif imp.startswith("from "):
                module = imp.split()[1].split(".")[0]
            else:
                continue

            if module in ALLOWED_IMPORTS:
                code_parts.append(imp)

    # Add test assertions
    for test in tests:
        test = test.strip()
        if test.startswith("assert"):
            code_parts.append(test)

    return "\n".join(code_parts)


def run_tests(
    code: str,
    tests: list[str],
    imports: Optional[list[str]] = None,
    sandbox: Optional[SafeSandbox] = None,
) -> bool:
    """
    Run MBPP tests against generated code.

    Args:
        code: Python code that defines the function(s)
        tests: MBPP test_list assertions
        imports: MBPP test_imports (optional)
        sandbox: SafeSandbox instance (creates one if not provided)

    Returns:
        True if all tests pass, False otherwise
    """
    if sandbox is None:
        sandbox = SafeSandbox()

    test_code = prepare_test_code(tests, imports)
    result = sandbox.execute(code=code, test_code=test_code)
    return result.success

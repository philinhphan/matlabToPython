import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional


def run_pytest(
    tests_path: Optional[str] = None,
    code_path: Optional[str] = None,
    pytest_args: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """
    Run pytest in a subprocess and capture its results.

    Args:
        tests_path: Path to the tests to run. Defaults to the current working directory.
        code_path: Directory to treat as the working directory while running pytest.
        pytest_args: Additional CLI arguments to pass through to pytest.

    Returns:
        Dictionary containing:
            - command: The shell-friendly pytest command that was executed.
            - cwd: Working directory used for the subprocess.
            - returncode: Integer exit code from pytest.
            - stdout: Captured standard output from pytest.
            - stderr: Captured standard error from pytest.
            - status: "passed" when pytest exits 0, otherwise "failed".
    """
    cmd = ["pytest"]

    if pytest_args:
        cmd.extend(pytest_args)

    if tests_path:
        cmd.append(tests_path)

    cwd = Path(code_path).resolve() if code_path else None

    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )

    command_str = " ".join(shlex.quote(part) for part in cmd)

    return {
        "command": command_str,
        "cwd": str(cwd) if cwd else "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "passed" if completed.returncode == 0 else "failed",
    }



"""
Dynamic Code Validator Tool

Executes tests to validate Python code runtime behavior.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def run_pytest(
    tests_path: Optional[str] = None,
    code_path: Optional[str] = None,
    pytest_args: Optional[Iterable[str]] = None,
) -> Dict[str, any]:
    """
    Run pytest in a subprocess and capture results.
    
    Args:
        tests_path: Path to tests to run (defaults to '.')
        code_path: Working directory for pytest
        pytest_args: Additional pytest CLI arguments
        
    Returns:
        Dictionary with test results
    """
    cmd = [sys.executable, "-m", "pytest"]
    
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


def write_files_to_disk(files_content: Dict[str, str], output_dir: Path) -> None:
    """Write files to disk for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files_content.items():
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def run_tests(
    files_content: Dict[str, str],
    output_dir: Optional[str] = None,
    pytest_args: Optional[Iterable[str]] = None,
) -> Dict:
    """
    Run tests on the provided Python files.
    
    Args:
        files_content: Dictionary mapping filename to code
        output_dir: Directory to write files (temp dir if not provided)
        pytest_args: Additional pytest arguments
        
    Returns:
        Dictionary with test execution results
    """
    import tempfile
    
    if output_dir:
        work_dir = Path(output_dir)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="matlab_convert_"))
    
    # Write files to disk
    write_files_to_disk(files_content, work_dir)
    
    # Run pytest with verbose output
    args = list(pytest_args) if pytest_args else []
    if "-v" not in args and "--verbose" not in args:
        args.append("-v")
    
    result = run_pytest(
        tests_path=".",
        code_path=str(work_dir),
        pytest_args=args,
    )
    
    result["output_dir"] = str(work_dir)
    result["files_written"] = list(files_content.keys())
    
    return result


def run_tests_tool(
    files_content: Dict[str, str],
    output_dir: Optional[str] = None,
    pytest_args: Optional[Iterable[str]] = None,
    write_files: bool = True,
) -> Dict:
    """
    Tool interface for test execution.
    
    Args:
        files_content: Dictionary mapping filename to code
        output_dir: Directory to write files
        pytest_args: Additional pytest arguments
        write_files: Whether to write files before testing
        
    Returns:
        Dictionary with test results for agent consumption
    """
    try:
        result = run_tests(files_content, output_dir, pytest_args)
        
        return {
            "success": result["status"] == "passed",
            "status": result["status"],
            "output_dir": result["output_dir"],
            "files_written": result["files_written"],
            "command": result["command"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e),
        }

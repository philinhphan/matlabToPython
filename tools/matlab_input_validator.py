"""
MATLAB Input Validator Tool

Validates MATLAB input files before conversion:
- File existence and readability
- Extension validation
- Basic syntax checks
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of MATLAB input validation."""
    is_valid: bool
    file_path: str
    content: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_file_exists(file_path: str) -> tuple[bool, Optional[str]]:
    """Check if file exists and is readable."""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    return True, None


def check_extension(file_path: str) -> tuple[bool, Optional[str]]:
    """Check if file has .m extension."""
    ext = Path(file_path).suffix.lower()
    if ext != ".m":
        return False, f"Invalid extension '{ext}'. Expected '.m' for MATLAB files."
    return True, None


def check_matlab_syntax(content: str, file_path: str) -> List[str]:
    """
    Perform basic MATLAB syntax sanity checks.
    
    Returns list of warning messages (empty if OK).
    """
    warnings = []
    
    # Check for balanced brackets
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for i, char in enumerate(content):
        if char in brackets:
            stack.append((char, i))
        elif char in brackets.values():
            if not stack:
                warnings.append(f"Unbalanced bracket '{char}' at position {i}")
            else:
                open_bracket, _ = stack.pop()
                if brackets[open_bracket] != char:
                    warnings.append(f"Mismatched brackets at position {i}")
    
    if stack:
        warnings.append(f"Unclosed brackets: {[b[0] for b in stack]}")
    
    # Check for common MATLAB constructs
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments and empty lines
        if not stripped or stripped.startswith('%'):
            continue
        
        # Check for function definitions without 'end' (warning only)
        if stripped.startswith('function ') and 'end' not in content:
            warnings.append(f"Line {i}: Function definition may be missing 'end' keyword")
    
    return warnings


def validate_matlab_input(
    file_paths: Union[str, List[str]],
    strict: bool = False
) -> Dict[str, ValidationResult]:
    """
    Validate one or more MATLAB input files.
    
    Args:
        file_paths: Single file path or list of file paths
        strict: If True, warnings are treated as errors
        
    Returns:
        Dictionary mapping file paths to ValidationResult objects
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    results = {}
    
    for file_path in file_paths:
        errors = []
        warnings = []
        content = None
        
        # Check file exists
        exists, error = check_file_exists(file_path)
        if not exists:
            results[file_path] = ValidationResult(
                is_valid=False,
                file_path=file_path,
                errors=[error] if error else []
            )
            continue
        
        # Check extension
        valid_ext, error = check_extension(file_path)
        if not valid_ext:
            errors.append(error)
        
        # Read content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                warnings.append("File was read with latin-1 encoding (not UTF-8)")
            except Exception as e:
                errors.append(f"Failed to read file: {e}")
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
        
        # Run syntax checks if content was read
        if content is not None:
            syntax_warnings = check_matlab_syntax(content, file_path)
            warnings.extend(syntax_warnings)
        
        # Determine validity
        is_valid = len(errors) == 0
        if strict and warnings:
            is_valid = False
            errors.extend(warnings)
            warnings = []
        
        results[file_path] = ValidationResult(
            is_valid=is_valid,
            file_path=file_path,
            content=content,
            errors=errors,
            warnings=warnings
        )
    
    return results


def validate_matlab_input_tool(
    file_paths: Union[str, List[str]],
    strict: bool = False
) -> Dict:
    """
    Tool interface for MATLAB input validation.
    
    This is the function exposed to the agent as a tool.
    
    Args:
        file_paths: Single file path or list of file paths
        strict: If True, warnings are treated as errors
        
    Returns:
        Dictionary with validation results suitable for agent consumption
    """
    results = validate_matlab_input(file_paths, strict)
    
    output = {
        "all_valid": all(r.is_valid for r in results.values()),
        "files": {}
    }
    
    for path, result in results.items():
        output["files"][path] = {
            "is_valid": result.is_valid,
            "content": result.content,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    
    return output

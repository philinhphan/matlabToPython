"""
Static Code Validator Tool

Performs static analysis on Python code:
- AST parsing for syntax validation
- Optional linting integration
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SyntaxError:
    """Details of a syntax error."""
    filename: str
    line: int
    column: int
    message: str
    code_context: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of static validation."""
    is_valid: bool
    errors: List[SyntaxError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def clean_code(code: str) -> str:
    """
    Clean code of common LLM artifacts.
    
    Removes markdown code fences and trailing whitespace.
    """
    code = code.strip()
    
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def validate_single_file(filename: str, code: str) -> ValidationResult:
    """
    Validate syntax of a single Python file.
    
    Args:
        filename: Name of the file (for error messages)
        code: Python code to validate
        
    Returns:
        ValidationResult with syntax check results
    """
    cleaned_code = clean_code(code)
    
    if not cleaned_code:
        return ValidationResult(
            is_valid=False,
            errors=[SyntaxError(
                filename=filename,
                line=0,
                column=0,
                message="Empty code"
            )]
        )
    
    try:
        ast.parse(cleaned_code)
        return ValidationResult(is_valid=True)
    except SyntaxError as e:
        # Get the line of code that caused the error
        lines = cleaned_code.split('\n')
        code_context = None
        if e.lineno and 0 < e.lineno <= len(lines):
            code_context = lines[e.lineno - 1]
        
        return ValidationResult(
            is_valid=False,
            errors=[SyntaxError(
                filename=filename,
                line=e.lineno or 0,
                column=e.offset or 0,
                message=str(e.msg) if hasattr(e, 'msg') else str(e),
                code_context=code_context
            )]
        )


def validate_syntax(files_content: Dict[str, str]) -> Dict[str, ValidationResult]:
    """
    Validate syntax of multiple Python files.
    
    Args:
        files_content: Dictionary mapping filename to code
        
    Returns:
        Dictionary mapping filename to ValidationResult
    """
    results = {}
    for filename, code in files_content.items():
        results[filename] = validate_single_file(filename, code)
    return results


def validate_syntax_tool(
    files_content: Dict[str, str],
    clean: bool = True
) -> Dict:
    """
    Tool interface for syntax validation.
    
    Args:
        files_content: Dictionary mapping filename to code
        clean: Whether to clean code before validation (remove markdown fences)
        
    Returns:
        Dictionary with validation results suitable for agent consumption
    """
    results = validate_syntax(files_content)
    
    all_valid = all(r.is_valid for r in results.values())
    
    output = {
        "all_valid": all_valid,
        "files": {},
        "cleaned_files": {} if clean else None
    }
    
    for filename, result in results.items():
        code = files_content[filename]
        cleaned = clean_code(code) if clean else code
        
        output["files"][filename] = {
            "is_valid": result.is_valid,
            "errors": [
                {
                    "line": e.line,
                    "column": e.column,
                    "message": e.message,
                    "code_context": e.code_context
                }
                for e in result.errors
            ],
            "warnings": result.warnings
        }
        
        if clean:
            output["cleaned_files"][filename] = cleaned
    
    return output

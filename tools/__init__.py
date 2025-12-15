# MATLAB to Python Converter Tools
"""
This package contains tools for the MATLAB to Python conversion agent.

Each tool is a self-contained module that can be called by the agent:
- matlab_input_validator: Validates MATLAB input files
- code_converter: LLM-based code conversion
- test_generator: Generates pytest test files
- static_code_validator: AST validation and static analysis
- dynamic_code_validator: Test execution
- report_generator: Conversion reports
"""

from .matlab_input_validator import validate_matlab_input
from .code_converter import convert_matlab_to_python
from .test_generator import generate_tests
from .static_code_validator import validate_syntax
from .dynamic_code_validator import run_tests
from .report_generator import generate_report

__all__ = [
    "validate_matlab_input",
    "convert_matlab_to_python",
    "generate_tests",
    "validate_syntax",
    "run_tests",
    "generate_report",
]

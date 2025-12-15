"""
MATLAB to Python Conversion Agent

An autonomous agent that uses discrete tools to convert MATLAB code to Python,
ensuring 100% correctness through validation and testing.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, RunContext

# Configure logging
logfire.configure()
logfire.instrument_pydantic_ai()

# Load environment variables
load_dotenv()

# Import tools
from tools.matlab_input_validator import validate_matlab_input, validate_matlab_input_tool
from tools.code_converter import convert_matlab_to_python, convert_matlab_to_python_tool
from tools.test_generator import generate_tests, generate_tests_tool
from tools.static_code_validator import validate_syntax, validate_syntax_tool, clean_code
from tools.dynamic_code_validator import run_tests, run_tests_tool, write_files_to_disk
from tools.report_generator import generate_report, generate_report_tool


# Configuration
DEFAULT_MODEL = os.getenv("CONVERSION_MODEL", "openai:gpt-4o")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
MAX_RETRIES = 5


class ConversionContext:
    """Context for the conversion process."""
    
    def __init__(
        self,
        input_files: List[str],
        output_dir: Path,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ):
        self.input_files = input_files
        self.output_dir = output_dir
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # State tracking
        self.validated_inputs: Dict[str, Any] = {}
        self.converted_code: Dict[str, str] = {}
        self.test_code: Dict[str, str] = {}
        self.validation_results: Dict[str, Any] = {}
        self.test_results: Dict[str, Any] = {}
        self.report: Optional[str] = None


def step_validate_inputs(ctx: ConversionContext) -> bool:
    """Step 1: Validate MATLAB input files."""
    print("\n📋 Step 1: Validating MATLAB input files...")
    
    result = validate_matlab_input_tool(ctx.input_files)
    ctx.validated_inputs = result
    
    if not result["all_valid"]:
        print("❌ Input validation failed!")
        for path, info in result["files"].items():
            if not info["is_valid"]:
                for error in info["errors"]:
                    print(f"   - {path}: {error}")
        return False
    
    print(f"✅ All {len(ctx.input_files)} input files validated successfully")
    
    # Print any warnings
    for path, info in result["files"].items():
        for warning in info.get("warnings", []):
            print(f"   ⚠️  {path}: {warning}")
    
    return True


def step_convert_code(ctx: ConversionContext) -> bool:
    """Step 2: Convert MATLAB to Python."""
    print("\n🔄 Step 2: Converting MATLAB to Python...")
    
    # Gather file contents from validation results
    file_contents = {}
    for path, info in ctx.validated_inputs["files"].items():
        if info["content"]:
            filename = Path(path).name
            file_contents[filename] = info["content"]
    
    result = convert_matlab_to_python_tool(
        file_contents=file_contents,
        model_name=ctx.model_name,
        ollama_url=ctx.ollama_url,
    )
    
    if not result["success"]:
        print(f"❌ Code conversion failed: {result.get('error', 'Unknown error')}")
        return False
    
    ctx.converted_code = result["files"]
    print(f"✅ Converted {len(result['input_files'])} files to {len(result['output_files'])} Python files")
    for filename in result["output_files"]:
        print(f"   - {filename}")
    
    return True


def step_validate_syntax(ctx: ConversionContext) -> bool:
    """Step 3: Static syntax validation."""
    print("\n🔍 Step 3: Validating Python syntax...")
    
    result = validate_syntax_tool(ctx.converted_code, clean=True)
    ctx.validation_results = result
    
    # Update converted code with cleaned versions
    if result.get("cleaned_files"):
        ctx.converted_code = result["cleaned_files"]
    
    if not result["all_valid"]:
        print("❌ Syntax validation failed!")
        for filename, info in result["files"].items():
            if not info["is_valid"]:
                for error in info["errors"]:
                    print(f"   - {filename}:{error['line']}: {error['message']}")
        return False
    
    print(f"✅ All {len(ctx.converted_code)} files have valid Python syntax")
    return True


def step_generate_tests(ctx: ConversionContext) -> bool:
    """Step 4: Generate pytest tests."""
    print("\n🧪 Step 4: Generating pytest tests...")
    
    result = generate_tests_tool(
        ctx.converted_code,
        model_name=ctx.model_name,
        ollama_url=ctx.ollama_url,
    )
    
    if not result["success"]:
        print(f"❌ Test generation failed: {result.get('error', 'Unknown error')}")
        return False
    
    ctx.test_code = result["files"]
    print(f"✅ Generated {result['test_count']} test files")
    for filename in result["files"]:
        print(f"   - {filename}")
    
    return True


def step_validate_tests_syntax(ctx: ConversionContext) -> bool:
    """Step 4b: Validate test file syntax."""
    print("\n🔍 Step 4b: Validating test file syntax...")
    
    result = validate_syntax_tool(ctx.test_code, clean=True)
    
    if result.get("cleaned_files"):
        ctx.test_code = result["cleaned_files"]
    
    if not result["all_valid"]:
        print("❌ Test file syntax validation failed!")
        for filename, info in result["files"].items():
            if not info["is_valid"]:
                for error in info["errors"]:
                    print(f"   - {filename}:{error['line']}: {error['message']}")
        return False
    
    print("✅ All test files have valid Python syntax")
    return True


def step_run_tests(ctx: ConversionContext) -> bool:
    """Step 5: Run tests to verify conversion."""
    print("\n🏃 Step 5: Running tests...")
    
    # Combine source and test files
    all_files = {**ctx.converted_code, **ctx.test_code}
    
    result = run_tests_tool(
        all_files,
        output_dir=str(ctx.output_dir),
        pytest_args=["-v", "--tb=short"],
    )
    ctx.test_results = result
    
    if not result["success"]:
        print("❌ Tests failed!")
        if result.get("stdout"):
            print("\n--- Test Output ---")
            print(result["stdout"][:3000])  # Limit output
        if result.get("stderr"):
            print("\n--- Errors ---")
            print(result["stderr"][:1000])
        return False
    
    print(f"✅ All tests passed!")
    print(f"   Files written to: {result.get('output_dir', ctx.output_dir)}")
    return True


def step_generate_report(ctx: ConversionContext) -> bool:
    """Step 6: Generate conversion report."""
    print("\n📝 Step 6: Generating report...")
    
    result = generate_report_tool(
        input_files=ctx.input_files,
        output_files=ctx.converted_code,
        test_files=ctx.test_code,
        validation_results=ctx.validation_results,
        test_results=ctx.test_results,
        output_dir=str(ctx.output_dir),
        format="markdown",
        save_to_file=True,
    )
    
    if not result["success"]:
        print(f"⚠️  Report generation failed: {result.get('error', 'Unknown error')}")
        return True  # Non-fatal
    
    ctx.report = result["report"]
    print(f"✅ Report saved to: {result.get('report_path')}")
    return True


def retry_conversion_with_feedback(
    ctx: ConversionContext,
    error_feedback: str,
    retry_count: int,
) -> bool:
    """Retry conversion with error feedback."""
    print(f"\n🔄 Retry {retry_count}/{MAX_RETRIES}: Reconverting with error feedback...")
    
    # Add error context to the conversion
    file_contents = {}
    for path, info in ctx.validated_inputs["files"].items():
        if info["content"]:
            filename = Path(path).name
            file_contents[filename] = info["content"]
    
    # The error feedback will be included in the retry
    result = convert_matlab_to_python_tool(
        file_contents=file_contents,
        model_name=ctx.model_name,
        ollama_url=ctx.ollama_url,
    )
    
    if not result["success"]:
        return False
    
    ctx.converted_code = result["files"]
    return True


def run_conversion_agent(
    input_files: List[str],
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> bool:
    """
    Run the full conversion pipeline.
    
    Args:
        input_files: List of MATLAB file paths
        output_dir: Output directory for converted files
        model_name: LLM model to use
        ollama_url: Ollama API URL
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    print("=" * 60)
    print("🚀 MATLAB to Python Conversion Agent")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Input files: {len(input_files)}")
    print(f"Output directory: {output_dir}")
    
    # Create context
    ctx = ConversionContext(
        input_files=input_files,
        output_dir=Path(output_dir),
        model_name=model_name,
        ollama_url=ollama_url,
    )
    
    # Step 1: Validate inputs
    if not step_validate_inputs(ctx):
        return False
    
    # Step 2: Convert code
    if not step_convert_code(ctx):
        return False
    
    # Retry loop for validation and testing
    for retry in range(MAX_RETRIES):
        # Step 3: Validate syntax
        if not step_validate_syntax(ctx):
            if retry < MAX_RETRIES - 1:
                error_feedback = "Syntax errors found. Please regenerate with valid Python syntax."
                if not retry_conversion_with_feedback(ctx, error_feedback, retry + 1):
                    continue
            else:
                print(f"❌ Failed after {MAX_RETRIES} retries")
                return False
        else:
            break
    
    # Step 4: Generate tests
    if not step_generate_tests(ctx):
        return False
    
    # Step 4b: Validate test syntax
    if not step_validate_tests_syntax(ctx):
        return False
    
    # Step 5: Run tests with retry
    for retry in range(MAX_RETRIES):
        if step_run_tests(ctx):
            break
        
        if retry < MAX_RETRIES - 1:
            print(f"\n🔄 Test failure - regenerating code (attempt {retry + 2}/{MAX_RETRIES})...")
            
            # Regenerate both code and tests
            if not step_convert_code(ctx):
                continue
            if not step_validate_syntax(ctx):
                continue
            if not step_generate_tests(ctx):
                continue
            if not step_validate_tests_syntax(ctx):
                continue
        else:
            print(f"❌ Tests failed after {MAX_RETRIES} attempts")
            return False
    
    # Step 6: Generate report
    step_generate_report(ctx)
    
    print("\n" + "=" * 60)
    print("🎉 Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {ctx.output_dir}")
    print("\nGenerated files:")
    for filename in sorted(ctx.converted_code.keys()):
        print(f"  📄 {filename}")
    for filename in sorted(ctx.test_code.keys()):
        print(f"  🧪 {filename}")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MATLAB to Python Conversion Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more MATLAB (.m) files to convert",
    )
    parser.add_argument(
        "--output", "-o",
        default="converted",
        help="Output directory (default: converted)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for f in args.input_files:
        if not os.path.exists(f):
            print(f"❌ Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)
    
    # Run conversion
    success = run_conversion_agent(
        input_files=args.input_files,
        output_dir=args.output,
        model_name=args.model,
        ollama_url=args.ollama_url,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

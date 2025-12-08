import argparse
import ast
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dotenv import load_dotenv

from pydantic import BaseModel
import logfire
from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Configure Logfire
logfire.configure()
logfire.instrument_pydantic_ai()

from testing_utils import run_pytest

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Configuration ---
# The default URL for the Ollama OpenAI-compatible API endpoint.
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
# The default model to use for the conversion.
DEFAULT_MODEL = "qwen:0.5b"

# System prompt for non-tool-calling models (like qwen:0.5b) that need to be prompted for JSON.
SYSTEM_PROMPT_JSON = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert the given Matlab code to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.

You MUST return the converted code inside a JSON object with a single key "code".
Do not include any other explanations, markdown formatting, or introductory text in your response.
Only provide the raw JSON object.
"""

# System prompt for models that support tool-calling (like OpenAI's).
# This prompt does NOT mention JSON, as the tool-calling mechanism handles the structure.
SYSTEM_PROMPT_TOOLS = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert the given Matlab code to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.
Directly return the converted Python code.
"""

# System prompt for multi-file batch conversion with JSON output
SYSTEM_PROMPT_MULTIFILE_JSON = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert ALL the given Matlab files to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.

You are converting multiple files from the same codebase in a SINGLE operation. Pay attention to:
- Function calls between files (convert to proper Python imports)
- Shared variables and data structures
- Maintaining consistent naming conventions across files

You MUST return a JSON object with a "files" key containing a dictionary where:
- Keys are the output Python filenames (e.g., "example_script.py")
- Values are the complete Python code for each file

Example format:
{
  "files": {
    "file1.py": "import numpy as np\\n\\ndef function1():\\n    pass",
    "file2.py": "from file1 import function1\\n\\nfunction1()"
  }
}

Do not include any other explanations, markdown formatting, or introductory text.
Only provide the raw JSON object.
"""

# System prompt for multi-file batch conversion with tool calling
SYSTEM_PROMPT_MULTIFILE_TOOLS = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert ALL the given Matlab files to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.

You are converting multiple files from the same codebase in a SINGLE operation. Pay attention to:
- Function calls between files (convert to proper Python imports)
- Shared variables and data structures
- Maintaining consistent naming conventions across files

Return a dictionary with a "files" key containing all converted files where:
- Keys are the output Python filenames (e.g., "example_script.py")
- Values are the complete Python code for each file
"""


class PythonCode(BaseModel):
    """A Pydantic model to structure the output, ensuring it's a JSON with a 'code' field."""

    code: str


class PythonFiles(BaseModel):
    """A Pydantic model for multi-file conversion output."""

    files: Dict[str, str]  # filename -> Python code


def normalize_pytest_args(pytest_args: Optional[Any]) -> Optional[List[str]]:
    """
    Normalize various pytest argument representations into a list of strings.
    """
    if not pytest_args:
        return None
    if isinstance(pytest_args, list):
        return pytest_args
    if isinstance(pytest_args, tuple):
        return list(pytest_args)
    return shlex.split(str(pytest_args))


def run_cli_tests(
    should_run: bool,
    tests_path: Optional[str],
    code_root: Path,
    pytest_args: Optional[str],
) -> None:
    """
    Optionally execute pytest based on CLI arguments.
    """
    if not should_run:
        return

    args_list = normalize_pytest_args(pytest_args)
    test_target = tests_path or "."

    print("\n🧪 Running pytest on generated code...")
    print(f"   Working directory: {code_root}")
    print(f"   Tests target: {test_target}")
    if args_list:
        print(f"   Extra pytest args: {' '.join(args_list)}")

    result = run_pytest(
        tests_path=test_target,
        code_path=str(code_root),
        pytest_args=args_list,
    )

    print(f"   Command: {result['command']}")
    print(f"   Status: {result['status']}")

    if result["stdout"].strip():
        print("\n--- pytest stdout ---")
        print(result["stdout"])
        print("--- end stdout ---")

    if result["stderr"].strip():
        print("\n--- pytest stderr ---")
        print(result["stderr"])
        print("--- end stderr ---")

    if result["status"] != "passed":
        print("❌ Tests failed. See logs above for details.", file=sys.stderr)
        sys.exit(result["returncode"] or 1)
    else:
        print("✅ Tests passed successfully.")


def create_agent(ollama_url: str, model_name: str, multi_file: bool = False) -> Agent:
    """
    Creates and configures the Pydantic AI agent for code conversion.

    This function conditionally selects the output mode (tool-calling vs. prompted JSON)
    based on the model name.

    The agent is configured with an output validator that checks for valid Python syntax.
    If the syntax is invalid, it automatically triggers a retry, asking the model to
    correct its own mistake.

    Args:
        ollama_url: The URL of the Ollama OpenAI-compatible v1 endpoint.
        model_name: The name of the model to use (e.g., 'qwen:0.5b' or 'openai:gpt-4o').
        multi_file: Whether this is a multi-file conversion (affects system prompt and output type).

    Returns:
        A configured Pydantic AI Agent instance.
    """
    print(f"🤖 Configuring agent for model '{model_name}'...")

    effective_model: OpenAIChatModel
    effective_output_type: Any
    effective_system_prompt: str

    supports_tool_calling = model_name.startswith("openai:")

    if supports_tool_calling:
        # Use OpenAI API with native tool calling
        print("   ... Using OpenAI API with tool calling.")
        # Strip the "openai:" prefix when passing the name to OpenAIChatModel
        model_name_without_prefix = model_name.split(":", 1)[-1]
        effective_model = OpenAIChatModel(model_name=model_name_without_prefix)
        if multi_file:
            effective_output_type = PythonFiles  # Multiple files
            effective_system_prompt = SYSTEM_PROMPT_MULTIFILE_TOOLS
        else:
            effective_output_type = PythonCode  # Single file
            effective_system_prompt = SYSTEM_PROMPT_TOOLS
    else:
        # Assume Ollama or another provider that may not support tool calling
        print(f"   ... Using Ollama at {ollama_url} with prompted JSON.")
        ollama_provider = OllamaProvider(base_url=ollama_url)
        effective_model = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
        if multi_file:
            effective_output_type = PromptedOutput(PythonFiles)  # Multiple files
            effective_system_prompt = SYSTEM_PROMPT_MULTIFILE_JSON
        else:
            effective_output_type = PromptedOutput(PythonCode)  # Single file
            effective_system_prompt = SYSTEM_PROMPT_JSON

    # Create the agent with instructions, a structured output type, and retries enabled
    agent = Agent(
        model=effective_model,
        output_type=effective_output_type,
        instructions=effective_system_prompt,
        retries=5,  
    )

    if multi_file:
        @agent.output_validator
        def validate_python_files(ctx: RunContext[Any], output: PythonFiles) -> PythonFiles:
            """
            Validates all generated Python files using ast.parse.
            If any file has invalid syntax, it raises ModelRetry.
            """
            if not output.files:
                print(f"❌ No files generated on attempt {ctx.retry + 1}. Asking model to retry.")
                raise ModelRetry("No files were generated. Please generate Python code for all input files.")
            
            print(f"📝 Validating {len(output.files)} generated files...")
            errors = []
            
            for filename, code in output.files.items():
                code_to_validate = code.strip()
                
                # Basic cleanup
                if code_to_validate.startswith("```python"):
                    code_to_validate = code_to_validate[9:]
                if code_to_validate.startswith("```"):
                    code_to_validate = code_to_validate[3:]
                if code_to_validate.endswith("```"):
                    code_to_validate = code_to_validate[:-3]
                code_to_validate = code_to_validate.strip()
                
                if not code_to_validate:
                    errors.append(f"{filename}: Empty code")
                    continue
                
                try:
                    ast.parse(code_to_validate)
                    print(f"   ✅ {filename}: Valid syntax")
                    output.files[filename] = code_to_validate
                except SyntaxError as e:
                    errors.append(f"{filename}: {e}")
                    print(f"   ❌ {filename}: Syntax error - {e}")
            
            if errors:
                error_msg = f"Syntax errors found in {len(errors)} file(s):\n" + "\n".join(errors)
                error_msg += "\n\nPlease fix these errors and regenerate ALL files."
                if not model_name.startswith("openai:"):
                    error_msg += ' Ensure you output the raw JSON object with the "files" key.'
                raise ModelRetry(error_msg)
            
            print("✅ All files have valid Python syntax.")
            return output
    else:
        @agent.output_validator
        def validate_python_syntax(ctx: RunContext[Any], output: PythonCode) -> PythonCode:
            """
            Validates the generated Python code using ast.parse.
            If syntax is invalid, it raises ModelRetry to ask the LLM to fix it.
            """
            code_to_validate = output.code.strip()

            # Basic cleanup in case the model includes markdown fences despite instructions
            if code_to_validate.startswith("```python"):
                code_to_validate = code_to_validate[9:]
            if code_to_validate.startswith("```json"):
                code_to_validate = code_to_validate[7:]
            if code_to_validate.startswith("```"):
                code_to_validate = code_to_validate[3:]
            if code_to_validate.endswith("```"):
                code_to_validate = code_to_validate[:-3]
            code_to_validate = code_to_validate.strip()

            # If the model returned the JSON wrapper (in prompted mode), try to parse it
            if code_to_validate.startswith("{"):
                try:
                    import json
                    data = json.loads(code_to_validate)
                    if 'code' in data and isinstance(data['code'], str):
                        code_to_validate = data['code'].strip()
                except Exception:
                    pass

            # Re-run cleanup for the extracted code
            if code_to_validate.startswith("```python"):
                code_to_validate = code_to_validate[9:]
            if code_to_validate.startswith("```"):
                code_to_validate = code_to_validate[3:]
            if code_to_validate.endswith("```"):
                code_to_validate = code_to_validate[:-3]
            code_to_validate = code_to_validate.strip()

            if not code_to_validate:
                print(f"❌ Empty code detected on attempt {ctx.retry + 1}. Asking model to retry.")
                raise ModelRetry("The generated code was empty. Please generate the Python code.")

            try:
                ast.parse(code_to_validate)
                print("✅ Python code syntax is valid.")
                output.code = code_to_validate
                return output
            except SyntaxError as e:
                print(f"❌ Invalid syntax detected on attempt {ctx.retry + 1}. Asking model to retry.")
                print(f"   Error: {e}")
                
                error_feedback = f"The generated Python code has a syntax error: {e}. " \
                                 "Review your previous output, fix the specific syntax error, and try again."
                                 
                if not model_name.startswith("openai:"):
                    error_feedback += " Ensure you ONLY output the raw JSON object with the 'code' key."

                raise ModelRetry(error_feedback)

    if supports_tool_calling:
        @agent.tool
        def run_tests(
            ctx: RunContext[Any],
            tests_path: Optional[str] = None,
            code_path: Optional[str] = None,
            pytest_args: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """
            Execute pytest inside the agent workflow so the model can react to failures.
            """
            print("🧪 Agent requested to run pytest...")
            pytest_result = run_pytest(
                tests_path=tests_path,
                code_path=code_path,
                pytest_args=pytest_args,
            )
            print(f"   ...pytest finished with status: {pytest_result['status']}")
            return pytest_result

    return agent


def read_all_files(file_paths: List[str]) -> Dict[str, str]:
    """
    Read all MATLAB files and return their contents.
    
    Args:
        file_paths: List of paths to Matlab files being converted.
        
    Returns:
        Dictionary mapping filename to file content.
    """
    files_content = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            filename = Path(file_path).name
            files_content[filename] = content
        except Exception as e:
            print(f"❌ Error reading file '{file_path}': {e}", file=sys.stderr)
            files_content[filename] = f"ERROR: Could not read file - {e}"
    
    return files_content


def build_batch_conversion_prompt(files_content: Dict[str, str]) -> str:
    """
    Build a prompt for batch conversion of all files together.
    
    Args:
        files_content: Dictionary mapping filename to file content.
        
    Returns:
        A formatted prompt with all files.
    """
    prompt_parts = [
        "Convert the following MATLAB files to Python. All files are from the same codebase.",
        "Pay attention to dependencies between files and generate appropriate Python imports.",
        "Return the converted code for ALL files.\n"
    ]
    
    for filename, content in files_content.items():
        prompt_parts.append(f"\n{'='*60}")
        prompt_parts.append(f"FILE: {filename}")
        prompt_parts.append('='*60)
        prompt_parts.append(content)
    
    return "\n".join(prompt_parts)


def convert_single_file(matlab_code: str, agent: Agent) -> str:
    """
    Convert a single Matlab file to Python using the agent.
    
    Args:
        matlab_code: The Matlab code to convert.
        agent: The configured Pydantic AI agent.
        
    Returns:
        The converted Python code.
    """
    result = agent.run_sync(matlab_code)
    return result.output.code


def convert_multiple_files(files_content: Dict[str, str], agent: Agent) -> Dict[str, str]:
    """
    Convert multiple Matlab files to Python in a single batch operation.
    
    Args:
        files_content: Dictionary mapping MATLAB filename to content.
        agent: The configured Pydantic AI agent.
        
    Returns:
        Dictionary mapping Python filename to converted code.
    """
    prompt = build_batch_conversion_prompt(files_content)
    result = agent.run_sync(prompt)
    return result.output.files


def main():
    """
    Main function to handle command-line arguments and file operations.
    """
    parser = argparse.ArgumentParser(
        description="Convert Matlab source file(s) to Python using an agentic process with Pydantic AI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more Matlab (.m) files to convert."
    )
    parser.add_argument(
        "--output",
        help="Output file (for single input) or directory (for multiple inputs). "
             "If not specified, creates <input>_converted.py for single file or 'converted/' directory for multiple files."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"The model to use (e.g., 'qwen:0.5b' or 'openai:gpt-4o'). (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help=f"The URL of the Ollama OpenAI-compatible API endpoint (default: {DEFAULT_OLLAMA_URL}). "
             "This is ignored if using a model with a provider prefix like 'openai:'.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest against the converted code after generation completes.",
    )
    parser.add_argument(
        "--tests-path",
        help="Path to the pytest tests to execute (defaults to '.' inside the output directory).",
    )
    parser.add_argument(
        "--pytest-args",
        help="Additional pytest CLI arguments (provide as a quoted string).",
    )

    args = parser.parse_args()

    # Validate input files exist
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file not found at '{input_file}'", file=sys.stderr)
            sys.exit(1)

    # Determine if this is single or multi-file conversion
    is_multi_file = len(args.input_files) > 1
    
    # Determine output path(s)
    if args.output:
        output_path = args.output
    else:
        if is_multi_file:
            output_path = "converted"
        else:
            # Single file: create <input>_converted.py
            input_path = Path(args.input_files[0])
            output_path = str(input_path.with_name(f"{input_path.stem}_converted.py"))
    
    # For multi-file, ensure output is a directory
    if is_multi_file:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Create the agent
    print("\n⏳ Starting conversion process...")
    converter_agent = create_agent(args.url, args.model, multi_file=is_multi_file)
    
    if is_multi_file:
        # BATCH CONVERSION: Convert all files together in one operation
        print(f"\n📚 Reading {len(args.input_files)} MATLAB files for batch conversion...")
        
        # Read all files
        files_content = read_all_files(args.input_files)
        
        # Create mapping from MATLAB filename to expected Python filename
        matlab_to_python = {}
        for input_file in args.input_files:
            matlab_name = Path(input_file).name
            python_name = Path(input_file).with_suffix(".py").name
            matlab_to_python[matlab_name] = python_name
        
        print(f"\n{'='*60}")
        print(f"Converting ALL {len(files_content)} files in a single batch operation...")
        print('='*60)
        
        # Perform batch conversion
        try:
            converted_files_dict = convert_multiple_files(files_content, converter_agent)
        except ModelRetry as e:
            print(
                f"❌ Batch conversion failed after multiple retries. Last error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"❌ An unexpected error occurred during batch conversion: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        
        if not converted_files_dict:
            print("🛑 Batch conversion failed to produce any files. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        print(f"\n🎉 Batch conversion successful! Generated {len(converted_files_dict)} files.")
        
        # Write all output files
        converted_files = []
        for filename, python_code in converted_files_dict.items():
            output_file = output_dir / filename
            
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(python_code)
                print(f"✅ Saved: {output_file}")
                converted_files.append(str(output_file))
            except IOError as e:
                print(f"❌ Error writing to file '{output_file}': {e}", file=sys.stderr)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"✨ Batch conversion complete!")
        print(f"   Successfully wrote {len(converted_files)} of {len(converted_files_dict)} files.")
        if converted_files:
            print(f"   Output files:")
            for cf in converted_files:
                print(f"     - {cf}")
        print('='*60)

        run_cli_tests(
            should_run=args.run_tests,
            tests_path=args.tests_path,
            code_root=output_dir,
            pytest_args=args.pytest_args,
        )
        
    else:
        # SINGLE FILE CONVERSION
        input_file = args.input_files[0]
        print(f"\n{'='*60}")
        print(f"Converting: {input_file}")
        print('='*60)
        
        # Read the input file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                matlab_code = f.read()
            print(f"✅ Successfully read Matlab code from '{input_file}'.")
        except IOError as e:
            print(f"❌ Error reading file '{input_file}': {e}", file=sys.stderr)
            sys.exit(1)
        
        # Perform conversion
        try:
            python_code = convert_single_file(matlab_code, converter_agent)
        except ModelRetry as e:
            print(
                f"❌ Conversion failed after multiple retries. Last error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"❌ An unexpected error occurred during conversion: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        
        if not python_code:
            print(f"🛑 Conversion failed to produce code. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        print("🎉 Conversion successful and code validated.")
        
        # Write the output file
        output_file = Path(output_path)
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(python_code)
            print(f"✅ Successfully saved Python code to '{output_file}'.")
        except IOError as e:
            print(f"❌ Error writing to file '{output_file}': {e}", file=sys.stderr)
            sys.exit(1)

        run_cli_tests(
            should_run=args.run_tests,
            tests_path=args.tests_path,
            code_root=output_file.parent,
            pytest_args=args.pytest_args,
        )


if __name__ == "__main__":
    main()


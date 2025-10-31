import argparse
import ast
import sys
from typing import Any, Type
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


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


class PythonCode(BaseModel):
    """A Pydantic model to structure the output, ensuring it's a JSON with a 'code' field."""

    code: str


def create_agent(ollama_url: str, model_name: str) -> Agent:
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

    Returns:
        A configured Pydantic AI Agent instance.
    """
    print(f"🤖 Configuring agent for model '{model_name}'...")

    effective_model: OpenAIChatModel
    effective_output_type: Any
    effective_system_prompt: str

    if model_name.startswith("openai:"):
        # Use OpenAI API with native tool calling
        print("   ... Using OpenAI API with tool calling.")
        # Strip the "openai:" prefix when passing the name to OpenAIChatModel
        model_name_without_prefix = model_name.split(":", 1)[-1]
        effective_model = OpenAIChatModel(model_name=model_name_without_prefix)
        effective_output_type = PythonCode  # Use direct tool calling
        effective_system_prompt = SYSTEM_PROMPT_TOOLS
    else:
        # Assume Ollama or another provider that may not support tool calling
        print(f"   ... Using Ollama at {ollama_url} with prompted JSON.")
        ollama_provider = OllamaProvider(base_url=ollama_url)
        effective_model = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
        effective_output_type = PromptedOutput(PythonCode)  # Use prompted JSON
        effective_system_prompt = SYSTEM_PROMPT_JSON

    # Create the agent with instructions, a structured output type, and retries enabled
    agent = Agent(
        model=effective_model,
        output_type=effective_output_type,
        instructions=effective_system_prompt,
        retries=5,  
    )

    @agent.output_validator
    def validate_python_syntax(ctx: Any, output: PythonCode) -> PythonCode:
        """
        Validates the generated Python code using ast.parse.
        If syntax is invalid, it raises ModelRetry to ask the LLM to fix it.
        """
        code_to_validate = output.code.strip()

        # Basic cleanup in case the model includes markdown fences despite instructions
        # This is more likely in PromptedOutput mode but harmless in tool-calling mode.
        if code_to_validate.startswith("```python"):
            code_to_validate = code_to_validate[9:]
        if code_to_validate.startswith("```json"):
            # Model might wrap the JSON in markdown
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
                # It wasn't valid JSON, so just proceed with the text as code
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
             print(
                f"❌ Empty code detected on attempt {ctx.retry + 1}. Asking model to retry."
            )
             raise ModelRetry(
                "The generated code was empty. Please generate the Python code."
            )

        try:
            ast.parse(code_to_validate)
            print("✅ Python code syntax is valid.")
            # Return the validated (and possibly cleaned) code
            output.code = code_to_validate
            return output
        except SyntaxError as e:
            # This is the agentic part: if validation fails, we provide feedback
            # and ask the model to try again.
            print(
                f"❌ Invalid syntax detected on attempt {ctx.retry + 1}. Asking model to retry."
            )
            print(f"   Error: {e}")
            
            error_feedback = f"The generated Python code has a syntax error: {e}. " \
                             "Review your previous output, fix the specific syntax error, and try again."
                             
            if not model_name.startswith("openai:"):
                 error_feedback += " Ensure you ONLY output the raw JSON object with the 'code' key."

            raise ModelRetry(error_feedback)

    return agent


def main():
    """
    Main function to handle command-line arguments and file operations.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Matlab source file to Python using an agentic process with Pydantic AI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="The path to the input Matlab (.m) file.")
    parser.add_argument(
        "output_file", help="The path to the output Python (.py) file."
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

    args = parser.parse_args()

    # 1. Read the input Matlab file
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            matlab_code = f.read()
        print(f"✅ Successfully read Matlab code from '{args.input_file}'.")
    except FileNotFoundError:
        print(
            f"❌ Error: Input file not found at '{args.input_file}'", file=sys.stderr
        )
        sys.exit(1)
    except IOError as e:
        print(f"❌ Error reading file '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Perform the conversion using the agentic process
    print("\n⏳ Starting agentic conversion process...")
    converter_agent = create_agent(args.url, args.model)

    try:
        # Pydantic AI's run_sync handles the async event loop and the retry logic
        result = converter_agent.run_sync(matlab_code)
        python_code = result.output.code
    except ModelRetry as e:
        print(
            f"❌ Conversion failed after multiple retries. Last error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"❌ An unexpected error occurred during conversion: {e}", file=sys.stderr
        )
        sys.exit(1)

    if not python_code:
        print("🛑 Conversion failed to produce code. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("🎉 Conversion successful and code validated.")

    # 3. Write the output Python file
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(python_code)
        print(f"✅ Successfully saved Python code to '{args.output_file}'.")
    except IOError as e:
        print(f"❌ Error writing to file '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


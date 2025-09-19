import argparse
import sys
import json
import requests
import re

# --- Configuration ---
# The default URL for the Ollama API.
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
# The default model to use for the conversion.
DEFAULT_MODEL = "qwen:0.5b"
# The system prompt that instructs the LLM on its task.
# This is crucial for getting good, clean output.
SYSTEM_PROMPT = """You are an expert programmer specializing in migrating Matlab code to Python. 
Your task is to convert the given Matlab code to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.
Do not include any explanations, markdown formatting, or introductory text in your response.
Only provide the raw Python code. Do not wrap the code in backticks or any other formatting.
"""

def parse_python_code(response_text: str) -> str:
    """
    Extracts raw Python code from a formatted response that may contain
    explanatory text, markdown formatting, backticks, etc.
    
    Args:
        response_text: The raw response from the LLM that may contain formatting
        
    Returns:
        Clean Python code with formatting and explanations removed
    """
    # First, try to extract code from markdown code blocks
    code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if code_blocks:
        # If we found code blocks, use the first one (assuming it's the main code)
        code = code_blocks[0].strip()
    else:
        # If no code blocks found, try to extract lines that look like Python code
        lines = response_text.split('\n')
        code_lines = []
        in_code_section = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines and obvious explanatory text
            if not stripped_line:
                continue
            if stripped_line.startswith(('Here\'s', 'This function', 'You can call', 'The function')):
                continue
            if stripped_line.startswith(('```', '**', '#', 'Note:', 'Example:')):
                continue
                
            # Check if line looks like Python code
            if (stripped_line.startswith(('def ', 'class ', 'import ', 'from ')) or
                re.match(r'^\s*(if|for|while|try|except|with|return|print|#)', stripped_line) or
                '=' in stripped_line and not stripped_line.endswith(':') or
                stripped_line.endswith((':')) or
                re.match(r'^\s+', line)):  # Indented lines are likely code
                code_lines.append(line)
                in_code_section = True
            elif in_code_section and not stripped_line:
                # Add empty lines within code sections
                code_lines.append('')
        
        code = '\n'.join(code_lines).strip()
    
    # Clean up common formatting issues
    code = re.sub(r'^Here\'s.*?:\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
    
    # Fix common syntax errors in the converted code
    # Fix malformed for loops like "for i in range(1,n+1)):"
    code = re.sub(r'range\([^)]+\)\)\s*:', lambda m: m.group(0).replace('))', ')'), code)
    
    # Remove duplicate return statements or malformed syntax
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Fix lines with semicolon-separated statements that should be on separate lines
        if '; ' in line and not line.strip().startswith('#'):
            parts = line.split('; ')
            base_indent = len(line) - len(line.lstrip())
            for i, part in enumerate(parts):
                if part.strip():
                    if i == 0:
                        cleaned_lines.append(part)
                    else:
                        cleaned_lines.append(' ' * base_indent + part)
        else:
            cleaned_lines.append(line)
    
    code = '\n'.join(cleaned_lines)
    
    # Remove any trailing explanatory text
    code = re.sub(r'\n\s*This function.*$', '', code, flags=re.DOTALL)
    code = re.sub(r'\n\s*You can call.*$', '', code, flags=re.DOTALL)
    
    return code.strip()

def convert_matlab_to_python(matlab_code: str, ollama_url: str, model: str) -> str | None:
    """
    Sends the Matlab code to the Ollama LLM and returns the Python conversion.

    Args:
        matlab_code: A string containing the Matlab source code.
        ollama_url: The URL of the Ollama '/api/generate' endpoint.
        model: The name of the model to use (e.g., 'qwen:0.5b').

    Returns:
        A string containing the converted Python code, or None if an error occurred.
    """
    print(f"🤖 Contacting Ollama at {ollama_url} using model '{model}'...")

    # Construct the payload for the Ollama API based on the documentation.
    # We use stream=False to get the full response in a single JSON object.
    payload = {
        "model": model,
        "prompt": matlab_code,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1, # Low temperature for more deterministic, less "creative" code
            "num_predict": 4096 # Max tokens to generate
        }
    }

    try:
        # Make the POST request to the Ollama API
        response = requests.post(ollama_url, json=payload, timeout=300) # 5-minute timeout
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        response_data = response.json()
        
        # Extract the converted code from the 'response' key
        raw_response = response_data.get("response", "").strip()
        
        if not raw_response:
            print("❌ Error: Received an empty response from the model.", file=sys.stderr)
            return None

        # Parse the response to extract clean Python code
        python_code = parse_python_code(raw_response)
        
        if not python_code:
            print("❌ Error: Could not extract valid Python code from the response.", file=sys.stderr)
            print(f"Raw response was: {raw_response[:200]}...", file=sys.stderr)
            return None

        return python_code

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ollama: {e}", file=sys.stderr)
        print("Please ensure Ollama is running and the URL is correct.", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print("❌ Error: Failed to decode the JSON response from Ollama.", file=sys.stderr)
        return None


def main():
    """
    Main function to handle command-line arguments and file operations.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Matlab source file to Python using an Ollama LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="The path to the input Matlab (.m) file.")
    parser.add_argument("output_file", help="The path to the output Python (.py) file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"The Ollama model to use for conversion (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help=f"The URL of the Ollama API endpoint (default: {DEFAULT_OLLAMA_URL})."
    )

    args = parser.parse_args()

    # 1. Read the input Matlab file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            matlab_code = f.read()
        print(f"✅ Successfully read Matlab code from '{args.input_file}'.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"❌ Error reading file '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Perform the conversion using the LLM
    print("⏳ Starting conversion process...")
    python_code = convert_matlab_to_python(matlab_code, args.url, args.model)

    if not python_code:
        print("🛑 Conversion failed. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    print("✅ Conversion successful.")

    # 3. Write the output Python file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(python_code)
        print(f"🎉 Successfully saved Python code to '{args.output_file}'.")
    except IOError as e:
        print(f"❌ Error writing to file '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
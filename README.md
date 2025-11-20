# MATLAB to Python Converter

An agentic code conversion tool that converts MATLAB code to Python using Pydantic AI.

## Features

- **Single File Conversion**: Convert individual MATLAB files to Python
- **Batch Multi-File Conversion**: Convert multiple MATLAB files in a single operation with full context
- **Automatic Dependency Handling**: Agent receives complete content of all files and generates proper imports
- **Syntax Validation**: Automatically validates generated Python code and retries on errors
- **Flexible Model Support**: Works with both OpenAI models and local Ollama models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single File Conversion

Convert a single MATLAB file to Python:

```bash
python matlab_to_python_converter.py input.m --output output.py
```

If `--output` is not specified, the output file will be named `<input>_converted.py`.

### Multi-File Batch Conversion

Convert multiple MATLAB files in a **single batch operation**:

```bash
python matlab_to_python_converter.py file1.m file2.m file3.m --output converted/
```

This will:
- Read ALL files completely (full content, not previews)
- Send all files to the agent in ONE request
- Agent converts all files together with full awareness of dependencies
- Automatically generates proper Python imports
- Creates the `converted/` directory with all converted files

**Key Advantage**: The agent receives the COMPLETE content of ALL files in a single operation, allowing it to:
- Understand the full dependency graph
- Automatically generate correct imports between files
- Maintain consistent naming and style across all files
- Make intelligent decisions about code structure

### Options

- `--model`: Specify the model to use (default: `qwen:0.5b`)
  - For OpenAI models: `--model openai:gpt-4o`
  - For Ollama models: `--model qwen:0.5b`
- `--url`: Ollama API endpoint URL (default: `http://localhost:11434/v1`)

### Examples

**Single file with OpenAI:**
```bash
python matlab_to_python_converter.py calculate_sum.m --output result.py --model openai:gpt-4o
```

**Multiple files with local Ollama:**
```bash
python matlab_to_python_converter.py example_script.m example_function.m --output converted/ --model qwen:0.5b
```

**Multiple files with default settings (creates 'converted/' directory):**
```bash
python matlab_to_python_converter.py example_script.m example_function.m
```

## How It Works

### Single File Mode
1. **Read** the MATLAB file
2. **Send** to agent for conversion
3. **Validate** Python syntax
4. **Write** output file

### Multi-File Batch Mode
1. **Read ALL** MATLAB files completely
2. **Send ALL files** to agent in a single request with full context
3. **Agent converts ALL files together** understanding dependencies
4. **Validate** all generated Python files
5. **Write ALL** output files

## Requirements

- Python 3.8+
- pydantic-ai
- python-dotenv
- OpenAI API key (for OpenAI models) or Ollama running locally

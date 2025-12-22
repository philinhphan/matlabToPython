"""
Code Converter Tool

LLM-based MATLAB to Python code conversion.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


# Default configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "openai:gpt-4o-mini"


# System prompts
SYSTEM_PROMPT_SINGLE = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert the given Matlab code to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.

Return a dictionary with a "files" key containing the converted files where:
- Keys are the output Python filenames (e.g., "example_script.py")
- Values are the complete Python code for each file

Do NOT include test files - those will be generated separately.
"""

SYSTEM_PROMPT_BATCH = """You are an expert programmer specializing in migrating Matlab code to Python.
Your task is to convert ALL the given Matlab files to clean, efficient, and idiomatic Python code.
Use common Python libraries like NumPy for matrix operations and Matplotlib for plotting.

You are converting multiple files from the same codebase in a SINGLE operation. Pay attention to:
- Function calls between files (convert to proper Python imports)
- Shared variables and data structures
- Maintaining consistent naming conventions across files

Return a dictionary with a "files" key containing all converted files where:
- Keys are the output Python filenames (e.g., "example_script.py")
- Values are the complete Python code for each file

Do NOT include test files - those will be generated separately.
"""


class PythonFiles(BaseModel):
    """Output model for converted Python files."""
    files: Dict[str, str]


def create_conversion_agent(
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    is_batch: bool = False
) -> Agent:
    """
    Create an LLM agent configured for code conversion.
    
    Args:
        model_name: Model to use (e.g., 'openai:gpt-4o' or 'qwen:0.5b')
        ollama_url: URL for Ollama API (used if not OpenAI)
        is_batch: Whether this is batch conversion (affects prompt)
        
    Returns:
        Configured Pydantic AI Agent
    """
    if model_name.startswith("openai:"):
        model_name_clean = model_name.split(":", 1)[-1]
        model = OpenAIChatModel(model_name=model_name_clean)
    else:
        ollama_provider = OllamaProvider(base_url=ollama_url)
        model = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
    
    system_prompt = SYSTEM_PROMPT_BATCH if is_batch else SYSTEM_PROMPT_SINGLE
    
    return Agent(
        model=model,
        output_type=PythonFiles,
        instructions=system_prompt,
        retries=3,
    )


def build_conversion_prompt(files_content: Dict[str, str]) -> str:
    """
    Build a conversion prompt from file contents.
    
    Args:
        files_content: Dictionary mapping filename to content
        
    Returns:
        Formatted prompt string
    """
    if len(files_content) == 1:
        filename, content = next(iter(files_content.items()))
        return f"Convert this MATLAB file ({filename}) to Python:\n\n{content}"
    
    prompt_parts = [
        "Convert the following MATLAB files to Python.",
        "All files are from the same codebase - handle dependencies appropriately.\n"
    ]
    
    for filename, content in files_content.items():
        prompt_parts.append(f"\n{'='*60}")
        prompt_parts.append(f"FILE: {filename}")
        prompt_parts.append('='*60)
        prompt_parts.append(content)
    
    return "\n".join(prompt_parts)


def convert_matlab_to_python(
    files_content: Dict[str, str],
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, str]:
    """
    Convert MATLAB code to Python using LLM.
    
    Args:
        files_content: Dictionary mapping MATLAB filename to content
        model_name: Model to use for conversion
        ollama_url: Ollama API URL (if using Ollama)
        
    Returns:
        Dictionary mapping Python filename to converted code
    """
    is_batch = len(files_content) > 1
    agent = create_conversion_agent(model_name, ollama_url, is_batch)
    prompt = build_conversion_prompt(files_content)
    
    result = agent.run_sync(prompt)
    return result.output.files


def convert_matlab_to_python_tool(
    file_paths: Optional[List[str]] = None,
    file_contents: Optional[Dict[str, str]] = None,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict:
    """
    Tool interface for MATLAB to Python conversion.
    
    Either file_paths or file_contents must be provided.
    
    Args:
        file_paths: List of MATLAB file paths to convert
        file_contents: Dictionary mapping filename to content (if already read)
        model_name: Model to use for conversion
        ollama_url: Ollama API URL
        
    Returns:
        Dictionary with conversion results
    """
    # Get file contents
    if file_contents is None:
        if file_paths is None:
            return {
                "success": False,
                "error": "Either file_paths or file_contents must be provided",
                "files": {}
            }
        
        file_contents = {}
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    filename = Path(path).name
                    file_contents[filename] = f.read()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read {path}: {e}",
                    "files": {}
                }
    
    try:
        converted = convert_matlab_to_python(file_contents, model_name, ollama_url)
        return {
            "success": True,
            "files": converted,
            "input_files": list(file_contents.keys()),
            "output_files": list(converted.keys())
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files": {}
        }

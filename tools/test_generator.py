"""
Test Generator Tool

Generates pytest test files for converted Python code.
"""

from typing import Dict, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


DEFAULT_MODEL = "openai:gpt-4o"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


SYSTEM_PROMPT = """You are an expert Python test engineer.
Your task is to generate comprehensive pytest test files for the given Python code.

For each Python file provided, create a corresponding test file that:
- Tests all public functions and classes
- Includes edge cases and error conditions
- Uses appropriate pytest features (fixtures, parametrize, etc.)
- Has clear, descriptive test names

Return a dictionary with a "files" key containing the test files where:
- Keys are test filenames (e.g., "test_example.py")
- Values are the complete test code

Follow pytest best practices and include necessary imports.
"""


class TestFiles(BaseModel):
    """Output model for generated test files."""
    files: Dict[str, str]


def create_test_agent(
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Agent:
    """Create an LLM agent for test generation."""
    if model_name.startswith("openai:"):
        model_name_clean = model_name.split(":", 1)[-1]
        model = OpenAIChatModel(model_name=model_name_clean)
    else:
        ollama_provider = OllamaProvider(base_url=ollama_url)
        model = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
    
    return Agent(
        model=model,
        output_type=TestFiles,
        instructions=SYSTEM_PROMPT,
        retries=3,
    )


def build_test_prompt(files_content: Dict[str, str]) -> str:
    """Build a prompt for test generation from Python files."""
    prompt_parts = ["Generate pytest tests for the following Python files:\n"]
    
    for filename, content in files_content.items():
        # Skip existing test files
        if filename.startswith("test_"):
            continue
        prompt_parts.append(f"\n{'='*60}")
        prompt_parts.append(f"FILE: {filename}")
        prompt_parts.append('='*60)
        prompt_parts.append(content)
    
    return "\n".join(prompt_parts)


def generate_tests(
    files_content: Dict[str, str],
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, str]:
    """
    Generate pytest test files for Python code.
    
    Args:
        files_content: Dictionary mapping Python filename to content
        model_name: Model to use for generation
        ollama_url: Ollama API URL
        
    Returns:
        Dictionary mapping test filename to test code
    """
    # Filter out existing test files
    source_files = {k: v for k, v in files_content.items() if not k.startswith("test_")}
    
    if not source_files:
        return {}
    
    agent = create_test_agent(model_name, ollama_url)
    prompt = build_test_prompt(source_files)
    
    result = agent.run_sync(prompt)
    return result.output.files


def generate_tests_tool(
    files_content: Dict[str, str],
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict:
    """
    Tool interface for test generation.
    
    Args:
        files_content: Dictionary mapping Python filename to content
        model_name: Model to use
        ollama_url: Ollama API URL
        
    Returns:
        Dictionary with generation results
    """
    try:
        test_files = generate_tests(files_content, model_name, ollama_url)
        return {
            "success": True,
            "files": test_files,
            "test_count": len(test_files)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files": {}
        }

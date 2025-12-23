"""
Test Generator Tool

Generates pytest test files for converted Python code.
Uses direct JSON parsing as fallback for models that struggle with structured output.
"""

import json
import re
from typing import Dict, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


# System prompt optimized for JSON output
SYSTEM_PROMPT = """You are an expert Python test engineer.
Generate pytest tests for the provided Python code.

Create test files that:
- Test all public functions
- Include edge cases
- Use pytest features appropriately
- Have descriptive test names

CRITICAL: You must respond with ONLY a valid JSON object in this exact format:
{"files": {"test_filename.py": "test code here"}}

No markdown, no explanations, ONLY the JSON object.
"""


class TestFiles(BaseModel):
    """Output model for generated test files."""
    files: Dict[str, str] = Field(description="Dictionary mapping test filename to test code")


def create_test_agent(
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Agent:
    """Create an LLM agent for test generation with string output."""
    if model_name.startswith("openai:"):
        model_name_clean = model_name.split(":", 1)[-1]
        model = OpenAIChatModel(model_name=model_name_clean)
    else:
        ollama_provider = OllamaProvider(base_url=ollama_url)
        model = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
    
    # Use string output and parse JSON manually for reliability
    return Agent(
        model=model,
        output_type=str,
        instructions=SYSTEM_PROMPT,
        retries=3,
    )


def extract_json_from_response(response: str) -> Dict[str, str]:
    """Extract JSON from LLM response, handling common issues."""
    # Clean the response
    text = response.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to find JSON object in text
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "files" in data:
            return data["files"]
        return data
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text using regex
    json_match = re.search(r'\{[^{}]*"files"\s*:\s*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("files", {})
        except json.JSONDecodeError:
            pass
    
    # Last resort: try to find any JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "files" in data:
                return data["files"]
            return data
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def build_test_prompt(files_content: Dict[str, str]) -> str:
    """Build a prompt for test generation from Python files."""
    prompt_parts = ["Generate pytest tests for this Python code:\n"]
    
    for filename, content in files_content.items():
        if filename.startswith("test_"):
            continue
        prompt_parts.append(f"\n# {filename}\n{content}")
    
    prompt_parts.append("\n\nRespond with ONLY JSON: {\"files\": {\"test_filename.py\": \"code\"}}")
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
    
    # Parse the JSON response manually
    return extract_json_from_response(result.output)


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

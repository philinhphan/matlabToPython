#!/bin/bash
# Test script for multi-file conversion

echo "Testing multi-file MATLAB to Python conversion"
echo "=============================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Test 1: Single file conversion (backward compatibility)
echo "Test 1: Single file conversion"
echo "Command: python matlab_to_python_converter.py example_function.m --output test_single.py"
echo ""

# Test 2: Multi-file conversion with dependencies
echo "Test 2: Multi-file conversion with dependencies"
echo "Command: python matlab_to_python_converter.py example_script.m example_function.m --output test_output/"
echo ""
echo "This will convert both files with context about their relationship."
echo "The agent will see that example_script.m calls add_numbers() from example_function.m"
echo "and should generate appropriate Python imports."
echo ""

# Test 3: Multi-file with default output directory
echo "Test 3: Multi-file with default output (creates 'converted/' directory)"
echo "Command: python matlab_to_python_converter.py example_script.m example_function.m"
echo ""

echo "To run these tests, uncomment the commands below and ensure you have:"
echo "  - An LLM running (Ollama or OpenAI API key)"
echo "  - The required Python packages installed"
echo ""

# Uncomment to actually run the tests:
# python matlab_to_python_converter.py example_function.m --output test_single.py
# python matlab_to_python_converter.py example_script.m example_function.m --output test_output/

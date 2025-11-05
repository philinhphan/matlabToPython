# Multi-File Conversion Example Workflow

This document demonstrates a complete workflow for converting multiple related MATLAB files to Python.

## Scenario

You have a MATLAB project with:
- `example_function.m` - A utility function that adds two numbers
- `example_script.m` - A script that uses the utility function

## Step 1: Review Your MATLAB Files

**example_function.m:**
```matlab
function result = add_numbers(a, b)
    % ADD_NUMBERS Add two numbers together
    result = a + b;
end
```

**example_script.m:**
```matlab
% Example script that uses add_numbers function
x = 5;
y = 10;
sum_result = add_numbers(x, y);
fprintf('The sum of %d and %d is %d\n', x, y, sum_result);
```

Notice that `example_script.m` calls `add_numbers()` which is defined in `example_function.m`.

## Step 2: Run Multi-File Conversion

Convert both files together so the agent understands their relationship:

```bash
python matlab_to_python_converter.py example_script.m example_function.m --output converted/
```

### What Happens:

1. **Context Building**: The converter reads both files and builds context
2. **Agent Awareness**: The agent sees that `add_numbers` is defined in one file and called in another
3. **Conversion**: Each file is converted with awareness of the full codebase
4. **Output**: Two Python files are created in the `converted/` directory

## Step 3: Review Generated Python Code

**Expected output in converted/example_function.py:**
```python
def add_numbers(a, b):
    """
    ADD_NUMBERS Add two numbers together
    """
    result = a + b
    return result
```

**Expected output in converted/example_script.py:**
```python
from example_function import add_numbers

# Example script that uses add_numbers function
x = 5
y = 10
sum_result = add_numbers(x, y)
print(f'The sum of {x} and {y} is {sum_result}')
```

Notice how the agent:
- Converted the MATLAB function to a Python function
- Added proper `import` statement in the script
- Converted `fprintf` to Python's `print` with f-string
- Maintained the relationship between the files

## Step 4: Test the Converted Code

```bash
cd converted/
python example_script.py
```

Expected output:
```
The sum of 5 and 10 is 15
```

## Alternative: Single File Conversion (Not Recommended for Related Files)

If you convert files separately:

```bash
python matlab_to_python_converter.py example_function.m --output func.py
python matlab_to_python_converter.py example_script.m --output script.py
```

The agent won't know about the relationship between files, and you may need to:
- Manually add import statements
- Adjust function names
- Fix module paths

## Using Different Models

### With OpenAI GPT-4:
```bash
export OPENAI_API_KEY="your-api-key"
python matlab_to_python_converter.py example_script.m example_function.m \
    --output converted/ \
    --model openai:gpt-4o
```

### With Local Ollama (CodeLlama):
```bash
python matlab_to_python_converter.py example_script.m example_function.m \
    --output converted/ \
    --model codellama:latest
```

## Tips for Success

1. **Convert related files together** - Always convert files that depend on each other in the same batch
2. **Check the output** - Review generated imports and function calls
3. **Test incrementally** - Test each converted module to ensure it works
4. **Use version control** - Keep your original MATLAB files and track changes to Python files

## Common Patterns

### Pattern 1: Utility Functions + Main Script
```bash
python matlab_to_python_converter.py utils.m main.m --output src/
```

### Pattern 2: Multiple Related Scripts
```bash
python matlab_to_python_converter.py data_loader.m processor.m visualizer.m --output pipeline/
```

### Pattern 3: Converting an Entire Directory
```bash
python matlab_to_python_converter.py matlab_project/*.m --output python_project/
```

## Troubleshooting

### Issue: Import not generated
- **Cause**: Files converted separately
- **Solution**: Re-convert all related files together

### Issue: Function name mismatch
- **Cause**: Inconsistent conversion
- **Solution**: Use a more capable model (e.g., GPT-4) or manually adjust

### Issue: Circular dependencies
- **Cause**: MATLAB files have circular references
- **Solution**: Refactor code structure or manually resolve in Python

# Batch Multi-File Conversion - Implementation Summary

## What Changed

The MATLAB to Python converter has been **completely redesigned** to support **true batch conversion** where multiple files are converted in a **single operation** with **full context**.

## Key Changes

### Previous Approach (Preview-Based, Individual)
- ❌ Only first 500 characters of each file provided as context
- ❌ Each file converted individually in separate operations
- ❌ Limited awareness of dependencies between files
- ❌ Imports often needed manual fixing

### New Approach (Full-Content, Batch)
- ✅ **Complete content** of ALL files provided to agent
- ✅ **All files converted together** in a SINGLE operation
- ✅ **Full awareness** of entire codebase and dependencies
- ✅ **Automatic import generation** - no manual fixing needed

## Architecture Changes

### 1. New Data Models

```python
class PythonFiles(BaseModel):
    """Model for multi-file batch conversion output."""
    files: Dict[str, str]  # filename -> Python code
```

### 2. New Functions

```python
def read_all_files(file_paths: List[str]) -> Dict[str, str]:
    """Read all MATLAB files completely."""
    
def build_batch_conversion_prompt(files_content: Dict[str, str]) -> str:
    """Build prompt with ALL files for batch conversion."""
    
def convert_multiple_files(files_content: Dict[str, str], agent: Agent) -> Dict[str, str]:
    """Convert ALL files in a single batch operation."""
```

### 3. Updated System Prompts

New prompts specifically designed for batch conversion:
- `SYSTEM_PROMPT_MULTIFILE_JSON` - For prompted JSON output
- `SYSTEM_PROMPT_MULTIFILE_TOOLS` - For tool-calling models

Both prompts instruct the agent to:
- Convert ALL files in a single operation
- Return a dictionary with all converted files
- Generate proper imports between files

### 4. Enhanced Validation

```python
@agent.output_validator
def validate_python_files(ctx: Any, output: PythonFiles) -> PythonFiles:
    """Validate ALL generated files. If any fail, regenerate ALL."""
```

### 5. Rewritten Main Function

Complete rewrite to support:
- Batch reading of all files
- Single conversion operation for multiple files
- Batch writing of all output files
- Clear distinction between single and batch modes

## How Batch Conversion Works

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH CONVERSION FLOW                     │
└─────────────────────────────────────────────────────────────┘

1. READ ALL FILES
   ┌──────────────────────────────────────────────────────┐
   │ file1.m: function result = add(a,b)                  │
   │          result = a + b;                             │
   │          end                                         │
   │                                                      │
   │ file2.m: x = 5; y = 10;                             │
   │          sum = add(x, y);                           │
   │          fprintf('Sum: %d\n', sum);                 │
   └──────────────────────────────────────────────────────┘

2. BUILD SINGLE PROMPT
   ┌──────────────────────────────────────────────────────┐
   │ Convert the following MATLAB files to Python.        │
   │ All files are from the same codebase.               │
   │                                                      │
   │ ========================================             │
   │ FILE: file1.m                                       │
   │ ========================================             │
   │ function result = add(a,b)                          │
   │ result = a + b;                                     │
   │ end                                                 │
   │                                                      │
   │ ========================================             │
   │ FILE: file2.m                                       │
   │ ========================================             │
   │ x = 5; y = 10;                                      │
   │ sum = add(x, y);                                    │
   │ fprintf('Sum: %d\n', sum);                          │
   └──────────────────────────────────────────────────────┘

3. AGENT CONVERTS ALL FILES TOGETHER
   ┌──────────────────────────────────────────────────────┐
   │ Agent analyzes:                                      │
   │ - file2.m calls add() which is in file1.m          │
   │ - Need to import add from file1 in file2           │
   │ - Convert fprintf to print                          │
   │                                                      │
   │ Agent returns:                                       │
   │ {                                                    │
   │   "file1.py": "def add(a, b):\n    return a + b",  │
   │   "file2.py": "from file1 import add\n\nx = 5..."  │
   │ }                                                    │
   └──────────────────────────────────────────────────────┘

4. VALIDATE ALL FILES
   ┌──────────────────────────────────────────────────────┐
   │ For each file:                                       │
   │   ✅ file1.py: Valid Python syntax                  │
   │   ✅ file2.py: Valid Python syntax                  │
   │                                                      │
   │ All files valid ✅                                   │
   └──────────────────────────────────────────────────────┘

5. WRITE ALL FILES
   ┌──────────────────────────────────────────────────────┐
   │ converted/file1.py ✅                                │
   │ converted/file2.py ✅                                │
   └──────────────────────────────────────────────────────┘
```

## Code Example

### Input Command
```bash
python matlab_to_python_converter.py example_function.m example_script.m
```

### What Happens Internally

```python
# 1. Read all files
files_content = {
    "example_function.m": "function result = add_numbers(a, b)\n    result = a + b;\nend",
    "example_script.m": "x = 5;\ny = 10;\nsum = add_numbers(x, y);\nfprintf('Sum: %d\\n', sum);"
}

# 2. Build batch prompt
prompt = """
Convert the following MATLAB files to Python...

============================================================
FILE: example_function.m
============================================================
function result = add_numbers(a, b)
    result = a + b;
end

============================================================
FILE: example_script.m
============================================================
x = 5;
y = 10;
sum = add_numbers(x, y);
fprintf('Sum: %d\n', sum);
"""

# 3. Send to agent (ONE request)
result = agent.run_sync(prompt)

# 4. Agent returns ALL files
converted = {
    "example_function.py": "def add_numbers(a, b):\n    return a + b",
    "example_script.py": "from example_function import add_numbers\n\nx = 5\ny = 10\nsum_result = add_numbers(x, y)\nprint(f'Sum: {sum_result}')"
}

# 5. Validate all files
for filename, code in converted.items():
    ast.parse(code)  # Validate syntax

# 6. Write all files
for filename, code in converted.items():
    with open(f"converted/{filename}", "w") as f:
        f.write(code)
```

## Benefits

### 1. Automatic Import Generation
The agent sees all files and automatically generates correct imports:
```python
# example_script.py
from example_function import add_numbers  # ← Automatically generated!
```

### 2. Full Context Understanding
Agent receives complete content of all files, not just previews:
- Understands full function signatures
- Sees all variable usage
- Recognizes all dependencies

### 3. Consistent Conversion
All files converted with same context and style:
- Consistent naming conventions
- Consistent code structure
- Consistent library usage

### 4. Single Operation
One API call instead of multiple:
- Faster execution
- Lower cost
- Better coherence

### 5. Better Results
Agent makes more intelligent decisions:
- Proper module structure
- Correct import paths
- Appropriate code organization

## Usage Examples

### Example 1: Simple Function and Script
```bash
python matlab_to_python_converter.py add.m script.m
```
Agent sees both files, generates import in script.py

### Example 2: Multiple Utilities
```bash
python matlab_to_python_converter.py utils1.m utils2.m utils3.m main.m
```
Agent understands all utilities, generates all necessary imports in main.py

### Example 3: Entire Project
```bash
python matlab_to_python_converter.py *.m --output python_project/
```
Complete codebase converted in one batch operation

## Validation Strategy

### Multi-File Validation
```python
def validate_python_files(ctx, output):
    errors = []
    for filename, code in output.files.items():
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"{filename}: {e}")
    
    if errors:
        # Ask agent to regenerate ALL files
        raise ModelRetry("Fix these errors and regenerate ALL files")
```

If ANY file has errors, ALL files are regenerated. This ensures consistency.

## Model Support

### OpenAI Models (Tool Calling)
```python
effective_output_type = PythonFiles  # Direct structured output
```

### Ollama Models (Prompted JSON)
```python
effective_output_type = PromptedOutput(PythonFiles)  # JSON wrapper
```

Both modes return the same structure: dictionary of filename → code

## Limitations

1. **Context Window**: Very large codebases may exceed model limits
   - Solution: Split into smaller batches
   - Use models with larger context (GPT-4 Turbo: 128K tokens)

2. **Model Capability**: Smaller models may struggle with complex dependencies
   - Solution: Use GPT-4 for complex codebases

3. **Validation**: If one file fails, all must be regenerated
   - This is intentional to maintain consistency

## Migration from Old Approach

### Old Code (Individual Conversion)
```python
for input_file in args.input_files:
    matlab_code = read_file(input_file)
    python_code = convert_single_file(matlab_code, agent, context)
    write_file(output_file, python_code)
```

### New Code (Batch Conversion)
```python
files_content = read_all_files(args.input_files)
converted_files = convert_multiple_files(files_content, agent)
for filename, code in converted_files.items():
    write_file(filename, code)
```

## Testing

### Test with Example Files
```bash
python matlab_to_python_converter.py example_function.m example_script.m
```

### Expected Behavior
1. Reads both files completely
2. Sends both to agent in one request
3. Agent returns both converted files with proper imports
4. Validates both files
5. Writes both to converted/ directory

### Verify Output
```bash
cat converted/example_function.py
# Should contain: def add_numbers(a, b): ...

cat converted/example_script.py
# Should contain: from example_function import add_numbers
```

## Documentation

- **[MULTIFILE_GUIDE.md](MULTIFILE_GUIDE.md)** - Comprehensive user guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet
- **[EXAMPLE_WORKFLOW.md](EXAMPLE_WORKFLOW.md)** - Step-by-step examples
- **[README.md](README.md)** - Main documentation

## Conclusion

The batch conversion approach provides:
- ✅ Better results through full context
- ✅ Automatic import generation
- ✅ Consistent code style
- ✅ Single operation efficiency
- ✅ True dependency awareness

This is a significant improvement over the previous preview-based, individual conversion approach.

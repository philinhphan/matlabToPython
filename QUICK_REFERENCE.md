# Multi-File Batch Conversion - Quick Reference

## Key Concept

**Multiple files are converted in a SINGLE batch operation** where the agent receives ALL files with their COMPLETE content and converts them together, automatically handling dependencies and imports.

## Basic Commands

### Single File
```bash
python matlab_to_python_converter.py input.m --output output.py
```

### Multiple Files (Batch Conversion - Default Output)
```bash
python matlab_to_python_converter.py file1.m file2.m file3.m
# Agent converts ALL files together in ONE operation
# Creates: converted/file1.py, converted/file2.py, converted/file3.py
```

### Multiple Files (Batch Conversion - Custom Output)
```bash
python matlab_to_python_converter.py file1.m file2.m --output my_project/
# Creates: my_project/file1.py, my_project/file2.py
```

### Convert All .m Files
```bash
python matlab_to_python_converter.py *.m
# ALL files converted together in single batch
```

## Model Options

### OpenAI GPT-4 (Recommended for Complex Code)
```bash
export OPENAI_API_KEY="your-key"
python matlab_to_python_converter.py *.m --model openai:gpt-4o
```

### Ollama (Local)
```bash
# Default model (qwen:0.5b)
python matlab_to_python_converter.py *.m

# CodeLlama
python matlab_to_python_converter.py *.m --model codellama:latest

# Llama 2
python matlab_to_python_converter.py *.m --model llama2
```

## How Batch Conversion Works

```
┌─────────────────────────────────────┐
│  Input: file1.m, file2.m, file3.m  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Read ALL files (complete content)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Send ALL files to agent in ONE     │
│  request with full context          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Agent converts ALL files together  │
│  - Understands dependencies         │
│  - Generates imports automatically  │
│  - Maintains consistency            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Returns ALL converted files        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Write all files to output dir      │
└─────────────────────────────────────┘
```

## Common Patterns

### Pattern 1: Function + Script
```bash
# Convert function and script that uses it
python matlab_to_python_converter.py utils.m main.m --output src/
# Agent sees both files and generates proper imports
```

### Pattern 2: Multiple Utilities
```bash
# Convert multiple utility functions together
python matlab_to_python_converter.py helper1.m helper2.m helper3.m
# All converted in single batch with awareness of each other
```

### Pattern 3: Entire Project
```bash
# Convert all MATLAB files in current directory
python matlab_to_python_converter.py *.m --output python_project/
# Complete codebase converted in one operation
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_files` | One or more .m files | Required |
| `--output` | Output file or directory | `<input>_converted.py` (single)<br>`converted/` (multiple) |
| `--model` | Model to use | `qwen:0.5b` |
| `--url` | Ollama API endpoint | `http://localhost:11434/v1` |

## Single vs Batch Conversion

| Aspect | Single File | Multiple Files (Batch) |
|--------|-------------|------------------------|
| Input | 1 file | 2+ files |
| Operation | Individual conversion | Single batch operation |
| Context | No context | Full content of ALL files |
| Imports | Manual | Automatic |
| API Calls | 1 per file | 1 for all files |
| Consistency | N/A | Guaranteed across files |

## Example: Batch Conversion with Dependencies

**Input Files:**
- `add_numbers.m` - Function definition
- `calculate.m` - Script that calls add_numbers

**Command:**
```bash
python matlab_to_python_converter.py add_numbers.m calculate.m
```

**What Happens:**
1. Both files read completely
2. Both sent to agent in ONE request
3. Agent sees `calculate.m` calls `add_numbers()`
4. Agent generates `calculate.py` with `from add_numbers import add_numbers`
5. Both files written to `converted/` directory

**Result:**
- `converted/add_numbers.py` - Python function
- `converted/calculate.py` - Python script with correct import

## Advantages of Batch Conversion

✅ **Automatic Import Generation** - Agent sees all files and generates correct imports  
✅ **Full Context** - Agent receives complete content of all files  
✅ **Consistent Style** - All files converted with same context  
✅ **Dependency Awareness** - Agent understands relationships  
✅ **Single Operation** - One API call for all files  
✅ **Better Results** - More intelligent conversion decisions  

## Troubleshooting

### Issue: "Module not found"
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
```

### Issue: "Connection refused"
```bash
# Solution: Start Ollama
ollama serve
```

### Issue: Large codebase fails
```bash
# Solution: Use model with larger context or split into batches
python matlab_to_python_converter.py batch1/*.m --output out1/
python matlab_to_python_converter.py batch2/*.m --output out2/
```

### Issue: Poor import generation
```bash
# Solution: Use a better model
python matlab_to_python_converter.py *.m --model openai:gpt-4o
```

## Tips

1. **Always convert related files together** - This is the key to good results
2. **Use GPT-4 for complex code** - Better understanding of dependencies
3. **Check output** - Review generated imports and structure
4. **Test converted code** - Ensure correctness
5. **Keep originals** - Don't delete MATLAB files

## Documentation

- **User Guide**: `MULTIFILE_GUIDE.md` - Comprehensive guide
- **Examples**: `EXAMPLE_WORKFLOW.md` - Step-by-step walkthrough
- **Architecture**: `ARCHITECTURE.md` - Technical details
- **Summary**: `IMPLEMENTATION_SUMMARY.md` - Overview

## Help

```bash
python matlab_to_python_converter.py --help
```

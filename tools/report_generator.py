"""
Report Generator Tool

Creates conversion reports summarizing the conversion process.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_report(
    input_files: List[str],
    output_files: Dict[str, str],
    test_files: Dict[str, str],
    validation_results: Dict[str, Any],
    test_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    format: str = "markdown",
) -> str:
    """
    Generate a conversion report.
    
    Args:
        input_files: List of input MATLAB file paths
        output_files: Dictionary mapping output filename to code
        test_files: Dictionary mapping test filename to code
        validation_results: Syntax validation results
        test_results: Test execution results
        output_dir: Directory where files were written
        format: Report format ('markdown' or 'json')
        
    Returns:
        Report content as string
    """
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "files": input_files,
            "count": len(input_files),
        },
        "output": {
            "source_files": list(k for k in output_files.keys() if not k.startswith("test_")),
            "test_files": list(test_files.keys()),
            "total_count": len(output_files) + len(test_files),
        },
        "validation": {
            "syntax_valid": validation_results.get("all_valid", False),
            "files_checked": len(validation_results.get("files", {})),
        },
        "tests": {
            "status": test_results.get("status", "not_run"),
            "passed": test_results.get("success", False),
        },
        "output_dir": output_dir,
    }
    
    if format == "json":
        return json.dumps(report_data, indent=2)
    
    return _generate_markdown_report(report_data, output_files, test_files, test_results)


def _generate_markdown_report(
    data: Dict[str, Any],
    output_files: Dict[str, str],
    test_files: Dict[str, str],
    test_results: Dict[str, Any],
) -> str:
    """Generate a markdown format report."""
    lines = [
        "# MATLAB to Python Conversion Report",
        "",
        f"**Generated:** {data['timestamp']}",
        "",
        "## Summary",
        "",
        f"- **Input Files:** {data['input']['count']}",
        f"- **Output Files:** {len(data['output']['source_files'])}",
        f"- **Test Files:** {len(data['output']['test_files'])}",
        f"- **Syntax Valid:** {'✅ Yes' if data['validation']['syntax_valid'] else '❌ No'}",
        f"- **Tests Passed:** {'✅ Yes' if data['tests']['passed'] else '❌ No'}",
        "",
    ]
    
    # Input files
    lines.extend([
        "## Input Files",
        "",
    ])
    for f in data['input']['files']:
        lines.append(f"- `{f}`")
    lines.append("")
    
    # Output files
    lines.extend([
        "## Generated Files",
        "",
        "### Source Files",
        "",
    ])
    for f in data['output']['source_files']:
        lines.append(f"- `{f}`")
    
    lines.extend([
        "",
        "### Test Files",
        "",
    ])
    for f in data['output']['test_files']:
        lines.append(f"- `{f}`")
    lines.append("")
    
    # Test results
    if test_results.get("stdout"):
        lines.extend([
            "## Test Output",
            "",
            "```",
            test_results.get("stdout", "")[:2000],  # Limit output length
            "```",
            "",
        ])
    
    # Output directory
    if data['output_dir']:
        lines.extend([
            "## Output Location",
            "",
            f"Files written to: `{data['output_dir']}`",
            "",
        ])
    
    return "\n".join(lines)


def generate_report_tool(
    input_files: List[str],
    output_files: Dict[str, str],
    test_files: Dict[str, str],
    validation_results: Dict[str, Any],
    test_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    format: str = "markdown",
    save_to_file: bool = True,
) -> Dict:
    """
    Tool interface for report generation.
    
    Args:
        input_files: List of input MATLAB file paths
        output_files: Dictionary mapping output filename to code
        test_files: Dictionary mapping test filename to code
        validation_results: Syntax validation results
        test_results: Test execution results
        output_dir: Directory where files were written
        format: Report format ('markdown' or 'json')
        save_to_file: Whether to save report to file
        
    Returns:
        Dictionary with report content and metadata
    """
    try:
        report = generate_report(
            input_files=input_files,
            output_files=output_files,
            test_files=test_files,
            validation_results=validation_results,
            test_results=test_results,
            output_dir=output_dir,
            format=format,
        )
        
        report_path = None
        if save_to_file and output_dir:
            ext = ".json" if format == "json" else ".md"
            report_path = Path(output_dir) / f"conversion_report{ext}"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return {
            "success": True,
            "report": report,
            "report_path": str(report_path) if report_path else None,
            "format": format,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

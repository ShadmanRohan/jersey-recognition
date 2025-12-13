"""
Common utilities for analysis scripts.
Consolidates duplicate functions used across multiple analysis scripts.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional


def extract_test_metrics_from_log(log_file: Path) -> Optional[Dict[str, float]]:
    """
    Extract test metrics from training log file.
    
    Args:
        log_file: Path to training log file
        
    Returns:
        Dictionary with metrics: loss, acc_number, acc_tens, acc_ones
        Returns None if metrics not found or file doesn't exist
    """
    if not log_file.exists():
        return None
    
    try:
        content = log_file.read_text()
        
        # Method 1: Look for "Test Metrics:" section (line-by-line parsing)
        if "Test Metrics:" in content:
            lines = content.split('\n')
            metrics = {}
            in_test_section = False
            
            for line in lines:
                if "Test Metrics:" in line:
                    in_test_section = True
                    continue
                
                if in_test_section:
                    if "Loss:" in line:
                        metrics['test_loss'] = float(line.split("Loss:")[1].strip())
                    elif "Acc Number:" in line:
                        metrics['test_acc_number'] = float(line.split("Acc Number:")[1].strip())
                    elif "Acc Tens:" in line:
                        metrics['test_acc_tens'] = float(line.split("Acc Tens:")[1].strip())
                    elif "Acc Ones:" in line:
                        metrics['test_acc_ones'] = float(line.split("Acc Ones:")[1].strip())
                        break  # Last metric (or continue if Acc Full exists)
                    elif "Acc Full:" in line:
                        metrics['test_acc_full'] = float(line.split("Acc Full:")[1].strip())
                        break
            
            # Return if we got the essential metrics
            if len(metrics) >= 4:  # At least loss, acc_number, acc_tens, acc_ones
                return metrics
        
        # Method 2: Regex fallback (more flexible)
        patterns = {
            'test_loss': r'Loss:\s+([\d.]+)',
            'test_acc_number': r'Acc Number:\s+([\d.]+)',
            'test_acc_tens': r'Acc Tens:\s+([\d.]+)',
            'test_acc_ones': r'Acc Ones:\s+([\d.]+)',
            'test_acc_full': r'Acc Full:\s+([\d.]+)'
        }
        
        metrics = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))
        
        return metrics if len(metrics) >= 4 else None
        
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def load_json_results(json_file: Path) -> Optional[dict]:
    """Load results from JSON file."""
    if not json_file.exists():
        return None
    
    try:
        with open(json_file) as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, list):
                return {"results": data}
            elif isinstance(data, dict) and "evaluation_results" in data:
                return data
            elif isinstance(data, dict):
                return data
            return None
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None


def format_metrics_table(results: list, headers: list = None) -> str:
    """
    Format results as a table string.
    
    Args:
        results: List of dictionaries with metric values
        headers: Optional list of header names (auto-detected if None)
        
    Returns:
        Formatted table string
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback if tabulate not available
        return format_metrics_table_simple(results, headers)
    
    if not results:
        return "No results to display"
    
    if headers is None:
        headers = list(results[0].keys())
    
    table_data = [[r.get(h, 'N/A') for h in headers] for r in results]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def format_metrics_table_simple(results: list, headers: list = None) -> str:
    """Simple table formatter without tabulate dependency."""
    if not results:
        return "No results to display"
    
    if headers is None:
        headers = list(results[0].keys())
    
    # Calculate column widths
    col_widths = {h: max(len(str(h)), max(len(str(r.get(h, ''))) for r in results)) for h in headers}
    
    # Build table
    lines = []
    # Header
    header_line = " | ".join(str(h).ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    # Rows
    for r in results:
        row_line = " | ".join(str(r.get(h, 'N/A')).ljust(col_widths[h]) for h in headers)
        lines.append(row_line)
    
    return "\n".join(lines)


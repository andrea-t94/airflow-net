# Airflow DAG Mining Script

Simple tool for extracting example DAGs from Apache Airflow repositories to create ML training datasets.

## Overview

Mines DAGs from official Apache Airflow repositories across versions, validates syntax, and outputs structured data for training ML models.

## Usage

### Basic Usage
```bash
python airflow_dag_miner.py --versions 2.7.2,2.8.4,2.9.3,3.0.0,3.0.1,3.0.6 --output dags_dataset.jsonl
```

### Quick Start
```bash
python run_dag_mining_example.py
```

## Output

### Dataset Format
Each line contains a DAG record:
```json
{
  "metadata": {
    "syntax_valid": true,
    "airflow_version": "3.0.6",
    "is_multifile": false,
    "line_count": 85,
    "operators": ["BashOperator", "PythonOperator"],
    "file_name": "example_bash_operator.py",
    "syntax_error": null
  },
  "content": "# Full DAG source code...",
  "extracted_at": "2024-11-05T10:30:00.123456"
}
```

### Summary Statistics
Generated alongside dataset:
```json
{
  "total_dags": 2514,
  "syntax_valid_count": 2200,
  "multifile_count": 149,
  "versions": {"2.7.2": 420, "3.0.6": 460, ...},
  "operators": {"@task": 129, "BashOperator": 54, ...},
  "length_distribution": {"short": 98, "medium": 237, "long": 125}
}
```

## Features

- **Multi-version Support**: Works with Airflow 2.x and 3.x
- **GitHub API**: Efficient file discovery and download
- **Syntax Validation**: Python AST parsing + compilation
- **Multi-file Detection**: Identifies DAGs spanning multiple files
- **Provider Examples**: Includes both core and provider examples (~460 DAGs per version)
- **Parallel Processing**: Multi-threaded downloads

## Requirements

- Python 3.8+
- Internet connection
- No external dependencies (uses stdlib only)

## Architecture

### DAG Discovery
- Uses GitHub Tree API to find all `example_*.py` files
- Validates files contain DAG patterns: `DAG(`, `@dag(`, etc.
- Removes license headers while preserving functional code

### Multi-file Detection
- Analyzes import patterns after last `airflow.*` import
- Tracks internal dependencies across repository
- Marks DAGs as multi-file with included file list

### Syntax Validation
- AST parsing for syntax errors
- Python compilation for semantic errors
- Captures failure reasons in metadata

## Performance
- ~460 DAGs per version in 10-15 seconds
- 89% syntax validity rate
- Handles 2,500+ total DAGs across all versions

## Troubleshooting

**Git Issues**: Script uses GitHub API, no local git required
**Syntax Errors**: 11% failure rate is normal (broken examples exist)
**Missing Versions**: Script skips unavailable version tags

## Dataset Purpose

First step in ML dataset creation workflow:
1. **DAG Mining** (this script) → Raw DAGs with metadata
2. **Instruction Generation** → Task/solution pairs
3. **Fine-tuning** → Train specialized Airflow model
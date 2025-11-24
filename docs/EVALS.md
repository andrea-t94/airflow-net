# Airflow DAG Evaluations

This document describes the validation checks performed on Airflow DAG files by the `DAGValidator` class.

## Syntax Validation

### SYNTAX_ERROR
**Description:** Python syntax errors that prevent AST parsing
**Trigger:** When `ast.parse()` fails on malformed Python code
**Example Issues:**
- Missing colons, parentheses, or quotes
- Invalid indentation
- Malformed Python statements

### PARSE_ERROR
**Description:** General parsing failures (non-syntax related)
**Trigger:** Exceptions during DAG content processing beyond syntax errors
**Example Issues:**
- Encoding issues
- Unexpected file structure
- Runtime parsing exceptions

## DAG Structure Validation

### DUPLICATE_TASK_ID
**Description:** Multiple tasks with the same `task_id` within a DAG
**Trigger:** When the same task_id appears on multiple lines
**Impact:** Airflow runtime failure - task_ids must be unique per DAG
**Example:**
```python
task1 = BashOperator(task_id="my_task", bash_command="echo 1")
task2 = PythonOperator(task_id="my_task", python_callable=func)  # ← Error
```

### CIRCULAR_DEPENDENCY
**Description:** Circular dependencies in task execution graph
**Trigger:** DFS cycle detection finds loops in `>>` or `<<` operators
**Impact:** DAG cannot execute - no valid topological ordering
**Example:**
```python
task1 >> task2 >> task3 >> task1  # ← Circular dependency
```

### EMPTY_DAG
**Description:** DAG definition without any tasks
**Trigger:** DAG constructor found but no operator tasks detected
**Impact:** Useless DAG that does nothing
**Example:**
```python
dag = DAG('empty_dag')
# No tasks defined
```

### MULTIPLE_DAGS
**Description:** Multiple DAG definitions in a single file
**Trigger:** More than one DAG constructor or `@dag` decorator found
**Impact:** Non-standard pattern, potential confusion
**Example:**
```python
dag1 = DAG('first_dag')
dag2 = DAG('second_dag')  # ← Warning: multiple DAGs
```

## Validation Process

The validation occurs in this order:
1. **Syntax Check** - AST parsing validation
2. **DAG Structure Extraction** - Find DAG and task definitions
3. **Task ID Validation** - Check for duplicates
4. **Dependency Analysis** - Detect circular dependencies
5. **Structure Validation** - Check for empty or multi-DAG files

## Error Output

Validation errors include:
- **Error Type** - One of the categories above
- **File/Source Name** - Location of the issue
- **Line Number** - Where applicable (syntax errors)
- **Description** - Human-readable error message

## Usage

The evaluations are performed by:
- **DAG Mining**: Enhanced metadata during dataset creation
- **Validation Script**: `scripts/run_dag_validation.py` for analysis
- **Standalone Usage**: `DAGValidator.validate_content(content, source_name)`
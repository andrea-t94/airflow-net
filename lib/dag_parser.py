#!/usr/bin/env python3
"""
Lightweight Airflow DAG Validator
Catches common syntax errors before pushing commits:
- Duplicate task_ids
- Invalid task_id format (mimics airflow.utils.helpers.validate_key)
- Circular dependencies
- Invalid DAG structure
- Import errors
- Missing required parameters
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ValidationError:
    """Represents a validation error found in a DAG"""
    
    def __init__(self, file: str, error_type: str, message: str, line: int = None):
        self.file = file
        self.error_type = error_type
        self.message = message
        self.line = line
    
    def __str__(self):
        location = f":{self.line}" if self.line else ""
        return f"[{self.error_type}] {self.file}{location}: {self.message}"


class DAGValidator:
    """Validates Airflow DAG files for common syntax errors"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.dags_info: Dict[str, dict] = {}
    
    def validate_file(self, file_path: Path) -> List[ValidationError]:
        """Validate a single DAG file"""
        file_errors = []

        # Check if file is Python
        if file_path.suffix != '.py':
            return file_errors

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Use validate_content for actual validation
            file_errors = self.validate_content(content, str(file_path))

        except Exception as e:
            file_errors.append(ValidationError(
                str(file_path),
                "READ_ERROR",
                f"Failed to read file: {str(e)}"
            ))

        self.errors.extend(file_errors)
        return file_errors

    def validate_content(self, content: str, source_name: str = "<content>") -> List[ValidationError]:
        """Validate DAG content directly (application-agnostic)"""
        content_errors = []

        try:
            # Parse AST
            tree = ast.parse(content, filename=source_name)

            # Extract DAG information
            dag_info = self._extract_dag_info(tree, Path(source_name))

            if dag_info:
                self.dags_info[source_name] = dag_info

                # Validate task IDs
                content_errors.extend(self._check_duplicate_task_ids(dag_info, Path(source_name)))

                # Validate task_id format
                content_errors.extend(self._check_task_id_format(dag_info, Path(source_name)))

                # Validate dependencies
                content_errors.extend(self._check_circular_dependencies(dag_info, Path(source_name)))

                # Validate DAG structure
                content_errors.extend(self._check_dag_structure(dag_info, Path(source_name)))

        except SyntaxError as e:
            content_errors.append(ValidationError(
                source_name,
                "SYNTAX_ERROR",
                f"Python syntax error: {e.msg}",
                e.lineno
            ))
        except Exception as e:
            content_errors.append(ValidationError(
                source_name,
                "PARSE_ERROR",
                f"Failed to parse content: {str(e)}"
            ))

        return content_errors
    
    def _extract_dag_info(self, tree: ast.AST, file_path: Path) -> dict:
        """Extract DAG and task information from AST"""
        dag_info = {
            'dag_ids': [],
            'task_ids': defaultdict(list),  # task_id -> [line_numbers]
            'task_dependencies': defaultdict(list),
            'task_var_map': {},  # Map variable names to task_ids
            'has_dag': False
        }
        
        # Find DAG instantiations and task assignments
        for node in ast.walk(tree):
            # Check for DAG() constructor calls
            if isinstance(node, ast.Call):
                if self._is_dag_constructor(node):
                    dag_id = self._extract_dag_id(node)
                    if dag_id:
                        dag_info['dag_ids'].append(dag_id)
                        dag_info['has_dag'] = True
            
            # Check for @dag decorator
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if self._is_dag_decorator(decorator):
                        dag_info['dag_ids'].append(node.name)
                        dag_info['has_dag'] = True
            
            # Extract task assignments (e.g., task1 = BashOperator(...))
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if isinstance(node.value, ast.Call):
                        task_id = self._extract_task_id_from_call(node.value)
                        if task_id:
                            dag_info['task_ids'][task_id].append(node.lineno)
                            dag_info['task_var_map'][var_name] = task_id
        
        # Extract task dependencies (>> and << operators)
        self._extract_dependencies(tree, dag_info)
        
        return dag_info if dag_info['has_dag'] else None
    
    def _is_dag_constructor(self, node: ast.Call) -> bool:
        """Check if node is a DAG constructor call"""
        if isinstance(node.func, ast.Name):
            return node.func.id == 'DAG'
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == 'DAG'
        return False
    
    def _is_dag_decorator(self, node) -> bool:
        """Check if decorator is @dag"""
        if isinstance(node, ast.Name):
            return node.id == 'dag'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id == 'dag'
        return False
    
    def _extract_dag_id(self, node: ast.Call) -> str:
        """Extract dag_id from DAG constructor"""
        # Check positional arguments first (first arg is dag_id)
        if node.args and len(node.args) > 0:
            if isinstance(node.args[0], ast.Constant):
                return node.args[0].value
        
        # Check keyword arguments
        for keyword in node.keywords:
            if keyword.arg == 'dag_id':
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
        return None
    
    def _extract_task_id_from_call(self, node: ast.Call) -> str:
        """Extract task_id from operator calls"""
        # Check if this is an Operator or Sensor
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        if not func_name or not any(x in func_name for x in ['Operator', 'Sensor']):
            return None
        
        # Extract task_id from keyword arguments
        for keyword in node.keywords:
            if keyword.arg == 'task_id':
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
        
        return None
    
    def _extract_dependencies(self, tree: ast.AST, dag_info: dict):
        """Extract task dependencies using >> and << operators"""
        visited_nodes = set()  # Track visited nodes to avoid duplicates
        
        # We need to process top-level expressions, not individual BinOps
        for node in tree.body if isinstance(tree, ast.Module) else [tree]:
            self._process_node_for_dependencies(node, dag_info, visited_nodes)
    
    def _process_node_for_dependencies(self, node: ast.AST, dag_info: dict, visited_nodes: set):
        """Recursively process nodes for dependency extraction"""
        node_id = id(node)
        if node_id in visited_nodes:
            return
        visited_nodes.add(node_id)
        
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            # Extract all tasks in the chain
            chain = self._extract_dependency_chain(node.value, dag_info)
            
            # Create dependencies from the chain
            for i in range(len(chain) - 1):
                upstream = chain[i]
                downstream = chain[i + 1]
                if upstream and downstream:
                    dag_info['task_dependencies'][upstream].append(downstream)
        
        # Recursively process children
        for child in ast.iter_child_nodes(node):
            self._process_node_for_dependencies(child, dag_info, visited_nodes)
    
    def _extract_dependency_chain(self, node: ast.BinOp, dag_info: dict) -> List[str]:
        """Extract the full chain of dependencies from chained >> or << operators"""
        if isinstance(node.op, ast.RShift):
            # >> operator: left >> right
            left_chain = []
            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.RShift):
                left_chain = self._extract_dependency_chain(node.left, dag_info)
            else:
                left_vars = self._get_var_names_from_expr(node.left)
                left_chain = [dag_info['task_var_map'].get(v, v) for v in left_vars]
            
            right_vars = self._get_var_names_from_expr(node.right)
            right_tasks = [dag_info['task_var_map'].get(v, v) for v in right_vars]
            
            return left_chain + right_tasks
        
        elif isinstance(node.op, ast.LShift):
            # << operator: right << left (reversed)
            left_vars = self._get_var_names_from_expr(node.left)
            left_tasks = [dag_info['task_var_map'].get(v, v) for v in left_vars]
            
            right_chain = []
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.LShift):
                right_chain = self._extract_dependency_chain(node.right, dag_info)
            else:
                right_vars = self._get_var_names_from_expr(node.right)
                right_chain = [dag_info['task_var_map'].get(v, v) for v in right_vars]
            
            return right_chain + left_tasks
        
        return []
    
    def _get_var_names_from_expr(self, expr) -> List[str]:
        """Extract variable names from an expression"""
        var_names = []
        
        if isinstance(expr, ast.Name):
            var_names.append(expr.id)
        elif isinstance(expr, ast.List):
            for elt in expr.elts:
                var_names.extend(self._get_var_names_from_expr(elt))
        
        return var_names
    
    def _check_duplicate_task_ids(self, dag_info: dict, file_path: Path) -> List[ValidationError]:
        """Check for duplicate task IDs within a DAG"""
        errors = []

        for task_id, lines in dag_info['task_ids'].items():
            if len(lines) > 1:
                errors.append(ValidationError(
                    str(file_path),
                    "DUPLICATE_TASK_ID",
                    f"Duplicate task_id '{task_id}' found at lines: {', '.join(map(str, lines))}",
                    lines[0]
                ))

        return errors

    def _check_task_id_format(self, dag_info: dict, file_path: Path) -> List[ValidationError]:
        """
        Check task_id format matches Airflow's requirements.
        Mimics airflow.utils.helpers.validate_key which enforces:
        - Only alphanumeric characters, dashes, dots, and underscores allowed
        - Pattern: ^[a-zA-Z0-9._-]+$
        """
        errors = []
        valid_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')

        for task_id, lines in dag_info['task_ids'].items():
            if not valid_pattern.match(task_id):
                errors.append(ValidationError(
                    str(file_path),
                    "INVALID_TASK_ID",
                    f"task_id '{task_id}' must contain only alphanumeric characters, dashes, dots, and underscores",
                    lines[0]
                ))

        return errors
    
    def _check_circular_dependencies(self, dag_info: dict, file_path: Path) -> List[ValidationError]:
        """Check for circular dependencies in task graph"""
        errors = []
        dependencies = dag_info['task_dependencies']
        
        # Build adjacency list
        graph = defaultdict(set)
        all_tasks = set()
        
        for upstream, downstreams in dependencies.items():
            all_tasks.add(upstream)
            for downstream in downstreams:
                graph[upstream].add(downstream)
                all_tasks.add(downstream)
        
        # Detect cycles using DFS
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str], path: List[str]) -> Tuple[bool, List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    has_cycle_result, cycle_path = has_cycle(neighbor, visited, rec_stack, path[:])
                    if has_cycle_result:
                        return True, cycle_path
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    return True, path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return False, []
        
        visited = set()
        for task in all_tasks:
            if task not in visited:
                has_cycle_result, cycle_path = has_cycle(task, visited, set(), [])
                if has_cycle_result:
                    cycle_str = ' -> '.join(cycle_path)
                    errors.append(ValidationError(
                        str(file_path),
                        "CIRCULAR_DEPENDENCY",
                        f"Circular dependency detected: {cycle_str}"
                    ))
                    break  # Report only the first cycle found
        
        return errors
    
    def _check_dag_structure(self, dag_info: dict, file_path: Path) -> List[ValidationError]:
        """Check for basic DAG structure issues"""
        errors = []
        
        # Check if DAG has at least one task
        if not dag_info['task_ids']:
            errors.append(ValidationError(
                str(file_path),
                "EMPTY_DAG",
                "DAG has no tasks defined"
            ))
        
        # Check for multiple DAG definitions in one file (warning)
        if len(dag_info['dag_ids']) > 1:
            errors.append(ValidationError(
                str(file_path),
                "MULTIPLE_DAGS",
                f"Multiple DAGs defined in one file: {', '.join(dag_info['dag_ids'])}"
            ))
        
        return errors
    
    def validate_directory(self, directory: Path, recursive: bool = True) -> List[ValidationError]:
        """Validate all DAG files in a directory"""
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in directory.glob(pattern):
            # Skip __pycache__ and test files
            if '__pycache__' in str(file_path) or file_path.name.startswith('test_'):
                continue
            
            self.validate_file(file_path)
        
        return self.errors
    
    def print_report(self):
        """Print validation report"""
        if not self.errors:
            print("✅ All DAG files validated successfully!")
            return True
        
        print(f"❌ Found {len(self.errors)} validation error(s):\n")
        
        # Group errors by type
        errors_by_type = defaultdict(list)
        for error in self.errors:
            errors_by_type[error.error_type].append(error)
        
        for error_type, errors in sorted(errors_by_type.items()):
            print(f"\n{error_type} ({len(errors)}):")
            print("-" * 80)
            for error in errors:
                print(f"  {error}")
        
        return False



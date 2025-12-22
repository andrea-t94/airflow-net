"""
Airflow DAG Miner Library

Extracts DAGs from Airflow repositories with minimal, focused metadata
for instruction generation and ML training.
"""

import ast
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging
# ...
try:
    from airflow_net.validation import DAGValidator
except ImportError:
    # Fallback/Dev mode: try to find it in src if not installed
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src')) # Adjust path to src
    from airflow_net.validation import DAGValidator

logger = logging.getLogger(__name__)


class SimpleAirflowDAGMiner:
    """Simple DAG miner focused on core metadata for instruction generation."""

    def __init__(self, config: Dict[str, Any] = None):
        # Set defaults or use config values
        if config:
            github_config = config.get('github', {})
            mining_config = config.get('mining', {})
            self.github_api_base = github_config.get('base_url', "https://api.github.com/repos/apache/airflow")
            self.max_workers = mining_config.get('max_workers', 8)
            self.timeout_seconds = mining_config.get('timeout_seconds', 30)
            # Get GitHub token for authentication
            self.github_token = github_config.get('token') or self._get_github_token()
        else:
            # Default values
            self.github_api_base = "https://api.github.com/repos/apache/airflow"
            self.max_workers = 8
            self.timeout_seconds = 30
            self.github_token = self._get_github_token()

        # Set up headers for GitHub API
        self.headers = {}
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
            logger.info("🔑 Using GitHub token for authenticated requests")
        else:
            logger.warning("⚠️ No GitHub token found - using unauthenticated requests (60/hour limit)")

        # Initialize DAG validator for enhanced validation
        self.validator = DAGValidator()

    def _get_github_token(self) -> str:
        """Get GitHub token from environment or .env file."""
        from config_loader import get_github_token
        return get_github_token()

    def get_version_major(self, version: str) -> str:
        """Get major version (2.x or 3.x)."""
        major = version.split('.')[0]
        return f"{major}.x"

    def fetch_dag_files(self, version: str) -> tuple:
        """Fetch all example_*.py files via GitHub API."""
        # Try only specific version tags, no fallback to main
        refs_to_try = [f"v{version}", f"{version}"]

        for ref in refs_to_try:
            try:
                # Use GitHub Tree API to get all files in one call
                url = f"{self.github_api_base}/git/trees/{ref}?recursive=1"
                logger.info(f"Fetching tree from {url}")

                response = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    tree_data = response.json()
                    dag_records = self._process_tree_files(tree_data, version, ref)
                    return dag_records, None
                else:
                    logger.info(f"Tag {ref} not found: HTTP {response.status_code}")

            except Exception as e:
                logger.info(f"Error fetching tree for {ref}: {e}")
                continue

        # No specific version tag found
        logger.warning(f"No tag found for version {version} (tried: v{version}, {version})")
        return [], f"No git tag found for version {version}"

    def _process_tree_files(self, tree_data: Dict, version: str, ref: str) -> List[Dict]:
        """Process files from GitHub Tree API response."""
        dag_records = []

        # Simple approach: get ALL example_*.py files across the entire repository
        python_files = []
        for item in tree_data.get('tree', []):
            if (item['type'] == 'blob' and
                item['path'].split('/')[-1].startswith('example_') and
                item['path'].endswith('.py')):
                python_files.append(item)

        logger.info(f"Found {len(python_files)} example_*.py files across repository")

        # Get all file paths for internal import detection
        all_repo_files = [item['path'] for item in tree_data.get('tree', []) if item['type'] == 'blob']

        # Process files in parallel
        dag_records = self._process_files_parallel(python_files, version, ref, all_repo_files)
        return dag_records

    def _download_file(self, raw_url: str, file_path: str) -> tuple:
        """Download single file with error handling."""
        try:
            response = requests.get(raw_url, headers=self.headers, timeout=self.timeout_seconds)
            if response.status_code == 200:
                return file_path, response.text, None
            else:
                return file_path, None, f"HTTP {response.status_code}"
        except Exception as e:
            return file_path, None, str(e)

    def _process_files_parallel(self, python_files: List[Dict], version: str, ref: str, all_repo_files: List[str]) -> List[Dict]:
        """Process files in parallel with graceful error handling."""
        dag_records = []

        # Prepare download tasks
        download_tasks = []
        for file_item in python_files:
            raw_url = f"https://raw.githubusercontent.com/apache/airflow/{ref}/{file_item['path']}"
            download_tasks.append((raw_url, file_item))

        logger.info(f"Downloading {len(download_tasks)} files in parallel...")

        # Download files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._download_file, task[0], task[1]['path']): task[1]
                for task in download_tasks
            }

            completed = 0
            for future in as_completed(future_to_file):
                file_item = future_to_file[future]
                file_path, content, error = future.result()

                completed += 1
                if completed % 10 == 0 or completed == len(download_tasks):
                    logger.info(f"Downloaded {completed}/{len(download_tasks)} files")

                if error:
                    logger.warning(f"Failed to download {file_path}: {error}")
                    continue

                if not content:
                    continue

                try:
                    # Check if it's actually a DAG
                    if self._is_dag_file(content):
                        # Remove license headers while preserving functional comments
                        cleaned_content = self._remove_license_headers(content)

                        # Create file_info dict for compatibility
                        file_info = {'name': file_item['path'].split('/')[-1]}

                        # Single-pass multifile detection and combination
                        final_content, included_files, is_multifile = self._process_multifile_dag(
                            cleaned_content, all_repo_files
                        )

                        metadata = self._extract_core_metadata(final_content, version, file_info, is_multifile, included_files)

                        dag_record = {
                            'metadata': metadata,
                            'content': final_content,
                            'extracted_at': datetime.now().isoformat(),
                        }

                        dag_records.append(dag_record)

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue

        logger.info(f"Successfully processed {len(dag_records)} DAG files")
        return dag_records

    def _is_dag_file(self, content: str) -> bool:
        """Check if file contains DAG definition."""
        dag_indicators = [
            r'from airflow import DAG',
            r'from airflow\.models import DAG',
            r'@dag\(',
            r'DAG\(',
        ]

        for pattern in dag_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False

    def _extract_core_metadata(self, content: str, version: str, file_info: Dict, is_multifile: bool = False, included_files: List[str] = None) -> Dict:
        """Extract core metadata fields for instruction generation."""
        # Use DAGValidator for comprehensive validation
        errors = self.validator.validate_content(content, file_info['name'])

        # Get extracted DAG info
        dag_info = self.validator.dags_info.get(file_info['name'])

        metadata = {
            'syntax_valid': len(errors) == 0,
            'airflow_version': version,
            'is_multifile': is_multifile,
            'line_count': len(content.splitlines()),
            'file_name': file_info['name'],
        }

        # Add validation errors if any
        if errors:
            metadata['validation_errors'] = [str(error) for error in errors]

        # Add DAG structure info from enhanced validation
        if dag_info:
            metadata.update({
                'task_count': len(dag_info.get('task_ids', {})),
                'has_dependencies': bool(dag_info.get('task_dependencies', {})),
                'dag_count': len(dag_info.get('dag_ids', [])),
            })

        # Add included files if any
        if included_files:
            metadata['included_files'] = included_files

        return metadata


    def _process_multifile_dag(self, content: str, all_repo_files: List[str]) -> tuple:
        """
        Single-pass detection of multi-file DAGs (metadata only).
        Returns tuple of (original_content, included_files, is_multifile)
        """
        # Find internal dependencies in one pass
        dependency_paths = self._find_internal_dependencies(content, all_repo_files)

        # If no dependencies found, it's single-file
        if not dependency_paths:
            return content, [], False

        # Multi-file detected - return metadata only (no file combination)
        included_files = [dep_path.split('/')[-1] for dep_path in dependency_paths]

        return content, included_files, True

    def _find_internal_dependencies(self, content: str, all_repo_files: List[str]) -> List[str]:
        """Find internal imports that come after the last airflow.* import."""
        lines = content.splitlines()

        # Find the last line with an airflow.* import
        last_airflow_import_line = -1
        for i, line in enumerate(lines):
            if re.search(r'from\s+airflow\b', line.strip()) or re.search(r'import\s+airflow\b', line.strip()):
                last_airflow_import_line = i

        # If no airflow imports found, assume no internal dependencies
        if last_airflow_import_line == -1:
            return []

        # Look for imports after the last airflow import
        internal_modules = set()
        for i in range(last_airflow_import_line + 1, len(lines)):
            line = lines[i].strip()

            # Stop at first non-import, non-comment, non-empty line
            if line and not line.startswith('#') and not line.startswith('from ') and not line.startswith('import '):
                break

            # Extract module from import statements
            if line.startswith('from ') and ' import ' in line:
                match = re.match(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import', line)
                if match:
                    module = match.group(1)
                    # Skip standard library and common packages
                    if not module.startswith(('airflow', 'datetime', 'os', 'sys', 'typing', 'json', 're', 'pathlib', 'logging')):
                        internal_modules.add(module)

        # Find actual file paths for these modules
        dependency_paths = []
        for module in internal_modules:
            module_path = module.replace('.', '/')
            potential_paths = [
                f"{module_path}.py",
                f"{module_path}/__init__.py",
            ]

            for path in potential_paths:
                if path in all_repo_files:
                    dependency_paths.append(path)
                    logger.debug(f"Found dependency: {module} -> {path}")
                    break

        if dependency_paths:
            logger.info(f"Found {len(dependency_paths)} internal dependencies")

        return dependency_paths


    def _remove_license_headers(self, content: str) -> str:
        """Remove license/copyright headers while preserving functional comments."""
        lines = content.splitlines()

        # Find first and last lines containing license-related keywords
        license_keywords = ['license', 'copyright', 'apache', 'licensed']
        first_license_line = None
        last_license_line = None

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in license_keywords):
                if first_license_line is None:
                    first_license_line = i
                last_license_line = i

        # If license block found, remove it
        if first_license_line is not None and last_license_line is not None:
            # Remove license block and any trailing empty lines
            remaining_lines = lines[last_license_line + 1:]

            # Skip empty lines after license block
            while remaining_lines and not remaining_lines[0].strip():
                remaining_lines.pop(0)

            return '\n'.join(remaining_lines)

        # No license found, return original content
        return content

    def mine_versions(self, versions: List[str]) -> tuple:
        """Mine DAGs from multiple Airflow versions."""
        all_dags = []
        missing_versions = []

        for version in versions:
            logger.info(f"Mining Airflow version {version}")
            dag_records, error = self.fetch_dag_files(version)

            if error:
                missing_versions.append({"version": version, "reason": error})
                logger.info(f"Skipped version {version}: {error}")
            else:
                all_dags.extend(dag_records)
                logger.info(f"Extracted {len(dag_records)} DAGs from version {version}")

        logger.info(f"Total DAGs extracted: {len(all_dags)}")
        if missing_versions:
            logger.info(f"Missing versions: {[v['version'] for v in missing_versions]}")

        return all_dags, missing_versions

    def save_results(self, dags: List[Dict], missing_versions: List[Dict], output_file: str, is_test: bool = False):
        """Save results to datasets/raw folder with metadata tracking."""
        # Create datasets/raw directory if it doesn't exist
        datasets_dir = Path("datasets/raw")
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Use consistent filename (add test prefix if test mode)
        if is_test:
            final_path = datasets_dir / f"test_{output_file}"
        else:
            final_path = datasets_dir / output_file

        # Add metadata to each DAG record with mining timestamp
        timestamp = datetime.now().isoformat()
        for dag in dags:
            dag['mining_metadata'] = {
                'mining_timestamp': timestamp,
                'is_test_run': is_test,
                'output_file': str(final_path)
            }

        with open(final_path, 'w', encoding='utf-8') as f:
            for dag in dags:
                json.dump(dag, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(dags)} DAGs to {final_path}")

        # Generate and save summary
        summary = self._generate_summary(dags, missing_versions)
        summary['mining_metadata'] = {
            'mining_timestamp': timestamp,
            'is_test_run': is_test,
            'total_dags': len(dags)
        }
        summary_path = final_path.with_suffix('.summary.json')

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary to {summary_path}")

        return str(final_path)

    def _generate_summary(self, dags: List[Dict], missing_versions: List[Dict]) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_dags': len(dags),
            'syntax_valid_count': 0,
            'multifile_count': 0,
            'versions': {},
            'length_distribution': {
                'short': 0,    # <50 lines
                'medium': 0,   # 50-150 lines
                'long': 0      # >150 lines
            },
            'missing_versions': missing_versions
        }

        for dag in dags:
            metadata = dag['metadata']

            # Count valid syntax
            if metadata['syntax_valid']:
                summary['syntax_valid_count'] += 1

            # Count multifile
            if metadata['is_multifile']:
                summary['multifile_count'] += 1

            # Version counts
            version = metadata['airflow_version']
            summary['versions'][version] = summary['versions'].get(version, 0) + 1

            # Length distribution
            line_count = metadata['line_count']
            if line_count < 50:
                summary['length_distribution']['short'] += 1
            elif line_count <= 150:
                summary['length_distribution']['medium'] += 1
            else:
                summary['length_distribution']['long'] += 1

        return summary
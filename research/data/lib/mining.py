"""
Airflow DAG Miner Library

Extracts DAGs from Airflow repositories with minimal, focused metadata
for instruction generation and ML training.
"""

import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
from airflow_net.validation import DAGValidator

logger = logging.getLogger(__name__)


class AirflowDAGMiner:
    """Simple DAG miner focused on core metadata for instruction generation."""

    def __init__(self, github_token: Optional[str] = None):
        self.github_api_base = "https://api.github.com/repos/apache/airflow"
        self.max_workers = 8
        self.timeout_seconds = 30
        
        # Initialize session with headers
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
            logger.info("🔑 Using GitHub token for authenticated requests")
        else:
            logger.warning("⚠️ No GitHub token found - using unauthenticated requests (60/hour limit)")

        # Initialize DAG validator
        self.validator = DAGValidator()

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

        return all_dags, missing_versions

    def fetch_dag_files(self, version: str) -> tuple:
        """Fetch all example_*.py files via GitHub API."""
        # Try specific version tags
        refs_to_try = [f"v{version}", f"{version}"]

        for ref in refs_to_try:
            try:
                # Use GitHub Tree API to get all files in one call
                url = f"{self.github_api_base}/git/trees/{ref}?recursive=1"
                logger.debug(f"Fetching tree from {url}")

                response = self.session.get(url, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    tree_data = response.json()
                    dag_records = self._process_tree_files(tree_data, version, ref)
                    return dag_records, None
                else:
                    logger.debug(f"Tag {ref} not found: HTTP {response.status_code}")

            except Exception as e:
                logger.warning(f"Error fetching tree for {ref}: {e}")
                continue

        return [], f"No git tag found for version {version}"

    def _process_tree_files(self, tree_data: Dict, version: str, ref: str) -> List[Dict]:
        """Process files from GitHub Tree API response."""
        # Filter for example DAGs
        python_files = []
        for item in tree_data.get('tree', []):
            if (item['type'] == 'blob' and
                item['path'].split('/')[-1].startswith('example_') and
                item['path'].endswith('.py')):
                python_files.append(item)

        logger.info(f"Found {len(python_files)} example_*.py files across repository")

        # Process files in parallel
        return self._process_files_parallel(python_files, version, ref)

    def _process_files_parallel(self, python_files: List[Dict], version: str, ref: str) -> List[Dict]:
        """Process files in parallel."""
        dag_records = []
        
        # Prepare download tasks
        download_tasks = []
        for file_item in python_files:
            raw_url = f"https://raw.githubusercontent.com/apache/airflow/{ref}/{file_item['path']}"
            download_tasks.append((raw_url, file_item))

        # Download files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._download_file, task[0]): task[1]
                for task in download_tasks
            }

            for future in as_completed(future_to_file):
                file_item = future_to_file[future]
                content, error = future.result()

                if error:
                    logger.debug(f"Failed to download {file_item['path']}: {error}")
                    continue

                if not content:
                    continue

                try:
                    # Check if it's actually a DAG
                    if self._is_dag_file(content):
                        # Remove license headers
                        cleaned_content = self._remove_license_headers(content)

                        file_info = {'name': file_item['path'].split('/')[-1]}
                        metadata = self._extract_core_metadata(cleaned_content, version, file_info)

                        dag_record = {
                            'metadata': metadata,
                            'content': cleaned_content,
                            'extracted_at': datetime.now().isoformat(),
                        }

                        dag_records.append(dag_record)

                except Exception as e:
                    logger.warning(f"Error processing {file_item['path']}: {e}")
                    continue

        return dag_records

    def _download_file(self, raw_url: str) -> tuple:
        """Download single file."""
        try:
            response = self.session.get(raw_url, timeout=self.timeout_seconds)
            if response.status_code == 200:
                return response.text, None
            else:
                return None, f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)

    def _is_dag_file(self, content: str) -> bool:
        """Check if file contains DAG definition."""
        dag_indicators = [
            r'from airflow import DAG',
            r'from airflow\.models import DAG',
            r'@dag\(',
            r'DAG\(',
        ]
        next_indicators = [
             r'from airflow import .*',
             r'import airflow'
        ]

        if not any(re.search(p, content, re.MULTILINE) for p in next_indicators):
             return False

        for pattern in dag_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False

    def _extract_core_metadata(self, content: str, version: str, file_info: Dict) -> Dict:
        """Extract core metadata fields for instruction generation."""
        # Use DAGValidator for comprehensive validation
        errors = self.validator.validate_content(content, file_info['name'])
        dag_info = self.validator.dags_info.get(file_info['name'])

        metadata = {
            'syntax_valid': len(errors) == 0,
            'airflow_version': version,
            'line_count': len(content.splitlines()),
            'file_name': file_info['name'],
        }

        # Add validation errors if any
        if errors:
            metadata['validation_errors'] = [str(error) for error in errors]

        # Add DAG structure info
        if dag_info:
            metadata.update({
                'task_count': len(dag_info.get('task_ids', {})),
                'has_dependencies': bool(dag_info.get('task_dependencies', {})),
                'dag_count': len(dag_info.get('dag_ids', [])),
            })

        return metadata

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
            # Simple heuristic: if the block is at the top of the file (first 20 lines)
            if first_license_line < 20: 
                remaining_lines = lines[last_license_line + 1:]
                # Skip empty lines after license block
                while remaining_lines and not remaining_lines[0].strip():
                    remaining_lines.pop(0)
                return '\n'.join(remaining_lines)

        return content
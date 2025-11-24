#!/usr/bin/env python3

import sys
import logging
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from dag_parser import DAGValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DAGValidationAnalyzer:
    """Analyzes DAG validation results from JSONL datasets."""

    def __init__(self):
        self.validator = DAGValidator()
        self.results = []
        self.filename_version_counter = defaultdict(int)

    def process_dataset(self, input_file: str, output_file: str) -> dict:
        """Process JSONL dataset and generate CSV validation report."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing dataset: {input_file}")

        # Process each DAG record
        processed_count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    dag_record = json.loads(line.strip())
                    self._process_dag_record(dag_record, line_num)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} DAGs...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        # Write CSV results
        self._write_csv_results(output_path)

        stats = {
            'total_dags': len(self.results),
            'passed_validation': sum(1 for r in self.results if r['validation_passed']),
            'failed_validation': sum(1 for r in self.results if not r['validation_passed']),
            'syntax_errors': sum(1 for r in self.results if 'SYNTAX_ERROR' in r['failure_cause']),
            'structure_errors': sum(1 for r in self.results if any(x in r['failure_cause']
                                   for x in ['DUPLICATE_TASK_ID', 'CIRCULAR_DEPENDENCY', 'EMPTY_DAG'])),
        }

        logger.info(f"Validation analysis completed. Results saved to: {output_path}")
        return stats

    def _process_dag_record(self, dag_record: dict, line_num: int):
        """Process a single DAG record from the dataset."""
        try:
            # Extract metadata
            metadata = dag_record.get('metadata', {})
            filename = metadata.get('file_name', f'unknown_line_{line_num}')
            airflow_version = metadata.get('airflow_version', 'unknown')
            content = dag_record.get('content', '')

            # Generate unique ID for same filename+version combinations
            key = f"{filename}_{airflow_version}"
            self.filename_version_counter[key] += 1
            record_id = self.filename_version_counter[key]

            # Validate DAG content
            validation_errors = self.validator.validate_content(content, filename)

            # Determine validation status and failure cause
            validation_passed = len(validation_errors) == 0
            failure_cause = ""
            if not validation_passed:
                # Combine all error messages
                error_types = [error.error_type for error in validation_errors]
                failure_cause = "; ".join(error_types)

            # Store result
            result = {
                'filename': filename,
                'airflow_version': airflow_version,
                'id': record_id,
                'validation_passed': validation_passed,
                'failure_cause': failure_cause,
                'line_number': line_num
            }

            self.results.append(result)

        except Exception as e:
            logger.warning(f"Error processing DAG record at line {line_num}: {e}")

    def _write_csv_results(self, output_path: Path):
        """Write validation results to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'airflow_version', 'id', 'validation_passed', 'failure_cause']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                # Write clean CSV row
                writer.writerow({
                    'filename': result['filename'],
                    'airflow_version': result['airflow_version'],
                    'id': result['id'],
                    'validation_passed': result['validation_passed'],
                    'failure_cause': result['failure_cause'] or ''
                })


def main():
    """Run DAG validation analysis."""
    parser = argparse.ArgumentParser(description="Airflow DAG Validation Analyzer")
    parser.add_argument('--input',
                       default='datasets/raw/dags.jsonl',
                       help='Input JSONL dataset file (default: datasets/raw/dags.jsonl)')
    parser.add_argument('--output',
                       default='datasets/eval/dag_validation_report.csv',
                       help='Output CSV file (default: datasets/eval/dag_validation_report.csv)')

    args = parser.parse_args()

    try:
        print("🔍 Starting DAG Validation Analysis")
        print(f"📁 Input: {args.input}")
        print(f"📄 Output: {args.output}")
        print()

        # Initialize analyzer
        analyzer = DAGValidationAnalyzer()

        # Process dataset
        stats = analyzer.process_dataset(args.input, args.output)

        print(f"\n✅ Validation analysis completed successfully!")
        print(f"📊 Statistics:")
        print(f"   Total DAGs: {stats['total_dags']}")
        print(f"   Passed validation: {stats['passed_validation']}")
        print(f"   Failed validation: {stats['failed_validation']}")
        print(f"   Syntax errors: {stats['syntax_errors']}")
        print(f"   Structure errors: {stats['structure_errors']}")
        print(f"📁 Results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"❌ Validation analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
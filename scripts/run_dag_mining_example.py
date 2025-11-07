#!/usr/bin/env python3
"""
Simple example to run Airflow DAG mining for dataset creation.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run DAG mining and show basic results."""

    # Airflow versions to mine
    versions = ["2.7.2", "2.8.4", "2.9.3", "3.0.0", "3.0.1", "3.0.6"]
    output_file = "simple_dags_dataset.jsonl"

    print(f"🚀 Mining {len(versions)} Airflow versions...")

    # Run the mining script
    script_path = Path(__file__).parent / "airflow_dag_miner.py"
    command = [
        sys.executable, str(script_path),
        "--versions", ",".join(versions),
        "--output", output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Mining completed! Check {output_file}")

        # Show basic stats
        import json
        with open(output_file, 'r') as f:
            dag_count = sum(1 for _ in f)
        print(f"📊 Extracted {dag_count} DAGs")

        # Show summary if available
        summary_file = output_file.replace('.jsonl', '.summary.json')
        if Path(summary_file).exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            print(f"🔧 Top operators: {list(summary['operators'].keys())[:5]}")
            print(f"📏 Complexity: {summary['length_distribution']['short']} short, "
                  f"{summary['length_distribution']['medium']} medium, "
                  f"{summary['length_distribution']['long']} long")

    except subprocess.CalledProcessError as e:
        print(f"❌ Mining failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
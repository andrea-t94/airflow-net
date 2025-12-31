#!/usr/bin/env python3

import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from research.data.lib.mining import AirflowDAGMiner
from research.data.lib.config_loader import load_mining_config, get_github_token

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_results(dags: List[Dict], missing_versions: List[Dict], output_file: str) -> str:
    """Save results to research/artifacts/01_raw_dags folder with metadata tracking."""
    # Create output directory if it doesn't exist
    datasets_dir = Path("research/artifacts/data/01_raw_dags")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    final_path = datasets_dir / output_file

    # Add metadata to each DAG record with mining timestamp
    timestamp = datetime.now().isoformat()
    for dag in dags:
        dag['mining_metadata'] = {
            'mining_timestamp': timestamp,
            'output_file': str(final_path)
        }

    with open(final_path, 'w', encoding='utf-8') as f:
        for dag in dags:
            json.dump(dag, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saved {len(dags)} DAGs to {final_path}")

    # Generate and save summary
    summary = generate_summary(dags, missing_versions)
    summary['mining_metadata'] = {
        'mining_timestamp': timestamp,
        'total_dags': len(dags)
    }
    summary_path = final_path.with_suffix('.summary.json')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved summary to {summary_path}")

    return str(final_path)


def generate_summary(dags: List[Dict], missing_versions: List[Dict]) -> Dict:
    """Generate summary statistics."""
    summary = {
        'total_dags': len(dags),
        'syntax_valid_count': 0,
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


def main():
    """Run DAG mining using configuration."""
    parser = argparse.ArgumentParser(description="Airflow DAG Miner")
    parser.add_argument('--test', action='store_true', help='Run in test mode (2 versions only)')
    parser.add_argument('--versions', nargs='+', help='Override versions to mine')
    parser.add_argument('--output', help='Override output filename')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_mining_config()

        # Get token explicitly
        github_token = get_github_token()

        # Determine versions to mine
        if args.test:
            # Test mode: use only first 2 versions
            versions = config['airflow_versions'][:2]
            output_file = 'test_dags.jsonl'
            logger.info("🧪 Running in TEST mode (2 versions)")
        else:
            versions = args.versions if args.versions else config['airflow_versions']
            output_file = args.output if args.output else config['mining']['output_file']
            logger.info("🚀 Running in FULL mode")

        print(f"📝 Versions: {', '.join(versions)}")
        print(f"📄 Output: {output_file}")
        print()

        # Initialize miner with just the token
        miner = AirflowDAGMiner(github_token=github_token)

        # Mine DAGs
        all_dags, missing_versions = miner.mine_versions(versions)

        if not all_dags:
            print("⚠️  No DAGs extracted (all versions may be missing tags)")
            if missing_versions:
                print(f"Missing versions: {[v['version'] for v in missing_versions]}")
            return 0  # Don't fail, just warn

        # Save results locally
        output_path = save_results(
            all_dags,
            missing_versions,
            output_file
        )

        print(f"\n✅ Mining completed successfully!")
        print(f"📊 Total DAGs: {len(all_dags)}")
        print(f"📁 Saved to: {output_path}")

        if missing_versions:
            print(f"⚠️  Missing versions: {[v['version'] for v in missing_versions]}")

        return 0

    except Exception as e:
        logger.error(f"❌ Mining failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
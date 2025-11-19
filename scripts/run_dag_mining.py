#!/usr/bin/env python3

import sys
import logging
import argparse
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from dag_miner import SimpleAirflowDAGMiner
from config_loader import load_mining_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run DAG mining using configuration."""
    parser = argparse.ArgumentParser(description="Airflow DAG Miner")
    parser.add_argument('--test', action='store_true', help='Run in test mode (limited versions)')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_mining_config()

        # Use test mode if specified
        if args.test:
            versions = ["2.7.2"]
            print("🧪 Starting Test DAG Mining")
        else:
            versions = config['airflow_versions']
            print("🚀 Starting Full DAG Mining")

        print(f"📝 Versions: {', '.join(versions)}")
        print(f"📄 Output: {config['mining']['output_file']}")
        print()

        # Initialize miner
        miner = SimpleAirflowDAGMiner(config)

        # Mine DAGs
        all_dags, missing_versions = miner.mine_versions(versions)

        if not all_dags:
            print("⚠️  No DAGs extracted (all versions may be missing tags)")
            if missing_versions:
                print(f"Missing versions: {[v['version'] for v in missing_versions]}")
            return 0  # Don't fail, just warn

        # Save results
        output_path = miner.save_results(
            all_dags,
            missing_versions,
            config['mining']['output_file'],
            is_test=args.test
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
#!/usr/bin/env python3
"""
Script to generate instruction datasets using Claude Batch API.
Run as module: python -m research.data.scripts.02_gen_instruct
"""

import logging
import time
import argparse
from pathlib import Path

from research.data.lib.instruction import ClaudeBatchInstructionGenerator
from research.data.lib.config_loader import load_generation_config, get_api_key, get_input_dataset_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("gen_instruct")

def main():
    parser = argparse.ArgumentParser(description="Airflow Instruction Generator")
    parser.add_argument('--maxdags', type=int, help='Override number of DAGs to process')
    parser.add_argument('--input', help='Override input dataset file')
    parser.add_argument('--test', action='store_true', help='Run in test mode (5 DAGs)')
    args = parser.parse_args()

    try:
        config = load_generation_config()
        api_key = get_api_key()

        # Determine input and limits
        input_file = args.input if args.input else get_input_dataset_path(config)
        
        if args.test:
            max_dags = 5
            logger.info("🧪 Running in TEST mode (max 5 DAGs)")
        else:
            max_dags = args.maxdags if args.maxdags else config['generation']['max_dags']
            logger.info("🚀 Running in FULL mode")

        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return 1

        logger.info(f"Input: {input_file}")
        logger.info(f"Output Config: {config['generation']['output_file']}")
        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Max DAGs: {max_dags if max_dags else 'ALL'}")

        # Initialize and run
        generator = ClaudeBatchInstructionGenerator(api_key, config)
        
        start_time = time.time()
        stats = generator.process_dataset_batch(
            input_file=input_file,
            max_dags=max_dags,
            is_test=args.test
        )
        elapsed = time.time() - start_time

        if stats:
            logger.info("✅ Generation completed successfully")
            for k, v in stats.items():
                if k != 'generation_metadata':
                    logger.info(f"   {k}: {v}")
            logger.info(f"⏱️ Total time: {elapsed:.1f}s")
            return 0
            
    except Exception as e:
        logger.exception("Generation failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
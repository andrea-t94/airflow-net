#!/usr/bin/env python3

import sys
import logging
import time
import argparse
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from instruction_generator import ClaudeBatchInstructionGenerator
from config_loader import load_generation_config, get_api_key, get_input_dataset_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run instruction generation using configuration."""
    parser = argparse.ArgumentParser(description="Airflow Instruction Generator")
    parser.add_argument('--maxdags', type=int, help='Number of DAGs to process (overrides config)')
    parser.add_argument('--input', help='Input dataset file (overrides auto-detection)')
    parser.add_argument('--test', action='store_true', help='Run in test mode (5 DAGs, test output)')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_generation_config()

        # Get API key
        api_key = get_api_key()

        # Handle test mode
        if args.test:
            # Test mode: use main dataset but limit to 5 DAGs
            input_file = get_input_dataset_path(config)
            max_dags = 5
            print("🧪 Starting Test Instruction Generation")
        else:
            # Normal mode
            if args.input:
                input_file = args.input
            else:
                input_file = get_input_dataset_path(config)
            max_dags = args.maxdags if args.maxdags else config['generation']['max_dags']
            print("🚀 Starting Full Instruction Generation")

        if not Path(input_file).exists():
            logger.error(f"❌ Input file not found: {input_file}")
            return 1
        print(f"📁 Input: {input_file}")
        print(f"📄 Output: {config['generation']['output_file']}")
        print(f"🤖 Model: {config['model']['name']}")
        if max_dags:
            print(f"📝 Max DAGs: {max_dags}")
        else:
            print(f"📝 Processing ALL DAGs in file")
        print()

        # Initialize generator
        generator = ClaudeBatchInstructionGenerator(api_key, config)

        start_time = time.time()

        # Run batch processing
        stats = generator.process_dataset_batch(
            input_file=input_file,
            output_file=config['generation']['output_file'],
            max_dags=max_dags,
            prompt_template=config.get('prompt_template'),
            instruction_source=config['generation']['instruction_source'],
            is_test=args.test
        )

        elapsed_time = time.time() - start_time

        if stats:
            print(f"\n✅ Generation completed successfully!")
            print(f"📊 Generated: {stats['successful_generations']} instructions")
            print(f"📈 Success rate: {stats['successful_generations']}/{stats['total_processed']} "
                  f"({stats['successful_generations']/stats['total_processed']*100:.1f}%)")
            print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
            print(f"⚡ Speed: {stats['total_processed']/elapsed_time:.1f} DAGs/second")

            if stats['failed_generations'] > 0:
                print(f"\n⚠️  {stats['failed_generations']} DAGs failed processing")

        return 0

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
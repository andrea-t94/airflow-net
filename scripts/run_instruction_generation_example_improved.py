#!/usr/bin/env python3
"""
High-Performance Claude Instruction Generator

This script uses concurrent processing to generate Airflow DAG instructions
much faster than the sequential approach.

Examples:
    # Basic usage (50 DAGs, 10 workers)
    python run_instruction_generation_example_improved.py

    # Custom settings
    python run_instruction_generation_example_improved.py --maxdags 100 --workers 15

    # Quick test
    python run_instruction_generation_example_improved.py --maxdags 5 --workers 3

    # High throughput
    python run_instruction_generation_example_improved.py --maxdags 500 --workers 20
"""

import argparse
import json
import time
from pathlib import Path
from claude_instruction_generator_fast import FastClaudeInstructionGenerator, load_env


def print_progress_bar(completed, total, width=50):
    """Print a progress bar with percentage."""
    if total == 0:
        return

    percentage = (completed / total) * 100
    filled_length = int(width * completed // total)
    bar = '█' * filled_length + '░' * (width - filled_length)
    print(f'\r📊 Progress: |{bar}| {completed}/{total} ({percentage:.1f}%) DAGs processed', end='', flush=True)


def count_valid_dags(input_file):
    """Count valid DAGs in the dataset for accurate progress tracking."""
    valid_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                dag_record = json.loads(line.strip())
                total_count += 1
                if dag_record['metadata'].get('syntax_valid', True):
                    valid_count += 1
            except json.JSONDecodeError:
                continue

    return valid_count, total_count


def main():
    """Run high-performance instruction generation with real-time progress and improved error handling."""

    parser = argparse.ArgumentParser(description="High-Performance Airflow Instruction Generator")
    parser.add_argument(
        '--maxdags',
        type=int,
        default=50,
        help='Number of DAGs to process (default: 50)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of concurrent workers (default: 10)'
    )
    parser.add_argument(
        '--input',
        default="simple_dags_dataset.jsonl",
        help='Input JSONL file (default: simple_dags_dataset.jsonl)'
    )
    parser.add_argument(
        '--output',
        default="airflow_instructions_dataset.jsonl",
        help='Output JSONL file (default: airflow_instructions_dataset.jsonl)'
    )

    args = parser.parse_args()

    # Configuration
    input_file = args.input
    output_file = args.output
    max_dags = args.maxdags
    max_workers = args.workers  

    # Input validation
    if not Path(input_file).exists():
        print(f"❌ Input file {input_file} not found!")
        print("Please run the DAG mining script first:")
        print("python run_dag_mining_example.py")
        return 1

    # API key validation
    api_key = load_env()
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("You can get an API key from: https://console.anthropic.com/")
        return 1

    print(f"🚀 High-Performance Instruction Generation")
    print(f"📁 Input: {input_file}")
    print(f"📄 Output: {output_file}")
    print(f"📝 Max DAGs: {max_dags}")
    print(f"👥 Workers: {max_workers}")

    # Count valid DAGs for accurate progress
    print("\n🔍 Analyzing dataset...")
    valid_count, total_count = count_valid_dags(input_file)
    actual_max = min(max_dags, valid_count)

    print(f"📊 Dataset stats: {total_count} total DAGs, {valid_count} valid ({(valid_count/total_count)*100:.1f}%)")
    print(f"🎯 Will process: {actual_max} DAGs")
    print()

    # Initialize fast generator
    generator = FastClaudeInstructionGenerator(
        api_key=api_key,
        max_workers=max_workers
    )

    # Progress tracking callback
    def progress_callback(completed, total):
        print_progress_bar(completed, total)

    start_time = time.time()

    try:
        print("⚡ Starting high-performance processing...")
        print_progress_bar(0, actual_max)

        # Use fast threaded processing
        stats = generator.process_dataset(
            input_file=input_file,
            output_file=output_file,
            max_dags=max_dags,
            progress_callback=progress_callback
        )

        print()  # New line after progress bar
        elapsed_time = time.time() - start_time

        # Load and show sample instructions
        print("\n📋 Sample generated instructions:")
        try:
            with open(output_file, 'r') as f:
                sample_instructions = []
                for i, line in enumerate(f):
                    if i >= 3:  # Show first 3 instructions
                        break
                    instruction = json.loads(line)
                    sample_instructions.append(instruction)

                for i, instruction in enumerate(sample_instructions):
                    print(f"\n{i+1}. {instruction['instruction']}")
                    print(f"   Source: {instruction['metadata']['instruction_source']}")
                    print(f"   Type: {instruction['metadata']['instruction_type']}")
                    print(f"   Complexity: {instruction['metadata']['claude_complexity_level']} (score: {instruction['metadata']['claude_complexity_score']})")
                    print(f"   Operators: {', '.join(instruction['input']['operators'][:3])}")

        except Exception as e:
            print(f"⚠️  Could not load sample instructions: {e}")

        # Final statistics from fast generator
        success_rate = (stats['successful_generations'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
        instructions_per_dag = stats['successful_generations'] / (stats['total_processed'] - stats['failed_generations']) if (stats['total_processed'] - stats['failed_generations']) > 0 else 0

        print(f"\n📊 Final Statistics:")
        print(f"✅ Total instructions: {stats['successful_generations']}")
        print(f"📈 DAGs processed: {stats['total_processed']}")
        print(f"🎯 Success rate: {stats['total_processed'] - stats['failed_generations']}/{stats['total_processed']} ({success_rate:.1f}%)")
        print(f"📊 Instructions per successful DAG: {instructions_per_dag:.1f}")
        print(f"🔧 Complexity distribution: {stats['complexity_distribution']}")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"⚡ Speed: {stats['total_processed']/elapsed_time:.1f} DAGs/second")
        
        if stats['failed_generations'] > 0:
            print(f"\n⚠️  {stats['failed_generations']} DAGs failed to generate instructions")
            print("Common failure reasons:")
            print("  • JSON parsing errors from Claude API")
            print("  • API timeouts or rate limits")
            print("  • Complex DAGs exceeding token limits")
            print("  • Network connectivity issues")

        stats_file = output_file.replace('.jsonl', '_stats.json')
        print(f"\n✅ Instructions saved to {output_file}")
        print(f"📊 Statistics saved to {stats_file}")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Generation interrupted by user")
        print("⚠️  Fast generator may have saved partial results")

        # Check if partial results exist
        if Path(output_file).exists():
            print(f"💾 Partial results available in {output_file}")

        return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
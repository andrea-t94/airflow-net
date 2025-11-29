# Dataset Statistics

Analysis of `datasets/processed/airflow_instructions.jsonl`

## Overview

Total samples: **6,738**

## Token Distribution

| Metric | Instruction | Output | Total |
|--------|------------|--------|-------|
| **Mean** | 69.79 | 1,048.49 | 1,118.28 |
| **Median** | 71 | 837 | 907 |
| **Min** | 37 | 110 | 157 |
| **Max** | 108 | 7,700 | 7,777 |
| **50th percentile** | 71 | 837 | 907 |
| **75th percentile** | 78 | 1,389 | 1,466 |
| **90th percentile** | 83 | 2,088 | 2,163 |
| **95th percentile** | 86 | 2,475 | 2,553 |
| **99th percentile** | 93 | 3,145 | 3,215 |
| **Total tokens** | 470,216 | 7,064,742 | 7,534,958 |

## Key Findings

- Instructions are concise and consistent (~70 tokens average)
- Outputs are approximately **15x longer** than instructions
- 90% of samples have total length under 2,163 tokens
- Total dataset contains ~7.5M tokens

*Note: Token counts use character-based approximation (~4 chars/token)*

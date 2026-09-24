"""Robustness benchmark on degraded audio (wrapper around ``ml.evaluation.robustness``).

python scripts/benchmark_robustness.py --checkpoint ml/checkpoints/guitar/best.pt --split test
"""

from ml.evaluation.robustness import main

if __name__ == "__main__":
    main()

"""Evaluate a checkpoint (wrapper around ``ml.evaluation.evaluate``).

python scripts/evaluate.py --checkpoint ml/checkpoints/guitar/best.pt --split test --plot
"""

from ml.evaluation.evaluate import main

if __name__ == "__main__":
    main()

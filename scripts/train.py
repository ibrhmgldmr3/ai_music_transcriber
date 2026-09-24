"""Train a transcription model (wrapper around ``ml.training.train``).

python scripts/train.py --config ml/configs/guitar.yaml
python scripts/train.py --config ml/configs/guitar.yaml --set training.epochs=20 model.rnn_type=gru
"""

from ml.training.train import main

if __name__ == "__main__":
    main()

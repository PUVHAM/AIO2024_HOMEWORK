import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from credit_card_fraud_detection import run_experiment

if __name__ == "__main__":
    run_experiment("Sentiment")
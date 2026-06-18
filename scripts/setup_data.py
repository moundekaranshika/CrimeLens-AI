"""Generate synthetic dataset and train ML models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import generate_synthetic_dataset, get_data_path, preprocess_data
from utils.prediction_utils import save_models


def main() -> None:
    """Generate dataset and train models."""
    data_path = get_data_path()
    print(f"Generating dataset at {data_path}...")
    df = generate_synthetic_dataset(n_records=5000)
    df.to_csv(data_path, index=False)
    print(f"Created {len(df)} records.")

    df = preprocess_data(df)
    print("Training ML models...")
    save_models(df)
    print("Models saved to models/")
    print("Done.")


if __name__ == "__main__":
    main()

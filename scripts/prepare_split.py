import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = ("normal", "agglutination")
TRAIN_RATIO = 0.8
VAL_RATIO = 1 - TRAIN_RATIO
RANDOM_SEED = 42

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--dataset-info-path", type=str, required=True, help="Path to the dataset info CSV file. Need columns 'processed_image_path' and 'label'.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory. Will create 'train.csv' and 'val.csv' in this directory.")
    
    return parser.parse_args()


def report(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Report the split results."""
    print(f"Train:")
    for label in LABELS:
        print(f"{label}: {train_df[train_df['label'] == label].shape[0]}")
    print()
    print("Val:")
    for label in LABELS:
        print(f"{label}: {val_df[val_df['label'] == label].shape[0]}")


def main() -> None:
    args = parse_args()
    dataset_info_path = Path(args.dataset_info_path)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(dataset_info_path)

    train_records = []
    val_records = []

    # Split dataset into train and validation sets for each label
    for label in LABELS:
        image_paths = df[df["label"] == label]["processed_image_path"].tolist()
        labels = [label] * len(image_paths)
        train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(image_paths, labels, \
                                                                                        test_size=VAL_RATIO, \
                                                                                        random_state=RANDOM_SEED, \
                                                                                        stratify=labels)

        for image_path, train_label in zip(train_image_paths, train_labels):
            train_records.append({"image_path": image_path.replace("\\", "/"), "label": train_label})

        for image_path, val_label in zip(val_image_paths, val_labels):
            val_records.append({"image_path": image_path.replace("\\", "/"), "label": val_label})


    output_dir.mkdir(parents=True, exist_ok=True)
    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)

    report(df_train, df_val)


if __name__ == "__main__":
    main()
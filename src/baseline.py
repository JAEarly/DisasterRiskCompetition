from utils import create_submission
import pandas as pd


if __name__ == "__main__":
    # Get training set class distribution from verified training labels
    class_distribution = pd.read_csv('../data/train_labels.csv').groupby(['verified']).mean()[-1:].values

    # Create submission based on training class distribution
    create_submission(class_distribution, "baseline")

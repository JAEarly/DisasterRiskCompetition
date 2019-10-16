from utils import create_submission

import pandas as pd
import os

# Get training set class distribution from verified training labels
class_distribution = pd.read_csv('../data/train_labels.csv').groupby(['verified']).mean()[-1:].values

# Create submission based on training class distribution
create_submission(class_distribution, "baseline")

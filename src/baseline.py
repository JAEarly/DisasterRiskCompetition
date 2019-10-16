import pandas as pd
import os

# Get training set class distribution from verified training labels
class_distribution = pd.read_csv('../data/train_labels.csv').groupby(['verified']).mean()[-1:].values

# Create submission based on training class distribution
submission = pd.read_csv('../data/submission_format.csv', index_col='id')
submission.loc[:, 0:5] = class_distribution

submission_folder = '../submissions'
if not os.path.exists(submission_folder):
    os.makedirs(submission_folder)
submission.to_csv(submission_folder + '/baseline.csv')

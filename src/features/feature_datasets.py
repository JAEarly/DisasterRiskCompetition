from torch.utils import data

from features import DatasetType, Datasets


class FeatureDatasets(Datasets):

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.feature_size = -1
        super().__init__()

    def create_datasets(self):
        train_features, train_labels = self.feature_extractor.extract(DatasetType.Train)
        validation_features, validation_labels = self.feature_extractor.extract(DatasetType.Validation)
        test_features, test_labels = self.feature_extractor.extract(DatasetType.Test)

        self.feature_size = train_features.shape[1]

        train_dataset = data.TensorDataset(train_features, train_labels)
        validation_dataset = data.TensorDataset(validation_features, validation_labels)
        test_dataset = data.TensorDataset(test_features, test_labels)

        return train_dataset, validation_dataset, test_dataset

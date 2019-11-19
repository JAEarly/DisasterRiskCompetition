"""Model evaluation script."""

import time

import features
import models
from features import DatasetType, FeatureDatasets
from models import FeatureTrainer

if __name__ == "__main__":
    _feature_extractor = features.AlexNet()
    _features_datasets = FeatureDatasets(_feature_extractor)
    _trainer = FeatureTrainer(_feature_extractor)
    _model = models.XGBModel(
        model_path=(
            "./models/"
            "grid_search_alexnet_xgb_smote/"
            "best.pth"
        ),
    )
    print("Running evaluation for", _feature_extractor.name, _model.name)

    use_softmax = True
    print("Training Set Results")
    train_acc, train_loss = _trainer.evaluate(
        _model,
        _features_datasets.get_loader(DatasetType.Train),
        apply_softmax=use_softmax,
        verbose=True
    )

    time.sleep(0.1)
    print("")
    print("Validation Set Results")
    val_acc, val_loss = _trainer.evaluate(
        _model,
        _features_datasets.get_loader(DatasetType.Validation),
        apply_softmax=use_softmax,
        verbose=True
    )

    time.sleep(0.1)
    print("")
    print("Test Set Results")
    test_acc, test_loss = _trainer.evaluate(
        _model,
        _features_datasets.get_loader(DatasetType.Test),
        apply_softmax=use_softmax,
        verbose=True
    )

    time.sleep(0.1)
    print("")
    print("Output for results.md")
    print(
        ("   {:.3f}   |" * 6).format(
            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss
        )
    )

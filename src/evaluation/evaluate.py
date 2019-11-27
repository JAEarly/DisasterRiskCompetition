"""Model evaluation script."""

import time

from torchvision import models as tv_models

import features
import models
import models.transfers as transfers
from features import DatasetType, FeatureDatasets, ImageDatasets
from models import FeatureTrainer
from models.cnn_model import PretrainedNNTrainer


def setup_feature_evaluation():
    feature_extractor = features.AlexNet()
    features_datasets = FeatureDatasets(feature_extractor)
    trainer = FeatureTrainer(feature_extractor)
    model = models.NNModel(
        models.BiggerNN,
        feature_extractor.feature_size,
        state_dict_path="./models/grid_search_alexnet_biggernn/best.pth",
        eval_mode=True,
    )

    # model = models.XGBModel(
    #     model_path="./models/grid_search_resnet_custom_smote_linearnn/best.pth"
    # )
    print("Running evaluation for", feature_extractor.name, model.name)
    return features_datasets, trainer, model


def setup_image_evaluation():
    image_datasets = ImageDatasets()
    trainer = PretrainedNNTrainer()
    model = models.PretrainedNNModel(
        tv_models.resnet152,
        transfers.final_layer_alteration_resnet,
        state_dict_path="./models/grid_search_resnet_custom/best.pth",
        eval_mode=True,
    )
    print("Running evaluation for", model.name)
    return image_datasets, trainer, model


if __name__ == "__main__":
    _datasets, _trainer, _model = setup_feature_evaluation()
    # _datasets, _trainer, _model = setup_image_evaluation()

    print("Training Set Results")
    train_acc, train_loss = _trainer.evaluate(
        _model,
        _datasets.get_loader(DatasetType.Train),
        verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Validation Set Results")
    val_acc, val_loss = _trainer.evaluate(
        _model,
        _datasets.get_loader(DatasetType.Validation),
        verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Test Set Results")
    test_acc, test_loss = _trainer.evaluate(
        _model,
        _datasets.get_loader(DatasetType.Test),
        verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Output for results.md")
    print(
        ("   {:.3f}   |" * 6).format(
            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss
        )
    )

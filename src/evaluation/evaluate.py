"""Model evaluation script."""

import time

from torchvision import models as tv_models
import os

import features
import models
import models.transfers as transfers
from features import (
    DatasetType,
    FeatureDatasets,
    ImageDatasets,
    BalanceMethod,
    SmoteExtractor,
)
from training import FeatureTrainer, PretrainedNNTrainer, Trainer
from models import ModelIterator


def setup_feature_evaluation():
    # Don't use SMOTE feature extractors, just usual normal version
    feature_extractor = features.ResNetCustomSMOTE()
    standard_datasets = FeatureDatasets(
        feature_extractor,
        balance_method=BalanceMethod.NoSample,
        override_balance_methods=False,
    )
    avg_datasets = FeatureDatasets(
        feature_extractor,
        balance_method=BalanceMethod.OverSample,
        override_balance_methods=True,
    )
    # custom_datasets = FeatureDatasets(
    #     feature_extractor,
    #     balance_method=BalanceMethod.CustomSample,
    #     override_balance_methods=True,
    # )

    model = models.NNModel(
        models.BiggerNN,
        feature_extractor.feature_size,
        state_dict_path="./models/oversample/grid_search_resnet_custom_smote2_biggernn_3/best.pth",
        eval_mode=True,
    )

    # model = models.XGBModel(
    #     model_path="./models/grid_search_resnet_custom_smote_xgb/best.pth"
    # )
    print("Running evaluation for", feature_extractor.name, model.name)
    return [standard_datasets, avg_datasets, None], model


def setup_image_evaluation():
    image_datasets = ImageDatasets()
    model = models.PretrainedNNModel(
        tv_models.resnet152,
        transfers.final_layer_alteration_resnet,
        state_dict_path="./models/grid_search_resnet_custom/best.pth",
        eval_mode=True,
    )
    print("Running evaluation for", model.name)
    return [image_datasets, None, None], model


def run_evaluation(datasets, model, avg_datasets=None, custom_datasets=None):
    results = []

    print("Training Set Results")
    train_acc, train_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Train), verbose=True,
    )
    results.append(train_acc)
    results.append(train_loss)

    time.sleep(0.1)
    print("")
    print("Validation Set Results")
    val_acc, val_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Validation), verbose=True,
    )
    results.append(val_acc)
    results.append(val_loss)

    time.sleep(0.1)
    print("")
    print("Test Set Results")
    test_acc, test_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Test), verbose=True,
    )
    results.append(test_acc)
    results.append(test_loss)

    if avg_datasets is not None:
        time.sleep(0.1)
        print("")
        print("Avg Test Set Results")
        avg_test_acc, avg_test_loss = Trainer.evaluate(
            model, avg_datasets.get_loader(DatasetType.Test), verbose=True,
        )
        results.append(avg_test_acc)
        results.append(avg_test_loss)

    if custom_datasets is not None:
        time.sleep(0.1)
        print("")
        print("Custom Test Set Results")
        cstm_test_acc, cstm_test_loss = Trainer.evaluate(
            model, custom_datasets.get_loader(DatasetType.Test), verbose=True,
        )
        results.append(cstm_test_acc)
        results.append(cstm_test_loss)

    time.sleep(0.1)
    print("")
    print("Output for results.md")
    print(
        ("   {:.3f}   |" * len(results)).format(*results)
    )


def evaluate_short(features_datasets, model):
    test_acc, test_loss = Trainer.evaluate(
        model, features_datasets.get_loader(DatasetType.Test)
    )
    print(("   {:.3f}   |" * 2).format(test_acc, test_loss))


def evaluate_all():
    print('Evaluating all')
    for model, datasets, desc in ModelIterator(
        balance_method=BalanceMethod.OverSample, override_balance_methods=True
    ):
        print(desc)
        evaluate_short(datasets, model)


if __name__ == "__main__":
    _datasets_list, _model = setup_feature_evaluation()
    # _datasets, _model = setup_image_evaluation()
    run_evaluation(_datasets_list[0], _model, avg_datasets=_datasets_list[1], custom_datasets=_datasets_list[2])

    #evaluate_all()

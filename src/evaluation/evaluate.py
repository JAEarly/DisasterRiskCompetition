"""Model evaluation script."""

import time

from torchvision import models as tv_models
import os

import features
import models
import models.transfers as transfers
from features import DatasetType, FeatureDatasets, ImageDatasets, BalanceMethod, SmoteExtractor
from training import FeatureTrainer, PretrainedNNTrainer, Trainer


def setup_feature_evaluation():
    # Don't use SMOTE feature extractors, just usual normal version
    feature_extractor = features.ResNetCustom()
    features_datasets = FeatureDatasets(
        feature_extractor,
        balance_method=BalanceMethod.AvgSample,
        override_balance_methods=True,
    )

    model = models.NNModel(
        models.LinearNN,
        feature_extractor.feature_size,
        state_dict_path="./models/grid_search_resnet_custom_linearnn/best.pth",
        eval_mode=True,
    )

    # model = models.XGBModel(
    #     model_path="./models/grid_search_resnet_custom_smote_xgb/best.pth"
    # )
    print("Running evaluation for", feature_extractor.name, model.name)
    return features_datasets, model


def setup_image_evaluation():
    image_datasets = ImageDatasets()
    model = models.PretrainedNNModel(
        tv_models.resnet152,
        transfers.final_layer_alteration_resnet,
        state_dict_path="./models/grid_search_resnet_custom/best.pth",
        eval_mode=True,
    )
    print("Running evaluation for", model.name)
    return image_datasets, model


def run_evaluation(datasets, model):
    print("Training Set Results")
    train_acc, train_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Train), verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Validation Set Results")
    val_acc, val_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Validation), verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Test Set Results")
    test_acc, test_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Test), verbose=True,
    )

    time.sleep(0.1)
    print("")
    print("Output for results.md")
    print(
        ("   {:.3f}   |" * 6).format(
            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss
        )
    )


def evaluate_short(features_datasets, model):
    test_acc, test_loss = Trainer.evaluate(
        model, features_datasets.get_loader(DatasetType.Test)
    )
    print(("   {:.3f}   |" * 2).format(test_acc, test_loss))


def evaluate_all():
    feature_extractors = [features.AlexNet(), features.AlexNetSMOTE(), features.AlexNetCustom(), features.AlexNetCustomSMOTE(),
                          features.ResNet(), features.ResNetSMOTE(), features.ResNetCustom(), features.ResNetCustomSMOTE()]

    for feature_extractor in feature_extractors:
        features_datasets = FeatureDatasets(
            feature_extractor,
            balance_method=BalanceMethod.AvgSample,
            override_balance_methods=True,
        )
        fe_name = feature_extractor.name
        if issubclass(type(feature_extractor), SmoteExtractor):
            fe_name += "_smote"

        # NN Models
        nn_models = [(models.LinearNN, "linearnn"), (models.BiggerNN, "biggernn")]
        for nn_model, model_name in nn_models:
            try:

                print(fe_name, model_name)
                model = models.NNModel(
                    nn_model,
                    feature_extractor.feature_size,
                    state_dict_path="./models/grid_search_" + fe_name + "_" + model_name + "/best.pth",
                    eval_mode=True,
                )
                evaluate_short(features_datasets, model)
            except:
                print("Not found")

        # XGB Model
        try:
            print(fe_name, "xgb")
            model = models.XGBModel(
                model_path="./models/grid_search_" + fe_name + "_xgb/best.pth"
            )
            evaluate_short(features_datasets, model)
        except:
            print("Not found")


if __name__ == "__main__":
    # _datasets, _model = setup_feature_evaluation()
    # _datasets, _model = setup_image_evaluation()

    # run_evaluation(_datasets, _model)
    evaluate_all()

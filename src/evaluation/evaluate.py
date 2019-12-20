"""Model evaluation script."""

import time

import os
from torchvision import models as tv_models

import features
import models
import models.transfers as transfers
from features import (
    DatasetType,
    FeatureDatasets,
    ImageDatasets,
)
from models import ModelIterator, EnsembleModel
from training import Trainer


def setup_feature_evaluation():
    # Don't use SMOTE feature extractors, just usual normal version
    feature_extractor = features.ResNetCustom()
    datasets = FeatureDatasets(feature_extractor)

    model = models.NNModel(
        models.LinearNN,
        feature_extractor.feature_size,
        state_dict_path="./models/verified/grid_search_resnet_custom_linearnn_14/best.pth",
        eval_mode=True,
    )

    # model = models.XGBModel(
    #     model_path="./models/verified/grid_search_resnet_custom_xgb_9/best.pth"
    # )

    # model = models.SVMModel(
    #     model_path="./models/verified/grid_search_resnet_custom_svm_5/best.pth"
    # )

    print("Running evaluation for", feature_extractor.name, model.name)
    return datasets, model


def setup_image_evaluation():
    image_datasets = ImageDatasets()
    model = models.PretrainedNNModel(
        tv_models.resnet152,
        transfers.final_layer_alteration_resnet,
        state_dict_path="./models/verified/grid_search_resnet_custom_3/all/images_resnet152_2019-12-17_06:23:56.pth",
        eval_mode=True,
    )
    print("Running evaluation for", model.name)
    return image_datasets, model


def setup_ensemble_evaluation():
    # Don't use SMOTE feature extractors, just usual normal version
    feature_extractor = features.ResNetCustom()
    datasets = FeatureDatasets(feature_extractor)

    name = "resnet_custom_linearnn_all_2"
    num_models = 4

    base_models = []
    for _ in range(num_models):
        model = models.NNModel(
            models.LinearNN,
            feature_extractor.feature_size,
        )
        # model = models.XGBModel()
        base_models.append(model)
    ensemble_model = EnsembleModel(base_models, name, load=True)

    print("Running evaluation for", feature_extractor.name, ensemble_model.name)
    return datasets, ensemble_model


def setup_ensemble_image_evaluation():
    image_datasets = ImageDatasets()

    name = "resnet_custom_3/best"
    num_models = 4

    base_models = []
    for _ in range(num_models):
        model = models.PretrainedNNModel(
            tv_models.resnet152,
            transfers.final_layer_alteration_resnet,
            eval_mode=True,
        )
        base_models.append(model)
    ensemble_model = EnsembleModel(base_models, name, load=True)

    print("Running ensemble image evaluation for", ensemble_model.name)
    return image_datasets, ensemble_model


def run_evaluation(datasets, model, verbose=True):
    results = []

    if verbose:
        print("Training Set Results")
    train_acc, train_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Train), verbose=verbose,
    )
    results.append(train_acc)
    results.append(train_loss)

    time.sleep(0.1)
    if verbose:
        print("")
        print("Validation Set Results")
    val_acc, val_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Validation), verbose=verbose,
    )
    results.append(val_acc)
    results.append(val_loss)

    time.sleep(0.1)
    if verbose:
        print("")
        print("Test Set Results")
    test_acc, test_loss = Trainer.evaluate(
        model, datasets.get_loader(DatasetType.Test), verbose=verbose,
    )
    results.append(test_acc)
    results.append(test_loss)

    time.sleep(0.1)
    if verbose:
        print("")
        print("Output for results.md")
        print(("   {:.3f}   |" * len(results)).format(*results))

    return results


def evaluate_all():
    print("Evaluating all")
    for model, datasets, desc in ModelIterator():
        print(desc)
        results = run_evaluation(datasets, model, verbose=False)
        print(("   {:.3f}   |" * len(results)).format(*results))


def evaluate_all_within_class_feature():
    base_dir = "./models/transfer/grid_search_resnet_custom_xgb/"
    all_dir = base_dir + "all/"
    best_filepath = base_dir + "best.pth"
    filepaths = [best_filepath]
    for filename in sorted(os.listdir(all_dir)):
        filepaths.append(all_dir + filename)

    feature_extractor = features.ResNetCustom(save_dir="./models/features/transfer/")
    datasets = FeatureDatasets(feature_extractor)

    filename_len = max(len(filepaths[1].split("/")[-1]) + 1, len("best.pth "))
    print(" " * filename_len + "| Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |")
    for model_path in filepaths:
        if os.path.exists(model_path):
            # model = models.NNModel(
            #     models.BiggerNN,
            #     feature_extractor.feature_size,
            #     state_dict_path=model_path,
            #     eval_mode=True,
            # )
            model = models.XGBModel(
                model_path=model_path
            )
            results = run_evaluation(datasets, model, verbose=False)
            print(("{:" + str(filename_len) + "s}").format(model_path.split("/")[-1]) + "|" + ("   {:.3f}   |" * len(results)).format(*results))
        else:
            print(("{:" + str(filename_len) + "s}").format(model_path.split("/")[-1]) + "| Missing")


def evaluate_all_within_class_image():
    base_dir = "./models/verified/grid_search_resnet_custom_2/"
    all_dir = base_dir + "all/"
    best_filepath = base_dir + "best.pth"
    filepaths = [best_filepath]
    for filename in sorted(os.listdir(all_dir)):
        filepaths.append(all_dir + filename)

    datasets = ImageDatasets()

    filename_len = max(len(filepaths[1].split("/")[-1]) + 1, len("best.pth "))
    print(" " * filename_len + "| Train Acc | Train LL  |  Val Acc  |   Val LL  |  Test Acc |  Test LL  |")
    for model_path in filepaths:
        if os.path.exists(model_path):
            model = models.PretrainedNNModel(
                tv_models.resnet152,
                transfers.final_layer_alteration_resnet,
                state_dict_path=model_path,
                eval_mode=True,
            )
            results = run_evaluation(datasets, model, verbose=False)
            print(("{:" + str(filename_len) + "s}").format(model_path.split("/")[-1]) + "|" + ("   {:.3f}   |" * len(results)).format(*results))
        else:
            print(("{:" + str(filename_len) + "s}").format(model_path.split("/")[-1]) + "| Missing")


if __name__ == "__main__":
    _datasets, _model = setup_feature_evaluation()
    # _datasets, _model = setup_image_evaluation()
    # _datasets, _model = setup_ensemble_evaluation()
    # _datasets, _model = setup_ensemble_image_evaluation()
    run_evaluation(_datasets, _model)

    # evaluate_all()
    # evaluate_all_within_class_image()

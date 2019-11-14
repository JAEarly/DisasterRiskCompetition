"""Grid search for hyper parameter tuning."""

import sys
import time
from typing import Type

import numpy as np
import os
from texttable import Texttable
from torch import nn

import features
import models
from features import BalanceMethod, FeatureExtractor
from models import ClassWeightMethod
from models.nn_model import NNTrainer, NNModel
from utils import (
    create_timestamp_str,
    create_dirs_if_not_found,
    DualLogger,
)

ROOT_DIR = "./models"


def run_nn_grid_search(
    nn_class: Type[nn.Module],
    feature_extractor: FeatureExtractor,
    repeats=3,
    grid_search_tag=None,
    **kwargs
) -> None:
    """
    Run a grid search for a neural network class.
    :param nn_class: Neural network class to use.
    :param feature_extractor: Feature extractor to use.
    :param repeats: Number of repeats for each parameter configuration.
    :param grid_search_tag: Name for this particular search instance.
    :param kwargs: Hyper parameter ranges.
    :return: None.
    """
    # Extract hyper parameter ranges
    epoch_range = _extract_range(kwargs, "epoch_range", [5])
    balance_methods = _extract_range(kwargs, "balance_methods", BalanceMethod.NoSample)
    class_weight_methods = _extract_range(kwargs, "class_weight_methods", [None])
    dropout_range = _extract_range(kwargs, "dropout_range", [0])

    # Overall results of grid search
    overall_accs = []
    overall_losses = []
    overall_params = []
    overall_models = []
    overall_filenames = []

    # Setup save dirs
    grid_search_tag = "grid_search_" + (
        create_timestamp_str() if grid_search_tag is None else grid_search_tag
    )
    save_dir = os.path.join(ROOT_DIR, grid_search_tag)
    create_dirs_if_not_found(save_dir)

    # Configure logging so output is saved to log file.
    log_path = os.path.join(save_dir, "log.txt")
    sys.stdout = DualLogger(log_path)

    # Create parameter configurations
    all_configs = (
        (num_epochs, balance_method, class_weight_method, dropout)
        for num_epochs in epoch_range
        for balance_method in balance_methods
        for class_weight_method in class_weight_methods
        for dropout in dropout_range
    )
    num_configs = (
        len(epoch_range)
        * len(balance_methods)
        * len(class_weight_methods)
        * len(dropout_range)
    )
    config_num = 1

    # Output parameter values
    print("         Epoch Range:", epoch_range)
    print("     Balance Methods:", [b.name for b in balance_methods])
    print("Class Weight Methods:", [c.name for c in class_weight_methods])
    print("       Dropout Range:", dropout_range)

    # Run grid search
    for (num_epochs, balance_method, class_weight_method, dropout) in all_configs:
        print("")
        print("-- Configuration " + str(config_num) + "/" + str(num_configs) + " --")
        print("         Num Epochs -", num_epochs)
        print("     Balance Method -", balance_method.name)
        print("Class weight method -", class_weight_method.name)
        print("            Dropout -", dropout)
        accs = []
        losses = []
        trained_models = []
        for r in range(repeats):
            print("Repeat " + str(r + 1) + "/" + str(repeats))
            trainer = NNTrainer(
                feature_extractor,
                num_epochs=num_epochs,
                balance_method=balance_method,
                class_weight_method=class_weight_method,
            )
            model = NNModel(nn_class, feature_extractor.feature_size, dropout=dropout)
            acc, loss = trainer.train(model)
            accs.append(acc)
            losses.append(loss)
            trained_models.append(model)
            time.sleep(0.1)

        best_model_idx = int(np.argmin(losses))
        best_acc = accs[best_model_idx]
        best_loss = losses[best_model_idx]
        best_model = trained_models[best_model_idx]

        overall_accs.append(best_acc)
        overall_losses.append(best_loss)
        overall_params.append(
            {
                "epochs": num_epochs,
                "balance_method": balance_method.name,
                "class_weight_method": class_weight_method.name,
                "dropout": dropout,
            }
        )
        overall_models.append(best_model)

        print("Best Loss:", best_loss)
        print(" Best Acc:", best_acc)
        print(" Avg Loss:", np.mean(losses))
        print("  Avg Acc:", np.mean(accs))

        file_name = _save_model(best_model, save_dir + "/all", feature_extractor.name)
        overall_filenames.append(file_name)
        config_num += 1

    # Sort best models
    print("")
    print("--- Final Results ---")
    results = zip(
        overall_losses, overall_accs, overall_params, overall_filenames, overall_models
    )
    sorted_results = sorted(results, key=lambda x: x[0])

    # Create results table
    table = Texttable(max_width=0)
    table.set_cols_align(["c", "c", "c", "c", "c"])
    rows = [["Pos", "Loss", "Acc", "Params", "Filename"]]
    for i, r in enumerate(sorted_results):
        rows.append([i + 1, *r[:4]])
    table.add_rows(rows)
    table_output = table.draw()
    print(table_output)

    # Save results
    results_file = os.path.join(save_dir, "results.txt")
    with open(results_file, "w") as file:
        file.write(table_output)

    # Save model
    _save_model(sorted_results[0][4], save_dir, feature_extractor.name, tag="best")


def _save_model(
    model: NNModel, save_dir: str, feature_extractor_name: str, tag=None
) -> str:
    """
    Save a neural network model.
    :param model: Model to save.
    :param save_dir: Path of save directory.
    :param feature_extractor_name: Name of feature extractor used.
    :param tag: Optional file tag.
    :return: Generated filename.
    """
    if tag is None:
        tag = create_timestamp_str()
    create_dirs_if_not_found(save_dir)
    file_name = feature_extractor_name + "_" + model.name + "_" + tag + ".pth"
    save_path = os.path.join(save_dir, file_name)
    model.save(save_path)
    return file_name


def _extract_range(ranges_dict, range_name, default_value):
    """
    Extract a range of hyper parameters from a dictionary.
    :param ranges_dict: Ranges dictionary to extract from.
    :param range_name: Key for range.
    :param default_value: Default value if range not found.
    :return: Extracted range.
    """
    if range_name in ranges_dict:
        return ranges_dict[range_name]
    return default_value


if __name__ == "__main__":
    run_nn_grid_search(
        models.LinearNN,
        features.ResNet(),
        repeats=3,
        grid_search_tag="resnet_linearnn",
        epoch_range=[1, 3, 5, 10, 15],
        balance_methods=[
            BalanceMethod.NoSample,
            BalanceMethod.AvgSample,
            BalanceMethod.OverSample,
        ],
        class_weight_methods=[
            ClassWeightMethod.Unweighted,
            ClassWeightMethod.SumBased,
            ClassWeightMethod.MaxBased,
        ],
        dropout_range=[0.0, 0.1, 0.25, 0.5],
    )

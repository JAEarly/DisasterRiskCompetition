"""Grid search for hyper parameter tuning."""

import sys
import time

import numpy as np
import os
from abc import ABC, abstractmethod
from texttable import Texttable

import features
import models
from features import BalanceMethod, FeatureExtractor
from models import ClassWeightMethod
from models.nn_model import NNTrainer, Model, NNModel
from utils import (
    create_timestamp_str,
    create_dirs_if_not_found,
    DualLogger,
)

ROOT_DIR = "./models"


class GridSearch(ABC):
    """Base class for grid search."""

    def __init__(self, feature_extractor: FeatureExtractor, tag=None, repeats=3):
        self.feature_extractor = feature_extractor
        self.grid_search_tag = "grid_search_" + (
            create_timestamp_str() if tag is None else tag
        )
        self.save_dir = os.path.join(ROOT_DIR, self.grid_search_tag)
        self.log_path = os.path.join(self.save_dir, "log.txt")
        self.repeats = repeats

        create_dirs_if_not_found(self.save_dir)

    def run(self, **kwargs) -> None:
        """
        Run a grid search.
        :param kwargs: Parameters to search over.
        :return: None.
        """
        # Overall results of grid search
        overall_accs = []
        overall_losses = []
        overall_params = []
        overall_models = []
        overall_filenames = []

        # Configure logging so output is saved to log file.
        sys.stdout = DualLogger(self.log_path)

        # Create all configs to search over
        all_configs = self._create_all_configs(kwargs)

        # Run grid search
        for i, config in enumerate(all_configs):
            print("")
            print(
                "-- Configuration " + str(i + 1) + "/" + str(len(all_configs)) + " --"
            )
            self._print_config(config)

            accs = []
            losses = []
            trained_models = []
            for r in range(self.repeats):
                print("Repeat " + str(r + 1) + "/" + str(self.repeats))
                acc, loss, model = self._train_model(config)
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
            overall_params.append(config)
            overall_models.append(best_model)

            print("Best Loss:", best_loss)
            print(" Best Acc:", best_acc)
            print(" Avg Loss:", np.mean(losses))
            print("  Avg Acc:", np.mean(accs))

            file_name = self._save_model(
                best_model, self.save_dir + "/all", self.feature_extractor.name
            )
            overall_filenames.append(file_name)

        # Sort best models
        print("")
        print("--- Final Results ---")
        results = zip(
            overall_losses,
            overall_accs,
            overall_params,
            overall_filenames,
            overall_models,
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
        results_file = os.path.join(self.save_dir, "results.txt")
        with open(results_file, "w") as file:
            file.write(table_output)

        # Save model
        self._save_model(
            sorted_results[0][4], self.save_dir, self.feature_extractor.name, tag="best"
        )

    @staticmethod
    def _save_model(
        model: Model, save_dir: str, feature_extractor_name: str, tag=None
    ) -> str:
        """
        Save a model.
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

    @staticmethod
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

    @abstractmethod
    def _create_all_configs(self, hyper_parameter_ranges):
        """
        Create the set of all possible configurations.
        :param hyper_parameter_ranges: Hyper parameter values for each parameter.
        :return: List of all possible parameters.
        """

    @abstractmethod
    def _print_config(self, config):
        """
        Print a config for this grid search.
        :param config: Config to print.
        :return: None.
        """

    @abstractmethod
    def _train_model(self, config) -> (float, float, Model):
        """
        Train and evaluate a model.
        :param config: Configuration of hyper parameters.
        :return: Accuracy, Loss and Model
        """


class NNGridSearch(GridSearch):
    """Grid search for NN models."""

    def __init__(
        self, nn_class, feature_extractor: FeatureExtractor, tag=None, repeats=3
    ):
        super().__init__(feature_extractor, tag, repeats)
        self.nn_class = nn_class

    def _create_all_configs(self, hyper_parameter_ranges):
        # Extract hyper parameter ranges
        epoch_range = self._extract_range(hyper_parameter_ranges, "epoch_range", [5])
        balance_methods = self._extract_range(
            hyper_parameter_ranges, "balance_methods", BalanceMethod.NoSample
        )
        class_weight_methods = self._extract_range(
            hyper_parameter_ranges, "class_weight_methods", [None]
        )
        dropout_range = self._extract_range(
            hyper_parameter_ranges, "dropout_range", [0]
        )

        # Output parameter values
        print("         Epoch Range:", epoch_range)
        print("     Balance Methods:", [b.name for b in balance_methods])
        print("Class Weight Methods:", [c.name for c in class_weight_methods])
        print("       Dropout Range:", dropout_range)

        # Create configs
        all_configs = (
            (num_epochs, balance_method, class_weight_method, dropout)
            for num_epochs in epoch_range
            for balance_method in balance_methods
            for class_weight_method in class_weight_methods
            for dropout in dropout_range
        )

        dict_configs = []
        for config in all_configs:
            dict_configs.append(
                {
                    "epochs": config[0],
                    "balance_method": config[1],
                    "class_weight_method": config[2],
                    "dropout": config[3],
                }
            )

        return dict_configs

    def _print_config(self, config):
        print("         Num Epochs -", config["epochs"])
        print("     Balance Method -", config["balance_method"].name)
        print("Class weight method -", config["class_weight_method"].name)
        print("            Dropout -", config["dropout"])

    def _train_model(self, config):
        trainer = NNTrainer(
            self.feature_extractor,
            num_epochs=config["epochs"],
            balance_method=config["balance_method"],
            class_weight_method=config["class_weight_method"],
        )
        model = NNModel(
            self.nn_class,
            self.feature_extractor.feature_size,
            dropout=config["dropout"],
        )
        acc, loss = trainer.train(model)
        return acc, loss, model


if __name__ == "__main__":
    grid_search = NNGridSearch(
        models.LinearNN,
        features.ResNetSMOTE(),
        repeats=3,
        tag="resnet_linearnn_smote_extended",
    )
    grid_search.run(
        epoch_range=[5, 10, 15],
        balance_methods=[BalanceMethod.NoSample,],
        class_weight_methods=[ClassWeightMethod.Unweighted,],
        dropout_range=[0.0],
    )

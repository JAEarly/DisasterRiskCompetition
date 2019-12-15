"""Grid search for hyper parameter tuning."""

import sys
import time

import numpy as np
import os
from abc import ABC, abstractmethod
from shutil import copyfile
from texttable import Texttable
from torchvision import models as tv_models
from features import SmoteType

import features
from features import BalanceMethod, FeatureExtractor
from model_manager import ModelManager
from models import transfers
import models
from models import (
    NNModel,
    Model,
    XGBModel,
    PretrainedNNModel,
)
from training import FeatureTrainer, ClassWeightMethod, PretrainedNNTrainer, NNTrainer
from utils import (
    create_timestamp_str,
    create_dirs_if_not_found,
    DualLogger,
)

ROOT_DIR = "./models/verified"


class GridSearch(ABC):
    """Base class for grid search."""

    def __init__(self, feature_name, tag=None, repeats=3):
        self.feature_name = feature_name
        self.grid_search_tag = "grid_search_" + (
            create_timestamp_str() if tag is None else tag
        )
        self.save_dir = os.path.join(ROOT_DIR, self.grid_search_tag)
        self.log_path = os.path.join(self.save_dir, "log.txt")
        self.repeats = repeats

        create_dirs_if_not_found(self.save_dir)
        print("Running", self.grid_search_tag)

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
                acc, loss, model = self._train_model(config, **kwargs)
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

            print("Best Loss:", best_loss)
            print(" Best Acc:", best_acc)
            print(" Avg Loss:", np.mean(losses))
            print("  Avg Acc:", np.mean(accs))

            file_name = self._save_model(best_model, self.save_dir + "/all")
            overall_filenames.append(file_name)

        # Sort best models
        print("")
        print("--- Final Results ---")
        results = zip(overall_losses, overall_accs, overall_params, overall_filenames,)
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

        # Save
        src_path = os.path.join(self.save_dir, "all", sorted_results[0][3])
        dst_path = os.path.join(self.save_dir, "best.pth")
        copyfile(src_path, dst_path)

        # Upload
        ModelManager().upload_model(dst_path)

    @staticmethod
    def _print_config(config):
        """
        Print a config for this grid search.
        :param config: Config to print.
        :return: None.
        """
        for key, value in config.items():
            print(key, "-", value)

    def _save_model(self, model: Model, save_dir: str, tag=None) -> str:
        """
        Save a model.
        :param model: Model to save.
        :param save_dir: Path of save directory.
        :param tag: Optional file tag.
        :return: Generated filename.
        """
        if tag is None:
            tag = create_timestamp_str()
        create_dirs_if_not_found(save_dir)
        file_name = self.feature_name + "_" + model.name + "_" + tag + ".pth"
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
    def _train_model(self, config, **kwargs) -> (float, float, Model):
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
        super().__init__(feature_extractor.name, tag=tag, repeats=repeats)
        self.feature_extractor = feature_extractor
        self.nn_class = nn_class

    def _create_all_configs(self, hyper_parameter_ranges):
        # Extract hyper parameter ranges
        epoch_range = self._extract_range(hyper_parameter_ranges, "epoch_range", [5])
        balance_methods = self._extract_range(
            hyper_parameter_ranges, "balance_methods", [BalanceMethod.NoSample]
        )
        class_weight_methods = self._extract_range(
            hyper_parameter_ranges,
            "class_weight_methods",
            [ClassWeightMethod.Unweighted],
        )
        dropout_range = self._extract_range(
            hyper_parameter_ranges, "dropout_range", [0]
        )
        smoothing_range = self._extract_range(
            hyper_parameter_ranges, "smoothing_range", [0]
        )

        # Output parameter values
        print("         Epoch Range:", epoch_range)
        print("     Balance Methods:", [b.name for b in balance_methods])
        print("Class Weight Methods:", [c.name for c in class_weight_methods])
        print("       Dropout Range:", dropout_range)
        print("     Smoothing Range:", smoothing_range)

        # Create configs
        all_configs = (
            (num_epochs, balance_method, class_weight_method, dropout, smoothing)
            for num_epochs in epoch_range
            for balance_method in balance_methods
            for class_weight_method in class_weight_methods
            for dropout in dropout_range
            for smoothing in smoothing_range
        )

        dict_configs = []
        for config in all_configs:
            dict_configs.append(
                {
                    "epochs": config[0],
                    "balance_method": config[1],
                    "class_weight_method": config[2],
                    "dropout": config[3],
                    "smoothing": config[4],
                }
            )

        return dict_configs

    def _train_model(self, config, **kwargs):
        trainer = NNTrainer(
            self.feature_extractor,
            num_epochs=config["epochs"],
            balance_method=config["balance_method"],
            class_weight_method=config["class_weight_method"],
            label_smoothing=config["smoothing"],
        )
        model = NNModel(
            self.nn_class,
            self.feature_extractor.feature_size,
            dropout=config["dropout"],
        )
        val_acc, val_loss = trainer.train(model)
        trainer.evaluate(model, trainer.feature_dataset.validation_loader)
        return val_acc, val_loss, model


class XGBGridSearch(GridSearch):
    def __init__(self, feature_extractor, tag=None, repeats=3):
        super().__init__(feature_extractor.name, tag=tag, repeats=repeats)
        self.feature_extractor = feature_extractor

    def _create_all_configs(self, hyper_parameter_ranges):
        # Extract hyper parameter ranges
        etas = self._extract_range(hyper_parameter_ranges, "etas", [0.3])
        gammas = self._extract_range(hyper_parameter_ranges, "gammas", [0])
        depths = self._extract_range(hyper_parameter_ranges, "depths", [5])
        c_weights = self._extract_range(hyper_parameter_ranges, "c_weights", [1])
        lambdas = self._extract_range(hyper_parameter_ranges, "lambdas", [1])
        rounds = self._extract_range(hyper_parameter_ranges, "num_rounds", [3])

        # Output parameter values
        print("             Etas:", etas)
        print("           Gammas:", gammas)
        print("           Depths:", depths)
        print("Min Child Weights:", c_weights)
        print("          Lambdas:", lambdas)
        print("           Rounds:", rounds)

        # Create configs
        all_configs = (
            (eta, gamma, depth, c_weight, reg_lambda, num_rounds)
            for eta in etas
            for gamma in gammas
            for depth in depths
            for c_weight in c_weights
            for reg_lambda in lambdas
            for num_rounds in rounds
        )

        dict_configs = []
        for config in all_configs:
            dict_configs.append(
                {
                    "eta": config[0],
                    "gamma": config[1],
                    "depth": config[2],
                    "c_weight": config[3],
                    "reg_lambda": config[4],
                    "num_rounds": config[5],
                }
            )

        return dict_configs

    def _train_model(self, config, **kwargs) -> (float, float, Model):
        trainer = FeatureTrainer(self.feature_extractor)
        model = XGBModel()
        config["pass_val"] = True
        val_acc, val_loss = trainer.train(model, **config)
        return val_acc, val_loss, model


class CNNGridSearch(GridSearch):
    def __init__(
        self,
        model_class,
        model_alteration_function,
        feature_name,
        root_dir="./data/processed/",
        num_classes=5,
        **kwargs
    ):
        super().__init__(feature_name, **kwargs)
        self.model_class = model_class
        self.model_alteration_function = model_alteration_function
        self.root_dir = root_dir
        self.num_classes = num_classes

    def _create_all_configs(self, hyper_parameter_ranges):
        # Extract hyper parameter ranges
        epoch_range = self._extract_range(hyper_parameter_ranges, "epoch_range", [5])
        class_weight_methods = self._extract_range(
            hyper_parameter_ranges,
            "class_weight_methods",
            [ClassWeightMethod.Unweighted],
        )

        # Output parameter values
        print("         Epoch Range:", epoch_range)
        print("Class Weight Methods:", [c.name for c in class_weight_methods])

        # Create configs
        all_configs = (
            (num_epochs, class_weight_method)
            for num_epochs in epoch_range
            for class_weight_method in class_weight_methods
        )

        dict_configs = []
        for config in all_configs:
            dict_configs.append(
                {"epochs": config[0], "class_weight_method": config[1],}
            )

        return dict_configs

    def _train_model(self, config, **kwargs) -> (float, float, Model):
        trainer = PretrainedNNTrainer(
            num_epochs=config["epochs"],
            class_weight_method=config["class_weight_method"],
            root_dir=self.root_dir,
        )
        model = PretrainedNNModel(
            self.model_class,
            self.model_alteration_function,
            num_classes=self.num_classes,
        )
        val_acc, val_loss = trainer.train(model)
        return val_acc, val_loss, model


class TransferGridSearch(CNNGridSearch):

    def __init__(self, model_class, model_alteration_function, feature_name, original_model_path,
                 root_dir="./data/processed/", num_classes=5, **kwargs):
        self.original_model_path = original_model_path
        super().__init__(model_class, model_alteration_function, feature_name, root_dir,
                         num_classes, **kwargs)

    def _train_model(self, config, **kwargs) -> (float, float, Model):
        trainer = PretrainedNNTrainer(
            num_epochs=config["epochs"],
            class_weight_method=config["class_weight_method"],
            root_dir=self.root_dir,
        )

        model = PretrainedNNModel.create_from_transfer(
            self.model_class,
            self.model_alteration_function,
            self.original_model_path,
            3,
            self.num_classes,
        )

        val_acc, val_loss = trainer.train(model)
        return val_acc, val_loss, model


if __name__ == "__main__":
    # grid_search = CNNGridSearch(
    #     tv_models.resnet152,
    #     transfers.final_layer_alteration_resnet,
    #     "images",
    #     tag="resnet_custom_2",
    #     repeats=1,
    #     root_dir="./data/processed_old/",
    #     num_classes=3
    # )
    # grid_search.run(
    #     epoch_range=[1, 2, 3, 4, 5],
    #     class_weight_methods=[
    #         ClassWeightMethod.Unweighted,
    #     ],
    # )

    grid_search = NNGridSearch(
        nn_class=models.LinearNN,
        feature_extractor=features.ResNetCustom(),
        tag="resnet_custom_linearnn_3",
        repeats=1,
    )
    grid_search.run(
        epoch_range=[2],
        class_weight_methods=[ClassWeightMethod.Unweighted],
        balance_methods=[BalanceMethod.NoSample],
        dropout_range=[0.0],
        smoothing_range=[0.0, 0.01, 0.05]
    )

    # grid_search = XGBGridSearch(
    #     feature_extractor=features.ResNetCustom(
    #         model_path="./models/transfer/grid_search_resnet_custom/best.pth",
    #         save_dir="./models/features/transfer/",
    #     ),
    #     tag="resnet_custom_xgb",
    #     repeats=1,
    # )
    # grid_search.run(num_rounds=[5, 10, 20, 30, 40],)

    # grid_search = TransferGridSearch(
    #     tv_models.resnet152,
    #     transfers.final_layer_alteration_resnet,
    #     "images",
    #     "./models/old_data/grid_search_resnet_custom_2/best.pth",
    #     tag="resnet_custom_3",
    #     repeats=2,
    #     root_dir="./data/processed/",
    #     num_classes=5
    # )
    # grid_search.run(
    #     epoch_range=[1, 2, 3],
    #     class_weight_methods=[
    #         ClassWeightMethod.Unweighted,
    #     ],
    # )

import sys

import numpy as np
import os
from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset, random_split, DataLoader, Subset

import features
import models
from evaluation import evaluate
from training import NNTrainer, BalanceMethod, ClassWeightMethod, FeatureTrainer
from utils import create_dirs_if_not_found, DualLogger
from model_manager import ModelManager

ROOT_DIR = "./models"


class KFoldTrainer(ABC):
    def __init__(self, k=10):
        self.k = k

    @abstractmethod
    def create_fresh_model(self):
        pass

    @abstractmethod
    def create_base_trainer(self, balance_method, override_balance_methods):
        pass

    @abstractmethod
    def train_model(self, trainer, model, train_loader, val_loader):
        pass


class FeatureKFoldTrainer(KFoldTrainer, ABC):
    def __init__(
        self, feature_extractor, save_tag, k=10, balance_method=BalanceMethod.NoSample, override_balance_methods=False
    ):
        super().__init__(k)
        self.feature_extractor = feature_extractor
        self.save_tag = "kfold_" + save_tag
        self.save_dir = os.path.join(ROOT_DIR, self.save_tag)
        self.log_path = os.path.join(self.save_dir, "log.txt")
        self.feature_trainer = self.create_base_trainer(balance_method, override_balance_methods)
        self.concat_dataset = self.combine_train_and_val_loaders()
        create_dirs_if_not_found(self.save_dir)
        print("Running", self.save_tag, "k=" + str(k))

    def combine_train_and_val_loaders(self) -> ConcatDataset:
        train_dataset = self.feature_trainer.feature_dataset.train_dataset
        val_dataset = self.feature_trainer.feature_dataset.validation_dataset
        return ConcatDataset([train_dataset, val_dataset])

    def train_kfold(self):
        accs = []
        losses = []
        trained_models = []

        sys.stdout = DualLogger(self.log_path)

        split_size = int(len(self.concat_dataset) / self.k)
        split_size = int(split_size)
        self.concat_dataset = Subset(self.concat_dataset, range(split_size * self.k))
        splits = random_split(self.concat_dataset, [split_size] * self.k)
        for i, val_dataset in enumerate(splits):
            print("")
            print("Repeat", str(i + 1) + "/" + str(self.k))
            training_splits = []
            for j, split in enumerate(splits):
                if i != j:
                    training_splits.append(split)
            train_dataset = ConcatDataset(training_splits)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.feature_trainer.feature_dataset.batch_size,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.feature_trainer.feature_dataset.batch_size
            )

            model = self.create_fresh_model()
            acc, loss = self.train_model(
                self.feature_trainer, model, train_loader, val_loader
            )

            accs.append(acc)
            losses.append(loss)
            trained_models.append(model)

        best_idx = int(np.argmin(losses))

        best_acc = accs[best_idx]
        best_loss = losses[best_idx]
        best_model = trained_models[best_idx]

        avg_acc = np.mean(accs)
        avg_loss = np.mean(losses)

        std_acc = np.std(accs)
        std_loss = np.std(losses)

        print("")
        print(" Best Acc:", best_acc)
        print("Best Loss:", best_loss)
        print("  Avg Acc:", avg_acc)
        print(" Avg Loss:", avg_loss)
        print("  Std Acc:", std_acc)
        print(" Std Loss:", std_loss)

        save_path = os.path.join(self.save_dir, "best.pth")
        best_model.save(save_path)
        ModelManager().upload_model(save_path)


class NNKFoldTrainer(FeatureKFoldTrainer):
    def __init__(self, feature_extractor, nn_model, save_tag, epochs, dropout, balance_method=BalanceMethod.NoSample, override_balance_methods=False, k=10):
        self.nn_model = nn_model
        self.epochs = epochs
        self.dropout = dropout
        super().__init__(feature_extractor, save_tag, balance_method=balance_method, override_balance_methods=override_balance_methods, k=k)

    def create_fresh_model(self):
        return models.NNModel(
            self.nn_model, self.feature_extractor.feature_size, dropout=self.dropout,
        )

    def create_base_trainer(self, balance_method, override_balance_methods):
        return NNTrainer(
            self.feature_extractor,
            num_epochs=self.epochs,
            balance_method=balance_method,
            class_weight_method=ClassWeightMethod.Unweighted,
            override_balance_methods=override_balance_methods
        )

    def train_model(self, trainer, model, train_loader, val_loader):
        return trainer.train(model, train_loader, val_loader)


class XGBKFoldTrainer(FeatureKFoldTrainer):
    def __init__(self, feature_extractor, save_tag, num_rounds, k=10):
        self.num_rounds = num_rounds
        super().__init__(feature_extractor, save_tag, k=k)

    def create_fresh_model(self):
        return models.XGBModel()

    def create_base_trainer(self, balance_method=BalanceMethod.AvgSample, override_balance_methods=True):
        return FeatureTrainer(self.feature_extractor)

    def train_model(self, trainer, model, train_loader, val_loader):
        return trainer.train(
            model, train_loader, val_loader, num_rounds=self.num_rounds, pass_val=True
        )


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustom()

    kfold_trainer = NNKFoldTrainer(
        _feature_extractor,
        models.LinearNN,
        save_tag="cstm_resnet_custom_linearnn",
        epochs=5,
        dropout=0,
        balance_method=BalanceMethod.CustomSample,
        override_balance_methods=True
    )

    # _num_rounds = 20
    # kfold_trainer = XGBKFoldTrainer(
    #     _feature_extractor, "resnet_custom_xgb", _num_rounds
    # )
    kfold_trainer.train_kfold()

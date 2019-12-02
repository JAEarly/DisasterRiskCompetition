import sys

import numpy as np
import os
from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset, random_split, DataLoader

import features
import models
from models import NNModel
from training import NNTrainer, BalanceMethod, ClassWeightMethod
from utils import create_dirs_if_not_found, DualLogger
from evaluation import evaluate

ROOT_DIR = "./models"


class KFoldTrainer(ABC):
    def __init__(self, k=10):
        self.k = k


class FeatureKFoldTrainer(KFoldTrainer, ABC):
    def __init__(self, feature_extractor, save_tag, k=10):
        super().__init__(k)
        self.feature_extractor = feature_extractor
        self.save_tag = "kfold_" + save_tag
        self.save_dir = os.path.join(ROOT_DIR, self.save_tag)
        self.log_path = os.path.join(self.save_dir, "log.txt")
        self.feature_trainer = self.create_base_trainer()
        self.concat_dataset = self.combine_train_and_val_loaders()
        create_dirs_if_not_found(self.save_dir)
        print("Running", self.save_tag, 'k=' + str(k))

    @abstractmethod
    def create_fresh_model(self):
        pass

    @abstractmethod
    def create_base_trainer(self):
        pass

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

            acc, loss = self.feature_trainer.train(model, train_loader, val_loader)
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

        best_model.save(os.path.join(self.save_dir, "best.pth"))

        print('')
        evaluate.run_evaluation(self.feature_trainer.feature_dataset, best_model)


class NNKFoldTrainer(FeatureKFoldTrainer):
    def __init__(self, feature_extractor, nn_model, save_tag, epochs, dropout, k=10):
        self.nn_model = nn_model
        self.epochs = epochs
        self.dropout = dropout
        super().__init__(feature_extractor, save_tag, k=k)

    def create_fresh_model(self):
        return NNModel(self.nn_model, self.feature_extractor.feature_size, dropout=self.dropout,)

    def create_base_trainer(self):
        return NNTrainer(
            self.feature_extractor,
            num_epochs=self.epochs,
            balance_method=BalanceMethod.NoSample,
            class_weight_method=ClassWeightMethod.Unweighted,
        )


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustom()
    _nn_model = models.BiggerNN
    _epochs = 3
    _dropout = 0.25
    kfold_trainer = NNKFoldTrainer(
        _feature_extractor, _nn_model, "resnet_custom_biggernn", _epochs, _dropout
    )
    kfold_trainer.train_kfold()

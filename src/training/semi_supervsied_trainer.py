import sys

import numpy as np
import os
from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset, random_split, DataLoader, Subset

import features
import models
from evaluation import evaluate
from training import NNTrainer, BalanceMethod, ClassWeightMethod, FeatureTrainer, Trainer
from utils import create_dirs_if_not_found, DualLogger
from model_manager import ModelManager
from features import PseudoFeatureDataset, DatasetType, ResNetCustom


ROOT_DIR = "./models/semisupervised"


class SemiSupervisedTrainer(ABC):

    @abstractmethod
    def create_fresh_model(self):
        pass

    @abstractmethod
    def create_base_trainer(self):
        pass

    @abstractmethod
    def train_model(self, trainer, model, train_loader, val_loader):
        pass


class FeatureSemiSupervisedTrainer(SemiSupervisedTrainer, ABC):

    def __init__(
        self, feature_extractor, save_tag
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.save_tag = "ss_" + save_tag
        self.save_dir = os.path.join(ROOT_DIR, self.save_tag)
        self.log_path = os.path.join(self.save_dir, "log.txt")
        self.feature_trainer = self.create_base_trainer()
        create_dirs_if_not_found(self.save_dir)
        create_dirs_if_not_found(self.save_dir + "/all")
        print("Running", self.save_tag)

    def train_semisupervised(self):
        training_losses = []
        pseudo_losses = []
        validation_losses = []
        trained_models = []

        train_dataset = self.feature_trainer.feature_dataset.train_dataset
        train_loader = self.feature_trainer.feature_dataset.train_loader
        val_loader = self.feature_trainer.feature_dataset.validation_loader
        sys.stdout = DualLogger(self.log_path)

        step = 0
        pseudo_dataset = None
        num_added = None
        prev_points = None
        while num_added is None or num_added > 0:
            print("")
            print("Step", str(step + 1))

            # Create training dataset with pseudo data included
            datasets = [train_dataset]
            if pseudo_dataset is not None:
                datasets.append(pseudo_dataset)
            train_with_pseudo_dataset = ConcatDataset(datasets)
            train_with_pseudo_loader = DataLoader(
                train_with_pseudo_dataset,
                batch_size=self.feature_trainer.feature_dataset.batch_size,
                shuffle=True,
            )

            model = self.create_fresh_model()
            _, val_loss = self.train_model(
                self.feature_trainer, model, train_with_pseudo_loader, val_loader
            )
            _, train_loss = Trainer.evaluate(model, train_loader)
            _, pseudo_loss = Trainer.evaluate(model, train_with_pseudo_loader)

            print('     Train Loss:', "{:.3f}".format(train_loss))
            print('    Pseudo Loss:', "{:.3f}".format(pseudo_loss))
            print('Validation Loss:', "{:.3f}".format(val_loss))

            training_losses.append(train_loss)
            pseudo_losses.append(pseudo_loss)
            validation_losses.append(val_loss)

            trained_models.append(model)
            save_path = os.path.join(self.save_dir + "/all", str(step) + ".pth")
            model.save(save_path)

            print('Creating new pseudo dataset')
            pseudo_dataset = PseudoFeatureDataset(model, self.feature_extractor.get_features_dir(DatasetType.Pseudo))
            if num_added is None:
                num_added = len(pseudo_dataset)
            else:
                num_added = len(pseudo_dataset) - prev_points
            prev_points = len(pseudo_dataset)
            if num_added < 0:
                num_added = 0
            print(num_added, 'datapoints added')

            step += 1

        best_idx = int(np.argmin(validation_losses))
        best_loss = validation_losses[best_idx]
        best_model = trained_models[best_idx]
        avg_loss = np.mean(validation_losses)
        std_loss = np.std(validation_losses)

        print("")
        print("Best Loss:", best_loss)
        print(" Avg Loss:", avg_loss)
        print(" Std Loss:", std_loss)
        print("")
        print("train_losses =", training_losses)
        print("pseudo_losses =", pseudo_losses)
        print("validation_losses =", validation_losses)

        save_path = os.path.join(self.save_dir, "best.pth")
        best_model.save(save_path)
        ModelManager().upload_model(save_path)


class NNSemiSupervisedTrainer(FeatureSemiSupervisedTrainer):
    def __init__(self, feature_extractor, nn_model, save_tag, epochs, dropout):
        self.nn_model = nn_model
        self.epochs = epochs
        self.dropout = dropout
        super().__init__(feature_extractor, save_tag)

    def create_fresh_model(self):
        return models.NNModel(
            self.nn_model, self.feature_extractor.feature_size, dropout=self.dropout,
        )

    def create_base_trainer(self):
        return NNTrainer(
            self.feature_extractor,
            num_epochs=self.epochs,
        )

    def train_model(self, trainer, model, train_loader, val_loader):
        return trainer.train(model, train_loader, val_loader)


if __name__ == "__main__":
    _feature_extractor = ResNetCustom()
    _trainer = NNSemiSupervisedTrainer(
        _feature_extractor,
        models.LinearNN,
        "resnet_custom_linearnn",
        epochs=5,
        dropout=0.4
    )
    _trainer.train_semisupervised()

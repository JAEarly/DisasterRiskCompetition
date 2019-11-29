from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset, random_split, DataLoader

import models
from models import NNModel
from training import FeatureTrainer, NNTrainer, BalanceMethod, ClassWeightMethod
import features
import numpy as np
from utils import create_dirs_if_not_found


class KFoldTrainer(ABC):
    def __init__(self, k=10):
        self.k = k


class FeatureKFoldTrainer(KFoldTrainer, ABC):
    def __init__(self, feature_extractor, k=10):
        super().__init__(k)
        self.feature_extractor = feature_extractor
        self.feature_trainer = self.create_base_trainer()
        self.concat_dataset = self.combine_train_and_val_loaders()

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
        models = []

        split_size = int(len(self.concat_dataset) / self.k)
        split_size = int(split_size)
        splits = random_split(self.concat_dataset, [split_size] * self.k)
        for i, val_dataset in enumerate(splits):
            print('')
            print('Repeat', str(i+1) + "/" + str(self.k))
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

            # TODO does this overwrite previous models?
            model = self.create_fresh_model()

            acc, loss = self.feature_trainer.train(model, train_loader, val_loader)
            accs.append(acc)
            losses.append(loss)
            models.append(model)

        best_idx = int(np.argmin(losses))

        best_acc = accs[best_idx]
        best_loss = losses[best_idx]
        best_model = models[best_idx]

        avg_acc = np.mean(accs)
        avg_loss = np.mean(losses)

        print(' Best Acc:', best_acc)
        print('Best Loss:', best_loss)
        print('  Avg Acc:', avg_acc)
        print(' Avg Loss:', avg_loss)


        create_dirs_if_not_found("./models/kfold_test")
        best_model.save("./models/kfold_test/best.pth")


class NNKFoldTrainer(FeatureKFoldTrainer):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)

    def create_fresh_model(self):
        return NNModel(models.LinearNN, self.feature_extractor.feature_size, dropout=0, )

    def create_base_trainer(self):
        return NNTrainer(
            self.feature_extractor,
            num_epochs=5,
            balance_method=BalanceMethod.NoSample,
            class_weight_method=ClassWeightMethod.Unweighted,
        )


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustom()
    kfold_trainer = NNKFoldTrainer(_feature_extractor)
    kfold_trainer.train_kfold()

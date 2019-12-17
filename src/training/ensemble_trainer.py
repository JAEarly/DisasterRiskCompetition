import numpy as np
import os
from texttable import Texttable
from torch.utils.data import ConcatDataset, random_split, DataLoader, Subset

import features
from features import FeatureExtractor
from models import EnsembleModel, NNModel, LinearNN
from training import NNTrainer, FeatureTrainer
from training import Trainer
from utils import create_dirs_if_not_found


class EnsembleTrainer(FeatureTrainer):
    def __init__(self, base_trainer: Trainer, feature_extractor: FeatureExtractor):
        super().__init__(feature_extractor)
        self.base_trainer = base_trainer

    def train(
        self,
        model: EnsembleModel,
        train_loader=None,
        validation_loader=None,
        kfold=True,
        verbose=False,
        **kwargs
    ) -> (float, float):
        k = len(model.models)

        if train_loader is None:
            train_loader = self.feature_dataset.train_loader
        if validation_loader is None:
            validation_loader = self.feature_dataset.validation_loader

        splits = None
        if k == 1:
            kfold = False
        if kfold:
            concat_dataset = ConcatDataset(
                [
                    self.feature_dataset.train_dataset,
                    self.feature_dataset.validation_dataset,
                    self.feature_dataset.test_dataset
                ]
            )
            split_size = int(len(concat_dataset) / k)
            split_size = int(split_size)
            concat_dataset = Subset(concat_dataset, range(split_size * k))
            splits = random_split(concat_dataset, [split_size] * k)

        self.save_dir = "./models/ensemble/" + model.name + "_" + str(len(model.models))
        print("Running ensemble trainer on", model.name)
        print("Kfold:", kfold, "\n")
        accs = []
        losses = []
        print("Training base models")
        for i, base_model in enumerate(model.models):
            print("Model", str(i + 1) + "/" + str(len(model.models)))

            if kfold:
                training_splits = []
                for j, split in enumerate(splits):
                    if i != j:
                        training_splits.append(split)
                train_dataset = ConcatDataset(training_splits)
                train_loader = DataLoader(
                    train_dataset, batch_size=self.feature_dataset.batch_size,
                )
                validation_loader = DataLoader(
                    splits[i], batch_size=self.feature_dataset.batch_size
                )

            acc, loss = self.base_trainer.train(
                base_model, train_loader, validation_loader, **kwargs
            )
            accs.append(acc)
            losses.append(loss)

        best_idx = int(np.argmin(losses))
        worst_idx = int(np.argmax(losses))
        acc, loss = Trainer.evaluate(model, self.feature_dataset.validation_loader)

        table = Texttable(max_width=0)
        table.set_cols_align(["c", "c", "c"])
        rows = [
            ["", "Acc", "Loss"],
            ["Best", accs[best_idx], losses[best_idx]],
            ["Avg", np.mean(accs), np.mean(losses)],
            ["Worst", accs[worst_idx], losses[worst_idx]],
            ["Std", np.std(accs), np.std(losses)],
            ["Ensemble", acc, loss],
        ]
        table.add_rows(rows)
        table_output = table.draw()
        if verbose:
            print(table_output)

        return acc, loss


def run_ensemble_trainer(
    num_base_models, tag, apply_softmax, base_trainer, base_model_func, kfold=True, verbose=False
):
    feature_extractor = base_trainer.feature_dataset.feature_extractor
    base_models = []
    for _ in range(num_base_models):
        base_model = base_model_func()
        base_models.append(base_model)
    ensemble_model = EnsembleModel(base_models, tag, apply_softmax)

    trainer = EnsembleTrainer(base_trainer, feature_extractor)
    acc, loss = trainer.train(ensemble_model, kfold=kfold, verbose=verbose)
    return ensemble_model, acc, loss


def run_ensemble_trainer_repeated(
    k, tag, apply_softmax, base_trainer, base_model_func, repeats=3, kfold=True, verbose=False
):
    models = []
    accs = []
    losses = []
    for r in range(repeats):
        print("Repeat", str(r + 1) + "/" + str(repeats))
        model, acc, loss = run_ensemble_trainer(
            k, tag, apply_softmax, base_trainer, base_model_func, kfold=kfold, verbose=verbose
        )
        models.append(model)
        accs.append(acc)
        losses.append(loss)
    return models, accs, losses


def run_ensemble_trainer_iterative(
    k_max, tag, apply_softmax, base_trainer, base_model_func, repeats=3, kfold=True, verbose=False
):
    accs = []
    losses = []
    for k in range(1, k_max + 1):
        _, repeat_accs, repeat_losses = run_ensemble_trainer_repeated(
            k, tag, apply_softmax, base_trainer, base_model_func, repeats=repeats, kfold=kfold, verbose=verbose
        )
        accs.append(np.mean(repeat_accs))
        losses.append(np.mean(repeat_losses))
    return accs, losses


def grid_search_ensemble_trainer(
    k, tag, apply_softmax, base_trainer, base_model_func, repeats=3, kfold=True, verbose=False
):
    models, accs, losses = run_ensemble_trainer_repeated(
        k, tag, apply_softmax, base_trainer, base_model_func, repeats=repeats, kfold=kfold, verbose=verbose
    )
    best_idx = int(np.argmin(losses))
    best_model = models[best_idx]
    best_acc = accs[best_idx]
    best_loss = losses[best_idx]
    print("")
    print(" Best Acc:", best_acc)
    print("Best Loss:", best_loss)

    models_dir = best_model.get_models_dir()
    create_dirs_if_not_found(models_dir)
    best_model.save(models_dir)
    results_file = os.path.join(models_dir, "results.txt")
    with open(results_file, "w") as file:
        file.write(" Best Acc: " + str(best_acc) + "\n")
        file.write("Best Loss: " + str(best_loss) + "\n")

    return best_model, best_acc, best_loss


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustom()
    _base_trainer = NNTrainer(_feature_extractor, num_epochs=6,)

    def _base_model_func():
        NNModel(LinearNN, _feature_extractor.feature_size, dropout=0.4)

    _num_base_models = 4
    _tag = "resnet_custom_linearnn_all"
    _apply_softmax = True

    grid_search_ensemble_trainer(4, _tag, _apply_softmax, _base_trainer, _base_model_func)

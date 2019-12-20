import numpy as np
import os
from texttable import Texttable
from torch.utils.data import ConcatDataset, random_split, DataLoader, Subset


import features
from features import FeatureExtractor
from models import EnsembleModel, NNModel, LinearNN, XGBModel, PretrainedNNModel
from training import NNTrainer, FeatureTrainer, ImageTrainer
from training import Trainer
from utils import create_dirs_if_not_found
import torch
from torchvision import models as tv_models
from models import transfers
from training.pretrained_nn_trainer import PretrainedNNTrainer
from distutils.dir_util import copy_tree


# TODO refactor ensemble feature and image trainers
class EnsembleFeatureTrainer(FeatureTrainer):
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
        **kwargs,
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
                    self.feature_dataset.test_dataset,
                ]
            )
            split_size = int(len(concat_dataset) / k)
            split_size = int(split_size)
            concat_dataset = Subset(concat_dataset, range(split_size * k))
            splits = random_split(concat_dataset, [split_size] * k)

        self.save_dir = "./models/ensemble/" + model.name + "/" + str(len(model.models))
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


class EnsemblePretrainedTrainer(ImageTrainer):

    def __init__(self, base_trainer):
        super().__init__()
        self.base_trainer = base_trainer

    def train(
        self,
        model: EnsembleModel,
        train_loader=None,
        validation_loader=None,
        kfold=True,
        verbose=False,
        **kwargs,
    ) -> (float, float):
        k = len(model.models)

        if train_loader is None:
            train_loader = self.image_datasets.train_loader
        if validation_loader is None:
            validation_loader = self.image_datasets.validation_loader

        splits = None
        if k == 1:
            kfold = False
        if kfold:
            concat_dataset = ConcatDataset(
                [
                    self.image_datasets.train_dataset,
                    self.image_datasets.validation_dataset,
                    # self.image_datasets.test_dataset,
                ]
            )
            split_size = int(len(concat_dataset) / k)
            split_size = int(split_size)
            concat_dataset = Subset(concat_dataset, range(split_size * k))
            splits = random_split(concat_dataset, [split_size] * k)

        print("Running ensemble trainer on", model.name)
        print("Kfold:", kfold, "\n")
        val_accs = []
        val_losses = []
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
                    train_dataset, batch_size=self.image_datasets.batch_size,
                )
                validation_loader = DataLoader(
                    splits[i], batch_size=self.image_datasets.batch_size
                )

            acc, loss = self.base_trainer.train(
                base_model, train_loader, validation_loader, **kwargs
            )
            val_accs.append(acc)
            val_losses.append(loss)

        best_idx = int(np.argmin(val_losses))
        worst_idx = int(np.argmax(val_losses))
        acc, loss = Trainer.evaluate(model, self.image_datasets.validation_loader)
        test_acc, test_loss = Trainer.evaluate(model, self.image_datasets.validation_loader)

        table = Texttable(max_width=0)
        table.set_cols_align(["c", "c", "c"])
        rows = [
            ["", "Acc", "Loss"],
            ["Best", val_accs[best_idx], val_losses[best_idx]],
            ["Avg", np.mean(val_accs), np.mean(val_losses)],
            ["Worst", val_accs[worst_idx], val_losses[worst_idx]],
            ["Std", np.std(val_accs), np.std(val_losses)],
            ["Ensemble Val", acc, loss],
            ["Ensemble Test", test_acc, test_loss],
        ]
        table.add_rows(rows)
        table_output = table.draw()
        if verbose:
            print(table_output)

        return model, acc, loss, test_acc, test_loss


def run_ensemble_trainer(
    num_base_models,
    tag,
    apply_softmax,
    base_trainer,
    base_model_func,
    kfold=True,
    verbose=False,
    **kwargs,
):
    base_models = []
    for _ in range(num_base_models):
        base_model = base_model_func()
        base_models.append(base_model)
    ensemble_model = EnsembleModel(base_models, tag, apply_softmax)

    trainer = EnsemblePretrainedTrainer(base_trainer)
    return trainer.train(ensemble_model, kfold=kfold, verbose=verbose, **kwargs)


def run_ensemble_trainer_repeated(
    k,
    tag,
    apply_softmax,
    base_trainer,
    base_model_func,
    repeats=3,
    kfold=True,
    verbose=False,
    **kwargs,
):
    accs = []
    losses = []
    test_accs = []
    test_losses = []
    best_model = None
    for r in range(repeats):
        print("Repeat", str(r + 1) + "/" + str(repeats))
        model, acc, loss, test_acc, test_loss = run_ensemble_trainer(
            k,
            tag,
            apply_softmax,
            base_trainer,
            base_model_func,
            kfold=kfold,
            verbose=verbose,
            **kwargs,
        )
        accs.append(acc)
        losses.append(loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        if best_model is None or loss < min(losses):
            best_model = model
    best_idx = int(np.argmin(losses))
    best_acc = accs[best_idx]
    best_loss = losses[best_idx]

    table = Texttable(max_width=0)
    table.set_cols_align(["c", "c", "c"])
    rows = [
        ["", "Acc", "Loss"],
        ["Best", accs[best_idx], losses[best_idx]],
        ["Avg", np.mean(accs), np.mean(losses)],
        ["Std", np.std(accs), np.std(losses)],
        ["Ensemble Val", best_acc, best_loss],
        ["Ensemble Test", test_accs[best_idx], test_losses[best_idx]],
    ]
    table.add_rows(rows)
    table_output = table.draw()
    save_dir = "./models/ensemble/" + best_model.name + "/" + str(len(best_model.models))
    save_model_and_results(save_dir, best_model, table_output)

    return best_acc, best_loss, test_accs[best_idx], test_losses[best_idx]


def run_ensemble_trainer_iterative(
    k_max,
    tag,
    apply_softmax,
    base_trainer,
    base_model_func,
    k_min=1,
    repeats=3,
    kfold=True,
    verbose=False,
    **kwargs,
):
    print("RUNNING ITERATIVE TRAINING FOR k in", k_min, "to", k_max)
    accs = []
    losses = []
    test_accs = []
    test_losses = []
    for k in range(k_min, k_max + 1):
        print("-- STEP k =", k, "--")
        acc, loss, test_acc, test_loss = run_ensemble_trainer_repeated(
            k,
            tag,
            apply_softmax,
            base_trainer,
            base_model_func,
            repeats=repeats,
            kfold=kfold,
            verbose=verbose,
            **kwargs,
        )
        accs.append(acc)
        losses.append(loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    return accs, losses, test_accs, test_losses


def grid_search_ensemble_trainer(
    k,
    tag,
    apply_softmax,
    base_trainer,
    base_model_func,
    repeats=3,
    kfold=True,
    verbose=False,
    iterative=True,
    **kwargs,
):
    search_func = (
        run_ensemble_trainer_iterative if iterative else run_ensemble_trainer_repeated
    )
    accs, losses, test_accs, test_losses = search_func(
        k,
        tag,
        apply_softmax,
        base_trainer,
        base_model_func,
        repeats=repeats,
        kfold=kfold,
        verbose=verbose,
        **kwargs,
    )
    if not isinstance(accs, list):
        accs = [accs]
        losses = [losses]

    print("  Accs:", accs)
    print("Losses:", losses)
    print("  Test Accs:", test_accs)
    print("Test Losses:", test_losses)
    best_idx = int(np.argmin(losses))
    best_acc = accs[best_idx]
    best_loss = losses[best_idx]
    print("")
    print(" Best Acc:", best_acc)
    print("Best Loss:", best_loss)

    model_dir = "./models/ensemble/ensemble_" + tag
    save_dir = model_dir + "/best"
    create_dirs_if_not_found(save_dir)
    src_dir = os.path.join(model_dir, str(best_idx+1))
    copy_tree(src_dir, save_dir)

    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, "w") as file:
        file.write(" Best Acc: " + str(best_acc) + "\n")
        file.write("Best Loss: " + str(best_loss) + "\n")
        file.write("  Accs:" + str(accs))
        file.write("Losses:" + str(losses))
        file.write("  Test Accs:" + str(test_accs))
        file.write("Test Losses:" + str(test_losses))

    return best_acc, best_loss


def save_model_and_results(dir_path, model, table):
    print('Saving', model.name, 'to', dir_path)
    create_dirs_if_not_found(dir_path)
    model.save(dir_path)
    results_file = os.path.join(dir_path, "results.txt")
    with open(results_file, "w") as file:
        file.write(table + "\n")


if __name__ == "__main__":
    _k_min = 1
    _k_max = 5
    _tag = "resnet_custom_4"
    _apply_softmax = True
    _iterative = True
    _num_epochs = 1

    _base_trainer = PretrainedNNTrainer(num_epochs=_num_epochs)

    def _base_model_func():
        return PretrainedNNModel(
            tv_models.resnet152,
            transfers.final_layer_alteration_resnet,
        )

    grid_search_ensemble_trainer(
        _k_max,
        _tag,
        _apply_softmax,
        _base_trainer,
        _base_model_func,
        repeats=3,
        iterative=_iterative,
        k_min=_k_min,
        verbose=True
    )

    # grid_search_ensemble_trainer(
    #     _num_base_models,
    #     _tag,
    #     _apply_softmax,
    #     _base_trainer,
    #     _base_model_func,
    #     repeats=3,
    #     iterative=_iterative,
    #     num_rounds=15,
    #     eta=0.34,
    #     gamma=0,
    #     depth=3,
    #     reg_lambda=0.9,
    #     c_weight=1.0,
    #     pass_val=True,
    # )

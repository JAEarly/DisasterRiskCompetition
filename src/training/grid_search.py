from models import Model
from typing import List, Type
import features
import models
from features import BalanceMethod, FeatureExtractor
from models.nn_model import NNTrainer, NNModel
from torch import nn
import numpy as np
from texttable import Texttable
import time


def run_nn_grid_search(
    nn_class: Type[nn.Module], feature_extractor: FeatureExtractor, repeats=3, **kwargs
):
    epoch_range = _extract_range(kwargs, "epoch_range", [5])
    balance_methods = _extract_range(kwargs, "balance_methods", BalanceMethod.NoSample)

    overall_accs = []
    overall_losses = []
    overall_params = []

    total_configs = len(epoch_range) * len(balance_methods)
    config_num = 1

    for epochs in epoch_range:
        for balance_method in balance_methods:
            print('')
            print("-- Configuration " + str(config_num) + "/" + str(total_configs) + " --")
            print(" Epochs -", epochs)
            print(" Balance Method -", balance_method)
            accs = []
            losses = []
            for r in range(repeats):
                print("Repeat " + str(r+1) + "/" + str(repeats))
                trainer = NNTrainer(
                    feature_extractor, num_epochs=epochs, balance_method=balance_method
                )
                model = NNModel(nn_class, feature_extractor.feature_size)
                acc, loss = trainer.train(model)
                accs.append(acc)
                losses.append(loss)

            best_model_idx = int(np.argmin(losses))
            best_acc = accs[best_model_idx]
            best_loss = losses[best_model_idx]
            overall_accs.append(best_acc)
            overall_losses.append(best_loss)
            overall_params.append({'epochs': epochs, 'balance_method': balance_method.name})

            time.sleep(0.1)
            print('Best Loss:', best_loss)
            print(' Best Acc:', best_acc)
            print(' Avg Loss:', np.mean(losses))
            print('  Avg Acc:', np.mean(accs))

            config_num += 1

            # TODO save best model for each param config

    # TODO save best overall model as its own file

    print('')
    print('--- Final Results ---')

    results = zip(overall_losses, overall_accs, overall_params)
    sorted_results = sorted(results, key=lambda x: x[0])

    table = Texttable()
    table.set_cols_align(["c", "c", "c", "c"])
    rows = [['Position', 'Loss', 'Acc', 'Params']]
    for i, r in enumerate(sorted_results):
        rows.append([i+1, *r])
    table.add_rows(rows)
    print(table.draw())


def _extract_range(ranges_dict, range_name, default_value):
    if range_name in ranges_dict:
        return ranges_dict[range_name]
    return default_value


if __name__ == "__main__":
    run_nn_grid_search(
        models.LinearNN,
        features.AlexNet(),
        repeats=3,
        epoch_range=[1, 3, 5, 7],
        balance_methods=[BalanceMethod.NoSample, BalanceMethod.AvgSample, BalanceMethod.OverSample],
    )

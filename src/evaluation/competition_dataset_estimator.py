import numpy as np
import torch
from tqdm import tqdm

from models import ModelIterator


def get_estimated_distribution(model, competition_loader):
    predicted_probas = []
    for batch, _ in competition_loader:
        y_pred = model.predict_batch(batch)
        predicted_probas.extend(y_pred.cpu().detach().numpy())

    counts = [0] * 5
    for label in predicted_probas:
        counts[int(np.argmax(label))] += 1
    return [x / sum(counts) for x in counts]


if __name__ == "__main__":
    print("Running competition dataset estimation")
    all_estimations = []
    for _model, _datasets, _ in tqdm(ModelIterator(), desc="Iterating over models"):
        estimated_dist = get_estimated_distribution(
            _model, _datasets.competition_loader
        )
        all_estimations.append(estimated_dist)

    print("")
    print("Aggregating values")
    mins = []
    maxs = []
    avgs = []
    stds = []
    for clz in range(5):
        clz_dist_values = []
        for estimation in all_estimations:
            clz_dist_values.append(estimation[clz])
        mins.append(min(clz_dist_values))
        maxs.append(max(clz_dist_values))
        avgs.append(np.mean(clz_dist_values))
        stds.append(np.std(clz_dist_values))

    print(" Mins:", ["{:.3f}".format(x) for x in mins])
    print(" Maxs:", ["{:.3f}".format(x) for x in maxs])
    print(" Stds:", ["{:.3f}".format(x) for x in stds])
    print(" Avgs:", ["{:.3f}".format(x) for x in avgs])

import numpy as np
import torch
from tqdm import tqdm

import features
import models
from features import FeatureDatasets, DatasetType
from sklearn.metrics import log_loss


def get_predicted_distribution_and_per_class_error(model, datasets, dataset_type):
    loader = datasets.get_loader(dataset_type)
    predicted_probas = []
    true_labels = []
    for batch, batch_labels in tqdm(
        loader, desc="Predicting error for " + dataset_type.name, leave=False
    ):
        y_outputs = model.predict_batch(batch)
        if model.apply_softmax:
            y_probas = torch.softmax(y_outputs, 1)
        else:
            y_probas = y_outputs
        predicted_probas.extend(y_probas.cpu().detach().numpy())

        if dataset_type is not DatasetType.Competition:
            true_labels.extend(batch_labels)

    class_losses = None
    if dataset_type is not DatasetType.Competition:
        class_losses = []
        for i in range(5):
            per_class_predictions = []
            for prediction, true_label in zip(predicted_probas, true_labels):
                if true_label == i:
                    per_class_predictions.append(prediction)
            total_class_loss = log_loss(
                [i] * len(per_class_predictions),
                per_class_predictions,
                labels=[0, 1, 2, 3, 4],
                normalize=False,
            )
            norm_class_loss = total_class_loss / len(predicted_probas)
            class_losses.append(norm_class_loss)

    counts = [0] * 5
    for label in predicted_probas:
        counts[int(np.argmax(label))] += 1
    return [x / sum(counts) for x in counts], class_losses


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustomSMOTE()
    _datasets = FeatureDatasets(_feature_extractor)

    _model = models.NNModel(
        models.BiggerNN,
        _feature_extractor.feature_size,
        state_dict_path="./models/oversample/grid_search_resnet_custom_smote2_biggernn_3/best.pth",
        eval_mode=True,
    )

    # _model = models.XGBModel(
    #     model_path="./models/kfold_resnet_custom_xgb/best.pth"
    # )

    test_dist, test_losses = get_predicted_distribution_and_per_class_error(
        _model, _datasets, DatasetType.Test
    )
    comp_dist = [0.083, 0.520, 0.037, 0.350, 0.010]

    expected_losses = []
    for i in range(5):
        test_loss = test_losses[i]
        diff_ratio = comp_dist[i] / test_dist[i]
        expected_loss = test_loss * diff_ratio
        expected_losses.append(expected_loss)

    print(
        "Test dist:", ["{:.3f}".format(x) for x in test_dist],
    )
    print(
        "Comp dist:", ["{:.3f}".format(x) for x in comp_dist],
    )

    print("")
    print(
        "Test losses:", ["{:.3f}".format(x) for x in test_losses],
    )
    print(
        "Comp losses:", ["{:.3f}".format(x) for x in expected_losses],
    )

    print("")
    print("    Total test loss:", sum(test_losses))
    print("Total expected loss:", sum(expected_losses))

import torch
from sklearn.metrics import log_loss
from tqdm import tqdm

import features
import models
from evaluation.submit import boost_labels
from features import DatasetType, FeatureDatasets


def run_boost_tuning():
    data_loader, model = setup_feature_evaluation()

    print('Getting base predictions')
    y_true = []
    y_pred = []
    for batch, labels in tqdm(data_loader, leave=False):
        y_pred.extend(model.predict_batch(batch).cpu().detach())
        y_true.extend(labels)

    # Format as tensors
    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)

    num_iters = 100
    lower_boost = 0
    higher_boost = 1

    original = evaluate_boosted(y_true, y_pred, 1.0, apply_softmax=model.apply_softmax)
    best_loss = None
    best_boost = None

    for i in range(num_iters):
        lower_loss = evaluate_boosted(y_true, y_pred, lower_boost, apply_softmax=model.apply_softmax)
        higher_loss = evaluate_boosted(y_true, y_pred, higher_boost, apply_softmax=model.apply_softmax)
        print(i, lower_boost, lower_loss, higher_boost, higher_loss)

        mid_point = lower_boost + (higher_boost - lower_boost)/2
        # Higher is better, move towards it
        if higher_loss < lower_loss:
            lower_boost = mid_point
            best_boost = higher_boost
            best_loss = higher_loss
        # Lower is better, move towards it
        else:
            higher_boost = mid_point
            best_boost = lower_boost
            best_loss = lower_loss

        if lower_boost == higher_boost:
            break

    print('Original loss:', original)
    print('     New loss:', best_loss)
    print('    Threshold:', best_boost)


def evaluate_boosted(y_true, y_pred, boost_threshold, apply_softmax):
    # Calculate prediction probabilities if required
    if apply_softmax:
        y_probabilities = torch.softmax(y_pred, 1)
    else:
        y_probabilities = y_pred

    y_probabilities = y_probabilities.tolist()

    # Boost
    y_probabilities = boost_labels(y_probabilities, boost_threshold, confirm=False, verbose=False)

    # Calculate and return loss
    return log_loss(y_true, y_probabilities, labels=[0, 1, 2, 3, 4])


def setup_feature_evaluation():
    feature_extractor = features.AlexNetCustomSMOTE("./models/grid_search_alexnet_custom/best.pth")
    data_loader = FeatureDatasets(feature_extractor).get_loader(DatasetType.Validation)
    model = models.NNModel(
        models.LinearNN,
        feature_extractor.feature_size,
        state_dict_path="./models/grid_search_alexnet_custom_smote_linearnn/best.pth",
        eval_mode=True,
    )
    print("Running boost tuning for", feature_extractor.name, model.name)
    return data_loader, model


if __name__ == "__main__":
    run_boost_tuning()

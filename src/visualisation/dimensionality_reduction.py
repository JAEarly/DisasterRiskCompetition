"""Dimensionality reduction visualisation script."""

import matplotlib.pyplot as plt
import pandas as pd

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Import must remain for projection='3d' to work
from sklearn.preprocessing import StandardScaler

import features
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import DatasetType, FeatureDatasets


def run_dimensionality_reduction(
    feature_dataset: FeatureDatasets,
    reduction_method,
    supervised=True,
    already_fit=False,
):
    """
    Run a dimensionality reduction method.

    :param feature_dataset: Dataset to run reduction over.
    :param reduction_method: Reduction method to use.
    :param supervised: If the reduction method is supervised or unsupervised.
    :param already_fit: If the reduction method has already been fit (so doesn't need training).
    :return: A dataframe containing the reduced dimension data.
    """
    print("Getting features")
    train_features, labels = feature_dataset.get_features_and_labels(DatasetType.Train)
    print("Scaling data")
    train_features = StandardScaler().fit_transform(train_features.cpu().detach())
    print("Running " + str(reduction_method))

    if already_fit:
        reduced_features = reduction_method.fit(train_features)
    else:
        if supervised:
            reduced_features = reduction_method.fit_transform(train_features, labels)
        else:
            reduced_features = reduction_method.fit_transform(train_features)

    print("Aggregating principal components")
    principal_df = pd.DataFrame(data=reduced_features, columns=["pc1", "pc2"])
    target_df = pd.DataFrame(data=labels, columns=["target"])
    final_df = pd.concat([principal_df, target_df], axis=1)
    return final_df


def visualise_reduction(dataframe) -> None:
    """
    Visualise a dimensionality reduction.

    :param dataframe: Reduced dimension dataframe.
    :return: None.
    """
    fig, axis = plt.subplots(nrows=1, ncols=1)

    targets = [0, 1, 2, 3, 4]
    for target in targets:
        indices_to_keep = dataframe["target"] == target
        # Attempt
        num_points = len(dataframe[(dataframe["target"] == target)])
        reduction_step = int(num_points / 100)
        axis.scatter(
            dataframe.loc[indices_to_keep, "pc1"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc2"][::reduction_step],
            marker="x",
        )
    plt.show()


if __name__ == "__main__":
    _feature_extractor = features.ResNetCustom("./models/grid_search_resnet_cnn/best.pth")
    _feature_dataset = FeatureDatasets(_feature_extractor)

    # PCA 2D
    # _final_df = run_dimensionality_reduction(
    #     _feature_dataset, PCA(n_components=2), supervised=False
    # )

    # LDA 2D - Decent
    _final_df = run_dimensionality_reduction(
        _feature_dataset, LinearDiscriminantAnalysis(n_components=2), supervised=True
    )

    # Visualise
    visualise_reduction(_final_df)

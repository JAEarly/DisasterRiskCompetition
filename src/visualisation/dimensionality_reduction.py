"""Dimensionality reduction visualisation script."""

import matplotlib.pyplot as plt
import pandas as pd

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Import must remain for projection='3d' to work
from sklearn.preprocessing import StandardScaler

import models
from features import DatasetType, AlexNet256, FeatureDatasets


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
    features, labels = feature_dataset.get_features_and_labels(DatasetType.Train)
    print("Scaling data")
    features = StandardScaler().fit_transform(features.cpu().detach())
    print("Running " + str(reduction_method))

    if already_fit:
        reduced_features = reduction_method.transform(features)
    else:
        if supervised:
            reduced_features = reduction_method.fit_transform(features, labels)
        else:
            reduced_features = reduction_method.fit_transform(features)

    print("Aggregating principal components")
    principal_df = pd.DataFrame(data=reduced_features, columns=["pc1", "pc2", "pc3"])
    target_df = pd.DataFrame(data=labels, columns=["target"])
    final_df = pd.concat([principal_df, target_df], axis=1)
    return final_df


def visualise_reduction(dataframe) -> None:
    """
    Visualise a dimensionality reduction.

    :param dataframe: Reduced dimension dataframe.
    :return: None.
    """
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    axis.set_xlabel("Principal Component 1", fontsize=15)
    axis.set_ylabel("Principal Component 2", fontsize=15)
    axis.set_zlabel("Principal Component 3", fontsize=15)

    targets = [0, 1, 2, 3, 4]
    for target in targets:
        indices_to_keep = dataframe["target"] == target
        # Attempt
        num_points = len(dataframe[(dataframe["target"] == target)])
        reduction_step = int(num_points / 100)
        axis.scatter(
            dataframe.loc[indices_to_keep, "pc1"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc2"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc3"][::reduction_step],
            marker="x",
        )
    axis.legend(targets)
    plt.show()


if __name__ == "__main__":
    _feature_dataset = FeatureDatasets(AlexNet256())

    # PCA 3D - Rubbish
    # _final_df = run_dimensionality_reduction(
    #     _feature_dataset, PCA(n_components=3), supervised=False
    # )

    # LDA 3D - Decent
    lda_model = models.LDAModel(
        "lda_alexnet256", model_path="./models/lda_alexnet256_2019-11-05_14:30:22.pkl"
    )
    _final_df = run_dimensionality_reduction(
        _feature_dataset, lda_model.lda, already_fit=True
    )

    # Visualise
    visualise_reduction(_final_df)

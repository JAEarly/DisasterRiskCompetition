"""Dimensionality reduction visualisation script."""

import matplotlib.pyplot as plt
import pandas as pd

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Import must remain for projection='3d' to work
from sklearn.preprocessing import StandardScaler

import features
import models
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
    axis3d = fig.add_subplot(111, projection="3d")
    axis2d = fig.add_subplot(211)

    axis3d.set_xlabel("Principal Component 1", fontsize=15)
    axis3d.set_ylabel("Principal Component 2", fontsize=15)
    axis3d.set_zlabel("Principal Component 3", fontsize=15)

    targets = [0, 1, 2, 3, 4]
    for target in targets:
        indices_to_keep = dataframe["target"] == target
        # Attempt
        num_points = len(dataframe[(dataframe["target"] == target)])
        reduction_step = int(num_points / 100)
        axis3d.scatter(
            dataframe.loc[indices_to_keep, "pc1"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc2"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc3"][::reduction_step],
            marker="x",
        )
        axis2d.scatter(
            dataframe.loc[indices_to_keep, "pc1"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc2"][::reduction_step],
            marker="x",
        )
    axis3d.legend(targets)
    plt.show()


if __name__ == "__main__":
    _feature_extractor = features.ResNet18t256()
    _feature_dataset = FeatureDatasets(_feature_extractor)

    # PCA 3D - Rubbish
    # _final_df = run_dimensionality_reduction(
    #     _feature_dataset, PCA(n_components=3), supervised=False
    # )

    # LDA 3D - Decent
    _model = models.LDAModel(
        "resnet18t256_lda",
        model_path="./models/resnet18t256_lda_2019-11-06_16:09:08.pkl",
    )
    _final_df = run_dimensionality_reduction(
        _feature_dataset, _model.lda, already_fit=True
    )

    # Visualise
    visualise_reduction(_final_df)

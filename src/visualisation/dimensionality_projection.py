"""Dimensionality reduction visualisation script."""

import matplotlib.pyplot as plt
import pandas as pd

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Import must remain for projection='3d' to work
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import features
from features import DatasetType, FeatureDatasets


def run_dimensionality_reduction(
    feature_extractor, dataset_types, reduction_model, already_fit=False,
):
    print("Getting features")
    feature_datasets = FeatureDatasets(feature_extractor)
    xs = []
    ys = []
    if type(dataset_types) == DatasetType:
        dataset_types = [dataset_types]
    for dataset_type in dataset_types:
        if dataset_type == DatasetType.Competition:
            new_xs, new_ys = feature_datasets.get_features(dataset_type), None
        else:
            new_xs, new_ys = feature_datasets.get_features_and_labels(dataset_type)
        xs.append(new_xs)
        if new_ys is not None:
            ys.append(new_ys)
    xs = torch.cat(xs)
    if ys:
        ys = torch.cat(ys)
    print("Scaling data")
    xs = StandardScaler().fit_transform(xs.cpu().detach())

    if already_fit:
        print("Using pre-fit model")
        reduced_features = reduction_model.transform(xs)
    else:
        print("Running " + str(reduction_model))
        reduced_features = reduction_model.fit_transform(xs, ys)
    print(reduction_model.explained_variance_ratio_)

    print("Aggregating principal components")
    principal_df = pd.DataFrame(data=reduced_features, columns=["pc1", "pc2"])
    target_df = pd.DataFrame(data=ys, columns=["target"])
    final_df = pd.concat([principal_df, target_df], axis=1)
    return final_df


def visualise_reduction(dataframe, targets, fe_name) -> None:
    fig, axis = plt.subplots(nrows=1, ncols=1)

    for target in targets:
        indices_to_keep = dataframe["target"] == target
        # Percentage to points to render
        reduce_to = 0.3
        reduction_step = int(1 / reduce_to)
        axis.scatter(
            dataframe.loc[indices_to_keep, "pc1"][::reduction_step],
            dataframe.loc[indices_to_keep, "pc2"][::reduction_step],
            marker="x",
            label=str(target),
            alpha=0.4,
        )
    axis.legend(loc="best")
    axis.set_title(fe_name)
    plt.show()


def plot_dataset(feature_extractor, dataset_type, reduction_model):
    final_df = run_dimensionality_reduction(
        feature_extractor, dataset_type, reduction_model
    )
    visualise_reduction(final_df, [0, 1, 2, 3, 4], feature_extractor.name)


def plot_dataset_comparison(feature_extractor):
    lda_model = LinearDiscriminantAnalysis(n_components=2)

    print("Fitting LDA model on labelled data")
    labelled_df = run_dimensionality_reduction(
        feature_extractor, DatasetType.Train, lda_model,
    )
    labelled_df["target"] = "train"

    print("")
    print("Reducing validation data")
    val_df = run_dimensionality_reduction(
        feature_extractor, DatasetType.Validation, lda_model, already_fit=True
    )
    val_df["target"] = "val"

    print("")
    print("Reducing testing data")
    test_df = run_dimensionality_reduction(
        feature_extractor, DatasetType.Test, lda_model, already_fit=True
    )
    test_df["target"] = "test"

    print("")
    print("Reducing competition data")
    comp_df = run_dimensionality_reduction(
        feature_extractor, DatasetType.Competition, lda_model, already_fit=True
    )
    comp_df["target"] = "comp"

    final_df = pd.concat([labelled_df, comp_df])
    visualise_reduction(
        final_df, ["train", "val", "test", "comp"], feature_extractor.name
    )


if __name__ == "__main__":
    for i in [2, 5, 10]:
        _feature_extractor = features.ResNetCustomReduced(
            num_components=i
            # model_path="./models/augmented/grid_search_resnet_custom/best.pth",
            # save_dir="./models/features/augmented/",
            # train_dir="./data/augmented/train",
        )
        plot_dataset(
            _feature_extractor,
            [DatasetType.Train],
            LinearDiscriminantAnalysis(n_components=2),
        )

    # Class comparison
    # plot_dataset(_feature_extractor, [DatasetType.Train], PCA(n_components=2))

    # Dataset comparison
    # plot_dataset_comparison(_feature_extractor)

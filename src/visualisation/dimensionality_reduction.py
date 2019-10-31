from features import AlexNet256, DatasetType
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def run_dimensionality_reduction(feature_extractor, reduction_method, supervised):
    print("Getting features")
    x, y = feature_extractor.extract(DatasetType.Train)
    print("Scaling data")
    x = StandardScaler().fit_transform(x)
    print("Running " + str(reduction_method))
    if supervised:
        reduced_features = reduction_method.fit_transform(x, y)
    else:
        reduced_features = reduction_method.fit_transform(x)
    print("Aggregating principal components")
    principal_df = pd.DataFrame(data=reduced_features, columns=['pc1', 'pc2', 'pc3'])
    target_df = pd.DataFrame(data=y, columns=['target'])
    final_df = pd.concat([principal_df, target_df], axis=1)
    return final_df


def visualise_reduction(dataframe):
    # Import must remain for projection='3d' to work
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')

    axis.set_xlabel('Principal Component 1', fontsize=15)
    axis.set_ylabel('Principal Component 2', fontsize=15)
    axis.set_zlabel('Principal Component 3', fontsize=15)

    targets = [0, 1, 2, 3, 4]
    for target in targets:
        indices_to_keep = dataframe['target'] == target
        # Attempt
        num_points = len(dataframe[(dataframe['target'] == target)])
        reduction_step = int(num_points / 100)
        axis.scatter(dataframe.loc[indices_to_keep, 'pc1'][::reduction_step],
                     dataframe.loc[indices_to_keep, 'pc2'][::reduction_step],
                     dataframe.loc[indices_to_keep, 'pc3'][::reduction_step],
                     marker='x')
    axis.legend(targets)
    plt.show()


if __name__ == "__main__":
    # Alexnet256 PCA
    visualise_reduction(run_dimensionality_reduction(AlexNet256(), PCA(n_components=3), supervised=False))

    # Alexnet256 LDA
    visualise_reduction(run_dimensionality_reduction(AlexNet256(), LinearDiscriminantAnalysis(n_components=3), supervised=True))








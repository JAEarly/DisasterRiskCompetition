from features import AlexNet256, DatasetType
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    print("Creating feature extractor")
    feature_extractor = AlexNet256()
    print("Getting features")
    x, y = feature_extractor.extract(DatasetType.Train)
    print("Scaling data")
    x = StandardScaler().fit_transform(x)
    print("Running PCA")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(x)
    print("Aggregating principal components")
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2', 'pc3'])
    target_df = pd.DataFrame(data=y, columns=['target'])
    final_df = pd.concat([principal_df, target_df], axis=1)

    # Import must remain for projection='3d' to work
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')

    axis.set_xlabel('Principal Component 1', fontsize=15)
    axis.set_ylabel('Principal Component 2', fontsize=15)
    axis.set_zlabel('Principal Component 3', fontsize=15)

    targets = [0, 1, 2, 3, 4]
    for target in targets:
        indicesToKeep = final_df['target'] == target
        # Attempt
        num_points = len(final_df[(final_df['target'] == target)])
        reduction_step = int(num_points / 100)
        axis.scatter(final_df.loc[indicesToKeep, 'pc1'][::reduction_step],
                     final_df.loc[indicesToKeep, 'pc2'][::reduction_step],
                     final_df.loc[indicesToKeep, 'pc3'][::reduction_step],
                     marker='x')

    axis.legend(targets)
    plt.show()

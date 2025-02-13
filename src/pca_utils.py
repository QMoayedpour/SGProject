import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def apply_dim_reduction(X, columns, model, name_suffix, n_dim):
    X_sub = X[columns]
    X_scaled = StandardScaler().fit_transform(X_sub)
    X_transformed = model.fit_transform(X_scaled)
    df_transformed = pd.DataFrame(X_transformed, columns=[f"{name_suffix}_pca_{i}" for i in range(1, n_dim+1)])
    return df_transformed


def plot_pca_2d(X_pca, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca.iloc[:, 0], 
        y=X_pca.iloc[:, 1], 
        hue=y, 
        palette={0: 'blue', 1: 'red'}, 
        alpha=0.7
    )
    plt.title(title)
    plt.legend(title='TARGET')
    plt.grid(True)
    plt.show()


def get_pca_features(X_encoded):
    q_columns = [col for col in X_encoded.columns if col.startswith('Q')]
    s_columns = [col for col in X_encoded.columns if col.startswith('S')]
    c_columns = [col for col in X_encoded.columns if col.startswith('C')]

    pca = PCA(n_components=10)

    q_pca = apply_dim_reduction(X_encoded, q_columns, pca, 'Q', 10)
    s_pca = apply_dim_reduction(X_encoded, s_columns, pca, 'S', 10)
    c_pca = apply_dim_reduction(X_encoded, c_columns, pca, 'C', 10)
    return q_pca, s_pca, c_pca


def plot_pca_3d(q_pca, s_pca, c_pca, y, title=''):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        q_pca.iloc[:, 0], 
        s_pca.iloc[:, 0], 
        c_pca.iloc[:, 0], 
        c=y, 
        cmap='coolwarm', 
        alpha=0.7
    )
    
    ax.set_xlabel('Q PCA 1')
    ax.set_ylabel('S PCA 1')
    ax.set_zlabel('C PCA 1')
    ax.set_title(title)
    
    legend_labels = {0: 'False', 1: 'True'}
    scatter_legend = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                                 markerfacecolor=('blue' if i == 0 else 'red'), markersize=10) for i in legend_labels]
    ax.legend(handles=scatter_legend, title="TARGET")

    plt.show()

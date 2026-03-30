'''
Author: Cameron Rosenthal
'''

from lets_plot import *
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# LetsPlot.setup_html() # For ipynb
LetsPlot.setup_show_ext() # For normal python environment (opens in browser).

# Prepare iris DataFrame ------------------------------------------------------

iris_raw = load_iris()

species = pd.DataFrame(data={
    "species": [iris_raw.target_names[x] for x in iris_raw.target]
})

features = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)

iris = pd.concat([species, features], axis=1)

iris.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width"
}, inplace=True)

# Dataset preprocessing ------------------------------------------------------------

iris_data = iris.drop(["species"], axis=1)

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(iris_data)

# K-Means Clustering ----------------------------------------------------------

k_means = KMeans(n_clusters=4)
labels = k_means.fit_predict(X_scaled)

# Root Mean Squared -----------------------------------------------------------

assigned_centroids = k_means.cluster_centers_[labels]

squared_diffs = (X_scaled - assigned_centroids) ** 2
rmse = np.sqrt(squared_diffs.sum(axis=1).mean())

print(f"{rmse=}")

# Compare predicted groupings to actual results -------------------------------

centroids = scaler.inverse_transform(k_means.cluster_centers_)

df_centroids = pd.DataFrame({
    'x':       centroids[:, 0],
    'y':       centroids[:, 1],
    'cluster': ['0', '1', '2', '3']
})

assigned_centroids = k_means.cluster_centers_[labels]
unscaled_x = scaler.inverse_transform(X_scaled)

iris_data_temp = pd.DataFrame(iris_data, columns=iris_data.columns)
labels_pd = pd.DataFrame({"label": labels})
labeled_pd = pd.concat([iris_data_temp, labels_pd], axis=1)

labeled_pd = labeled_pd.astype({
    "label": 'str'
})

# Graph actual results and predicted groupings --------------------------------

(
    ggplot(iris) +
    geom_point(aes(x="sepal_length", y="sepal_width", color="species")) +
    # geom_point(data=df_centroids, mapping=aes(x='x', y='y'), shape=21, size=8, stroke=2, fill='white') +
    ggtitle("Original Iris Species")
).show()


(
    ggplot(labeled_pd) +
    geom_point(aes(x="sepal_length", y="sepal_width", color="label")) +
    geom_point(data=df_centroids, mapping=aes(x='x', y='y'), shape=21, size=8, stroke=2, fill='white') +
    ggtitle("Predicted Groups")
).show()

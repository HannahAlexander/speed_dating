import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import shap
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import copy 
from sklearn.model_selection import train_test_split

def gen_shap_summary_plot(clf: lgb.LGBMClassifier, X_test: pd.DataFrame) -> plt:
    """
    Plots feature importance feature importance using Shapely values for a feature and an instance.
    Input arguments:
    clf: A lightgbm classifer object.
    X_test: A dataframe containing the test data feature values.

    Returns:
    A plot of the feature SHAP feature importance.
    A dataframe containing the feature importances of feature in the model in descending order.

    """
    plt.figure()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, plot_size = [30, 12])
    plt.show()

    vals= np.abs(shap_values[1]).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_test.columns,vals)),columns=['col_name','feature_importance_vals'])

    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    print(feature_importance.head())

    return plt, feature_importance

def gen_lgbm_feat_importance(clf: lgb.LGBMClassifier, num_features: int=30) -> plt:
    """
    Plots feature importance using built in LightGBM method
    Input arguments:
    clf: A lightgbm classifer object.
    num_features: An integer specifying the top number of features to display.

    Returns:
    A plot of the importance of each feature in the top num_features.
    A dataframe containing the feature importances of the top num_features in the model in descending order.

    """
    plt.figure()
    feat_imp = pd.Series(clf.feature_importances_, index=clf.booster_.feature_name())
    feat_imp.nlargest(num_features).plot(kind = "barh", figsize = (25, 12))
    plt.show()

    return plt, feat_imp.nlargest(num_features)

def dbscan_clustering(df, df_orig, eps, min_samples, colour_by_conversion = False, show_legend=True):
    """
    Plots clustering graph with sectioned out cluster areas.

    Input arguments:
    df: The umap clustered dataframe to be considered
    df_orig: The total dataset 
    eps: The eps value to pass to DBSCAN
    min_samples: The minimum samples to include in each cluster 
    colour_by_conversion: Whether to colour the plot points by converted type
    show_legend: Whether to show the legend on the plot to provide a label for each colour

    Returns:
    The dataframe containing the resulting plot points 

    """

    db = DBSCAN(eps=eps, min_samples= min_samples).fit(df)
    labels = db.labels_

    print(np.unique(labels))

    y_true = df_orig["match_a"]
    
    # Plot result
    unique_labels = set(labels)
    colors = ['olive', 'aqua', 'aquamarine', 'darksalmon', "darkblue", 'red', 'pink', 'green', 'yellow', 'black', 'grey', 'white']

    df = pd.DataFrame(list(zip(df[:, 0], df[:, 1], labels, y_true)),
                 columns =['x_val', 'y_val', "labels", 'match_a'])

    fig = go.Figure()

    for k, col in zip(unique_labels, colors):

        class_member_mask = (labels == k)
    
        xy = df[class_member_mask]

        if colour_by_conversion == True:
            col = xy["match_a"]

        fig.add_trace(go.Scatter(x = xy["x_val"], y = xy["y_val"], customdata=xy['match_a'], mode = "markers", marker =dict(color=col)))

    
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{x}",
            "ColY: %{y}",
            "Match: %{customdata}",
        ])
    )
    fig.update_layout(height=800)
    fig.update_traces(showlegend=show_legend)
    
    fig.show()
    
    #evaluation metrics
    #sc = metrics.silhouette_score(df, labels)
    # ranges between -1 and 1. Closer to 1 is better, values around 0 indicate overlapping clusters
    #print("Silhouette Coefficient:%0.2f"%sc)
    # indicates how closely clusters match actual labels (in this case leads that do and dont convert)
    #ari = metrics.adjusted_rand_score(y_true, labels)
    #print("Adjusted Rand Index: %0.2f"%ari)
    return db, df
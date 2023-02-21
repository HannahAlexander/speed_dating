import numpy as np
import pandas as pd
# feature selection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    f1_score
)
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

def select_features(X_train, y_train, X_test, k, df):
    np.random.seed(10)
    # select k most predicitve features
    fs = SelectKBest(score_func=chi2, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    # extract names of these features
    names = df.columns.values[fs.get_support(indices=True)]

    # extract importance score for each feature
    scores = fs.scores_[fs.get_support()]
    # make into dataframe
    names_scores = list(zip(names, scores))
    ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])

    return X_train_fs, X_test_fs, names, ns_df_sorted

def log_reg(X_train, Y_train, X_test, Y_test, feature_variables):

    classifier = LogisticRegression(random_state=42, max_iter= 10000)#, class_weight = "balanced"
    classifier.fit(X_train, Y_train)

    # get accuracy score
    score = classifier.score(X_test, Y_test)

    importance_df = pd.DataFrame(
        list(zip(feature_variables, classifier.coef_[0])),
        columns=["Feature", "importance"],
    ).sort_values(by="importance", ascending=False)

    y_pred = classifier.predict(X_test)

    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    cm_display.plot()

    f1_score = metrics.f1_score(Y_test, y_pred)


    return {"clf_model": classifier, "accuracy": score, "F1_score": f1_score, "feature_importance": importance_df}

def scoring_func(y_true, y_pred):
    """
    Print evaluation metrics of trained model, including: precision, recall and accuracy.
    Plots confusion matrix.

    Input arguments:
    y_true: The true values of the varaible being predicted
    y_pred: The predicted value

    Returns:
    None

    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()

    print('Precision:', metrics.precision_score(y_true, y_pred))
    print('Recall:', metrics.recall_score(y_true, y_pred))
    print('Accuracy:', metrics.accuracy_score(y_true, y_pred))
    print('F1 Score:', metrics.f1_score(y_true, y_pred))

# balance datasets
def resample_data(train_df, multiplier, X_test, y_test, classifier):
    """
    Applies upsampling to dataset

    Input arguments:
    train_df- training dataset
    multiplier- how much to multiply number of leads by
    X_test- test dataset features only
    y_test- test target variable
    classifier- classifier object

    Returns:
    Model fitted on new upsampled data

    """
    upsample_pos = resample(train_df[train_df['match_a']==1], replace=True, n_samples=round(sum(train_df['match_a']==1)*multiplier), random_state=123)
    neg_df = train_df[train_df['match_a']==0]
    train_df = pd.concat([neg_df, upsample_pos], axis=0)

    X_train = train_df.drop(['match_a'], axis = 1)
    y_train = train_df['match_a']

    y_train = y_train.values
    y_test = y_test

    print("X_train shape is: ", X_train.shape, "X_train shape is: ", y_train.shape)

    clf = classifier
    clf.fit(X_train, y_train)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    scoring_func(y_test, y_pred)

    return clf
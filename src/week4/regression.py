from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# load dataset (same pattern as r-sample.py)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
data: pd.DataFrame = pd.read_csv(ASSETS_DIR / "HousingData.csv")

x = data[["HouseAge", "HouseSize"]]
y = data["HousePrice"]

# split into train/test so x_train/y_train are available for model.fit
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# choose an option 1..5 to select a model implementation
model_option: int = int(
    input(
        "Choose one model from the following: 1- SVR, 2- Decision Tree, 3- Random Forest, 4- KNN, 5- Linear Regression\n your choice is: "
    )
)

if model_option == 1:
    # SVR imported at top

    """ Task
    Call the constructor SVR() to create a SVR object, name it as 'model', by passing the following key parameters:

    (i)  'kernel': Specifies the kernel type to be used in the algorithm. possible values are "linear", "poly", "rbf", "sigmoid" and "precomputed".
         Default is "rbf".
    (ii) 'degree': Degree of the polynomial kernel function ("poly"). Must be non-negative. Ignored by all other kernels. Default=3
    (iii) 'gamma': Kernel coefficient for "rbf", "poly" and "sigmoid".
          Possible values are "scale" (1 / (n_features * X.var())) and "auto" (1 / n_features).
          Default is "scale".
    (iv)  'C': Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        Default=1.0
    (v) 'epsilon': Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
        Must be non-negative. Default=0.1
    (vi) 'max_iter': Hard limit on iterations within solver, or -1 for no limit. Default is -1.
    """
    model = SVR(gamma="auto")
    # Train this model using the training dataset (x_train, y_train).
    model.fit(x_train, y_train)
elif model_option == 2:
    # LinearRegression imported at top

    """
    Call the constructor LinearRegression() to create a linear regression object, name it as 'model', by passing the following parameters:
    (i) 'fit_intercep': Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered). Default=True
    (ii) 'copy_X': If True, X will be copied; else, it may be overwritten. Default=True
    """
    model = LinearRegression()
    # Train this model using the training dataset (x_train, y_train).
    model.fit(x_train, y_train)
elif model_option == 3:
    # KNeighborsRegressor imported at top

    """
    Call the constructor KNeighborsRegressor() to create a KNN regressor, name it as 'model', by passing the following parameters:
    (i) 'n_neighbors': Number of neighbors to use by default for kneighbors queries. Default is 5.
    (ii) 'weights': Weight function used in prediction. Possible values are "uniform" (uniform weights),
         "distance" (weight points by the inverse of their distance) and
         [callable] (a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights). Default = 'uniform'
    (iii) 'algorithm': Algorithm used to compute the nearest neighbors. Possible values are "auto" (attempt to decide the most appropriate algorithm based on the values passed to fit method),
          "ball_tree" (use BallTree), "kd_tree" (use KDTree), and "brute" (use a brute-force search). Default is "auto".
    (iv) 'metric': Metric to use for distance computation. Default is “minkowski”, which results in the standard Euclidean distance when p = 2.
    """
    model = KNeighborsRegressor(n_neighbors=3)
    # Train this model using the training dataset (x_train, y_train).
    model.fit(x_train, y_train)

elif model_option == 4:
    # DecisionTreeRegressor imported at top

    """
    Call the constructor DecisionTreeRegressor () to create a decision tree regressor, name it as 'model', by passing the following key parameters:
    (i) "criterion": The function to measure the quality of a split.
        Possible options are “squared_error” (the mean squared error), “friedman_mse” ( mean squared error with Friedman"s improvement score),
        “absolute_error” (the mean absolute error, which minimizes the L1 loss using the median of each terminal node), “poisson” (uses reduction in Poisson deviance to find splits).
        Default=”squared_error”
    (ii) "splitter": The strategy used to choose the split at each node.
          Supported strategies are “best” to choose the best split and “random” to choose the best random split. Default is ”best”.
    (iii) "max_depth": Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contains fewer than `min_samples_split` samples
    (iv) "min_samples_split": Minimum number of samples required to split an internal node, default is 2
    (v) "min_samples_leaf" Minimum number of samples required to be at a leaf node. default is 1.
    (vi) "max_features": number of features to consider when looking for the best split, default is None (max_features=n_features).
        Other values are 'sqrt' (max_features=sqrt(n_features)) and "log2” (max_features=log2(n_features))
    (vii) "random_state": Controls the randomness of the estimator. default is None.
          To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer, say random_state = 42.
    (viii) 'min_impurity_decrease': A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default=0.0.
    """
    model = DecisionTreeRegressor()
    # Train this DT regressor using the training data set (x_train, y_train).
    model.fit(x_train, y_train)
elif model_option == 5:
    # RandomForestRegressor imported at top

    """
    Call the constructor RandomForestRegressor () to create a random forest regreesor, name it as 'model', by passing the following key parameters:
    (i) 'n_estimator': Number of trees in the forest, default is 100
    (ii) "criterion": The function to measure the quality of a split.
        Possible options are “squared_error” (the mean squared error), “friedman_mse” ( mean squared error with Friedman"s improvement score),
        “absolute_error” (the mean absolute error, which minimizes the L1 loss using the median of each terminal node), “poisson” (uses reduction in Poisson deviance to find splits).
        Default=”squared_error”
    (iii) "max_depth": Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contains fewer than `min_samples_split` samples
    (iv) "min_samples_split": Minimum number of samples required to split an internal node, default is 2
    (v) "min_samples_leaf" Minimum number of samples required to be at a leaf node. default is 1.
    (vi) "max_features": number of features to consider when looking for the best split, default is None (max_features=n_features).
        Other values are 'sqrt' (max_features=sqrt(n_features)) and "log2” (max_features=log2(n_features)).
    (vii) 'min_impurity_decrease': A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default=0.0.
    (viii)'bootstrap': Whether to use bootstrap samples when building trees, default is 'True'.
          If 'False', the entire dataset is used to build each tree, which may lead to overfitting.
    (ix)   'oob_score': Whether to use out-of-bag samples to estimate the generalisation accuracy.
          default value is 'False'. If 'True', an unbiased estimate of the model performance is provided.
    (x) 'max_samples': If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draw X.shape[0] samples. Default=None
    (xi) "random_state": Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and
         the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
         Default is None.
    """
    model = RandomForestRegressor(
        n_estimators=3, max_depth=3, max_features=2, max_samples=100, random_state=12
    )
    # Train this model using the training dataset (x_train, y_train).
    model.fit(x_train, y_train)
else:
    print("invalid option number. Try again")

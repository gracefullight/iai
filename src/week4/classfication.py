from __future__ import annotations

# Common imports and dataset preparation shared by all model options
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset from assets
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
iris_df: pd.DataFrame = pd.read_csv(ASSETS_DIR / "iris.csv")

# Encode species labels
le: LabelEncoder = LabelEncoder()
iris_df["species"] = le.fit_transform(iris_df["species"])

# Features and target
X: NDArray[np.float64] = iris_df.drop("species", axis=1).to_numpy(dtype=float)
y: NDArray[np.int64] = iris_df["species"].to_numpy()

# Train/val/test split with random_state for reproducibility
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Prepare a small future sample from the test set (last two rows)
futureSample_X: NDArray[np.float64] = X_test[-2:]
_futureSample_y: NDArray[np.int64] = y_test[-2:]
X_test = X_test[:-2]
y_test = y_test[:-2]

model_option: int = int(
    input(
        "Choose one model from the following: 1- decision tree, 2- Random forest, 3- logistic regression, 4-K nearest neighbours, or 5- Support vector classifier \n your choice is: "
    )
)

if model_option == 1:
    #  Task: Import `DecisionTreeClassifier` from `sklearn.tree`.

    """
    The constructor `DecisionTreeClassifier()` is used to create a decision tree classifier object by passing the following key parameters:
    (i) "criterion", which represents the measure used in expanning the decision tree.
        There are a number of options for this parameter, such as `gini` (Gini impurity) and `entropy` or 'log_loss(information gain).
        By default, it is `gini`. We use `entropy`.
    (ii) "splitter": The strategy used to choose the split at each node.
          Supported strategies are “best” to choose the best split and “random” to choose the best random split. Default is ”best”.
    (iii) "max_depth": Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contains fewer than `min_samples_split`
    (iv) "min_sample_split": Minimum number of samples required to split an internal node, default is 2
    (v) "min_samples_leaf" Minimum number of samples required to be at a leaf node. default is 1.
    (vi) "max_features": number of features to consider when looking for the best split, default is None (max_features=n_features).
        Other values are 'sqrt' (max_features=sqrt(n_features)) and "log2” (max_features=log2(n_features))
    (vii) "random_state": Controls the randomness of the estimator. default is None.
          To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer, say random_state = 42.
    """
    # Define model + hyperparameter grid (tuned *on validation set*)
    dt_param_grid: dict[str, list[Any]] = {
        "criterion": ["gini", "entropy"],  # impurity metrics
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    dt_fixed_params: dict[str, Any] = {"random_state": 42}

    # Define evaluation function
    def dt_evaluate(params: dict[str, Any]) -> tuple[float, DecisionTreeClassifier]:
        clf = DecisionTreeClassifier(**dt_fixed_params, **params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        return accuracy_score(y_val, preds), clf

    # Manual grid search using validation accuracy (break ties by shallower depth)
    dt_results: list[dict[str, Any]] = []
    best_acc: float = float(-np.inf)
    best_params: dict[str, Any] | None = None
    best_tie_metric: tuple[int, int] | None = None  # smaller = better (depth, node_count)

    dt_keys, dt_values = zip(*dt_param_grid.items(), strict=False)
    for combo in product(*dt_values):
        params = dict(zip(dt_keys, combo, strict=False))
        try:
            acc, model = dt_evaluate(params)
            depth = model.get_depth()
            nodes = model.tree_.node_count
            dt_results.append({**params, "val_acc": acc, "depth": depth, "nodes": nodes})
            tie_metric = (depth, nodes)

            if (acc > best_acc) or (
                np.isclose(acc, best_acc)
                and (best_tie_metric is None or tie_metric < best_tie_metric)
            ):
                best_acc = acc
                best_params = params
                best_tie_metric = tie_metric
        except Exception as e:
            dt_results.append({**params, "val_acc": np.nan, "depth": np.nan, "nodes": np.nan})
            print(f"Skipped {params} due to: {e}")

    # Show validation results sorted
    dt_df_results = pd.DataFrame(dt_results).sort_values(
        ["val_acc", "depth", "nodes"], ascending=[False, True, True]
    )
    print("\nValidation results (top 10):")
    print(dt_df_results.head(10).to_string(index=False))

    print("\nBest on validation:")
    print(
        best_params,
        "val_acc=",
        round(best_acc, 4),
        "depth=",
        best_tie_metric[0] if best_tie_metric is not None else None,
        "nodes=",
        best_tie_metric[1] if best_tie_metric is not None else None,
    )

    # Refit best model on TRAIN + VAL
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    final_clf = DecisionTreeClassifier(**dt_fixed_params, **best_params)
    final_clf.fit(X_train_val, y_train_val)

    # Final evaluation on TEST
    y_pred = final_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[DecisionTree] Test accuracy:", round(test_acc, 4))
    # Future sample prediction
    future_pred = final_clf.predict(futureSample_X)
    print("[DecisionTree] Future predictions:", future_pred.tolist())

elif model_option == 2:
    # Task: Import RandomForestClassifier from sklearn.ensemble.

    """
    The constructor RandomForestClassifier() is used to create a RandomForestClassifier object by passing the following parameters:
    (i) n_estimator: Number of trees in the forest, default is 100
    (ii) "criterion", which represents the measure used in expanning the decision tree.
         There are a number of options for this parameter, such as `gini` (Gini impurity) and `entropy` or 'log_loss(information gain).
         By default, it is `gini`. We use `entropy`.
    (iii) max_depth: Maximum depth of the trees,
         default is None (nodes are expanded until all leaves are pure or
         until they contain fewer than 'min_sample_split' samples)
    (iv) min_samples_split: Minum number of samples required to split an internal node, default is 2
    (v) min_samples_leaf: Minum number of samples required to be at a leaf node, default is 1
    (vi) max_features: number of features to consider when looking for the best split, default is None (max_features=n_features).
        Other values are 'sqrt' (max_features=sqrt(n_features)) and "log2” (max_features=log2(n_features))
    (vii) 'bootstrap: Whether to use bootstrap samples when building trees, default is 'True'.
          If 'False', the entire dataset is used to build each tree, which may lead to overfitting.
    (viii) oob_score: Whether to use out-of-bag samples to estimate the generalisation accuracy.
          default value is 'False'. If 'True', an unbiased estimate of the model performance is provided.
    """
    # Define model + hyperparameter grid (tuned *on validation set*)
    rf_param_grid: dict[str, list[Any]] = {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 3, 5, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],  # 'auto' is deprecated
    }

    rf_fixed_params: dict[str, Any] = {
        "random_state": 42,
        #   "n_jobs": -1,
        "oob_score": True,  # OOB uses training data only; helpful as an internal check
        "bootstrap": True,
    }

    # Define evaluation function
    def rf_evaluate(params: dict[str, Any]) -> tuple[float, float, RandomForestClassifier]:
        clf = RandomForestClassifier(**rf_fixed_params, **params)
        clf.fit(X_train, y_train)  # fit on TRAIN only
        val_preds = clf.predict(X_val)  # evaluate on VALIDATION
        val_acc = accuracy_score(y_val, val_preds)
        oob = getattr(clf, "oob_score_", np.nan)  # OOB score (training internal estimate)
        return val_acc, oob, clf

    # Manual grid search using validation accuracy (break ties by shallower depth)
    rf_results: list[dict[str, Any]] = []
    rf_best: dict[str, Any] = {"acc": float(-np.inf), "params": None, "clf": None}

    def rf_tie_key(p: dict[str, Any]) -> tuple[int, int, int, int]:
        # Prefer simpler models when validation accuracies tie:
        # fewer trees, shallower depth, larger leaves/splits (more regularization)
        depth_val = 10**6 if p["max_depth"] is None else p["max_depth"]
        return (p["n_estimators"], depth_val, -p["min_samples_leaf"], -p["min_samples_split"])

    rf_keys, rf_values = zip(*rf_param_grid.items(), strict=False)
    for combo in product(*rf_values):
        params = dict(zip(rf_keys, combo, strict=False))
        try:
            acc, oob, clf = rf_evaluate(params)
            row = {**params, "val_acc": acc, "oob_score": oob}
            rf_results.append(row)

            if (acc > rf_best["acc"]) or (
                np.isclose(acc, rf_best["acc"])
                and rf_tie_key(params) < rf_tie_key(rf_best["params"])
            ):
                rf_best = {"acc": acc, "params": params, "clf": clf}
        except Exception as e:
            rf_results.append({**params, "val_acc": np.nan, "oob_score": np.nan})
            print(f"Skipped {params} due to: {e}")

    # Show validation results sorted
    rf_df_results = pd.DataFrame(rf_results).sort_values(
        ["val_acc", "n_estimators", "max_depth"], ascending=[False, True, True]
    )
    print("\nValidation results (top 10):")
    print(rf_df_results.head(10).to_string(index=False))

    print("\nBest on validation:")
    print(rf_best["params"], "val_acc=", round(rf_best["acc"], 4))

    # Refit best model on TRAIN + VAL (finalize before testing)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    final_clf = RandomForestClassifier(**rf_fixed_params, **rf_best["params"])
    final_clf.fit(X_train_val, y_train_val)

    # Final evaluation on TEST
    y_pred = final_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[RandomForest] Test accuracy:", round(test_acc, 4))
    # Future sample prediction
    future_pred = final_clf.predict(futureSample_X)
    print("[RandomForest] Future predictions:", future_pred.tolist())

elif model_option == 3:
    # Task: Import LogisticRegression from sklearn.linear_model.

    """
    The constructor LogisticRegression() is uded to create a LogisticRegresion object by passing the following parameters:.
    (i) 'penalty':   Specify the norm of the penalty.
         Possible values: None (no penalty); 'l2' (add a L2 penalty term);
         'l1' (add a L1 penalty term); 'elasticnet' (both L1 and L2 penalty terms are added).  Default is l2
    (ii) 'dual': Dual (constrained) or primal (regularized) formulation. Default is False.
         Dual formulation is only implemented for l2 penalty with liblinear solver.
    (iii) 'random_state': Used when solver == "sag", "saga" or "liblinear" to shuffle the data. Default is None

    (iv)'solver': Algorithm to use in the optimization problem.
         possible values: "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", and "saga". Default is "lbfgs"
         To choose a solver, you might want to consider the following aspects
         (more info see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html):
          -- For small datasets, "liblinear" is a good choice, whereas "sag" and "saga" are faster for large ones;
          -- For multiclass problems, only "newton-cg", "sag", "saga" and "lbfgs" handle multinomial loss;
          -- "liblinear" and "newton-cholesky" can only handle binary classification by default.
          -- To apply a one-versus-rest scheme for the multiclass setting one can wrapt it with the OneVsRestClassifier.
          -- "newton-cholesky" is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories.
             Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.
    (v) 'max_iter': Maximum number of iterations taken for the solvers to converge. Default is 100.
    """
    # Define model + hyperparameter grid (tuned *on validation set*)
    lr_param_grid: dict[str, list[Any]] = {
        "C": [0.01, 0.1, 1.0, 3.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    }
    lr_fixed_params: dict[str, Any] = {
        "max_iter": 1000,
        # NOTE: 'multi_class' is deprecated in sklearn>=1.5 and removed in 1.7.
        # Leave it to default to avoid FutureWarning and ensure compatibility.
        "penalty": "l2",
    }

    lr_scaler = StandardScaler()
    X_train_s = lr_scaler.fit_transform(X_train)
    X_val_s = lr_scaler.transform(X_val)

    # Define evaluation function
    def lr_evaluate(C: float, solver: str) -> tuple[float, LogisticRegression]:
        clf = LogisticRegression(C=C, solver=solver, **lr_fixed_params)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_val_s)
        return accuracy_score(y_val, preds), clf

    # Manual grid search using validation accuracy
    lr_results: list[dict[str, Any]] = []
    lr_best: dict[str, Any] = {"acc": float(-np.inf), "params": None, "model": None}
    for C_val, solver_val in product(lr_param_grid["C"], lr_param_grid["solver"]):
        try:
            acc, model = lr_evaluate(C_val, solver_val)
            lr_results.append({"C": C_val, "solver": solver_val, "val_acc": acc})
            if acc > lr_best["acc"]:
                lr_best = {"acc": acc, "params": {"C": C_val, "solver": solver_val}, "model": model}
        except Exception as e:
            # Some solver/param combos can fail depending on data; keep going.
            lr_results.append({"C": C_val, "solver": solver_val, "val_acc": np.nan})
            print(f"Skipped C={C_val}, solver={solver_val} due to: {e}")

    # Show validation results sorted
    lr_df_results = pd.DataFrame(lr_results).sort_values("val_acc", ascending=False)
    print("\nValidation results (top rows):")
    print(lr_df_results.head(10).to_string(index=False))

    print("\nBest on validation:")
    print(lr_best["params"], "val_acc=", round(lr_best["acc"], 4))

    # Refit the best model on TRAIN + VAL (finalize before test)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    # IMPORTANT: Refit scaler on train+val, then transform both train+val and test
    lr_scaler_final = StandardScaler()
    X_train_val_s: NDArray[np.float64] = lr_scaler_final.fit_transform(X_train_val)
    X_test_s: NDArray[np.float64] = lr_scaler_final.transform(X_test)
    futureSample_X_s: NDArray[np.float64] = lr_scaler_final.transform(futureSample_X)

    final_clf = LogisticRegression(
        C=lr_best["params"]["C"],
        solver=lr_best["params"]["solver"],
        **lr_fixed_params,
    )
    final_clf.fit(X_train_val_s, y_train_val)

    # Final evaluation on TEST
    y_pred = final_clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[LogisticRegression] Test accuracy:", round(test_acc, 4))
    # Future sample prediction (scaled)
    future_pred = final_clf.predict(futureSample_X_s)
    print("[LogisticRegression] Future predictions:", future_pred.tolist())

elif model_option == 4:
    # Task: Import `KNeighborsClassifier` from sklearn.neighbors.

    """
    Call the constructor `KNeighborsClassifier()` to create a KNN classifier object, name it as 'model', by passing the following key parameters:.
    Set the number of neighbors as 3
    (i) 'n_neighbors': Number of neighbors to use by default for kneighbors queries. Default is 5.
    (ii) 'weights': Weight function used in prediction. Possible values are "uniform" (uniform weights),
         "distance" (weight points by the inverse of their distance) and
         [callable] (a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights).
    (iii) 'algorithm': Algorithm used to compute the nearest neighbors. Possible values are "auto" (attempt to decide the most appropriate algorithm based on the values passed to fit method),
          "ball_tree" (use BallTree), "kd_tree" (use KDTree), and "brute" (use a brute-force search). Default is"auto".
    """
    # Define model + hyperparameter grid (tuned *on validation set*)
    knn_param_grid: dict[str, list[Any]] = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1=Manhattan, 2=Euclidean
    }

    knn_scaler = StandardScaler()
    X_train_s = knn_scaler.fit_transform(X_train)
    X_val_s = knn_scaler.transform(X_val)

    # Define evaluation function
    def knn_evaluate(params: dict[str, Any]) -> tuple[float, KNeighborsClassifier]:
        clf = KNeighborsClassifier(**params)
        clf.fit(X_train_s, y_train)  # fit on TRAIN only
        val_preds = clf.predict(X_val_s)  # evaluate on VALIDATION
        val_acc = accuracy_score(y_val, val_preds)
        return val_acc, clf

    def knn_tie_key(p: dict[str, Any]) -> tuple[int, int, int]:
        # Lower tuple is preferred on ties:
        # (-k) → prefer larger k (more smoothing),
        # weight_rank: uniform over distance,
        # p_rank: Euclidean (2) over Manhattan (1)
        k = p["n_neighbors"]
        weight_rank = 0 if p["weights"] == "uniform" else 1
        p_rank = 0 if p["p"] == 2 else 1
        return (-k, weight_rank, p_rank)

    # Manual grid search using validation accuracy
    knn_results: list[dict[str, Any]] = []
    knn_best: dict[str, Any] = {"acc": float(-np.inf), "params": None, "clf": None, "tie": None}

    knn_keys, knn_values = zip(*knn_param_grid.items(), strict=False)
    for combo in product(*knn_values):
        params = dict(zip(knn_keys, combo, strict=False))
        try:
            acc, clf = knn_evaluate(params)
            knn_results.append({**params, "val_acc": acc})
            if (acc > knn_best["acc"]) or (
                np.isclose(acc, knn_best["acc"])
                and (knn_best["tie"] is None or knn_tie_key(params) < knn_best["tie"])
            ):
                knn_best = {"acc": acc, "params": params, "clf": clf, "tie": knn_tie_key(params)}
        except Exception as e:
            knn_results.append({**params, "val_acc": np.nan})
            print(f"Skipped {params} due to: {e}")

    # Show validation results sorted
    knn_df_results = pd.DataFrame(knn_results).sort_values(
        ["val_acc", "n_neighbors", "weights", "p"], ascending=[False, False, True, True]
    )
    print("\nValidation results (top 10):")
    print(knn_df_results.head(10).to_string(index=False))

    print("\nBest on validation:")
    print(knn_best["params"], "val_acc=", round(knn_best["acc"], 4))

    # Refit best model on TRAIN + VAL (finalize before testing)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    # Refit scaler on train+val to avoid leakage
    knn_scaler_final = StandardScaler()
    X_train_val_s = knn_scaler_final.fit_transform(X_train_val)
    X_test_s = knn_scaler_final.transform(X_test)
    futureSample_X_s = knn_scaler_final.transform(futureSample_X)

    final_clf = KNeighborsClassifier(**knn_best["params"])
    final_clf.fit(X_train_val_s, y_train_val)

    # Final evaluation on TEST
    y_pred = final_clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[KNN] Test accuracy:", round(test_acc, 4))
    # Future sample prediction (scaled)
    future_pred = final_clf.predict(futureSample_X_s)
    print("[KNN] Future predictions:", future_pred.tolist())

elif model_option == 5:
    # Task: Import SVC from sklearn.svm.

    """
    The constructor SVC() is used to create a SVC object by passing the following key parameters:

    (i) 'C': Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        Default=1.0
    (ii) 'kernel': Specifies the kernel type to be used in the algorithm. possible values are "linear", "poly", "rbf", "sigmoid" and "precomputed".
         Default is "rbf".
    (iii) 'degree': Degree of the polynomial kernel function ("poly"). Must be non-negative. Ignored by all other kernels. Default=3
    (iv) 'gamma': Kernel coefficient for "rbf", "poly" and "sigmoid".
          Possible values are "scale" (1 / (n_features * X.var())) and "auto" (1 / n_features).
          Default is "scale".
    (v) 'probability': Whether to enable probability estimates. Default is False.
         This must be enabled prior to calling fit. If use 'True', it will slow down that method as it internally uses 5-fold cross-validation.
    (vi) 'max_iter': Hard limit on iterations within solver, or -1 for no limit. Default is -1.
    (vii) 'decision_function_shape': Whether to return a one-vs-rest ("ovr") decision function of shape (n_samples, n_classes) as all other classifiers,
          or the original one-vs-one ("ovo") decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
          Default is "ovr". The parameter is ignored for binary classification.
    (viii) 'random_state': Controls the pseudo random number generation for shuffling the data for probability estimates.
          Ignored when probability is False. Default is None.
          Pass an int for reproducible output across multiple function calls.
    """
    # Define model + hyperparameter grid (tuned *on validation set*)
    #    Keep it compact but representative; gamma is ignored for 'linear' kernel.
    svc_param_grid: dict[str, list[Any]] = {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 3, 10, 30, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1.0],  # used only if kernel != 'linear'
    }

    svc_scaler = StandardScaler()
    X_train_s = svc_scaler.fit_transform(X_train)
    X_val_s = svc_scaler.transform(X_val)

    def svc_evaluate(params: dict[str, Any]) -> tuple[float, SVC]:
        clf = SVC(**params)
        clf.fit(X_train_s, y_train)  # fit on TRAIN only
        val_preds = clf.predict(X_val_s)  # evaluate on VALIDATION only
        val_acc = accuracy_score(y_val, val_preds)
        return val_acc, clf

    def svc_gamma_rank(g: str | float) -> tuple[int, float]:
        # Prefer 'scale' > 'auto' > numeric (smaller better)
        if g == "scale":
            return (0, 0.0)
        if g == "auto":
            return (1, 0.0)
        return (2, float(g))

    def svc_tie_key(p: dict[str, Any]) -> tuple[int, float, tuple[int, float]]:
        # Prefer simpler models on ties:
        # linear kernel over rbf; smaller C; 'scale' gamma over others, then smaller gamma
        kernel_rank = 0 if p["kernel"] == "linear" else 1
        C_rank = float(p["C"])
        g_rank = svc_gamma_rank(p["gamma"])
        return (kernel_rank, C_rank, g_rank)

    # Manual grid search using validation accuracy
    svc_results: list[dict[str, Any]] = []
    svc_best: dict[str, Any] = {"acc": float(-np.inf), "params": None, "clf": None, "tie": None}

    svc_keys, svc_values = zip(*svc_param_grid.items(), strict=False)
    for combo in product(*svc_values):
        params = dict(zip(svc_keys, combo, strict=False))
        # SVC ignores gamma for linear kernel, that's fine.
        try:
            acc, clf = svc_evaluate(params)
            svc_results.append({**params, "val_acc": acc})
            if (acc > svc_best["acc"]) or (
                np.isclose(acc, svc_best["acc"])
                and (svc_best["tie"] is None or svc_tie_key(params) < svc_best["tie"])
            ):
                svc_best = {"acc": acc, "params": params, "clf": clf, "tie": svc_tie_key(params)}
        except Exception as e:
            svc_results.append({**params, "val_acc": np.nan})
            print(f"Skipped {params} due to: {e}")

    # Show validation results sorted
    svc_df_results = pd.DataFrame(svc_results).sort_values(
        ["val_acc", "kernel", "C", "gamma"], ascending=[False, True, True, True]
    )
    print("\nValidation results (top 10):")
    print(svc_df_results.head(10).to_string(index=False))

    print("\nBest on validation:")
    print(svc_best["params"], "val_acc=", round(svc_best["acc"], 4))

    # Refit best model on TRAIN + VAL (finalize before testing)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    # Refit scaler on train+val to avoid leakage
    svc_scaler_final = StandardScaler()
    X_train_val_s = svc_scaler_final.fit_transform(X_train_val)
    X_test_s = svc_scaler_final.transform(X_test)
    futureSample_X_s = svc_scaler_final.transform(futureSample_X)

    final_clf = SVC(**svc_best["params"])
    final_clf.fit(X_train_val_s, y_train_val)

    # Final evaluation on TEST
    y_pred = final_clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[SVC] Test accuracy:", round(test_acc, 4))
    # Future sample prediction (scaled)
    future_pred = final_clf.predict(futureSample_X_s)
    print("[SVC] Future predictions:", future_pred.tolist())
else:
    print("invalid option number. Try again")

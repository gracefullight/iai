from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Import Sequential from keras.models

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
dataframe: pd.DataFrame = pd.read_csv(
    ASSETS_DIR / "airline-passengers.csv", usecols=[1], engine="python"
)

problem_code = int(
    input(
        "Input a problem code: problem_code = 1 -- a regression problem with single output, problem_code = 2 -- a regression problem with multi-outputs, problem_code = 3 -- a multi-class classification problem. Your choice: "
    )
)

if problem_code == 1:
    """
    Read a CSV file (the target csv file) into a DataFrame, called df, using pandas' read_csv function. The read_csv function loads the data from the CSV file into a pandas DataFrame, which is a tabular data structure with labeled axes. This method takes two parameters:
    the csv file name with its path and names, which lists the column names, such as ['x1','x2','y']
    """
    df = pd.read_csv(ASSETS_DIR / "HousingData.csv", names=["x1", "x2", "y"])
    # Display the dimensions of the dataframe df
    print(df.shape)
    # Display the first 5 rows of this dataframe df
    df.head()
    # Create the input data X and target y
    """
    Separate the dataset into inputs and output parts and save the inputs in a
    dataframe X and output in a series y. Use iloc() method to selects all rows
    and all columns starting from the second column to the end of the DataFrame
    and assign them to the inputs X. Use iloc() method to select the first
    column and assign it to the output y.
    """
    X = df.iloc[:, 0:2]
    y = df.iloc[:, -1]
    # print(y)
elif problem_code == 2:
    from sklearn.datasets import load_linnerud

    # Load and return the physical exercise Linnerud dataset.
    X, y = load_linnerud(return_X_y=True, as_frame=True)
    # print(y)
elif problem_code == 3:
    """
    Read a CSV file (the target csv file) into a DataFrame, called df, using pandas' read_csv function. The read_csv function loads the data from the CSV file into a pandas DataFrame, which is a tabular data structure with labeled axes. This method takes two parameters:
    the csv file name with its path and names, which lists the column names, such as
    ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']
    """
    df = pd.read_csv(
        ASSETS_DIR / "wine.csv",
        names=[
            "y",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
        ],
    )
    # Display the dimensions of the dataframe df
    print(df.shape)
    # Display the first 5 rows of this dataframe df
    df.head()
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    y = y - 1  # Decrementing by 1 to make labels start from 0
    # print(y)
else:
    print("Invalid problem code. Try a correct one: 1 or 2")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

futureSample_data = X_test[-2:]
futureSample_label = y_test[-2:]
print(futureSample_data)
print(futureSample_label)
X_test = X_test[:-2]
y_test = y_test[:-2]

if problem_code == 1:
    # Set input dimension for the FNN
    input_dim = X_train.shape[1]
    # Define the layers and number of neurons in each layer
    n_neurons_1 = 512
    n_neurons_2 = 512
    n_neurons_3 = 100
    # Set the output dimension
    output_dim = 1
    # Set up parameters for running the FNN model
    epochNo = 20
    batchSize = 10
elif problem_code == 2:
    # Set input dimension for the FNN
    input_dim = X_train.shape[1]
    # Define the layers and number of neurons in each layer
    n_neurons_1 = 512
    n_neurons_2 = 512
    n_neurons_3 = 100
    # Set the output dimension
    output_dim = 3
    # Set up parameters for running the FNN model
    epochNo = 20
    batchSize = 10
elif problem_code == 3:
    # Set input dimension for the FNN
    input_dim = X_train.shape[1]
    # Define the layers and number of neurons in each layer
    n_neurons_1 = 512
    n_neurons_2 = 512
    n_neurons_3 = 100
    # Define the output dimension
    output_dim = 3
    # Set up parameters for running the FNN model
    epochNo = 20
    batchSize = 16
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(input_dim,)))
model.add(Dense(n_neurons_1, activation="relu"))
model.add(Dense(n_neurons_2, activation="relu"))
model.add(Dense(n_neurons_3, activation="relu"))
if problem_code == 1 or problem_code == 2:
    model.add(Dense(output_dim))
elif problem_code == 3:
    model.add(Dense(output_dim, activation=tf.keras.activations.softmax))
else:
    print("invalide code")

# Display the layers in the newly created NN model
print(f"The model layers is {len(model.layers)}: ")


# Compile the constructed model
if problem_code == 1 or problem_code == 2:
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
elif problem_code == 3:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")
model.summary()

model.fit(X_train, y_train, epochs=epochNo, batch_size=batchSize, verbose=0)

# Predict the output of the test set using method predict() from the model
if problem_code == 1 or problem_code == 2:
    pred = model.predict(X_test)
elif problem_code == 3:
    predictions = model.predict(
        X_test
    )  # each row contains the probability scores for the 3 classes for a particular sample.
    y_preds = predictions.argmax(axis=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")


if problem_code == 1 or problem_code == 2:
    # Calculate the R2 score using method r2_score() from metrics
    r_square_test = metrics.r2_score(y_test, pred)
    print(f"r_square_test is {r_square_test}:")
    # Calculate the mean absolute error using method mean_absolute_error() from metrics
    mean_absolute_error_test = metrics.mean_absolute_error(y_test, pred)
    print(f"mean_absolute_error_test is {mean_absolute_error_test}:")
    # Calculate the mean squared error of test set using method mean_squared_error() from metrics
    mean_squared_error_test = metrics.mean_squared_error(y_test, pred)
    print(f"mean_squared_error_test is {mean_squared_error_test}:")

    # Present the regression plot and import required packages
    import matplotlib.pyplot as plt

    plt.scatter(y_test, pred)
    plt.xlabel("y_test")
    plt.ylabel("pred")
    plt.show()
elif problem_code == 3:
    accuracy_test = metrics.accuracy_score(y_preds, y_test)
    print(f"accuracy_test is {accuracy_test}:")
    precision_test = metrics.precision_score(y_test, y_preds, average="weighted")
    print(f"precision_test is {precision_test}:")
    recall_test = metrics.recall_score(y_test, y_preds, average="weighted")
    print(f"recall_test is {recall_test}:")
    f1_score_test = metrics.f1_score(y_test, y_preds, average="weighted")
    print(f"f1_score_test is {f1_score_test}:")
    # import `pyplot` from matplotlib and give it an alias as `plt`
    import matplotlib.pyplot as plt

    # import `ConfusionMatrixDisplay` and `confusion_matrix` from `sklearn.metrics`
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    cm = confusion_matrix(y_test, y_preds)
    cmp = ConfusionMatrixDisplay(cm, display_labels=["class_0", "class_1", "class_2"])
    cmp.plot(ax=ax)
    # Create the evaluation report, called evaluation_report, and display it
    evaluation_report = metrics.classification_report(y_test, y_preds)
    print(evaluation_report)
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")

model.save(ASSETS_DIR / "model.keras")


loaded_model = load_model(ASSETS_DIR / "model.keras")
loaded_model.summary()

# Predict the output for the future samples using method predict() from the loaded model, loaded_model
if problem_code == 1 or problem_code == 2:
    preds_future = loaded_model.predict(futureSample_data)
elif problem_code == 3:
    predictions_val = loaded_model.predict(futureSample_data)
    preds_future = predictions_val.argmax(axis=1)
    print(
        f"The predicated classes are {preds_future} vs the true classes are {futureSample_label.values}"
    )
else:
    print("Invalid code")

for i in range(2):
    print(
        f"The future data is {futureSample_data.values[i]}, the predicted value is {preds_future[i]} and the acutal value is {futureSample_label.values[i]}"
    )

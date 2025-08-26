from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# assets live in the repository-level `src/assets` directory (not inside week4/)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# filenames in src/assets use British spelling 'Versicolour'
img1 = plt.imread(ASSETS_DIR / "Iris_Setosa.png")
img2 = plt.imread(ASSETS_DIR / "Iris_Versicolour.png")
img3 = plt.imread(ASSETS_DIR / "Iris_Virginica.png")


# fig, axes = plt.subplots(1, 3)
# axes[0].imshow(img1)
# axes[0].set_title("Iris Setosa")
# axes[0].axis("off")

# axes[1].imshow(img2)
# axes[1].set_title("Iris Versicolor")
# axes[1].axis("off")

# axes[2].imshow(img3)
# axes[2].set_title("Iris Virginica")
# axes[2].axis("off")


# plt.show()

iris = pd.read_csv(ASSETS_DIR / "Iris.csv")
print(iris.head(5))

le = LabelEncoder()
iris["species"] = le.fit_transform(iris["species"])
print(le.classes_)
print(iris.head(5))

# Define the input features X as all columns except 'species' column by using drop() method for a dataframe and target variable y as the "species" column
x = iris.drop("species", axis=1)
y = iris["species"]


# Using train_test_split function to split the dataset into the train and temp (validation plus test) datasets, the percentage of samples in the temp dataset is 40% and the seed for the random number generator is 42
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

# Using train_test_split function to split the temp (validation plus test) datasets into the validation and test datasets, the percentage of samples in the test dataset is 20% and the seed for the random number generator is 42
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Make a futureSample test dataset
# Take two samples from the dataset as the future data samples, called futureSample_X, and futureSample_y,
# as the inputs from the real-world cases when the classifier is deployed.

# Get the last two samples from the test  to be the future data samples
futureSample_X = x_test[-2:]
futureSample_y = y_test[-2:]

# Remove the last two samples from the test dataset
x_test = x_test[:-2]
y_test = y_test[:-2]
plt.figure(figsize=(8, 6))
plt.scatter(
    x_train["sepal.length"],
    x_train["sepal.width"],
    c=y_train,
    cmap=plt.get_cmap("Set1"),
    edgecolor="k",
)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Train data (sepal attributes)")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    x_train["petal.length"],
    x_train["petal.width"],
    c=y_train,
    cmap=plt.get_cmap("Set1"),
    edgecolor="k",
)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Train data (petal attributes)")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    x_val["sepal.length"],
    x_val["sepal.width"],
    c=y_val,
    cmap=plt.get_cmap("Set1"),
    edgecolor="k",
)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Validation data (sepal attributes)")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    x_val["petal.length"],
    x_val["petal.width"],
    c=y_val,
    cmap=plt.get_cmap("Set1"),
    edgecolor="k",
)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Validation data (petal attributes)")
plt.show()

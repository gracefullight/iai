from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
data: pd.DataFrame = pd.read_csv(ASSETS_DIR / "HousingData.csv")

print(data.head())

x = data[["HouseAge", "HouseSize"]]
y = data["HousePrice"]

# x = data.drop("HousePrice", axis=1)
# y = data["HousePrice"]

# Using train_test_split function from sklearn to split the dataset into the training and test datasets, the percentage of samples in the test dataset is 20%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

futureSample_x = x_test[-2:]
futureSample_y = y_test[-2:]

x_test = x_test[:-2]
y_test = y_test[:-2]

plt.figure(1, figsize=(8, 6))
plt.scatter(x_train["HouseAge"], y_train, c=y_train, cmap=plt.get_cmap("Set1"), edgecolor="k")
plt.xlabel("House Age")
plt.ylabel("House Price")
plt.title("Training data (age-price)")
plt.show()

plt.figure(1, figsize=(8, 6))
plt.scatter(x_train["HouseSize"], y_train, c=y_train, cmap=plt.get_cmap("Set1"), edgecolor="k")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Training data (size-price)")
plt.show()

plt.figure(3, figsize=(8, 6))
plt.scatter(x_test["HouseAge"], y_test, c=y_test, cmap=plt.get_cmap("Set1"), edgecolor="k")
plt.xlabel("House Age")
plt.ylabel("House Price")
plt.title("Test data (age-price)")
plt.show()

plt.figure(4, figsize=(8, 6))
plt.scatter(x_test["HouseSize"], y_test, c=y_test, cmap=plt.get_cmap("Set1"), edgecolor="k")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Test data (size-price)")
plt.show()

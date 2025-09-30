import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

max_features = 20000  # Consider only the top 50,000 words from the dataset
maxlen = 500  # Truncate or pad sequences to this length
batch_size = 32  # Number of samples processed before the model is updated

print("Loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), "train sequences")
print(len(input_test), "test sequences")

print("Before padding:")
print("Length of the first training sequence: ", len(input_train[0]))
print("First training sequence: ", input_train[0])

# Decode the content to words
word_index = imdb.get_word_index()
print(f"The length of word index is {len(word_index)}")

print(f"word index is \n {word_index}\n")


# Convert a review in a form of list of IDs back into a human-readable string (words separated by spaces).
def decode_review(review: list[int]) -> str:
    # Remove any padding tokens
    review = [token for token in review if token != 0]
    reverse_word_index = {value: key for key, value in word_index.items()}
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in review]
    )  # For each token i, it looks up i - 3 in reverse_word_index. The - 3 offset is typical in datasets like IMDB where the first few indices are reserved.


decode_review(input_train[0])

# Get lengths of the reviews
review_lengths = [len(review) for review in input_train]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(review_lengths, bins=30)
plt.title("Histogram of Review Lengths")
plt.xlabel("Review Length")
plt.ylabel("Frequency")
plt.show()

# Pad an input sequence to a given length
print("Pad sequences into samples with maxlen")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)

# Print some sequences after padding
print("\nAfter padding:")
print("Length of the first training sequence: ", len(input_train[0]))
print("First training sequence: ", input_train[0])

"""
The model will consist of an embedding layer to convert word indices into dense vectors,
followed by an LSTM layer, and then a fully connected layer to produce the final sentiment probabilities.
"""
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adamW", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(input_train, y_train, epochs=9, batch_size=128, validation_split=0.2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, acc, "b", label="Training acc", linewidth=2)
ax1.plot(epochs, val_acc, "r", label="Validation acc", linewidth=2)
ax1.set_title("Training and Validation Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(epochs, loss, "b", label="Training loss", linewidth=2)
ax2.plot(epochs, val_loss, "r", label="Validation loss", linewidth=2)
ax2.set_title("Training and Validation Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()

plt.show()

for i in range(100, 110):
    # Convert the i-th review to a numpy array and expand its dimensions
    review = np.expand_dims(input_test[i], axis=0)

    # Get the prediction for the i-th review
    prediction = model.predict(review)[0][0]
    predicted_label = "positive" if prediction > 0.5 else "negative"

    # Get the actual label
    actual_label = "positive" if y_train[i] == 1 else "negative"

    # Print the original review, the predicted result, and the actual label
    print(f"Review {i + 1}:")
    print(decode_review(input_test[i]))
    print("Predicted sentiment: ", predicted_label)
    print("Actual sentiment: ", actual_label)
    print("\n")

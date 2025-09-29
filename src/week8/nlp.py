import string
from collections import Counter
from pathlib import Path

import nltk
from heading import Vectors
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
nltk.download("popular")
nltk.download("punkt_tab")

sent_tokenize_result = sent_tokenize(EXAMPLE_TEXT)
print(sent_tokenize_result)

word_tokenize_result = word_tokenize(EXAMPLE_TEXT)
print(word_tokenize_result)

all_punctuation = string.punctuation
print(f"All punctuation: {all_punctuation}")

# You can configure for the language you need. In this example, you can use 'English'
stops = set(stopwords.words("english"))
print(f"All stopwords: {stops}")

# Filter the stop words and punctuations etc by removing the stop words and punctuations in the word list
# removales = list(stops)+list(all_punctuation)+list("n't")
wordsWithoutStopWords = []

for w in word_tokenize_result:
    w_lower = w.lower()
    if (w_lower not in stops) and (w_lower != "n't") and (w_lower not in all_punctuation):
        wordsWithoutStopWords.append(w)

print(wordsWithoutStopWords)

WEEK8_FOLDER = Path(__file__).resolve().parent
file_directory = WEEK8_FOLDER / "data"

num_words = 2


class Ngrams:
    def __init__(self, directory_path: Path, n: int) -> None:
        self.directory_path = directory_path
        self.n = n
        self.ngram_counts = self._generate_ngrams()

    def _generate_ngrams(self) -> Counter[tuple[str, ...]]:
        all_words = []
        for file_path in self.directory_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".txt":
                with file_path.open(encoding="utf-8") as f:
                    text = f.read()
                    # Simple tokenization and lowercasing
                    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
                    all_words.extend(words)

        # Generate n-grams and count frequencies
        n_grams = list(nltk.ngrams(all_words, self.n))
        return Counter(n_grams)

    def top_term_frequencies(self, top_n: int) -> None:
        """Display the top N n-grams and their frequencies."""
        for ngram, freq in self.ngram_counts.most_common(top_n):
            print(f"{freq}: {ngram}")


ngrams = Ngrams(file_directory / "holmes", num_words)
ngrams.top_term_frequencies(10)

ngrams_test = Counter(nltk.ngrams(wordsWithoutStopWords, num_words))
print(ngrams_test)

for ngram, freq in ngrams_test.most_common(5):
    print(f"{freq}: {ngram}")

vectors = Vectors(file_directory / "words.txt")
words = vectors.words

print(words["city"])

print(vectors.distance(words["city"], words["book"]))

print(vectors.closest_words(words["book"])[:10])

print(words["paris"] - words["france"] + words["england"])

print(vectors.closest_word(words["paris"] - words["france"] + words["england"]))

# A corpus D
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are great pets.",
    "The quick brown fox jumps over the lazy dog.",
]

# a query term
query = "A dog on the mat."

# All documents including the query
all_documents = documents + [query]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

# Convert TF-IDF matrix to a Numpy array
X_array = tfidf_matrix.toarray()


print(
    f"Number of documents including the query document is {tfidf_matrix.shape[0]} and the size of vocabulary is {tfidf_matrix.shape[1]}\n."
)
print(f"the first document is '{all_documents[0]}'")
print(f"The tf-idf representation of this document in array format is \n {X_array[0]} ")
print(f"The tf-idf representation of this document is \n {tfidf_matrix[0]} ")

print(f"the query document is '{all_documents[-1]}'")
print(f"The tf-idf representation of this document in array format is \n {X_array[-1]} ")
print(f"The tf-idf representation of this document is \n {tfidf_matrix[-1]} \n")


# Compute cosine similarity between query and given documents
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Display similarity scores
for i, score in enumerate(similarity_scores[0]):
    print(f"Similarity to document {i + 1}: {score:.4f}")

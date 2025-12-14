# Corrected train_chatbot.py
# Robust preprocessing and proper conversion of training -> train_x, train_y
# Replace paths/names as needed to match your project structure.

import json
import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
SGD = keras.optimizers.SGD
import os

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Load intents (adjust filename if different)
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

# Build words, classes, documents
for intent in intents.get("intents", []):
    tag = intent.get("tag")
    if tag not in classes:
        classes.append(tag)
    for pattern in intent.get("patterns", []):
        # tokenize each pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, tag))

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes {classes}")
print(f"{len(words)} unique lemmatized words {words}")

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    # output is a one-hot for the class
    output_row = list(output_empty)  # copy
    tag_index = classes.index(doc[1])
    output_row[tag_index] = 1

    training.append([bag, output_row])

# Diagnostic: check each training item for consistent lengths
expected_bag_len = len(words)
expected_output_len = len(classes)

for i, item in enumerate(training):
    try:
        bag, output = item
    except Exception:
        print(f"Malformed training item at index {i}: {item}")
        raise

    bag_len = len(bag) if hasattr(bag, "__len__") else None
    out_len = len(output) if hasattr(output, "__len__") else None

    if bag_len != expected_bag_len or out_len != expected_output_len:
        print(f"Inconsistent sizes at index {i}: bag_len={bag_len}, expected={expected_bag_len}; "
              f"output_len={out_len}, expected={expected_output_len}")
        raise ValueError("Inconsistent training vector sizes detected. Fix preprocessing.")

# Split features and labels and convert to numpy arrays
train_x = []
train_y = []

for bag, output in training:
    # Ensure plain Python lists (robustness)
    if isinstance(bag, np.ndarray):
        bag = bag.tolist()
    if isinstance(output, np.ndarray):
        output = output.tolist()

    train_x.append(bag)
    train_y.append(output)

train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)

print("train_x shape:", train_x.shape)  # expected (n_samples, n_words)
print("train_y shape:", train_y.shape)  # expected (n_samples, n_classes)

# Build a simple model (optional — keep if you train here)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model (adjust epochs/batch_size as needed)
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model and data
model.save("chatbot_model.h5")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Note about TensorFlow oneDNN logs
# If you want to suppress the oneDNN informational messages, set:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# before importing tensorflow. The messages themselves are informational and not the cause of the numpy error.

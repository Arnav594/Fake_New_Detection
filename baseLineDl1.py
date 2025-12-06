# fake_news_dense.py
import pandas as pd, re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===== Load and preprocess =====
df = pd.read_excel("bharatfakenewskosh.xlsx")

text_col = 'Statement' if 'Statement' in df.columns else df.columns[0]
label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
df = df[[text_col, label_col]].dropna().rename(columns={text_col:'text', label_col:'label'})

def clean(s):
    s = str(s).lower()
    s = re.sub(r'http\S+|[^a-z\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

df['text'] = df['text'].map(clean)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# ===== Tokenize =====
max_words = 30000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# ===== Model =====
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_seq, y_train, validation_split=0.2, epochs=5, batch_size=64)

# ===== Evaluation =====
test_loss, test_acc = model.evaluate(X_test_seq, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = (model.predict(X_test_seq) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend(); plt.show()

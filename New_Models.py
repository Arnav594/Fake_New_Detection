# ==================== CELL 1: Imports ====================
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, GRU, LSTM, Dense, Dropout, Bidirectional, 
    BatchNormalization, Conv1D, GlobalMaxPooling1D, 
    MaxPooling1D, Flatten, Concatenate, Input, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# Machine Learning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Found: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
        
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled")

# ==================== CELL 2: Load and Prepare Data ====================
df = pd.read_excel("bharatfakenewskosh.xlsx")

statement_col = 'Eng_Trans_Statement'
body_col = 'Eng_Trans_News_Body'
label_col = 'Label'

df_clean = df[[statement_col, body_col, label_col]].dropna()

def clean(s):
    s = str(s).lower()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

df_clean['statement_clean'] = df_clean[statement_col].map(clean)
df_clean['body_clean'] = df_clean[body_col].map(clean)
df_clean['combined_text'] = df_clean['statement_clean'] + " " + df_clean['body_clean']

le = LabelEncoder()
df_clean['label'] = le.fit_transform(df_clean[label_col])

print(f"✓ Data prepared: {len(df_clean)} samples")
print(f"Label distribution:\n{df_clean['label'].value_counts()}")

# ==================== CELL 3: Split Data ====================
X_train, X_test, y_train, y_test = train_test_split(
    df_clean['combined_text'], 
    df_clean['label'], 
    test_size=0.2, 
    stratify=df_clean['label'], 
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ==================== CELL 4: Tokenization for Deep Learning ====================
max_words = 30000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

print(f"✓ Tokenization complete")
print(f"Sequence shape: {X_train_seq.shape}")

# ==================== CELL 5: MODEL 1 - Deep GRU (Better than LSTM) ====================
def create_gru_model():
    """Deep Bidirectional GRU - Faster and often better than LSTM"""
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(128, return_sequences=True, dropout=0.3)),
        Bidirectional(GRU(64, dropout=0.3)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

print("\n" + "="*80)
print("TRAINING MODEL 1: BIDIRECTIONAL GRU")
print("="*80)

gru_model = create_gru_model()
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

gru_history = gru_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

gru_loss, gru_acc = gru_model.evaluate(X_test_seq, y_test, verbose=0)
gru_pred = (gru_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
gru_f1 = f1_score(y_test, gru_pred)

print(f"\n✓ GRU Results: Accuracy={gru_acc:.4f}, F1={gru_f1:.4f}")

# ==================== CELL 6: MODEL 2 - CNN for Text (Fast & Effective) ====================
def create_cnn_model():
    """1D CNN - Excellent for text classification, very fast"""
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

print("\n" + "="*80)
print("TRAINING MODEL 2: 1D CNN")
print("="*80)

cnn_model = create_cnn_model()
cnn_history = cnn_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_seq, y_test, verbose=0)
cnn_pred = (cnn_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
cnn_f1 = f1_score(y_test, cnn_pred)

print(f"\n✓ CNN Results: Accuracy={cnn_acc:.4f}, F1={cnn_f1:.4f}")

# ==================== CELL 7: MODEL 3 - Hybrid CNN-GRU ====================
def create_cnn_gru_model():
    """Hybrid: CNN for feature extraction + GRU for sequence modeling"""
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(GRU(64, dropout=0.3)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

print("\n" + "="*80)
print("TRAINING MODEL 3: HYBRID CNN-GRU")
print("="*80)

hybrid_model = create_cnn_gru_model()
hybrid_history = hybrid_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

hybrid_loss, hybrid_acc = hybrid_model.evaluate(X_test_seq, y_test, verbose=0)
hybrid_pred = (hybrid_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
hybrid_f1 = f1_score(y_test, hybrid_pred)

print(f"\n✓ Hybrid Results: Accuracy={hybrid_acc:.4f}, F1={hybrid_f1:.4f}")

# ==================== CELL 8: MODEL 4 - Multi-Channel CNN ====================
def create_multichannel_cnn():
    """Multi-channel CNN with different filter sizes (like Inception)"""
    input_layer = Input(shape=(max_len,))
    embedding = Embedding(max_words, 128, input_length=max_len)(input_layer)
    dropout = SpatialDropout1D(0.2)(embedding)
    
    # Channel 1: 3-gram
    conv1 = Conv1D(64, 3, activation='relu')(dropout)
    pool1 = GlobalMaxPooling1D()(conv1)
    
    # Channel 2: 4-gram
    conv2 = Conv1D(64, 4, activation='relu')(dropout)
    pool2 = GlobalMaxPooling1D()(conv2)
    
    # Channel 3: 5-gram
    conv3 = Conv1D(64, 5, activation='relu')(dropout)
    pool3 = GlobalMaxPooling1D()(conv3)
    
    # Concatenate all channels
    merged = Concatenate()([pool1, pool2, pool3])
    dense1 = Dense(128, activation='relu')(merged)
    bn = BatchNormalization()(dense1)
    drop1 = Dropout(0.4)(bn)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.3)(dense2)
    output = Dense(1, activation='sigmoid', dtype='float32')(drop2)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

print("\n" + "="*80)
print("TRAINING MODEL 4: MULTI-CHANNEL CNN")
print("="*80)

multicnn_model = create_multichannel_cnn()
multicnn_history = multicnn_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

multicnn_loss, multicnn_acc = multicnn_model.evaluate(X_test_seq, y_test, verbose=0)
multicnn_pred = (multicnn_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
multicnn_f1 = f1_score(y_test, multicnn_pred)

print(f"\n✓ Multi-CNN Results: Accuracy={multicnn_acc:.4f}, F1={multicnn_f1:.4f}")

# ==================== CELL 9: MODEL 5 - XGBoost with TF-IDF ====================
print("\n" + "="*80)
print("TRAINING MODEL 5: XGBoost + TF-IDF")
print("="*80)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='gpu_hist' if gpus else 'hist'
)

xgb_model.fit(X_train_tfidf, y_train, 
              eval_set=[(X_test_tfidf, y_test)],
              verbose=50)

xgb_pred = xgb_model.predict(X_test_tfidf)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"\n✓ XGBoost Results: Accuracy={xgb_acc:.4f}, F1={xgb_f1:.4f}")

# ==================== CELL 10: MODEL 6 - LightGBM ====================
print("\n" + "="*80)
print("TRAINING MODEL 6: LightGBM + TF-IDF")
print("="*80)

lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    device='gpu' if gpus else 'cpu',
    verbose=-1
)

lgbm_model.fit(X_train_tfidf, y_train,
               eval_set=[(X_test_tfidf, y_test)],
               callbacks=[])

lgbm_pred = lgbm_model.predict(X_test_tfidf)
lgbm_acc = accuracy_score(y_test, lgbm_pred)
lgbm_f1 = f1_score(y_test, lgbm_pred)

print(f"\n✓ LightGBM Results: Accuracy={lgbm_acc:.4f}, F1={lgbm_f1:.4f}")

# ==================== CELL 11: MODEL 7 - Ensemble (Voting) ====================
print("\n" + "="*80)
print("TRAINING MODEL 7: ENSEMBLE (Voting Classifier)")
print("="*80)

# Get predictions from deep learning models
gru_proba = gru_model.predict(X_test_seq, verbose=0).flatten()
cnn_proba = cnn_model.predict(X_test_seq, verbose=0).flatten()
hybrid_proba = hybrid_model.predict(X_test_seq, verbose=0).flatten()
multicnn_proba = multicnn_model.predict(X_test_seq, verbose=0).flatten()

# Get predictions from ML models
xgb_proba = xgb_model.predict_proba(X_test_tfidf)[:, 1]
lgbm_proba = lgbm_model.predict_proba(X_test_tfidf)[:, 1]

# Weighted ensemble
ensemble_proba = (
    0.20 * gru_proba + 
    0.15 * cnn_proba + 
    0.20 * hybrid_proba + 
    0.15 * multicnn_proba +
    0.15 * xgb_proba +
    0.15 * lgbm_proba
)

ensemble_pred = (ensemble_proba > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)

print(f"\n✓ Ensemble Results: Accuracy={ensemble_acc:.4f}, F1={ensemble_f1:.4f}")

# ==================== CELL 12: Results Comparison ====================
results = pd.DataFrame([
    {'Model': 'Bidirectional GRU', 'Accuracy': gru_acc, 'F1 Score': gru_f1, 'Type': 'Deep Learning'},
    {'Model': '1D CNN', 'Accuracy': cnn_acc, 'F1 Score': cnn_f1, 'Type': 'Deep Learning'},
    {'Model': 'Hybrid CNN-GRU', 'Accuracy': hybrid_acc, 'F1 Score': hybrid_f1, 'Type': 'Deep Learning'},
    {'Model': 'Multi-Channel CNN', 'Accuracy': multicnn_acc, 'F1 Score': multicnn_f1, 'Type': 'Deep Learning'},
    {'Model': 'XGBoost + TF-IDF', 'Accuracy': xgb_acc, 'F1 Score': xgb_f1, 'Type': 'Machine Learning'},
    {'Model': 'LightGBM + TF-IDF', 'Accuracy': lgbm_acc, 'F1 Score': lgbm_f1, 'Type': 'Machine Learning'},
    {'Model': 'Weighted Ensemble', 'Accuracy': ensemble_acc, 'F1 Score': ensemble_f1, 'Type': 'Ensemble'}
])

results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("📊 FINAL RESULTS COMPARISON")
print("="*80)
print(results.to_string(index=False))
print("\n🏆 Best Model: " + results.iloc[0]['Model'])
print(f"   Accuracy: {results.iloc[0]['Accuracy']:.4f}")
print(f"   F1 Score: {results.iloc[0]['F1 Score']:.4f}")

results.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")

# ==================== CELL 13: Visualizations ====================
# Plot 1: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7', '#DFE6E9']
axes[0].barh(results['Model'], results['Accuracy'], color=colors)
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].axvline(x=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

# F1 Score comparison
axes[1].barh(results['Model'], results['F1 Score'], color=colors)
axes[1].set_xlabel('F1 Score', fontsize=12)
axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Best Model Confusion Matrix
best_model_name = results.iloc[0]['Model']
if 'GRU' in best_model_name:
    best_pred = gru_pred
elif 'Hybrid' in best_model_name:
    best_pred = hybrid_pred
elif best_model_name == '1D CNN':
    best_pred = cnn_pred
elif 'Multi' in best_model_name:
    best_pred = multicnn_pred
elif 'XGBoost' in best_model_name:
    best_pred = xgb_pred
elif 'LightGBM' in best_model_name:
    best_pred = lgbm_pred
else:
    best_pred = ensemble_pred

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'{best_model_name} - Confusion Matrix\nAccuracy: {results.iloc[0]["Accuracy"]:.4f}', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Training History (Best Deep Learning Model)
best_dl_model = results[results['Type'] == 'Deep Learning'].iloc[0]['Model']
if 'GRU' in best_dl_model:
    history = gru_history
elif 'Hybrid' in best_dl_model:
    history = hybrid_history
elif best_dl_model == '1D CNN':
    history = cnn_history
else:
    history = multicnn_history

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train', marker='o', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title(f'{best_dl_model} - Training History', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train', marker='o', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', marker='s', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_dl_model_training.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== CELL 14: Classification Reports ====================
print("\n" + "="*80)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("="*80)
print(classification_report(y_test, best_pred, target_names=le.classes_))

# ==================== CELL 15: Save Best Models ====================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save deep learning models
gru_model.save('gru_model.keras')
cnn_model.save('cnn_model.keras')
hybrid_model.save('hybrid_model.keras')
multicnn_model.save('multicnn_model.keras')

# Save ML models
import pickle
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✓ All models saved successfully!")
print("\n" + "="*80)
print("✅ COMPLETE! All 7 models trained and evaluated!")
print("="*80)

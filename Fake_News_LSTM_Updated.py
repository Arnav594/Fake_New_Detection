# ==================== CELL 1: Imports ====================
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!")

# ==================== CELL 2: Load Data ====================
df = pd.read_excel("bharatfakenewskosh.xlsx")

# Check columns
print("Available columns:", df.columns.tolist())
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check label distribution
print("\nLabel distribution:")
print(df['Label'].value_counts())

# ==================== CELL 3: Data Preprocessing ====================
statement_col = 'Eng_Trans_Statement'
body_col = 'Eng_Trans_News_Body'
label_col = 'Label'

# Remove rows with missing values
df_clean = df[[statement_col, body_col, label_col]].dropna()

def clean(s):
    s = str(s).lower()
    s = re.sub(r'http\S+|[^a-z\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# Clean both columns
df_clean['statement_clean'] = df_clean[statement_col].map(clean)
df_clean['body_clean'] = df_clean[body_col].map(clean)

# Encode labels (TRUE/FALSE)
le = LabelEncoder()
df_clean['label_encoded'] = le.fit_transform(df_clean[label_col])

print(f"✓ Data cleaned successfully!")
print(f"Total samples: {len(df_clean)}")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ==================== CELL 4: Choose Training Approach ====================
# You can choose between:
# OPTION A: Separate models for Statement and Body
# OPTION B: Combined text (Statement + Body)
# OPTION C: Multi-input model (processes both separately, then combines)

TRAINING_MODE = 'SEPARATE'  # Change to 'COMBINED' or 'MULTI_INPUT'

print(f"Training Mode: {TRAINING_MODE}")
print("\nAvailable options:")
print("- 'SEPARATE': Train models on Statement and Body independently")
print("- 'COMBINED': Combine Statement + Body into single text")
print("- 'MULTI_INPUT': Advanced model with separate LSTM branches for each column")

# ==================== CELL 5: Tokenization Setup ====================
max_words = 50000
max_len = 300

if TRAINING_MODE == 'SEPARATE':
    # Create separate tokenizers
    tokenizer_statement = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer_body = Tokenizer(num_words=max_words, oov_token='<OOV>')
    
    # Split data
    X_train_stmt, X_test_stmt, X_train_body, X_test_body, y_train, y_test = train_test_split(
        df_clean['statement_clean'], 
        df_clean['body_clean'],
        df_clean['label_encoded'], 
        test_size=0.2, 
        stratify=df_clean['label_encoded'], 
        random_state=42
    )
    
    # Fit tokenizers
    tokenizer_statement.fit_on_texts(X_train_stmt)
    tokenizer_body.fit_on_texts(X_train_body)
    
    # Create sequences
    X_train_stmt_seq = pad_sequences(tokenizer_statement.texts_to_sequences(X_train_stmt), maxlen=max_len)
    X_test_stmt_seq = pad_sequences(tokenizer_statement.texts_to_sequences(X_test_stmt), maxlen=max_len)
    X_train_body_seq = pad_sequences(tokenizer_body.texts_to_sequences(X_train_body), maxlen=max_len)
    X_test_body_seq = pad_sequences(tokenizer_body.texts_to_sequences(X_test_body), maxlen=max_len)
    
    print("✓ Separate tokenization complete!")
    print(f"Statement sequences shape: {X_train_stmt_seq.shape}")
    print(f"Body sequences shape: {X_train_body_seq.shape}")

elif TRAINING_MODE == 'COMBINED':
    # Combine text
    df_clean['combined_text'] = df_clean['statement_clean'] + " " + df_clean['body_clean']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['combined_text'], 
        df_clean['label_encoded'], 
        test_size=0.2, 
        stratify=df_clean['label_encoded'], 
        random_state=42
    )
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
    
    print("✓ Combined tokenization complete!")
    print(f"Combined sequences shape: {X_train_seq.shape}")

elif TRAINING_MODE == 'MULTI_INPUT':
    # Multi-input approach (same as SEPARATE for data prep)
    tokenizer_statement = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer_body = Tokenizer(num_words=max_words, oov_token='<OOV>')
    
    X_train_stmt, X_test_stmt, X_train_body, X_test_body, y_train, y_test = train_test_split(
        df_clean['statement_clean'], 
        df_clean['body_clean'],
        df_clean['label_encoded'], 
        test_size=0.2, 
        stratify=df_clean['label_encoded'], 
        random_state=42
    )
    
    tokenizer_statement.fit_on_texts(X_train_stmt)
    tokenizer_body.fit_on_texts(X_train_body)
    
    X_train_stmt_seq = pad_sequences(tokenizer_statement.texts_to_sequences(X_train_stmt), maxlen=max_len)
    X_test_stmt_seq = pad_sequences(tokenizer_statement.texts_to_sequences(X_test_stmt), maxlen=max_len)
    X_train_body_seq = pad_sequences(tokenizer_body.texts_to_sequences(X_train_body), maxlen=max_len)
    X_test_body_seq = pad_sequences(tokenizer_body.texts_to_sequences(X_test_body), maxlen=max_len)
    
    print("✓ Multi-input tokenization complete!")
    print(f"Statement sequences shape: {X_train_stmt_seq.shape}")
    print(f"Body sequences shape: {X_train_body_seq.shape}")

# ==================== CELL 6: Model Architectures ====================
def create_model_v1(input_shape):
    """Deep BiLSTM with ReLU"""
    model = Sequential([
        Embedding(max_words, 256, input_length=input_shape),
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_v2(input_shape):
    """Deep BiLSTM with ELU"""
    model = Sequential([
        Embedding(max_words, 256, input_length=input_shape),
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='elu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_v3(input_shape):
    """Deep BiLSTM with LeakyReLU"""
    from tensorflow.keras.layers import LeakyReLU
    model = Sequential([
        Embedding(max_words, 256, input_length=input_shape),
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
        Dense(128),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_v4(input_shape):
    """Deep BiLSTM with SELU"""
    model = Sequential([
        Embedding(max_words, 256, input_length=input_shape),
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
        Dense(128, activation='selu'),
        Dropout(0.4),
        Dense(64, activation='selu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_multi_input_model():
    """Multi-input model with separate branches"""
    # Statement branch
    input_stmt = Input(shape=(max_len,), name='statement_input')
    emb_stmt = Embedding(max_words, 128, input_length=max_len)(input_stmt)
    lstm_stmt = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2))(emb_stmt)
    
    # Body branch
    input_body = Input(shape=(max_len,), name='body_input')
    emb_body = Embedding(max_words, 128, input_length=max_len)(input_body)
    lstm_body = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2))(emb_body)
    
    # Combine branches
    combined = Concatenate()([lstm_stmt, lstm_body])
    dense1 = Dense(128, activation='relu')(combined)
    bn = BatchNormalization()(dense1)
    drop1 = Dropout(0.4)(bn)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.3)(dense2)
    output = Dense(1, activation='sigmoid')(drop2)
    
    model = Model(inputs=[input_stmt, input_body], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("✓ Model architectures defined!")

# ==================== CELL 7: Training Configuration ====================
model_creators = [create_model_v1, create_model_v2, create_model_v3, create_model_v4]
model_names = ['BiLSTM_ReLU', 'BiLSTM_ELU', 'BiLSTM_LeakyReLU', 'BiLSTM_SELU']

results = []
histories = []

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

print("Training Configuration:")
print(f"- Number of architectures: {len(model_creators)}")
print(f"- Runs per architecture: 5")
print(f"- Total models to train: {len(model_creators) * 5}")
print(f"- Epochs per model: 12")
print(f"- Training mode: {TRAINING_MODE}")

# ==================== CELL 8: Training Loop ====================
print("="*80)
print(f"STARTING TRAINING - {TRAINING_MODE} MODE")
print("="*80)

if TRAINING_MODE == 'SEPARATE':
    # Train on Statement
    print("\n" + "="*80)
    print("TRAINING ON STATEMENT COLUMN")
    print("="*80)
    for idx, (model_creator, model_name) in enumerate(zip(model_creators, model_names)):
        for run in range(5):
            print(f"\n{'='*80}")
            print(f"Statement - {model_name} - Run {run+1}/5 (Model {idx*5 + run + 1}/20)")
            print(f"{'='*80}")
            
            model = model_creator(max_len)
            history = model.fit(
                X_train_stmt_seq, y_train,
                validation_split=0.2,
                epochs=12,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            test_loss, test_acc = model.evaluate(X_test_stmt_seq, y_test, verbose=0)
            y_pred = (model.predict(X_test_stmt_seq, verbose=0) > 0.5).astype(int)
            
            results.append({
                'data_source': 'Statement',
                'model': f"{model_name}_stmt_run{run+1}",
                'architecture': model_name,
                'run': run+1,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1]
            })
            
            histories.append({
                'model': f"{model_name}_stmt_run{run+1}",
                'history': history.history
            })
            
            print(f"\nResults: Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Train on Body
    print("\n" + "="*80)
    print("TRAINING ON BODY COLUMN")
    print("="*80)
    for idx, (model_creator, model_name) in enumerate(zip(model_creators, model_names)):
        for run in range(5):
            print(f"\n{'='*80}")
            print(f"Body - {model_name} - Run {run+1}/5 (Model {idx*5 + run + 21}/40)")
            print(f"{'='*80}")
            
            model = model_creator(max_len)
            history = model.fit(
                X_train_body_seq, y_train,
                validation_split=0.2,
                epochs=12,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            test_loss, test_acc = model.evaluate(X_test_body_seq, y_test, verbose=0)
            y_pred = (model.predict(X_test_body_seq, verbose=0) > 0.5).astype(int)
            
            results.append({
                'data_source': 'Body',
                'model': f"{model_name}_body_run{run+1}",
                'architecture': model_name,
                'run': run+1,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1]
            })
            
            histories.append({
                'model': f"{model_name}_body_run{run+1}",
                'history': history.history
            })
            
            print(f"\nResults: Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

elif TRAINING_MODE == 'COMBINED':
    for idx, (model_creator, model_name) in enumerate(zip(model_creators, model_names)):
        for run in range(5):
            print(f"\n{'='*80}")
            print(f"Combined - {model_name} - Run {run+1}/5 (Model {idx*5 + run + 1}/20)")
            print(f"{'='*80}")
            
            model = model_creator(max_len)
            history = model.fit(
                X_train_seq, y_train,
                validation_split=0.2,
                epochs=12,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=0)
            y_pred = (model.predict(X_test_seq, verbose=0) > 0.5).astype(int)
            
            results.append({
                'data_source': 'Combined',
                'model': f"{model_name}_run{run+1}",
                'architecture': model_name,
                'run': run+1,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1]
            })
            
            histories.append({
                'model': f"{model_name}_run{run+1}",
                'history': history.history
            })
            
            print(f"\nResults: Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

elif TRAINING_MODE == 'MULTI_INPUT':
    for run in range(5):
        print(f"\n{'='*80}")
        print(f"Multi-Input Model - Run {run+1}/5")
        print(f"{'='*80}")
        
        model = create_multi_input_model()
        history = model.fit(
            [X_train_stmt_seq, X_train_body_seq], y_train,
            validation_split=0.2,
            epochs=12,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        test_loss, test_acc = model.evaluate([X_test_stmt_seq, X_test_body_seq], y_test, verbose=0)
        y_pred = (model.predict([X_test_stmt_seq, X_test_body_seq], verbose=0) > 0.5).astype(int)
        
        results.append({
            'data_source': 'Multi-Input',
            'model': f"MultiInput_run{run+1}",
            'architecture': 'Multi-Input',
            'run': run+1,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1]
        })
        
        histories.append({
            'model': f"MultiInput_run{run+1}",
            'history': history.history
        })
        
        print(f"\nResults: Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

print("\n" + "="*80)
print("✓ ALL TRAINING COMPLETE!")
print("="*80)

# ==================== CELL 9: Results Analysis ====================
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
if TRAINING_MODE == 'SEPARATE':
    summary = results_df.groupby(['data_source', 'architecture']).agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_loss': ['mean', 'std', 'min']
    }).round(4)
else:
    summary = results_df.groupby('architecture').agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_loss': ['mean', 'std', 'min']
    }).round(4)
print(summary)

# Best model
best_idx = results_df['test_accuracy'].idxmax()
best_model_info = results_df.iloc[best_idx]
print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
print(f"Model: {best_model_info['model']}")
print(f"Architecture: {best_model_info['architecture']}")
if 'data_source' in best_model_info:
    print(f"Data Source: {best_model_info['data_source']}")
print(f"Test Accuracy: {best_model_info['test_accuracy']:.4f}")
print(f"Test Loss: {best_model_info['test_loss']:.4f}")

# Save results
results_df.to_csv('training_results.csv', index=False)
print("\n✓ Results saved to 'training_results.csv'")

# ==================== CELL 10: Visualizations ====================
# Plot 1: Accuracy comparison
plt.figure(figsize=(16, 6))
if TRAINING_MODE == 'SEPARATE':
    stmt_results = results_df[results_df['data_source'] == 'Statement']
    body_results = results_df[results_df['data_source'] == 'Body']
    
    x = np.arange(len(stmt_results))
    width = 0.35
    
    plt.bar(x - width/2, stmt_results['test_accuracy'], width, label='Statement', alpha=0.8)
    plt.bar(x + width/2, body_results['test_accuracy'], width, label='Body', alpha=0.8)
    plt.xlabel('Model Run')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy: Statement vs Body')
    plt.legend()
else:
    x = range(len(results_df))
    plt.bar(x, results_df['test_accuracy'], alpha=0.7, edgecolor='black')
    plt.axhline(y=results_df['test_accuracy'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["test_accuracy"].mean():.4f}')
    plt.xlabel('Model Run')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy Across All Runs ({TRAINING_MODE} Mode)')
    plt.legend()

plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Best model training history
best_history = next(h for h in histories if h['model'] == best_model_info['model'])
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(best_history['history']['accuracy'], label='Train', linewidth=2)
plt.plot(best_history['history']['val_accuracy'], label='Validation', linewidth=2)
plt.title(f'Best Model: {best_model_info["model"]}\nAccuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(best_history['history']['loss'], label='Train', linewidth=2)
plt.plot(best_history['history']['val_loss'], label='Validation', linewidth=2)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_training.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Architecture comparison boxplot
if TRAINING_MODE != 'MULTI_INPUT':
    plt.figure(figsize=(12, 6))
    if TRAINING_MODE == 'SEPARATE':
        results_df.boxplot(column='test_accuracy', by=['data_source', 'architecture'], figsize=(14, 6))
        plt.xticks(rotation=45, ha='right')
    else:
        results_df.boxplot(column='test_accuracy', by='architecture', figsize=(12, 6))
        plt.xticks(rotation=30, ha='right')
    
    plt.title('Test Accuracy Distribution')
    plt.suptitle('')
    plt.ylabel('Test Accuracy')
    plt.tight_layout()
    plt.savefig('architecture_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n✓ All visualizations saved!")

# ==================== CELL 11: Final Summary ====================
print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)
print(f"Training Mode: {TRAINING_MODE}")
print(f"Total Models Trained: {len(results_df)}")
print(f"Average Test Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
print(f"Best Test Accuracy: {results_df['test_accuracy'].max():.4f}")
print(f"Worst Test Accuracy: {results_df['test_accuracy'].min():.4f}")
print("\n✓ Training pipeline complete!")
print("="*80)

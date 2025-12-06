import os
import torch
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ============================================================
# ✅ 1. Setup
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/bart-base"
MODEL_DIR = "./bart_fake_news"

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"🔥 Using device: {DEVICE}")

# ============================================================
# ✅ 2. Load Dataset
# ============================================================
df = pd.read_excel("bharatfakenewskosh.xlsx")
df = df.dropna(subset=["Eng_Trans_News_Body", "Label"])

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])  # TRUE/FALSE → 1/0

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Eng_Trans_News_Body"].tolist(),
    df["Label"].tolist(),
    test_size=0.2,
    random_state=42,
)

# ============================================================
# ✅ 3. Tokenization
# ============================================================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ============================================================
# ✅ 4. Model Initialization
# ============================================================
model = BartForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))
model.to(DEVICE)

# ============================================================
# ✅ 5. Metrics Function
# ============================================================
def compute_metrics(p):
    preds = torch.tensor(p.predictions)
    preds = torch.argmax(preds, dim=1)
    labels = torch.tensor(p.label_ids)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ============================================================
# ✅ 6. Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{MODEL_DIR}/logs",
    load_best_model_at_end=True,
    logging_steps=100,
)

# ============================================================
# ✅ 7. Trainer Setup
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ============================================================
# ✅ 8. Train Model
# ============================================================
print("\n🚀 Starting training...\n")
trainer.train()

# ============================================================
# ✅ 9. Evaluate Model
# ============================================================
print("\n📊 Evaluating on test data...\n")
metrics = trainer.evaluate()
print(metrics)

preds = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(preds.predictions), dim=1)

print("\nClassification Report:")
print(classification_report(test_labels, pred_labels, target_names=[str(c) for c in le.classes_]))

# ============================================================
# ✅ 10. Save Model
# ============================================================
trainer.save_model(MODEL_DIR)
print(f"\n✅ Model and tokenizer saved to {MODEL_DIR}")

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm

# todo check if cuda is available
# todo add save embedding
# todo add log info?
# todo tsne plot

# Initialize tokenizer and model on GPU
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to("cuda")

# Load datasets
train_dataset = load_dataset("ag_news", split="train")
val_dataset = load_dataset("ag_news", split="test")

# Set batch size
batch_size = 128  # 32 ok

# Extract features for training dataset
train_features = []
train_labels = []
for i in tqdm(range(0, len(train_dataset["text"]), batch_size)):
    batch = train_dataset["text"][i:i + batch_size]
    encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        batch_output = model(encoded_batch.input_ids, encoded_batch.attention_mask)
    batch_features = batch_output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    train_features.append(batch_features)
    train_labels.append(train_dataset["label"][i:i + batch_size])
train_features = np.concatenate(train_features, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

# Train logistic regression classifier
clf = LogisticRegression()
clf.fit(train_features, train_labels)

# Extract features for validation dataset
val_features = []
for i in tqdm(range(0, len(val_dataset["text"]), batch_size)):
    batch = val_dataset["text"][i:i + batch_size]
    encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        batch_output = model(encoded_batch.input_ids, encoded_batch.attention_mask)
    batch_features = batch_output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    val_features.append(batch_features)
val_features = np.concatenate(val_features, axis=0)

# Make predictions on validation dataset
preds = clf.predict(val_features)
acc = accuracy_score(val_dataset["label"], preds)
print(f"Validation accuracy: {acc:.4f}")

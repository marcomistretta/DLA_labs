import itertools
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

labels = ["World", "Sports", "Business", "Science/Tech"]
writer = SummaryWriter("./runs/lab2_3_1-ovr")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
save = False

# Initialize tokenizer and model on GPU
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
clf = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="ovr")

if save:
    # Create folder for data
    if not os.path.exists("ag_news_data"):
        os.makedirs("ag_news_data")

    # Load datasets
    train_dataset = load_dataset("ag_news", split="train")
    val_dataset = load_dataset("ag_news", split="test")

    # Set batch size
    batch_size = 128

    # Extract features for training dataset
    train_features = []
    train_labels = []
    for i in tqdm(range(0, len(train_dataset["text"]), batch_size)):
        batch = train_dataset["text"][i:i + batch_size]
        encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_output = model(encoded_batch.input_ids, encoded_batch.attention_mask)
        batch_features = batch_output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        train_features.append(batch_features)
        train_labels.append(train_dataset["label"][i:i + batch_size])
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Extract features for validation dataset
    val_features = []
    for i in tqdm(range(0, len(val_dataset["text"]), batch_size)):
        batch = val_dataset["text"][i:i + batch_size]
        encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_output = model(encoded_batch.input_ids, encoded_batch.attention_mask)
        batch_features = batch_output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        val_features.append(batch_features)
    val_features = np.concatenate(val_features, axis=0)

    val_labels = val_dataset["label"]
    # Save data
    np.save("ag_news_data/train_features.npy", train_features)
    np.save("ag_news_data/train_labels.npy", train_labels)
    np.save("ag_news_data/val_features.npy", val_features)
    np.save("ag_news_data/val_labels.npy", val_dataset["label"])

else:
    train_features = np.load("ag_news_data/train_features.npy")
    train_labels = np.load("ag_news_data/train_labels.npy")
    val_features = np.load("ag_news_data/val_features.npy")
    val_labels = np.load("ag_news_data/val_labels.npy")

print("fitting classifier")
# Train logistic regression classifier
clf.fit(train_features, train_labels)
print("classifier fitted")

tsne_train = TSNE(n_components=2, random_state=42).fit_transform(train_features[:1000])
plt.figure(figsize=(10, 10))
plt.scatter(tsne_train[:, 0], tsne_train[:, 1], c=train_labels[:1000], label=labels, cmap="viridis")
plt.title("TSNE plot of train features")
plt.colorbar()
plt.legend()
plt.savefig('images/tsne_train.png')
plt.show()

tsne_val = TSNE(n_components=2, random_state=42).fit_transform(val_features[:1000])
plt.figure(figsize=(10, 10))
plt.scatter(tsne_val[:, 0], tsne_val[:, 1], c=val_labels[:1000], label=labels, cmap="magma")
plt.title("TSNE plot of validation features")
plt.colorbar()
plt.legend()
plt.savefig('images/tsne_val.png')
plt.show()

# Make predictions on validation dataset
preds = clf.predict(val_features)
acc = accuracy_score(val_labels, preds)
precision = precision_score(val_labels, preds, average='macro')
recall = recall_score(val_labels, preds, average='macro')
f1 = f1_score(val_labels, preds, average='macro')

# Create confusion matrix
cm = confusion_matrix(val_labels, preds)
# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix for AG News Classification")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = "d"
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig('images/confusion_matrix.png')
plt.show()

writer.add_scalar('Validation Accuracy', acc)
writer.add_scalar('Validation Precision', precision)
writer.add_scalar('Validation Recall', recall)
writer.add_scalar('Validation F1 Score', f1)

writer.close()
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

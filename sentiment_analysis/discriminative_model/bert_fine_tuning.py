import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

max_class_samples = 25000

type = "real_data"  # or 'synthetic_data
random_state = 12321
model_type = "prajjwal1/bert-tiny"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

if type == "synthetic_data":
    dataset_path = "/workspaces/esg-controversy-tracker/sentiment_analysis/generative_model/synthetic_data_12_13_2022_23_34_50.csv"
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index()
    pos_sample = dataset[dataset["sentiment"] == 1][0:max_class_samples]
    neg_sample = dataset[dataset["sentiment"] == 0][0:max_class_samples]
    dataset = pd.concat([pos_sample, neg_sample])
    model_output = "./sentiment-analysis-synthetic-data"
    model_checkpoint = "./sentiment-analysis-synthetic-data/checkpoint-90000"
else:
    dataset_path = "/workspaces/esg-controversy-tracker/dataset/news_sentiment.csv"
    dataset = pd.read_csv(dataset_path)
    dataset["confidence"] = dataset["confidence"].abs()
    dataset = dataset[dataset["confidence"] >= 0.99]
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index()
    pos_sample = dataset[dataset["sentiment"] == "POSITIVE"][0:max_class_samples]
    neg_sample = dataset[dataset["sentiment"] == "NEGATIVE"][0:max_class_samples]
    dataset = pd.concat([pos_sample, neg_sample])
    dataset["sentiment"] = [0 if x == "NEGATIVE" else 1 for x in dataset["sentiment"]]
    model_output = "./sentiment-analysis"
    model_checkpoint = "./sentiment-analysis/checkpoint-90000"


dataset = dataset.sample(frac=1, random_state=random_state).reset_index()

train_set = dataset[:30000]
valid_set = dataset[30000:37500]
test_set = dataset[37500:]

# Create The Dataset Class.
class TheDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, sentiments, tokenizer):
        self.reviews = reviews
        self.sentiments = sentiments

        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = str(self.reviews[index])
        sentiments = self.sentiments[index]

        encoded_review = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": encoded_review["input_ids"][0],
            "attention_mask": encoded_review["attention_mask"][0],
            "labels": torch.tensor(sentiments, dtype=torch.long),
        }


tokenizer = AutoTokenizer.from_pretrained(model_type)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Create Dataset objects for train/validation sets.
train_set_dataset = TheDataset(
    reviews=train_set.Title.tolist(),
    sentiments=train_set.sentiment.tolist(),
    tokenizer=tokenizer,
)

valid_set_dataset = TheDataset(
    reviews=valid_set.Title.tolist(),
    sentiments=valid_set.sentiment.tolist(),
    tokenizer=tokenizer,
)

# Create DataLoader for train/validation sets.
train_set_dataloader = torch.utils.data.DataLoader(
    train_set_dataset, batch_size=16, num_workers=4
)

valid_set_dataloader = torch.utils.data.DataLoader(
    valid_set_dataset, batch_size=16, num_workers=4
)

model = BertForSequenceClassification.from_pretrained(model_type)

training_args = TrainingArguments(
    output_dir=model_output,
    num_train_epochs=300,
    per_device_train_batch_size=100,
    per_device_eval_batch_size=50,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set_dataset,
    eval_dataset=valid_set_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
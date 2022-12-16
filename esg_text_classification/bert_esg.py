
from datetime import datetime as dt
import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'nbroad/ESG-BERT'
model_output_path = './bert-esg-outputs'
dataset_path = '/workspaces/esg-controversy-tracker/dataset/us_equities_news_dataset.csv'
config_path = 'bert_esg_config.json'

dataset = pd.read_csv(dataset_path)

# Create The Dataset Class.
class TheDataset(torch.utils.data.Dataset):
    def __init__(self, news_articles, tokenizer=AutoTokenizer.from_pretrained(model_type)):
        self.news_articles = news_articles

        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.news_articles)

    def __getitem__(self, index):
        news_articles = str(self.news_articles[index])

        encoded_news_articles = self.tokenizer.encode_plus(
            news_articles,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoded_news_articles["input_ids"][0],
            "attention_mask": encoded_news_articles["attention_mask"][0]
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Create Dataset objects for prediction
prediction_dataset = TheDataset(
    news_articles=dataset.content.tolist())

model = AutoModelForSequenceClassification.from_pretrained(model_type)

training_args = TrainingArguments(output_dir=model_output_path, do_predict=True)

trainer = Trainer(
    model=model,
    args=training_args
)

predictions = trainer.predict(prediction_dataset)
bert_config = json.load(open(config_path))
predictions_df = pd.DataFrame(predictions.predictions, columns=bert_config['id2label'].values())


result_df = pd.merge(dataset, predictions_df, left_index=True, right_index=True)
result_df.to_csv(f'{model_output_path}/bert_esg_tagged_articles_{dt.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv', index=False)
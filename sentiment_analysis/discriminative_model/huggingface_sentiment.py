import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

random_state=12321
dataset_path = '/workspaces/esg-controversy-tracker/dataset/news_sentiment.csv'

# Limit the dataset to n rows
data = pd.read_csv(dataset_path)
data['confidence'] = data['confidence'].abs()
data = data[data['confidence'] >= 0.99]

max_class_samples = 50
data = data.sample(frac=1, random_state=random_state).reset_index()
pos_sample = data[data['sentiment'] == 'POSITIVE'][0:max_class_samples]
neg_sample = data[data['sentiment'] == 'NEGATIVE'][0:max_class_samples]
dataset = pd.concat([pos_sample, neg_sample])


dataset["sentiment"] = [0 if x=='NEGATIVE' else 1 for x in dataset['sentiment']]

train_set = dataset[0:40]
valid_set = dataset[40:45]
test_set  = dataset[45:50]

#print( train_set.head() )



# Create The Dataset Class.
class TheDataset(torch.utils.data.Dataset):

    def __init__(self, reviews, sentiments, tokenizer):
        self.reviews    = reviews
        self.sentiments = sentiments
        self.tokenizer  = tokenizer
        self.max_len    = tokenizer.model_max_length
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, index):
        review = str(self.reviews[index])
        sentiments = self.sentiments[index]

        encoded_review = self.tokenizer.encode_plus(
            review,
            add_special_tokens    = True,
            max_length            = self.max_len,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors        = "pt",
            padding               = "max_length",
            truncation            = True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],
            'attention_mask': encoded_review['attention_mask'][0],
            'labels': torch.tensor(sentiments, dtype=torch.long)
        }

# Load the tokenizer for the BERT model.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create Dataset objects for train/validation sets.
train_set_dataset = TheDataset(
    reviews    = train_set.Title.tolist(),
    sentiments = train_set.sentiment.tolist(),
    tokenizer  = tokenizer,
)

valid_set_dataset = TheDataset(
    reviews    = valid_set.Title.tolist(),
    sentiments = valid_set.sentiment.tolist(),
    tokenizer  = tokenizer,
)

# Create DataLoader for train/validation sets.
train_set_dataloader = torch.utils.data.DataLoader(
    train_set_dataset,
    batch_size  = 16,
    num_workers = 4
)

valid_set_dataloader = torch.utils.data.DataLoader(
    valid_set_dataset,
    batch_size  = 16,
    num_workers = 4
)

# Get one batch as example.
train_data = next(iter(train_set_dataloader))
valid_data = next(iter(valid_set_dataloader))

# Print the output sizes.
print( train_data["input_ids"].size(), valid_data["input_ids"].size() )

model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

# for name, param in model.bert.named_parameters():
#     param.requires_grad = False

# Freeze the first 23 layers of the BERT
for name, param in model.bert.named_parameters():
    if ( not name.startswith('pooler') ) and "layer.23" not in name :
        param.requires_grad = False



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir                  = "./sentiment-analysis",
    num_train_epochs            = 1,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size  = 64,
    warmup_steps                = 500,
    weight_decay                = 0.01,
    save_strategy               = "epoch",
    evaluation_strategy         = "steps"
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_set_dataset,
    eval_dataset    = valid_set_dataset,
    compute_metrics = compute_metrics
)

trainer.train()

# Load the checkpoint
model = BertForSequenceClassification.from_pretrained("./sentiment-analysis/checkpoint-1")

# Make the test set ready
test_set_dataset = TheDataset(
    reviews    = test_set.Title.tolist(),
    sentiments = test_set.sentiment.tolist(),
    tokenizer  = tokenizer,
)

training_args = TrainingArguments(
    output_dir = "./sentiment-analysis",
    do_predict = True
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    compute_metrics =compute_metrics,
)

predictions = trainer.predict(test_set_dataset)

print(predictions)
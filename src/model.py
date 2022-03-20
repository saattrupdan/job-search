'''Model that detects whether a job listing is relevant or not'''

from datasets import Dataset, load_metric
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)

import numpy as np
from pathlib import Path
import pandas as pd
import json
import os


def train_model():
    '''Trains the model and stores it to disk'''

    # Disable tokenizers parallelization
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Set up pretrained model ID
    model_id = 'xlm-roberta-base'

    # Load the paragraph data
    data_path = Path('data') / 'job_listing_paragraphs.jsonl'
    with data_path.open('r') as f:
        job_listings = [json.loads(line) for line in f]

    # Convert data to DataFrame
    df = pd.DataFrame.from_records(job_listings)
    df = df[['cleaned_text', 'title_or_tasks', 'requirements']]
    df = df.explode(['cleaned_text', 'title_or_tasks', 'requirements'])

    # Create `labels` column
    labels = list()
    for _, row in df.iterrows():
        if row.title_or_tasks:
            labels.append(1)
        elif row.requirements:
            labels.append(2)
        else:
            labels.append(0)
    df['labels'] = labels
    df = df.drop(columns=['title_or_tasks', 'requirements'])

    # Convert the data to a HuggingFace dataset
    dataset = Dataset.from_dict(dict(text=df.cleaned_text.tolist(),
                                     label=df.labels.tolist()))

    # Split the dataset into training and validation sets
    splits = dataset.train_test_split(train_size=0.8)
    train = splits['train']
    val = splits['test']

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Initialise the data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Tokenize the corpus
    def tokenize(example: dict):
        return tokenizer(example['text'], truncation=True, max_length=512)
    train = train.map(tokenize)
    val = val.map(tokenize)

    # Initialise the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        hidden_dropout_prob=0.5,
        classifier_dropout=0.5,
    )

    # Load the F1 metric and define the `compute_metrics` function
    f1_metric = load_metric('f1')
    precision_metric = load_metric('precision')
    recall_metric = load_metric('recall')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        params = dict(predictions=preds, references=labels, average='macro')
        f1 = f1_metric.compute(**params)
        precision = precision_metric.compute(**params)
        recall = recall_metric.compute(**params)
        return dict(f1=f1, precision=precision, recall=recall)

    # Initialise the training arguments
    training_args = TrainingArguments(
        output_dir='.',
        num_train_epochs=1000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_steps=100,
        eval_steps=500,
        report_to='none',
    )

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    train_model()

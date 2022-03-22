'''Model that detects whether a job listing is relevant or not'''

from datasets import Dataset
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments)
import torchmetrics as tm
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import json
import os
from trainers import MultiLabelTrainer, ClassWeightTrainer


def train_filtering_model():
    '''Trains the filtering model and stores it to disk'''

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

    # Convert the data to a HuggingFace dataset
    labels = df[['title_or_tasks', 'requirements']].values
    dataset = Dataset.from_dict(dict(text=df.cleaned_text.tolist(),
                                     label=labels))

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
        num_labels=2,
        hidden_dropout_prob=0.1,
        classifier_dropout=0.5,
    )

    # Create the model directory if it doesn't exist
    model_dir = Path('models')
    if not model_dir.exists():
        model_dir.mkdir()

    # Initialise the training arguments
    training_args = TrainingArguments(
        output_dir='models/filtering_model',
        hub_model_id='saattrupdan/job-listing-filtering-model',
        max_steps=1,
        #num_train_epochs=25,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_steps=50,
        eval_steps=50,
        report_to='none',
        save_total_limit=0,
        push_to_hub=True,
    )

    # Initialise the trainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Initialise the metrics
    params = dict(average='none', num_classes=2, multiclass=True)
    f1_metric = tm.FBetaScore(beta=1, **params)
    precision_metric = tm.Precision(**params)
    recall_metric = tm.Recall(**params)

    # Get the predictions and labels for the validation set
    model.cpu().eval()
    all_labels = torch.zeros(len(val), 2).long()
    all_preds = torch.zeros(len(val), 2)
    for idx in range(len(val)):
        inputs = data_collator(val.remove_columns(['text'])[idx:idx+1])
        all_labels[idx] = inputs.labels
        inputs.pop('labels')
        preds = model(**inputs).logits > 0
        all_preds[idx] = preds

    # Evaluate the model
    for idx, task in enumerate(['title_or_tasks', 'requirements']):

        # Compute the metrics
        args = [all_preds[:, idx], all_labels[:, idx]]
        f1 = f1_metric(*args)[1].item()
        precision = precision_metric(*args)[1].item()
        recall = recall_metric(*args)[1].item()

        # Print the results
        print(f'\n*** Scores for {task} ***')
        print(f'F1-score: {100 * f1:.2f}')
        print(f'Precision: {100 * precision:.2f}')
        print(f'Recall: {100 * recall:.2f}')

    breakpoint()

    # Save the model
    model.save_pretrained()

    # Push to hub
    trainer.push_to_hub()


def train_relevance_model():
    '''Trains the relevance model and stores it to disk'''

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
    df = df[['url', 'cleaned_text', 'title_or_tasks', 'requirements', 'bad']]
    df = (df.explode(['cleaned_text', 'title_or_tasks', 'requirements', 'bad'])
            .query('title_or_tasks or requirements')
            .drop(columns=['title_or_tasks', 'requirements']))

    # Add more data
    dfs = list()
    for i in range(100):
        extra_df = (df.sample(frac=0.5, replace=False, random_state=i)
                      .groupby('url')
                      .agg(dict(cleaned_text=lambda x: '\n'.join(x),
                                bad=lambda x: any(x))))
        dfs.append(extra_df)
    df = pd.concat(dfs)

    # Convert the data to a HuggingFace dataset
    labels = 1 - df.bad.astype(float)
    dataset = Dataset.from_dict(dict(text=df.cleaned_text.tolist(),
                                     label=labels.tolist()))

    # Split the dataset into training and validation sets
    splits = dataset.train_test_split(train_size=0.9)
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
        num_labels=1,
        hidden_dropout_prob=0.1,
        classifier_dropout=0.5,
    )

    # Create the model directory if it doesn't exist
    model_dir = Path('models')
    if not model_dir.exists():
        model_dir.mkdir()

    # Initialise the training arguments
    training_args = TrainingArguments(
        output_dir='models/relevance_model',
        hub_model_id='saattrupdan/job-listing-relevance-model',
        #num_train_epochs=10,
        max_steps=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_steps=10,
        eval_steps=50,
        report_to='none',
        save_total_limit=0,
        push_to_hub=True,
    )

    # Initialise the trainer
    trainer = ClassWeightTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        pos_weight=2,
    )

    # Train the model
    trainer.train()

    # Initialise the metrics
    params = dict(average='none', num_classes=2)
    f2_metric = tm.FBetaScore(beta=2, **params)
    precision_metric = tm.Precision(**params)
    recall_metric = tm.Recall(**params)

    # Get the predictions and labels for the validation set
    model.cpu().eval()
    all_labels = torch.zeros(len(val), 1).long()
    all_preds = torch.zeros(len(val), 1)
    for idx in range(len(val)):
        inputs = data_collator(val.remove_columns(['text'])[idx:idx+1])
        all_labels[idx] = inputs.labels
        inputs.pop('labels')
        preds = model(**inputs).logits[0] > 0
        all_preds[idx] = preds

    # Compute the metrics
    args = [all_preds, all_labels]
    f2 = f2_metric(*args)[1].item()
    precision = precision_metric(*args)[1].item()
    recall = recall_metric(*args)[1].item()

    # Print the results
    print(f'\n*** Scores ***')
    print(f'F2-score: {100 * f2:.2f}')
    print(f'Precision: {100 * precision:.2f}')
    print(f'Recall: {100 * recall:.2f}')

    # Save the model
    model.save_pretrained()

    # Push to hub
    trainer.push_to_hub()


if __name__ == '__main__':
    #train_filtering_model()
    train_relevance_model()

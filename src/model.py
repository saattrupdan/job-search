'''Model that detects whether a job listing is relevant or not'''

from datasets import Dataset, load_metric
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)

import numpy as np
from pathlib import Path
import warnings
import json


def train_model():
    '''Trains the model and stores it to disk'''

    # Set up pretrained model ID
    model_id = 'markussagen/xlm-roberta-longformer-base-4096'

    # Load the data
    data_path = Path('data') / 'job_listings.jsonl'
    with data_path.open('r') as f:
        job_listings = [json.loads(line) for line in f]

    # If there are job listings with no label then raise a warning and remove
    # them from the dataset
    num_no_label = sum(1 for job in job_listings if 'label' not in job)
    if num_no_label > 0:
        warnings.warn(f'{num_no_label} job listings have no label and '
                      f'have been removed from the training data.')
        job_listings = [job for job in job_listings if 'label' in job]

    # Extract the raw features and labels
    corpus = [job['cleaned_text'] for job in job_listings]
    labels = [1 if job['label'] == 'Relevant' else 0 for job in job_listings]

    # Convert the data to a HuggingFace dataset
    dataset = Dataset.from_dict(dict(text=corpus, label=labels))

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
        return tokenizer(example['text'], truncation=True)
    train = train.map(tokenize)
    val = val.map(tokenize)

    # Initialise the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2
    )

    # Load the F1 metric and define the `compute_metrics` function
    f1 = load_metric('f1')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return f1.compute(predictions=preds, references=labels)

    # Initialise the training arguments
    training_args = TrainingArguments(
        output_dir='.',
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_steps=10,
        eval_steps=50,
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

'''Main script which regularly queries for jobs, and filters and emails them'''

from emailbot import EmailBot
from jobscraper import JobScraper
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification)


def main():
    # Create list of relevant queries
    queries = [
        'analytical chemistry',
        'biosensor',
        'gc-ms',
        'electrochemistry',
        'electroanalytical chemistry',
        'separation science',
        'chromatography',
        'lab on a chip',
        'electrochemical sensor',
        'ceramic sensors',
        'metal oxide sensor',
        'electronic nose',
        'VOC',
        'volatile',
        'mVOC',
        'bVOC',
        'metabolite',
        'gas analysis'
    ]

    # Create job scraper
    job_scraper = JobScraper(queries=queries)

    # Create email bot
    email_bot = EmailBot()

    # Load filtering and relevance tokenizers
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # Load filtering and relevance models
    filtering_model = AutoModelForSequenceClassification.from_pretrained(
        './models/filtering_model'
    ).cpu().eval()
    relevance_model = AutoModelForSequenceClassification.from_pretrained(
        './models/relevance_model'
    ).cpu().eval()

    # Initialise data collators
    data_collator = DataCollatorWithPadding(tokenizer)

    # Update file with job listings
    #new_job_listings = job_scraper.scrape_jobs()
    new_job_listings = [dict(url='https://www.google.com',
                             cleaned_text='This is a test job\nYou need to have 10+ years of experience with programming')]

    # Split up the job listings into paragraphs
    df = pd.DataFrame.from_records(new_job_listings).drop_duplicates('url')
    df['cleaned_text'] = df.cleaned_text.str.split('\n')
    df = df.explode('cleaned_text').reset_index(drop=True)

    # Use the filtering model to filter out irrelevant paragraphs
    paragraphs = data_collator([
        tokenizer(p, truncation=True, max_length=512)
        for p in df.cleaned_text
    ])
    mask = (filtering_model(**paragraphs).logits > 0).numpy()
    mask = np.logical_or(mask[:, 0], mask[:, 1])
    df = (df.loc[mask]
            .groupby('url')
            .agg(dict(cleaned_text=lambda x: '\n'.join(x))))

    # Use the relevance model on the resulting filtered job listings to arrive
    # at the relevant ones
    filtered_job_listings = data_collator([
        tokenizer(listing, truncation=True, max_length=512)
        for listing in df.cleaned_text
    ])
    mask = (relevance_model(**filtered_job_listings).logits > 0).numpy()
    filtered_job_listings = (df.loc[mask, ['url', 'cleaned_text']]
    filtered_job_listings = (df.reset_index()
                               .loc[mask, ['url', 'cleaned_text']]
                               .to_dict('records'))

    # Send the relevant new job listings by email
    email_bot.send_job_listings(relevant_job_listings,
                                to='saattrupdan@gmail.com')
    email_bot.send_job_listings(relevant_job_listings,
                                to='amy.smart1@btinternet.com')

    # Close the job_scraper
    job_scraper.close()


if __name__ == '__main__':
    main()

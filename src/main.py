'''Main script which regularly queries for jobs, and filters and emails them'''

from emailbot import EmailBot
from jobscraper import JobScraper
import pandas as pd
import numpy as np
import logging
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification)


# Set up logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


def main():

    # Create email bot
    email_bot = EmailBot()

    logger.info('Loading tokenizer and model')

    # Load filtering and relevance tokenizers
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # Load filtering and relevance models
    filtering_model = AutoModelForSequenceClassification.from_pretrained(
        'saattrupdan/job-listing-filtering-model'
    ).cpu().eval()
    relevance_model = AutoModelForSequenceClassification.from_pretrained(
        'saattrupdan/job-listing-relevance-model'
    ).cpu().eval()

    # Initialise data collators
    data_collator = DataCollatorWithPadding(tokenizer)

    logger.info('Starting job scraper')

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

    logger.info('Scraping new job listings')

    # Update file with job listings
    new_job_listings = job_scraper.scrape_jobs()

    logger.info(f'Found {len(new_job_listings):,} new job listings')
    logger.info('Filtering paragraphs of new job listings')

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
            .agg(dict(cleaned_text=lambda x: ' '.join(x))))
    df['cleaned_text'] = df.cleaned_text.str.lower()

    logger.info(f'Found {int(mask.sum()):,} relevant paragraphs out '
                f'of {mask.shape[0]:,}.')
    logger.info(f'Classifying the relevance of the resulting {len(df):,} '
                f'job listings')

    # Use the relevance model on the resulting filtered job listings to arrive
    # at the relevant ones
    filtered_job_listings = data_collator([
        tokenizer(listing, truncation=True, max_length=512)
        for listing in df.cleaned_text
    ])
    mask = (relevance_model(**filtered_job_listings).logits > 0).numpy()
    relevant_job_listings = (df.reset_index()
                               .loc[mask, ['url', 'cleaned_text']]
                               .to_dict('records'))

    logger.info(f'Found {len(relevant_job_listings):,} relevant job listings')

    # Send the relevant new job listings by email
    if len(relevant_job_listings) > 0:
        logger.info('Sending email with relevant job listings')
        email_bot.send_job_listings(relevant_job_listings,
                                    to='saattrupdan@gmail.com')
        email_bot.send_job_listings(relevant_job_listings,
                                    to='amy.smart1@btinternet.com')

    # Close the job_scraper
    job_scraper.close()

    logger.info('All done!')


if __name__ == '__main__':
    main()

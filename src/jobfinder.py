'''Class that queries job listings from various sites and filters them'''

from jobindex import JobIndex
from typing import List, Union
from pathlib import Path
import json
from tqdm.auto import tqdm
from utils import clean_job_listing
import pandas


# Enable tqdm with pandas
tqdm.pandas()


class JobScraper:
    '''Class that queries job listings from various sites and filters them.

    Args:
        queries (list of str):
            List of queries to search for.
        num_pages (int, optional):
            Number of pages to search for each query. Defaults to 10.
        listing_path (str or Path, optional):
            Path to save job listings to. Defaults to 'data/job_listings.jsonl'.
        overwrite (bool, optional):
            Whether to overwrite the listing_path if it already exists.
            Defaults to False.

    Attributes:
        queries (list of str): List of queries to search for.
        num_pages (int): Number of pages to search for each query.
        listing_path (str or Path): Path to save job listings to.
        overwrite (bool): Whether to overwrite existing listings.
    '''
    def __init__(self,
                 queries: List[str],
                 num_pages: int = 10,
                 listing_path: Union[str, Path] = 'data/job_listings.jsonl',
                 overwrite: bool = False):
        self.queries = queries
        self.num_pages = num_pages
        self.listing_path = Path(listing_path)
        self.overwrite = overwrite
        self._job_sites = [JobIndex(num_pages=num_pages)]
        self._urls = list()

        # If we are overwriting then delete the file
        if self.overwrite and self.listing_path.exists():
            self.listing_path.unlink()

        # If the file exists then load in all the stored URLs, to ensure that
        # we're not duplicating any jobs
        if self.listing_path.exists():
            with self.listing_path.open('r') as f:
                self._urls = [json.loads(line)['url'] for line in f]

    def scrape_jobs(self):
        '''Finds and cleans job listings from all job sites'''

        # Query all the job sites for all the queries and save the job listings
        # to disk
        for query in tqdm(self.queries, desc='Fetching and parsing jobs'):
            for job_site in self._job_sites:
                job_listings = job_site.query(query, urls_to_ignore=self._urls)
                self._store_jobs(job_listings)

        # Clean all the job listings on disk
        self.clean_jobs()

    def clean_jobs(self):
        '''Cleans all the stored job listings'''
        if self.listing_path.exists():

            # Open the file and read in all the job listings
            with self.listing_path.open('r') as f:
                job_listings = [json.loads(line) for line in f]

            # Convert the job listings to a Pandas DataFrame
            df = pandas.DataFrame.from_records(job_listings)

            # Drop duplicates in the job listings
            df.drop_duplicates(subset='url', inplace=True)

            # Truncate the `text` column to 100,000 characters
            df['text'] = df.text.apply(lambda x: x[:100_000])

            # Add a `cleaned_text` column to the DataFrame
            df['cleaned_text'] = df.text.progress_apply(clean_job_listing)

            # Store the cleaned job listings
            self._store_jobs(df.to_dict('records'), overwrite=True)

    def _store_jobs(self, job_listings: List[dict], overwrite: bool = False):
        '''Stores job listings to a JSONL file.

        Args:
            job_listings (list of dict):
                List of job listings to store.
            overwrite (bool, optional):
                Whether to overwrite the listing_path if it already exists.
                Defaults to False.
        '''
        with self.listing_path.open('w' if overwrite else 'a') as f:
            for job_listing in job_listings:
                f.write(json.dumps(job_listing))
                f.write('\n')


if __name__ == '__main__':
    # Create list of relevant queries
    queries = [
        'analytical chemistry',
        'biosensors',
        'gc-ms',
        'electrochemistry',
        'electroanalytical chemistry',
        'separation science',
        'chromatography',
        'lab on a chip',
        'biosensors',
        'electrochemical sensors',
        'ceramic sensors',
        'metal oxide sensors',
        'electronic nose',
        'VOC',
        'volatile',
        'mVOC',
        'bVOC',
        'metabolite volatile',
        'gas analysis'
    ]

    # Create JobFinder
    job_finder = JobScraper(queries=queries, overwrite=False)

    # Update file with job listings
    job_finder.scrape_jobs()

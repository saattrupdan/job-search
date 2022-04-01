'''Class that queries job listings from various sites and filters them'''

from typing import List, Union, Optional
from pathlib import Path
import json
from tqdm.auto import tqdm
from utils import clean_job_listing
import pandas

from jobindex import JobIndex
from dtu import DTU
from thehub import TheHub
from ku import KU


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
                 overwrite: bool = False,
                 headless: bool = True):
        self.queries = queries
        self.num_pages = num_pages
        self.listing_path = Path(listing_path)
        self.overwrite = overwrite
        self.headless = headless
        self._job_site_classes = [
            JobIndex,
            TheHub,
            KU,
            DTU,
        ]
        self._urls = list()

        # If we are overwriting then delete the file
        if self.overwrite and self.listing_path.exists():
            self.listing_path.unlink()

        # If the file exists then load in all the stored URLs, to ensure that
        # we're not duplicating any jobs
        if self.listing_path.exists():
            with self.listing_path.open('r') as f:
                self._urls = [json.loads(line)['url'] for line in f]

    def scrape_jobs(self) -> List[dict]:
        '''Finds and cleans job listings from all job sites.

        Returns:
            list of dict:
                List of job listings.
        '''
        # Query all the job sites for all the queries and save the job listings
        # to disk
        all_job_listings = list()
        for job_site_class in self._job_site_classes:

            # Initialise the job site
            job_site = job_site_class(num_pages=self.num_pages,
                                      headless=self.headless)

            # Query the job site for all the queries
            if job_site.uses_queries:
                desc = f'Fetching and parsing jobs from {job_site.name}'
                for query in tqdm(self.queries, desc=desc):
                    job_listings = job_site.query(query=query,
                                                  urls_to_ignore=self._urls)
                    all_job_listings.extend(job_listings)
            else:
                job_listings = job_site.query(urls_to_ignore=self._urls)
                all_job_listings.extend(job_listings)

            # Close the job site
            job_site.close()

        if len(all_job_listings) > 0:

            # Clean all the new job listings
            all_job_listings = self.clean_jobs(all_job_listings)

            # Store the cleaned new job listings to disk
            self._store_jobs(all_job_listings)

        # Return the new job listings
        return all_job_listings

    def clean_jobs(self,
                   job_listings: Optional[List[dict]] = None) -> List[dict]:
        '''Cleans all the stored job listings.

        Args:
            job_listings (list of dict or None, optional):
                List of job listings to clean. If None then all job listings on
                disk will be loaded instead. Defaults to None.

        Returns:
            list of dict:
                List of cleaned job listings.

        Raises:
            FileNotFoundError:
                If the listing_path does not exist and `job_listings` is None.
        '''
        if not self.listing_path.exists() and job_listings is None:
            raise FileNotFoundError(f'{self.listing_path} does not exist')

        # Open the file and read in all the job listings if `job_listings` is
        # not specified
        if job_listings is None:
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

        # Replace the `job_listings` list with the cleaned listings
        job_listings = df.to_dict('records')

        # Store the cleaned job listings if no `job_listings` were specified
        if job_listings is None:
            self._store_jobs(job_listings, overwrite=True)

        return job_listings

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

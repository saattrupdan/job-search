'''Class that queries job listings from various sites and filters them'''

from jobindex import JobIndex
from typing import List, Union
from pathlib import Path
import json
from tqdm.auto import tqdm


class JobFinder:
    '''Class that queries job listings from various sites and filters them.

    Args:
        queries (list of str):
            List of queries to search for.
        num_pages (int, optional):
            Number of pages to search for each query. Defaults to 3.
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
                 num_pages: int = 3,
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

    def update_jobs(self):
        '''Finds jobs from all job sites'''

        # Collect job listings from all job sites
        for query in tqdm(self.queries, desc='Fetching and parsing jobs'):
            for job_site in self._job_sites:
                job_listings = job_site.query(query, urls_to_ignore=self._urls)
                self._store_job_listings(job_listings)

    def _store_job_listings(self, job_listings: List[dict]):
        '''Stores job listings to a JSONL file.

        Args:
            job_listings (list of dict):
                List of job listings to store.
        '''
        with self.listing_path.open('a') as f:
            for job_listing in job_listings:
                job_listing['text'] = job_listing['text'][:100_000]
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
    job_finder = JobFinder(queries=queries, overwrite=True)

    # Update file with job listings
    job_finder.update_jobs()

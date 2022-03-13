'''Class that queries jobindex.dk for job listings'''

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List
from utils import clean_job_listing


class JobIndex:
    '''Class that queries jobindex.dk for job listings.

    Args:
        num_pages (int, optional):
            Number of pages to query. Defaults to 3.

    Attributes:
        num_pages (int): Number of pages to query.
    '''
    base_url: str = 'https://www.jobindex.dk'

    def __init__(self, num_pages: int = 3):
        self.num_pages = num_pages

    def query(self,
              query: str,
              urls_to_ignore: List[str] = list(),
              area: str = 'storkoebenhavn') -> List[dict]:
        '''Query jobindex.dk for job listings.

        Args:
            query (str):
                The query to search for.
            urls_to_ignore (list of str, optional):
                A list of urls to ignore. Defaults to an empty list.
            area (str, optional):
                The area to search in. Defaults to 'storkoebenhavn'.

        Returns:
            list of dict:
                A list of job listings, with each listing being dicts with keys
                'url' and 'text'.
        '''
        # Initialise the list of urls to the job listings
        urls = list()

        # Iterate over the search result pages
        desc = f'Querying jobindex.dk for {query}'
        for page in tqdm(range(1, self.num_pages + 1), desc=desc, leave=False):

            # Query jobindex.dk for job listings
            url = f'{self.base_url}/jobsoegning/{area}'
            response = requests.get(url,
                                    params=dict(q=query, page=page),
                                    allow_redirects=True)

            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get the div tag that contains the job listings, having the id
            # 'result_list_box'
            result_list_box = soup.find('div', id='result_list_box')

            # Get the div tags that contain the job listings, being the
            # children div tags of the result_list_box div tag with class
            # 'PaidJob-inner'
            jobs = result_list_box.find_all('div', class_='PaidJob-inner')

            # Get the URLs to the job listings, being the href attribute of the
            # first a tag which is the child of the job listing that has a <b>
            # tag as a child
            for job in jobs:
                url = [a['href'] for a in job.find_all('a') if a.b][0]
                if url not in urls_to_ignore:
                    urls.append(url)

        # For each URL, get the job listing
        job_listings = list()
        desc = f'Parsing jobindex.dk job listings for {query}'
        for url in tqdm(urls, desc=desc, leave=False):

            # Query jobindex.dk for the job listing. If this results in an
            # error then skip the job listing
            try:
                response = requests.get(url, allow_redirects=True)
            except requests.exceptions.RequestException:
                continue

            # Parse the response if the response is successful
            if response.status_code == 200:

                # Parse the response
                job_listing = BeautifulSoup(response.text, 'html.parser')

                # Extract the text of the job listing
                job_listing = job_listing.get_text()

                # Clean the job listing
                job_listing = clean_job_listing(job_listing)

                # Store the cleaned job listing in the list of job listings
                job_listings.append(dict(url=url, text=job_listing))

        # Return the list of job listings
        return job_listings

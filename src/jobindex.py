'''Class that queries jobindex.dk for job listings'''

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List
import re


class JobIndex:
    '''Class that queries jobindex.dk for job listings'''

    base_url: str = 'https://www.jobindex.dk'

    def query(self,
              query: str,
              area: str = 'storkoebenhavn',
              num_pages: int = 3) -> List[str]:
        '''Query jobindex.dk for job listings.

        Args:
            query (str):
                The query to search for.
            area (str, optional):
                The area to search in. Defaults to 'storkoebenhavn'.
            num_pages (int, optional):
                The number of pages to search. Defaults to 3.

        Returns:
            list of str:
                A list of job listings.
        '''
        # Initialise the list of urls to the job listings
        urls = list()

        # Iterate over the search result pages
        for page in tqdm(range(1, num_pages + 1), desc='Finding job ads'):

            # Query jobindex.dk for job listings
            url = f'{self.base_url}/jobsoegning/{area}'
            response = requests.get(url, params=dict(q=query, page=page))

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
                urls.append(url)

        # For each URL, get the job listing
        job_listings = list()
        for url in tqdm(urls, desc='Parsing job ads'):

            # Query jobindex.dk for the job listing
            response = requests.get(url)

            # Parse the response if the response is successful
            if response.status_code == 200:

                # Parse the response
                job_listing = BeautifulSoup(response.text, 'html.parser')

                # Extract the text of the job listing
                job_listing = job_listing.get_text()

                # Clean the job listing
                job_listing = self._clean_job_listing(job_listing)

                # Store the cleaned job listing in the list of job listings
                job_listings.append(job_listing)

        # Return the list of job listings
        return job_listings

    @staticmethod
    def _clean_job_listing(job_listing: str) -> str:
        '''Clean the job listing.

        Args:
            job_listing (str):
                The job listing to clean. This is the text of the job listing,
                and not the raw HTML.

        Returns:
            str:
                The cleaned job listing.
        '''
        # Replace emails by 'EMAIL'
        email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        job_listing = re.sub(email_regex, 'EMAIL', job_listing)

        # Replace phone numbers by 'PHONE'
        phone_regex = r'(\+[0-9]{1,2} ?)?([0-9]{2} ?){3}[0-9]{2}'
        job_listing = re.sub(phone_regex, 'PHONE', job_listing)

        # Replace URLs by 'URL'
        url_regex = r'(https?://|www\.)[^\s]+'
        job_listing = re.sub(url_regex, 'URL', job_listing)

        # Convert tabs to spaces
        job_listing = re.sub('\t', ' ', job_listing)

        # Strip whitespace
        job_listing = job_listing.strip()

        # Remove whitespace after newlines
        job_listing = re.sub('\n +', '\n', job_listing)

        # Remove more than two consecutive newlines
        job_listing = re.sub('\n\n+', '\n\n', job_listing)

        # Remove consecutive whitespace
        job_listing = re.sub(' +', ' ', job_listing)

        # Return the cleaned job listing
        return job_listing

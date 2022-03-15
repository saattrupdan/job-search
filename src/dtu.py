'''Class that queries dtu.dk for job listings'''

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List


class DTU:
    '''Class that queries dtu.dk for job listings'''

    name = 'DTU'
    base_url: str = ('https://www.dtu.dk/english/about/'
                     'job-and-career/vacant-positions')
    uses_queries: bool = False

    def query(self, urls_to_ignore: List[str] = list()) -> List[dict]:
        '''Query dtu.dk for job listings.

        Args:
            urls_to_ignore (list of str, optional):
                A list of urls to ignore. Defaults to an empty list.

        Returns:
            list of dict:
                A list of job listings, with each listing being dicts with keys
                'url' and 'text'.
        '''
        # Initialise the list of urls to the job listings
        urls = list()

        # Query dtu.dk for job listings
        institutes = [
            55004761,  # Chemistry
            56000002,  # Health Tech
            55004755,  # Environment
        ]
        for inst in institutes:

            # Query the institute for vacant positions
            params = dict(type='Videnskabeligt',  # Academic position
                          category=5372,  # Postdoc
                          inst=inst)
            response = requests.get(self.base_url,
                                    params=params,
                                    allow_redirects=True)

            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get the table of job listings
            job_list = soup.find('table', id='jobList').find('tbody')

            # Iterate over the contents of the table
            jobs = job_list.find_all('tr')

            # Get the URLs of the job listings, being the href attribute of
            # the first <a> tag in the <td> tag
            for job in jobs:
                url = job.find('td').find('a').get('href')
                if url not in urls_to_ignore:
                    urls.append(url)

        # For each URL, get the job listing
        job_listings = list()
        for url in tqdm(urls, desc='Parsing dtu.dk job listings', leave=False):

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

                # Store the cleaned job listing in the list of job listings
                job_listings.append(dict(url=url, text=job_listing))

        # Return the list of job listings
        return job_listings

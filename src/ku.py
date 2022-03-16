'''Class that queries ku.dk for job listings'''

from tqdm.auto import tqdm
from selenium.webdriver.common.by import By
from typing import List
from base_scraper import BaseScraper
from bs4 import BeautifulSoup


class KU(BaseScraper):
    '''Class that queries ku.dk for job listings.

    Args:
        num_pages (int, optional):
            Number of pages to query. Defaults to 3.

    Attributes:
        num_pages (int): Number of pages to query.
    '''
    base_url: str = 'https://employment.ku.dk'
    uses_queries: bool = True

    def __init__(self, num_pages: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_pages = num_pages

        # Accept cookies and remove popups
        self._get(self.base_url + '/all-vacancies')
        (self._driver.find_element(by=By.ID, value='ccc')
                     .find_element(by=By.CLASS_NAME, value='ccc-buttons')
                     .find_elements(by=By.TAG_NAME, value='button')[1]
                     .click())

    def query(self,
              query: str,
              urls_to_ignore: List[str] = list()) -> List[dict]:
        '''Query jobindex.dk for job listings.

        Args:
            query (str):
                The query to search for.
            urls_to_ignore (list of str, optional):
                A list of urls to ignore. Defaults to an empty list.

        Returns:
            list of dict:
                A list of job listings, with each listing being dicts with keys
                'url' and 'text'.
        '''
        # Initialise the list of urls to the job listings
        urls = list()

        # Query ku.dk for job listings
        self._get(self.base_url + '/all-vacancies')
        search = self._driver.find_element(by=By.ID, value='pxs_search')
        btn = self._driver.find_element(by=By.CLASS_NAME,
                                        value='pxs_search_button')
        search.clear()
        search.send_keys(query)
        btn.click()

        # Get next button
        paginate_div = 'DataTables_Table_0_paginate'
        next_btn = (self._driver.find_element(by=By.ID, value=paginate_div)
                                 .find_element(by=By.CLASS_NAME,
                                               value='next'))

        # Iterate over the search result pages
        desc = f'Querying ku.dk for {query}'
        for _ in tqdm(range(1, self.num_pages + 1), desc=desc, leave=False):

            # Parse the page
            soup = BeautifulSoup(self._driver.page_source, 'html.parser')

            # Get the table tag that contains the job listings, having the
            # class 'vacancies'
            job_table = soup.find('table', class_='vacancies')

            # Get the tr tags that contain the job listings, being the
            # children tr tags of the job_table table tag with class
            # 'vacancy-specs'
            jobs = job_table.find_all('tr', class_='vacancy-specs')

            # Get the URLs to the job listings, being the href attribute of the
            # first a tag of the first td tag of each tr tag
            for job in jobs:
                url = self.base_url + job.td.a['href']
                if url not in urls_to_ignore:
                    urls.append(url)
                    urls_to_ignore.append(url)

            # Click the next button
            next_btn.click()

        # For each URL, get the job listing
        job_listings = list()
        desc = f'Parsing jobindex.dk job listings for {query}'
        for url in tqdm(urls, desc=desc, leave=False):

            # Query jobindex.dk for the job listing
            text = self._get(url).get_text()

            # Store the cleaned job listing in the list of job listings
            job_listings.append(dict(url=url, text=text))

        # Return the list of job listings
        return job_listings

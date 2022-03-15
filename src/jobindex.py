'''Class that queries jobindex.dk for job listings'''

from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List
import chromedriver_autoinstaller as chrome_installer
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from time import sleep


class JobIndex:
    '''Class that queries jobindex.dk for job listings.

    Args:
        num_pages (int, optional):
            Number of pages to query. Defaults to 3.

    Attributes:
        num_pages (int): Number of pages to query.
    '''
    name = 'jobindex'
    base_url: str = 'https://www.jobindex.dk/jobsoegning'
    uses_queries: bool = True

    def __init__(self, num_pages: int = 3):
        self.num_pages = num_pages
        chrome_installer.install()

        # Initialise Chrome web driver
        options = ChromeOptions()
        options.headless = True
        self._driver = webdriver.Chrome(chrome_options=options)

        # Accept cookies and remove popups
        self._driver.get(self.base_url)
        sleep(10)
        consent_btn_id = 'jix-cookie-consent-accept-all'
        self._driver.find_element(by=By.ID, value=consent_btn_id).click()
        (self._driver.find_element(by=By.ID, value='jobmail_popup_block')
                     .find_element(by=By.CLASS_NAME, value='close')
                     .click())

    def close(self):
        '''Close the web driver'''
        self._driver.close()

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
        # Ensure that the query has quotation marks around it
        query = '"' + query.strip('"') + '"'

        # Initialise the list of urls to the job listings
        urls = list()

        # Iterate over the search result pages
        desc = f'Querying jobindex.dk for {query}'
        for page in tqdm(range(1, self.num_pages + 1), desc=desc, leave=False):

            # Query jobindex.dk for job listings
            url = f'{self.base_url}/{area}?q={query}&page={page}'
            self._driver.get(url)
            sleep(10)

            # Parse the response
            soup = BeautifulSoup(self._driver.page_source, 'html.parser')

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

            # Query jobindex.dk for the job listing
            self._driver.get(url)
            sleep(10)

            # Parse the response
            job_listing = BeautifulSoup(self._driver.page_source,
                                        'html.parser')

            # Extract the text of the job listing
            job_listing = job_listing.get_text()

            # Store the cleaned job listing in the list of job listings
            job_listings.append(dict(url=url, text=job_listing))

        # Return the list of job listings
        return job_listings

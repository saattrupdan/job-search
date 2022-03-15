'''Class that queries thehub.io for job listings'''

from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List
import chromedriver_autoinstaller as chrome_installer
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from time import sleep


class TheHub:
    '''Class that queries thehub.io for job listings.

    Args:
        num_pages (int, optional):
            Number of pages to query. Defaults to 3.

    Attributes:
        num_pages (int): Number of pages to query.
    '''
    name = 'the Hub'
    base_url: str = 'https://thehub.io'
    uses_queries: bool = True

    def __init__(self, num_pages: int = 3):
        self.num_pages = num_pages
        chrome_installer.install()

        # Initialise Chrome web driver
        options = ChromeOptions()
        options.headless = True
        self._driver = webdriver.Chrome(chrome_options=options)

        # Accept cookies
        self._driver.get(self.base_url)
        sleep(10)
        (self._driver.find_element(by=By.ID, value='coiConsentBannerBase')
             .find_elements(by=By.TAG_NAME, value='div')[0]
             .find_elements(by=By.TAG_NAME, value='button')[0]
             .click())

    def close(self):
        '''Close the web driver'''
        self._driver.close()

    def query(self,
              query: str,
              urls_to_ignore: List[str] = list(),
              area: str = 'København, Denmark') -> List[dict]:
        '''Query thehub.io for job listings.

        Args:
            query (str):
                The query to search for.
            urls_to_ignore (list of str, optional):
                A list of urls to ignore. Defaults to an empty list.
            area (str, optional):
                The area to search in. Defaults to 'København, Denmark'.

        Returns:
            list of dict:
                A list of job listings, with each listing being dicts with keys
                'url' and 'text'.
        '''
        # Initialise the list of urls to the job listings
        all_urls = list()

        # Iterate over the search result pages
        desc = f'Querying thehub.io for {query}'
        for page in tqdm(range(1, self.num_pages + 1), desc=desc, leave=False):

            # Query thehub.io for job listings
            url = (f'{self.base_url}/jobs?search={query}&location={area}&'
                   f'countryCode=DK&sorting=mostPopular&page={page}')
            self._driver.get(url)
            sleep(10)

            # Parse the response
            soup = BeautifulSoup(self._driver.page_source, 'html.parser')

            # If there are no results then break
            if soup.find('div', class_='no-results'):
                break

            # Get the URLs of the job listings
            a_tags = soup.find_all('a', class_='card-job-find-list__link')
            urls = [self.base_url + a.get('href') for a in a_tags]
            urls = [url for url in urls if url not in urls_to_ignore]
            all_urls.extend(urls)

        # For each URL, get the job listing
        job_listings = list()
        desc = f'Parsing thehub.io job listings for {query}'
        for url in tqdm(all_urls, desc=desc, leave=False):

            # Query thehub.io for the job listing
            self._driver.get(url)
            sleep(10)

            # Parse the response
            job_listing = BeautifulSoup(self._driver.page_source,
                                        'html.parser')

            # Extract the title of the job listing
            class_name = 'view-job-details__title'
            title = job_listing.find('h2', class_=class_name).get_text()

            # Extract the text of the job listing
            class_name = 'text-block__content text-block__content--default'
            content = job_listing.find('content', class_=class_name).get_text()

            # Concatenate the title and text
            text = title + '\n' + content

            # Store the cleaned job listing in the list of job listings
            job_listings.append(dict(url=url, text=text))

        # Return the list of job listings
        return job_listings

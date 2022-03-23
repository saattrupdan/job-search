'''Base class for scraping job listings from websites'''

from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from time import sleep
from bs4 import BeautifulSoup
from typing import Union


class BaseScraper:
    '''Base class for scraping job listings from websites.

    Args:
        headless (bool, optional):
            Whether to run the web driver in headless mode. Defaults to True.

    Attributes:
        headless (bool): Whether to run the web driver in headless mode.
    '''
    def __init__(self, headless: bool = True):
        self.headless = headless

        # Initialise web driver
        options = FirefoxOptions()
        options.headless = headless
        self._driver = webdriver.Firefox(options=options)

    @property
    def name(self) -> str:
        '''The name of the scraper.

        Returns:
            str: The name of the scraper.
        '''
        return self.__class__.__name__.lower()

    def close(self):
        '''Close the web driver'''
        self._driver.close()

    def _get(self, url: str,
             params: dict = dict(),
             return_soup: bool = True)  -> Union[BeautifulSoup, str]:
        '''Get the page at the given url.

        Args:
            url (str):
                The url to get.
            params (dict, optional):
                The parameters to pass to the url. Defaults to an empty dict.
            return_soup (bool, optional):
                Whether to return the page as a BeautifulSoup object. Defaults
                to True.

        Returns:
            BeautifulSoup:
                The page at the given url.
        '''
        # Form the URL
        get_str = '&'.join(f'{k}={v}' for k, v in params.items())
        url = url + '?' + get_str

        # Get the page
        self._driver.get(url)
        sleep(10)

        # Parse the page and return it
        if return_soup:
            return BeautifulSoup(self._driver.page_source, 'html.parser')
        else:
            return self._driver.page_source

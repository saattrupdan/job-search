'''Automatic sending of emails with newest job listings'''

import yagmail
import random
from dotenv import load_dotenv
from typing import List
import os


# Load .env file
load_dotenv()


class EmailBot:
    '''Automatic sending of emails with newest job listings'''

    def __init__(self):
        pwd = os.environ.get('GMAIL_PASSWORD')
        self.email = yagmail.SMTP('amysjobbot', pwd)

    def _random_greeting(self) -> str:
        '''Generate a random greeting.

        Returns:
            str:
                Random greeting.
        '''
        greetings = ['Hi',
                     'Hello',
                     'Hey',
                     'Howdy',
                     'Hiya',
                     'Sup',
                     'Yo',
                     'Hiiiiii',
                     'Hej',
                     'Dav',
                     'Davs',
                     'Halløj',
                     'Halløjsa']
        return random.choice(greetings)

    def _random_farewell(self) -> str:
        '''Generate a random farewell.

        Returns:
            str:
                Random farewell.
        '''
        farewells = ['Bye',
                     'Cheerio',
                     'See ya',
                     'Later',
                     'Hilsen',
                     'Med venlig hilsen',
                     'Kærlig hilsen',
                     'Vi ses',
                     'Hej hej']
        return random.choice(farewells)

    def send_job_listings(self, job_listings: List[dict], to: str):
        '''Send job listings an email.

        Args:
            job_listings (list of str):
                List of job listings to send.
            to (str):
                Email address to send to.
        '''
        subject = '[YourFavoriteJobBot] New Job Listings'
        content = (f'{self._random_greeting()}, it\'s me again!\n\n'
                   'Here are the latest job listings:\n\n')
        for idx, job_listing in enumerate(job_listings):
            content += f'    {1 + idx}. {job_listing["url"]}\n'
        content += f'\n{self._random_farewell()},\nYour fav job bot x'
        self.send_email(subject=subject, content=content, to=to)

    def send_email(self, subject: str, content: str, to: str):
        '''Send email with subject and content.

        Args:
            subject (str):
                Subject of email.
            content (str):
                Content of email.
            to (str):
                Email address to send to.
        '''
        # receiver = 'amy.smart1@btinternet.com'
        self.email.send(to=to, subject=subject, contents=content)

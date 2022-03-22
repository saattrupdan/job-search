'''Automatic sending of emails with newest job listings'''

import yagmail
from typing import List


class EmailBot:
    '''Automatic sending of emails with newest job listings'''

    def __init__(self):
        self.email = yagmail.SMTP('amysjobbot')

    def send_job_listings(self, job_listings: List[dict], to: str):
        '''Send job listings an email.

        Args:
            job_listings (list of str):
                List of job listings to send.
            to (str):
                Email address to send to.
        '''
        subject = '[YourFavoriteJobBot] New Job Listings'
        content = ('Hiiii, it\'s me again!\n\n'
                   'Here are the latest job listings:\n\n')
        for idx, job_listing in enumerate(job_listings):
            content += f'    {1 + idx}. {job_listing["url"]}\n'
        content += '\nCheerio,\nYour fav job bot x'
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


if __name__ == '__main__':
    content = ('Dear Amy,\n\nThis is a test email. Wowza.\n\n'
               'Best,\n Your favorite job bot')
    send_email(subject='Look at this cool email', content=content)

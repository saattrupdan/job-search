'''Utility functions and variables for the project'''

import re


IGNORED_PARAGRAPHS = [
    'skip to main content',
    'job i dag',
    'jobsøgning',
    'din indbakke',
    'opret dit cv',
    'søg job',
    'jobagent',
    'næste arbejdsplads',
    'jobsamtale',
    'send me alerts',
    'start applying with linkedin',
    'please contact your manager',
    'press tab to ',
    'job req number',
    'terms and conditions',
    'existing user',
    'apply',
    'enter a',
    'copyright',
    'cookie',
    'up and down arrows',
    'please enter',
    'please select',
    'current mailing address',
    'privacy information',
    'regulation (eu)',
    'policy',
    'type of data processed',
    'personal data',
    'purpose and legal basis',
    'data controller',
    'gdpr',
    'rights',
    'more information',
    'du kan søge',
    'application',
    'cover letter',
    'read more',
    'questions about the position',
    'join our talent community',
    'keyword',
    'alerts',
    'browser',
    'hvordan skriver du et godt cv?',
    'hjælp til at skrive en jobansøgning',
    'tjek din løn og se lønniveauet for profiler som din',
    'værktøjer og tips til dit nye arbejdsliv',
    'bliv klar til din lønforhandling',
    'få gode råd til dit cv',
    'inspiration til din ansøgning',
    'sæt jobglæde på agendaen',
    'kender du dine styrker?',
    'hvordan er din jobglæde?',
    'hvad tjener andre som dig?',
    'hvilken arbejdsplads passer du ind hos',
    'se virksomheder med graduate programmer',
    'se virksomheder med elevpladser',
    'hvordan er din arbejdsplads?',
    'find inspiration i udløbne jobopslag',
    'hjælper ukrainske flygtninge med at få job',
    'vælg mellem flere søgekriterier',
    'indrykket',
    'follow (the|a) company',
    'only show interest in job',
    'it is a good idea',
    'show interest in this job',
    'show interest in job',
    'more on showing interest',
    'you must have an online cv',
    'baseret på [0-9]+ evalueringer',
    'ansvarlig brug af dine oplysninger',
    'jobindex a/s, carl jacobsens vej 29-31',
    'i confirm that i have read and agree',
    '(cvr og ean)',
    'ean numre',
    'search jobs by location',
    'privacy',
    'address, city and state',
    'show interest',
    'more on following companies',
    'allow the company to contact you',
    'the company will have access',
    'view all jobs',
    'snak med en specialist',
    'search by postal code',
    'job portal',
    'add to saved jobs',
    'create a profile',
    'send to a friend',
    'e-mail this job ad to your friend',
    'select the language',
    'recipient e-mail address',
    'your recruitment process',
    'satisfied with your job',
]


def clean_job_listing(job_listing: str) -> str:
    '''Clean the job listing.

    Args:
        job_listing (str):
            The job listing to clean. This is the text of the job listing,
            and not the raw HTML.

    Returns:
        str:
            The cleaned job listing.
    '''
    # Replace emails with email tag
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    job_listing = re.sub(email_regex, '<email>', job_listing)

    # Replace phone numbers with phone tag
    phone_regex = r'(\+[0-9]{1,2} ?)?([0-9]{2} ?){3}[0-9]{2}'
    job_listing = re.sub(phone_regex, '<phone>', job_listing)

    # Replace URLs by url tag
    url_regex = r'(https?://|www\.)[^\s]+'
    job_listing = re.sub(url_regex, '<url>', job_listing)

    # Convert tabs to spaces
    job_listing = re.sub('\t', ' ', job_listing)

    # Convert \r to \n
    job_listing = re.sub('\r', '\n', job_listing)

    # Remove empty whitespace
    job_listing = re.sub('\xa0', ' ', job_listing)

    # Convert job listing to lowercase
    job_listing = job_listing.lower()

    # Remove paragraphs that are shorter than or equal to 3 words
    all_paragraphs = job_listing.split('\n')
    filtered_paragraphs = [p.strip() for p in all_paragraphs
                           if len(p.split()) > 3]
    job_listing = '\n'.join(filtered_paragraphs)

    # Remove all paragraphs that matches the ignored paragraphs
    for ignored_paragraph in IGNORED_PARAGRAPHS:
        regex_first_paragraph = f'^.*{ignored_paragraph}.*(?=\n)'
        regex_middle_paragraphs = f'(?<=\n).*{ignored_paragraph}.*(?=\n)'
        regex_last_paragraph = f'(?<=\n).*{ignored_paragraph}.*$'
        job_listing = re.sub(regex_first_paragraph, '', job_listing)
        job_listing = re.sub(regex_middle_paragraphs, '', job_listing)
        job_listing = re.sub(regex_last_paragraph, '', job_listing)

    # Remove consecutive whitespace and newlines
    job_listing = re.sub(' +', ' ', job_listing)
    job_listing = re.sub('\n+', '\n', job_listing)
    job_listing = job_listing.strip()

    # Return the cleaned job listing
    return job_listing

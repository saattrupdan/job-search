'''Automatic sending of emails with newest job listings'''

import yagmail


def send_email(subject: str, content: str):
    '''Send email with subject and content.

    Args:
        subject (str):
            Subject of email
        content (str):
            Content of email
    '''
    # receiver = 'amy.smart1@btinternet.com'
    yag = yagmail.SMTP('amysjobbot')
    yag.send(to='saattrupdan@gmail.com', subject=subject, contents=content)


if __name__ == '__main__':
    content = ('Dear Amy,\n\nThis is a test email. Wowza.\n\n'
               'Best,\n Your favorite job bot')
    send_email(subject='Look at this cool email', content=content)

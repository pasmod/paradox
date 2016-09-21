import logging
import httplib2

h = httplib2.Http(".cache")


def fetch(url):
    """Downloads a URL.
    Args:
        url: The URL.
    Returns:
        The HTML at the URL or None if the request failed.
    """
    if not url:
        return None
    try:
        (_, content) = h.request(url, "GET")
        return content
    except:
        logging.debug('Fetching url failed: {}'.format(url))
    return None

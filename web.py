import requests
import time


class RateLimitedRequest(object):
    def __init__(self, limit_per_second=1):
        self.limit_per_second = limit_per_second
        self.last_request = 0

    def __enter__(self):
        wait_time = self.limit_per_second - (time.time() - self.last_request)
        if wait_time > 0:
            time.sleep(wait_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_request = time.time()


def safe_get(url, timeout=10, max_retries=4, **kwargs):
    success = False
    retries = 0
    response = None
    while not success and retries < max_retries:
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            success = True
        except requests.exceptions.Timeout:
            # LOGGER.debug(f"GET '{url}' failed, retrying...")
            retries = retries + 1
    return response
import os
import pprint

from web import safe_get, RateLimitedRequest

pp = pprint.PrettyPrinter(indent=4)


def t_request(url, json=None):
    r = safe_get(url, timeout=5, max_retries=4, json=json)
    res = r.json()
    if not res['ok']:
        if 'description' in res:
            raise Exception(f"Get '{r.url}' API failure: '{res['description']}'")
    return res


class TBot:
    def __init__(self, botname):
        self.apikey = os.getenv(f"{botname}_API_KEY", None)
        self.limiter = RateLimitedRequest(2)

    def acquire_last_user_id(self):
        url = f"https://api.telegram.org/bot{self.apikey}/getUpdates"
        data = t_request(url)
        try:
            last_update = data['result'][-1]
        except IndexError:
            return None

        if 'message' not in last_update:
            pp.pprint(f"No 'message' in last update: {last_update}")
            return None
        if 'chat' not in last_update['message']:
            pp.pprint(f"No 'chat' in last update['message']: {last_update}")
            return None

        return last_update['message']['chat']['id']

    def send_msg(self, text, userid, **kwargs):
        url = f"https://api.telegram.org/bot{self.apikey}/sendMessage"
        payload = {
            'chat_id': userid,
            'text': text
        }
        payload.update(kwargs)
        with self.limiter:
            res = t_request(url, json=payload)


def escape_message(msg):
    chars = set(msg)
    for char in chars:
        if 1 <= ord(char) <= 126:
            msg = msg.replace(char, '\\' + char)
    return msg
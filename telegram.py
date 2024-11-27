import os
import logging
import pprint
import traceback
import threading
from queue import Queue, Empty

from web import safe_get, RateLimitedRequest

pp = pprint.PrettyPrinter(indent=4)


telegram_user_id = os.getenv(f"MY_TELEGRAM_USER_ID", None)


class TelegramHandler(logging.Handler):

    def __init__(self )-> None:
        logging.Handler.__init__(self=self)
        self.bot = TBot("GenericLogger")
        self.log_queue = Queue()  # Thread-safe queue for log messages
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()

    def emit(self, record) -> None:
        try:
            # Format the record and put it in the queue
            text = self.format(record)
            self.log_queue.put(text)
        except Exception:
            self.handleError(record)

    def _process_logs(self):
        while True:
            try:
                # Retrieve a log message from the queue
                text = self.log_queue.get()
                if text is None:  # Exit signal
                    break
                self.bot.send_msg(text, telegram_user_id)
            except Exception as e:
                # Handle exceptions in the background thread
                print(f"Error sending log message: {e}")

    def close(self):
        # Signal the worker thread to exit and wait for it to finish
        self.log_queue.put(None)
        self.worker_thread.join()
        super().close()


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


if __name__ == "__main__":
    bot = TBot("TELEGRAM")
    print(bot.acquire_last_user_id())
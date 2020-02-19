import json

import requests


class ITest:

    @staticmethod
    def local_server_up(server_url):
        response = requests.get(server_url)
        return response.status_code == 200

    @staticmethod
    def do_request(server_url, action, source, file='', model='', mode='', data='', domain='', lang='', dictionary='',
                   force=0, bucket=''):
        body = {
            "action": action,
            "source": source,
            "file": file,
            "data": data,
            "domain": domain,
            "model": model,
            "mode": mode,
            "dictionary": dictionary,
            "lang": lang,
            "force": force,
            "bucket": bucket
        }
        payload = json.dumps(body)
        header = {"Content-Type": "application/json"}
        response = requests.post(server_url, data=payload, headers=header)
        if response.status_code != 200:
            return False

        return response

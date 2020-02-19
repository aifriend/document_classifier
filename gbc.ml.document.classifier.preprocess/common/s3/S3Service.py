import base64
import io
import os
import warnings

import boto3

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


class S3File:
    def __init__(self, s3Element):
        self.Key = s3Element['Key']
        self.Size = s3Element['Size']


class S3Service(object):
    TIMEOUT = 99999

    def __init__(self, bucket, domain=""):
        self.bucket = bucket
        self.domain = domain
        self.s3Session = self.getS3Session()

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3

    def get_files_from_s3(self, altDomain=None):
        documents = []
        doma = self.domain
        if altDomain:
            doma = altDomain
        for doc in self.__get_all_s3_objects(Bucket=self.bucket, Prefix=doma):
            documents.append(S3File(doc))
        return documents

    def __get_all_s3_objects(self, **base_kwargs):
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = self.s3Session.list_objects_v2(**list_kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')

    def get_txt_file(self, key):
        bytes_buffer = io.BytesIO()

        self.s3Session.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=bytes_buffer)
        dataDecoded = bytes_buffer.getvalue()
        return base64.b64decode(dataDecoded).decode('utf-8')

    def get_byte_file(self, key):
        bytes_buffer = io.BytesIO()

        self.s3Session.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=bytes_buffer)
        dataDecoded = bytes_buffer.getvalue()
        return dataDecoded

    def upload_file(self, key, content):
        response = self.s3Session.put_object(
            Bucket=self.bucket,
            Key=str(key),
            Body=content
        )
        return response

    @staticmethod
    def s3_check_by_extension(s3_elements, extension):
        extension = extension.upper()
        for obj in s3_elements:
            if obj.Size > 0 and obj.Key.upper().endswith(f'.{extension}'):
                return True
        return False

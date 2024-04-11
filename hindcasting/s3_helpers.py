import boto3
import json
import pandas as pd
import io
import os
from constants import ACCESS_KEY, SECRET_KEY, S3_BUCKET_NAME

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)


def load_data_from_s3(file_path):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Failed to load data from S3: {e}")
        return pd.DataFrame()


def file_exists(bucket, key):
    """Check if file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def get_completed_dates(user_id):
    """Get completed dates for a user."""
    file_key = f'users/{user_id}_completed_dates'
    if file_exists(S3_BUCKET_NAME, file_key):
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        dates = json.loads(obj['Body'].read())
        return dates
    else:
        return []


def write_completed_dates(user_id, dates):
    """Write completed dates for a user."""
    file_key = f'users/{user_id}_completed_dates'
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_key, Body=json.dumps(dates))


def get_s3_object(filename, format='json',default={}):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        if format == 'json':
            data = json.loads(response['Body'].read())
        elif format == 'utf-8':
            data = response['Body'].read().decode('utf-8')
    except s3_client.exceptions.NoSuchKey:
        data = default
    return data


def put_s3_object(data, filename):
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=filename, Body=data)

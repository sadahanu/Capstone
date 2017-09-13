#!/usr/bin/python
import boto3
import requests
import pandas as pd

def download_review_pics(df):
    N = df.shape[0]
    for i in xrange(N):
        if i%100 == 1:
            print "processed till {} images".format(i+1)
        pic_url = df.iloc[i]['pic_url']
        pic_name = df.iloc[i]['pic_name']
        r = requests.get(test_link)
        if r.status_code == 200:
            img_data = r.content
            image_name = "reviews/review_pics/"+pic_name
            s3.put_object(Bucket="dogfaces", Key=image_name, Body=img_data);
        else:
            print "images {} could not download".format(pic_name)

if __name__=='__main__':
    df = pd.read_csv("s3://dogfaces/reviews/pictures_log.csv")
    download_review_pics(df)

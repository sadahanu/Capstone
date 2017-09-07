#!/usr/bin/python
import pandas as pd
import boto3
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import re

DF_CAT = pd.read_csv("s3://dogfaces/reviews/category.csv")

def get_cat_link(cat_id):
    '''
    get information regarding toy category on chewy
    '''
    try:
        df = DF_CAT
    except NameError:
        df = pd.read_csv("s3://dogfaces/reviews/category.csv")
    link = df[df['cat_id']==cat_id]['link'].values[0]
    page_range = df[df['cat_id']==cat_id]['page range'].values[0]
    return link, page_range

def get_toys(cat_id):
    '''
    from each category, get toys information
    '''
    link, page_range = get_cat_link(cat_id)
    res = []
    for i in xrange(page_range):
        toys_url = link+'&page={}'.format(i+1)
        r = requests.get(toys_url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content,"lxml")
            for item in soup.select("article.product-holder.cw-card.cw-card-hover"):
                try:
                    num_reviews = int(item.select('p.rating.item-rating')[0].find('span').get_text())
                except:
                    continue
                if num_reviews > 10:
                    toy = {}
                    toy['num_reviews'] = num_reviews
                    raw_id = item.select('a')[0]['href']
                    toy['toy_link'] = "https://www.chewy.com"+item.select('a')[0]['href']
                    toy['toy_id'] = raw_id.split('/')[-1]
                    toy['toy_name'] = raw_id.split('/')[1]
                    toy['picture_link'] = "https:" + item.select('img')[0]['src']
                    toy['price'] = item.select('p.price')[0].get_text().split()[0]
                    toy['cat_id'] = cat_id
                    res.append(toy)
    return res

def get_all_toys():
    try:
        df = DF_CAT
    except NameError:
        df = pd.read_csv("s3://dogfaces/reviews/category.csv")
    toy_res = []
    for cat_id in df['cat_id'].values:
        cat_record = get_toys(cat_id)
        print "getting {} records for category{}".format(len(cat_record),cat_id)
        toy_res.extend(cat_record)
    res_df = pd.DataFrame.from_dict(toy_res)
    res_df.drop_duplicates(subset='toy_link',inplace=True)
    df_data = res_df.to_csv(index=False)
    s3_res = boto3.resource('s3')
    s3_res.Bucket('dogfaces').put_object(Key='reviews/toys.csv', Body=df_data)

if __name__ == '__main__':
    get_all_toys()

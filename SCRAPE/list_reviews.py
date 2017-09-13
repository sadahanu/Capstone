#!/usr/bin/python
import pandas as pd
import boto3
import requests
from bs4 import BeautifulSoup
import re
import time
import sys

DF_CAT = pd.read_csv("s3://dogfaces/reviews/toys.csv")

def get_df(cat_id):
    try:
        df = DF_CAT
    except NameError:
        df = pd.read_csv("s3://dogfaces/reviews/toys.csv")
    return df[df['cat_id']==cat_id]

def get_review_content(toy_row):
    toy_id = toy_row['toy_id']
    num_reviews = toy_row['num_reviews']
    toy_link = toy_row['toy_link']
    toy_name = toy_row['toy_name']
    total_pages = num_reviews/10+1
    base_url = toy_link.replace('/dp/','/product-reviews/') +'?'+'reviewSort=NEWEST&reviewFilter=ALL_STARS&pageNumber='
    count = 1
    review_per_item = []
    pic_per_item = []
    for i in xrange(total_pages):
        if count > num_reviews:
            break
        review_url = base_url+'{}'.format(i+1)
        r = requests.get(review_url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'lxml')
            for review in soup.select("li.js-content"):
                review_record = {}
                picture_record = {}
                review_record['review_id'] = review['data-content-id']
                rating_raw = review.select_one("span.ugc-list__list__stars").select_one("source")['srcset']
                review_record['rating'] = int(re.findall('rating-(\S*)\.svg',rating_raw)[0].split('_')[0])
                try:
                    review_record['user_name'] = review.find_all("span",{"itemprop":"author"})[0].get_text()
                except:
                    review_record['user_name'] = "chewy..."
                review_record['review_content'] = review.select_one("span.ugc-list__review__display").get_text()
                review_record['toy_id'] = toy_id
                review_record['toy_name'] = toy_name
                review_record['review_time'] = review.find_all("span",{"itemprop":"datePublished"})[0].get_text()
                review_per_item.append(review_record)
                count += 1
                pic = review.select_one('a.js-open-modal.js-swap')
                if pic:
                    picture_record['review_id'] = review_record['review_id']
                    picture_record['toy_id'] = toy_id
                    picture_record['toy_name'] = toy_name
                    picture_record['star_rating'] = review_record['rating']
                    pic_link = pic["data-image"]
                    picture_record['pic_url'] = pic_link
                    pic_items = pic_link.split("/")
                    picture_record['pic_id'] =  "_".join(pic_items[-3:-1])
                    picture_record['pic_name'] = "_".join(pic_items[-3:])
                    pic_per_item.append(picture_record)
    return review_per_item, pic_per_item

def store_res(lst, file_name):
    res_df = pd.DataFrame.from_dict(lst)
    df_data = res_df.to_csv(index=False, encoding='utf-8')
    s3_res = boto3.resource('s3')
    s3_res.Bucket('dogfaces').put_object(Key='reviews/'+file_name, Body=df_data)

def fetch_and_store_reviews(df_toys, save_code):
    N = df_toys.shape[0]
    chunks = 20000
    review_store = []
    pic_store = []
    for i in xrange(N):
        review_per_item, pic_per_item = get_review_content(df_toys.iloc[i])
        review_store.extend(review_per_item)
        pic_store.extend(pic_per_item)
        if len(review_store) > chunks:
            save_target = "reviews-"+save_code+"-"+str(i+1)+".csv"
            print "save reviews till record "+save_target
            store_res(review_store, save_target)
            review_store = []
        if len(pic_store) > chunks:
            save_target = "pictures-"+save_code+"-"+str(i+1)+".csv"
            print "save pictures till record "+save_target
            store_res(pic_store, save_target)
            pic_store = []
    save_reviews_target = "reviews-"+save_code+"-final"+".csv"
    save_pictures_target = "pictures-"+save_code+"-final"+".csv"
    store_res(review_store, save_reviews_target)
    store_res(pic_store, save_pictures_target)
    print "fetched and stored all records"

if __name__ == '__main__':
    try:
        cat_id = int(sys.argv[1])
        if cat_id not in range(1,6):
            print "category number not valid, stop"
        else:
            save_code = str(int(time.time()))+"_cat"+str(cat_id)
            df = get_df(cat_id)
            fetch_and_store_reviews(df, save_code)
    except:
        print "please input valid category id"

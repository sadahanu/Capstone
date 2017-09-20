import pandas as pd
import os.path
import requests

def get_toys_info(df):
    '''
     input: toy datarams as ['toy_name','price', 'toy_link', 'picture_link']
     return:
    '''
    img_path = "./static/toy_icons/"
    list_dic = df.to_dict(orient='record')
    for toy in list_dic:
        img_name = toy['picture_link'].split('/')[-1]
        icon_path = img_path + img_name
        print icon_path
        if not os.path.exists(icon_path):
            img_data = requests.get(toy['picture_link']).content
            with open(icon_path, 'wb') as f:
                f.write(img_data)
        toy['img'] = "."+icon_path
    return list_dic

def get_random_dogToys(n=5):
    '''
    get random toys from S3
    '''
    toy_df = pd.read_csv("s3://dogfaces/reviews/toys.csv")
    sample_toys_df = toy_df.sample(n).copy()
    sample_toys = sample_toys_df.to_dict(orient='record')
    img_path = "./static/toy_icons/"
    for toy in sample_toys:
        img_name = toy['picture_link'].split('/')[-1]
        icon_path = img_path + img_name
        print icon_path
        if not os.path.exists(icon_path):
            img_data = requests.get(toy['picture_link']).content
            with open(icon_path, 'wb') as f:
                f.write(img_data)
        toy['img'] = "."+icon_path
    return sample_toys

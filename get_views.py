import pandas as pd
import json
import requests
from tqdm import tqdm

DEVELOPER_KEY = "AIzaSyDYXyxiBB7RAp4GtYk5Pa4PmIrQ_QwxZzs"

def youtube_search(id):
    payload = {'id': id, 'part': 'statistics', 'key': DEVELOPER_KEY}
    l = requests.Session().get('https://www.googleapis.com/youtube/v3/videos', params=payload)    
    resp_dict = json.loads(l.content)
    try:
        return resp_dict['items'][0]['statistics']['viewCount']
    except Exception as e:
        print(id)
        return 0

df = pd.read_hdf('./flask_app_df.h5')
ids = df.youtube_id
views = [youtube_search(x) for x in tqdm(ids)]
df['views'] = views
df.to_hdf('./flask_app_df2.h5', 'df')

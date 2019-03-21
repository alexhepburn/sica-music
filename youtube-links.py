import pandas as pd 
import numpy as np 
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import time
import json
from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
from tqdm import tqdm


with open('./youtube_keys.txt', 'r') as f:
    keys = f.read().splitlines()
DEVELOPER_KEY = keys[0]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

def youtube_search(q):
  search_response = youtube.search().list(q=q, type="video", order = "relevance", 
    part="id", maxResults=2).execute()
  videos = []
  for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
      videos.append(search_result)
  try:
      nexttok = search_response["nextPageToken"]
      return(nexttok, videos)
  except Exception as e:
      nexttok = "last_page"
      return(nexttok, videos)

def find_youtube(artist, song, f):
    search = (artist + ' ' + song)
    #print(search)
    res = youtube_search(search)
    try:
        f.write(search + ': ' + "http://www.youtube.com/watch?v=" + res[1][0]['id']['videoId'] + '\n')
        return("http://www.youtube.com/watch?v=" + res[1][0]['id']['videoId'])
    except Exception as e:
        print(e)
        return('None')
    

df = pd.read_hdf('/Users/ah13558/Documents/PhD/interesting-svd/data/df_magd_with_chroma_partition.h5', 'df')
df = df.drop_duplicates('song')
#df.at[1427956, 'album'] = 'Orange Trägt Nur Die Müllabfuhr' # get rid of some invalid characters
#df.at[2490446, 'album'] = 'Vegas Collie'
#df.at[2500513, 'album'] = 'Swing, Swing'

ids = list(df['song'])
songs = list(df['song_title'])
art = list(df['artist_name'])
links = []
f = open('links.txt', 'r')

for line in f:
    links.append(line.split(': ')[-1][:-1])
f.close()
f = open('links.txt', 'a', 100)

while len(links) != len(art):
    links.append('None')
for l in tqdm(range(0, len(links))):
    if links[l] == 'None':
        links[l] = find_youtube(art[l], songs[l], f)

#df2 = pd.DataFrame({'song':ids, 'album':songs, 'artist':art, 'youtube':links})
#df['youtube'] = links
df.to_hdf('/Users/ah13558/Documents/PhD/interesting-svd/data/SICA-songs.h5', 'df')

#f = open('links.txt', 'a')
#df2['youtube'] = df2.apply(lambda x: find_youtube(x.artist, x.album, f) if  x.youtube == 'None' else x.youtube, axis=1)
#f.close()
#print(df2.head())
#df2.to_hdf('./songs-youtube.h5', 'df')

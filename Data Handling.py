# Installing dependencies
import pathlib
import os
import pandas as pd
import json

from google.colab import drive
drive.mount('/content/drive')

#accessing match data
path = '/content/drive/MyDrive/english_football_data/matches_England.json'
with open(path) as f:
    data = json.load(f)
df_matches = pd.DataFrame(data)

#accessing player data
path = '/content/drive/MyDrive/english_football_data/players.json'
with open(path) as f:
    data = json.load(f)
df_players = pd.DataFrame(data)

#accessing event data
path = '/content/drive/MyDrive/english_football_data/events_England.json'
with open(path) as f:
    data = json.load(f)
df_events = pd.DataFrame(data)

# splitting positions
df_events['positions/0/x'] = df_events['positions'].apply(lambda x: x[0]['x'] if len(x) > 0 else None)
df_events['positions/0/y'] = df_events['positions'].apply(lambda x: x[0]['y'] if len(x) > 0 else None)
df_events['positions/1/x'] = df_events['positions'].apply(lambda x: x[1]['x'] if len(x) > 1 else None)
df_events['positions/1/y'] = df_events['positions'].apply(lambda x: x[1]['y'] if len(x) > 1 else None)

# splitting tags
tags_list = df_events['tags'].apply(lambda x: [t['id'] for t in x] if isinstance(x, list) else [])
tag_columns = pd.DataFrame(tags_list.tolist(), index=df_events.index).add_prefix('tag_')

df_events = pd.concat([df_events, tag_columns], axis=1)
df_events = df_events.drop(columns=['positions', 'tags'])


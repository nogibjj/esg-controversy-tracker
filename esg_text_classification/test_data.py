#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax
df = pd.read_csv('/workspaces/esg-controversy-tracker/esg_text_classification/bert-esg-outputs/bert_esg_tagged_articles_12_16_2022_01_18_09.csv')
df['content_length'] = [len(x) if x==x else 0 for x in df['content']]#[len(x) for x in df['content'] if x == x]
df = df[df['content_length'] > 4500]
all_columns = df.columns
non_esg_cols = ['id', 'ticker', 'title', 'category', 'content', 'release_date', 'provider', 'url', 'article_id', 'content_length']
esg_cols = list(set(all_columns) - set(non_esg_cols))
df[esg_cols] = softmax(np.asarray(df[esg_cols]))
df_describe = pd.DataFrame(df[esg_cols].quantile([0.0, .9]))
print(df_describe)
print(df.shape)

#%%
retain_rows = set()
for col in esg_cols:
    avg = df_describe[col][0.9]
    temp = df[df[col] > avg]
    retain_rows.update(temp.index)
    temp.to_csv(f"filtered_outputs/{col}.csv")
#%%
import pandas as pd
import numpy as np
from scipy.special import softmax
df = pd.read_csv('/workspaces/esg-controversy-tracker/esg_text_classification/bert-esg-outputs/bert_esg_tagged_articles_12_16_2022_01_18_09.csv')
df['content_length'] = [len(x) if x==x else 0 for x in df['content']]#[len(x) for x in df['content'] if x == x]
df = df[df['content_length'] > 4500]
all_columns = df.columns
non_esg_cols = ['id', 'ticker', 'title', 'category', 'content', 'release_date', 'provider', 'url', 'article_id', 'content_length']
esg_cols = list(set(all_columns) - set(non_esg_cols))
df[esg_cols] = softmax(np.asarray(df[esg_cols]))

lower_limit = 0.95
# df_describe = pd.DataFrame(df[esg_cols].quantile([0.0, lower_limit]))
df_describe = df[esg_cols].stack().quantile([0.0, lower_limit])


for col in esg_cols:
    #threshold = df_describe[col][lower_limit]
    threshold = df_describe[lower_limit]
    temp = df[df[col] > threshold]
    print(col, temp.shape)
    temp.to_csv(f"filtered_outputs/{col}.csv")
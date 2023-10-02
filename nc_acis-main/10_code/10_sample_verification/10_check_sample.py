# from dask.distributed import Client

# client = Client()
# client

# import dask.dataframe as dd


import pandas as pd
import numpy as np

pd.set_option("mode.copy_on_write", True)

# Make sure attorneys are not anonymized
df = pd.read_csv(
    "../../00_raw_data/sean_newsample_2023/Eubanks_sample/sample_2013_2022_cr_attorneys.csv"
)

df.head(10)
df.sample().T

# Make sure defendants are anonymized bit ages included

df = pd.read_csv("../../00_raw_data/2013_2022_cr_cases.csv", nrows=100_000)
df.CRRDOB.value_counts(dropna=False)
pd.set_option("display.max_rows", 115)
df.sample().T
df.age.value_counts()

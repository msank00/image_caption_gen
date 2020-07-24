import random

import pandas as pd


def get_sample_data_from_clean(df: pd.DataFrame):

    grouped = df.groupby(["IMAGE_ID"])
    image_id = random.sample(list(grouped.indices), 1)

    df_list = map(lambda df_i: grouped.get_group(df_i), image_id)

    sampled_df = pd.concat(df_list, axis=0, join="outer")
    all_captions = sampled_df.CAPTION.values.tolist()

    caption = " <SEP> ".join(all_captions)

    return image_id[0], caption


def get_sample_data_from_clean_flat(df: pd.DataFrame):
    df_sample = df.sample(n=1)

    image_id = df_sample.iloc[0]["IMAGE_ID"]
    caption = df_sample.iloc[0]["CAPTION"]

    return image_id, caption

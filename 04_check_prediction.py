import random

import pandas as pd
import streamlit as st
from PIL import Image

from src.data_validation import (
    get_sample_data_from_clean,
    get_sample_data_from_clean_flat,
)
from src.utils import Config


def pick_random_sample(df: pd.DataFrame):
    df_sample = df.sample(n=1)

    image_id = df_sample.iloc[0]["IMAGE_ID"]
    true_caption = df_sample.iloc[0]["TRUE_CAPTION"]
    pred_caption = df_sample.iloc[0]["PRED_CAPTION"]

    return image_id, true_caption, pred_caption


def init_streamlit():

    st.title("Image Caption Generation Task")
    st.header("Data Validation")
    st.subheader("Showing random sample from data")


if __name__ == "__main__":

    config = Config("config.yaml")
    # config.set_attribute()

    # filename = f"{config.DATA_DIR}/flickr_caption_data_processed/Flickr8k_token_processed.csv"
    filename = "model/predictions_20200809.csv"
    df = pd.read_csv(filename, sep=",")

    # get_sample_data_from_clean_flat(df_flat)

    init_streamlit()

    n = st.slider("How many images want to see", 0, 10)

    for i in range(int(n)):
        image_name, true_caption, pred_caption = pick_random_sample(df)
        image_file = f"{config.IMAGE_DATA_DIR}{image_name}"

        image = Image.open(image_file)

        st.markdown(f":rocket: **True Caption:** `{true_caption}`")
        # st.text_area(label="", value=true_caption)
        st.markdown(f":dart: **Predicted Caption:** `{pred_caption}`")
        # st.text_area(label="", value=pred_caption)
        # st.markdown("hi friend...")
        st.markdown(f"### :camera: {image_name}")

        st.image(image, use_column_width=True)

        st.markdown("----")

    st.markdown("## :santa: FINISH")

    print("DONE")

import random

import pandas as pd
import streamlit as st
from PIL import Image

from src.data_validation import (
    get_sample_data_from_clean,
    get_sample_data_from_clean_flat,
)
from src.utils import Config


def init_streamlit():

    st.title("Image Caption Generation Task")
    st.header("Data Validation")
    st.subheader("Showing random sample from data")


if __name__ == "__main__":

    config = Config("config.yaml")
    # config.set_attribute()

    # filename = f"{config.DATA_DIR}/flickr_caption_data_processed/Flickr8k_token_processed.csv"
    filename = f"{config.CAPTION_FILE}"
    df = pd.read_csv(filename, sep="\t")

    # get_sample_data_from_clean_flat(df_flat)

    init_streamlit()

    n = st.slider("How many images want to see", 0, 10)

    for i in range(int(n)):
        image_name, caption = get_sample_data_from_clean_flat(df)
        image_file = f"{config.IMAGE_DATA_DIR}{image_name}"

        image = Image.open(image_file)

        st.markdown("## :rocket: Caption")
        st.text_area(label="", value=caption)
        st.markdown(f"## :camera: {image_name}")

        st.image(image, use_column_width=True)

        st.markdown("----")

    st.markdown("## :santa: FINISH")

    print("DONE")

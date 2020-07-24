import os

import streamlit as st

from src.process_raw_data import prepare_training_data, process_data
from src.utils import Config

if __name__ == "__main__":

    config = Config("config.yaml")
    print(config)
    # config.set_attribute()

    print("Converting raw data into clean data...")
    df = process_data(f"{config.DATA_DIR}Flickr8k.token.txt",dev_mode=config.DEV_MODE)

    print("Converting clean data into flat clean data for training...")
    df_train_test = prepare_training_data(df, dev_mode=config.DEV_MODE)

    print("========df clean==========")
    print(df.head())
    print(f"df clean data shape: {df.shape}")

    print("========df clean flat==========")
    print(df_train_test.head())
    print(f"df train-test data shape: {df_train_test.shape}")

    df.to_csv(
        f"{config.CAPTION_FILE}",
        sep="\t",
        index=False,
    )

    df_train_test.to_csv(
        f"{config.DATA_DIR}Flickr8k_token_processed_flat.csv",
        sep="\t",
        index=False,
    )

    print(f"\nDONE, ran in dev_mode: {config.DEV_MODE}")

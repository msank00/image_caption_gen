import json
import os

if __name__ == "__main__":

    assert (
        os.environ.get("KAGGLE_USER_NAME") != None
    ), "Please export KAGGLE_USER_NAME"
    assert (
        os.environ.get("KAGGLE_API_KEY") != None
    ), "Please export KAGGLE_API_KEY"

    KAGGLE_USER_NAME = os.environ.get("KAGGLE_USER_NAME")
    KAGGLE_API_KEY = os.environ.get("KAGGLE_API_KEY")

    kaggle_credentials = {"username": KAGGLE_USER_NAME, "key": KAGGLE_API_KEY}

    with open("kaggle.json", "w") as fp:
        json.dump(kaggle_credentials, fp)

    print("Success...")

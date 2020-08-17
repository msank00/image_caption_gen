import json
import os

if __name__ == "__main__":
    
    assert (
        os.environ.get("KAGGLE_DATASET_ID") != None
    ), "Please export KAGGLE_DATASET_ID"
    

    KAGGLE_DATASET_ID = os.environ.get("KAGGLE_DATASET_ID")
    LICENSE_NAME = "CC0-1.0"
    TITLE = "Image Caption"
    DESCRIPTION = "To host trained models and other outputs"
    

    dataset_metadata = {
        "licenses": [
            {"name": LICENSE_NAME}
            ],
        "id": KAGGLE_DATASET_ID,
        "title": TITLE,
        "description": DESCRIPTION
    }

    with open("dataset-metadata.json", "w") as fp:
        json.dump(dataset_metadata, fp)

    print("Success...")
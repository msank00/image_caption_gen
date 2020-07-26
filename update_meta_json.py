import json
import os
import pprint

KAGGLE_DATASET_ID = "sankarshan7/image-caption"

if __name__ == "__main__":

    filename = "model/dataset-metadata.json"

    try:
        with open(filename, "r") as fin:
            data = json.load(fin)
    except FileNotFoundError as identifier:
            print(f"{filename} not found !!")
            exit(1)


    tmp = data["id"]
    data["id"] = KAGGLE_DATASET_ID
    print(f"old id: {tmp}")
    print(f"new id: {data['id']}")

    data["title"] = "Tween Sentiment Analysis"
    data["description"] = "To host trained models and other outputs"

    with open(filename, "w") as jsonFile:
        json.dump(data, jsonFile)

    pprint.pprint(data)
    print(f"{filename} update successful !!")
    
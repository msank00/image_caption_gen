import json
import os
import pprint

KAGGLE_DATASET_ID = os.environ.get("KAGGLE_DATASET_ID")
assert KAGGLE_DATASET_ID != None, "Please export KAGGLE_DATASET_ID"

def update_kaggle_metadata(filename: str, 
                           dataset_id: str, 
                           title: str, 
                           description: str):

    try:
        with open(filename, "r") as fin:
            data = json.load(fin)
    except FileNotFoundError as identifier:
            print(f"{filename} not found !!")
            exit(1)

    tmp = data["id"]
    data["id"] = dataset_id
    print(f"old id: {tmp}")
    print(f"new id: {data['id']}")

    data["title"] = title
    data["description"] = description

    # overwrite old information with new information
    with open(filename, "w") as jsonFile:
        json.dump(data, jsonFile)
        
    pprint.pprint(data)


if __name__ == "__main__":

    filename = "model/dataset-metadata.json"
    update_kaggle_metadata(filename, 
                           KAGGLE_DATASET_ID, 
                           title="Image Caption Generation", 
                           description="To host trained models and other outputs")
    
    print(f"{filename} update successful !!")
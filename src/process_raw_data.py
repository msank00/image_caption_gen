import pandas as pd
from tqdm import tqdm


def process_data(filename: str, dev_mode=False):

    caption_ids = []
    captions = []
    image_ids = []
    caption_nos = []

    with open(filename, "r") as fin:
        all_lines = fin.readlines()
        for i, line in tqdm(enumerate(all_lines), total=len(all_lines)):
            w = line.split("\t")
            cid = w[0].strip()
            t = cid.split("#")

            image_id = t[0].strip()
            caption_no = t[1].strip()

            caption = w[1].strip()

            caption_ids.append(cid)
            captions.append(caption)

            image_ids.append(image_id)
            caption_nos.append(caption_no)

            if dev_mode and i == 10:
                print("Running in dev mode...")
                break

    df = pd.DataFrame(
        {
            "CAPTION_ID": caption_ids,
            "IMAGE_ID": image_ids,
            "CAPTION_NO": caption_nos,
            "CAPTION": captions,
        }
    )

    return df


def prepare_training_data(df: pd.DataFrame, dev_mode: bool = False):

    """Create training dataset where each row contains 
       image id and all the captions concatenated in a single string
       and thus create a flat version of the input dataframe.

    """

    grouped = df.groupby(["IMAGE_ID"])
    image_ids = []
    captions = []

    for i, (image_id, group) in tqdm(enumerate(grouped), total=len(grouped)):

        all_captions = group.CAPTION.values.tolist()

        caption = " ".join(all_captions)
        image_ids.append(image_id)
        captions.append(caption)

        if dev_mode and i == 10:
            print("Running in dev mode...")
            break

    df = pd.DataFrame({"IMAGE_ID": image_ids, "CAPTION": captions,})

    return df

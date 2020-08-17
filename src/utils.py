import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
import time
from torchvision import transforms
from src.model import DecoderRNN, DecoderRNNUpdated, EncoderCNN
from typing import List
from PIL import Image

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

def parse_config_file(config_file: str):

    with open(config_file) as f:
        config = yaml.load(f)  # , Loader=yaml.FullLoader)

    return config


def seed_everything(seed: int = 42):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

# all timezone neame: https://bit.ly/2MDvGuT
def set_timezone(tz_name:str='Asia/Calcutta'):
    os.environ['TZ'] = tz_name 
    time.tzset()
    print(f"Time-zone: {time.tzname}")

def tag_date():
    # https://docs.python.org/2/library/time.html
    return time.strftime("%Y%m%d")

def tag_date_time():
    # https://docs.python.org/2/library/time.html
    return time.strftime("%Y%m%d%H%M")

def sentence_similarity(sentence1: str, sentence2: str):
    
    # https://www.geeksforgeeks.org/python-measure-similarity-between-two-sentences-using-cosine-similarity/
    
    X = sentence1
    Y = sentence2
    
    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
    
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return np.round(cosine, 4)

def pick_random_test_image(df: pd.DataFrame):
    idx = np.random.randint(low=0, high=len(df))
    image_id = df.iloc[idx]["IMAGE_ID"]
    caption = df.iloc[idx]["CAPTION"]
    return image_id, caption


def copy_file_to_correct_folder(image_id: str):
    file_src_path = f"{config.IMAGE_DATA_DIR}{image_id}"
    file_destination_path = f"asset/test_image/{image_id}"
    shutil.copy(file_src_path, file_destination_path)


def predict_image_caption(
    image_file: str,
    transform_image: transforms,
    model_encoder: EncoderCNN,
    model_decoder: DecoderRNN,
    test_data_loader,
    device,
):

    assert os.path.exists(
        image_file
    ), f"Image file: '{image_file}' doesn't not exist."
    PIL_image = Image.open(image_file).convert("RGB")
    transformed_image = transform_image(PIL_image)
    transformed_image = transformed_image.to(device)
    transformed_image = transformed_image.unsqueeze(
        dim=0
    )  # convert size [3, 224, 224] -> [1, 3, 224, 224]
    features = model_encoder(transformed_image).unsqueeze(1)
    output = model_decoder.predict_token_ids(features)
    sentence = process_predicted_tokens(output, test_data_loader)

    return sentence

def find_bleu_score(hypothesis: str, reference: List[str]):
    hypothesis = hypothesis.split(" ")
    reference = [sent.split(" ") for sent in reference]
    if len(hypothesis) >= 2:
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights = (0.5, 0.5))
    else:
        BLEUscore = -1.0
        
    return np.round(BLEUscore, 4)

def process_predicted_tokens(output: list, test_data_loader):
    """Map list of token ids to list of corresponding words/tokens
       using the vocabulary dictionary idx2word. 

    :param output: list of predicted token ids
    :type output: list
    :return: list of tokens
    :rtype: list
    """
    words_sequence = []

    for i in output:
        if i == 1:
            continue
        token = test_data_loader.dataset.vocab.idx2word[i]
        words_sequence.append(token)
        
        if token == ".":
            break

    # words_sequence = words_sequence[1:-1]
    sentence = " ".join(words_sequence)
    # sentence = sentence.capitalize()

    return sentence.capitalize()


class Config:
    def __init__(self, filename: str):
        self.config_file = filename
        self.config = parse_config_file(filename)
        self.DATA_DIR = self.config["config"]["DATA_DIR"]
        self.CAPTION_DATA_DIR = self.config["config"]["CAPTION_DATA_DIR"]
        self.CAPTION_FILE = self.config["config"]["CAPTION_FILE"]
        self.IMAGE_ID_FILE_TRAIN = self.config["config"]["IMAGE_ID_FILE_TRAIN"]
        self.IMAGE_ID_FILE_VAL = self.config["config"]["IMAGE_ID_FILE_VAL"]
        self.IMAGE_ID_FILE_TEST = self.config["config"]["IMAGE_ID_FILE_TEST"]
        self.IMAGE_DATA_DIR = self.config["config"]["IMAGE_DATA_DIR"]
        self.MODEL_DIR = self.config["config"]["MODEL_DIR"]
        self.DEV_MODE = self.config["config"]["DEV_MODE"]
        self.BATCH_SIZE = self.config["config"]["BATCH_SIZE"]
        self.VOCAB_THRESHOLD = self.config["config"]["VOCAB_THRESHOLD"]
        self.IMG_EMBED_SIZE = self.config["config"]["IMG_EMBED_SIZE"]
        self.WORD_EMBED_SIZE = self.config["config"]["WORD_EMBED_SIZE"]
        self.HIDDEN_SIZE = self.config["config"]["HIDDEN_SIZE"]
        self.NUM_EPOCHS = self.config["config"]["NUM_EPOCHS"]
        self.SAVE_EVERY = self.config["config"]["SAVE_EVERY"]
        self.PRINT_EVERY = self.config["config"]["PRINT_EVERY"]
        self.VERBOSE = self.config["config"]["VERBOSE"]
        self.VOCAB_FILE = self.config["config"]["VOCAB_FILE"]
        self.VOCAB_FROM_FILE = self.config["config"]["VOCAB_FROM_FILE"]
        self.DEV_MODE = self.config["config"]["DEV_MODE"]

    def __str__(self):
        return f"{self.config}"


def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_training_data(image_id_file: str, caption_file: str):
    """Returns the training/test/validation data depending on the image_id_file
    and the caption file 

    :param image_id_file: Contains list of image ids 
    :type image_id_file: str
    :param caption_file: Master file contains all the image ids and the captions
    :type caption_file: str
    """

    df_train_ids = pd.read_csv(image_id_file, names=["IMAGE_ID"])
    df_full_data = pd.read_csv(caption_file, sep="\t")
    df_train = pd.merge(
        df_train_ids, df_full_data, how="left", on=["IMAGE_ID"]
    )

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    return df_train


if __name__ == "__main__":
    cfg = Config("config.yaml")
    print(cfg)

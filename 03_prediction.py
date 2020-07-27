from torchvision import transforms
from src.data_loader import get_data_loader
from src.utils import Config, get_training_data

from PIL import Image

import os
import torch
from src.model import EncoderCNN, DecoderRNN

import numpy as np
import pandas as pd 
import shutil



def pick_random_test_image(df: pd.DataFrame):
    idx = np.random.randint(low=0, high=len(df))
    image_id = df.iloc[idx]["IMAGE_ID"]
    caption = df.iloc[idx]["CAPTION"]
    return image_id, caption


def copy_file_to_correct_folder(image_id: str):
    file_src_path = f"{config.IMAGE_DATA_DIR}{image_id}"
    file_destination_path = f"asset/test_image/{image_id}"
    shutil.copy(file_src_path, file_destination_path)



def predict_image_caption(image_file: str, 
                          transform_image: transforms, 
                          model_encoder: EncoderCNN,
                          model_decoder: DecoderRNN, 
                          device):
    
    assert os.path.exists(image_file), f"Image file: '{image_file}' doesn't not exist."
    PIL_image = Image.open(image_file).convert("RGB")
    transformed_image = transform_image(PIL_image)  
    transformed_image = transformed_image.to(device)
    transformed_image = transformed_image.unsqueeze(dim=0) # convert size [3, 224, 224] -> [1, 3, 224, 224]
    features = model_encoder(transformed_image).unsqueeze(1)
    output = model_decoder.predict_token_ids(features)    
    sentence = process_predicted_tokens(output)
    
    return sentence

def process_predicted_tokens(output:list):
    """Map list of token ids to list of corresponding words/tokens
       using the vocabulary dictionary idx2word. 

    :param output: list of predicted token ids
    :type output: list
    :return: list of tokens
    :rtype: list
    """
    words_sequence = []
    
    for i in output:
        if (i == 1):
            continue
        words_sequence.append(test_data_loader.dataset.vocab.idx2word[i])
    
    # words_sequence = words_sequence[1:-1] 
    sentence = ' '.join(words_sequence) 
    # sentence = sentence.capitalize()
    
    return sentence

if __name__ == "__main__":

    transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                        ])

    config = Config("config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data_loader = get_data_loader(transform=transform_test,
                                   caption_file=config.CAPTION_FILE,
                                   image_id_file=config.IMAGE_ID_FILE_TEST, 
                                   image_folder=config.IMAGE_DATA_DIR, 
                                   config=config,
                                   mode='test')

    # TODO #2: Specify the saved models to load.
    encoder_file = f"{config.MODEL_DIR}encoder-1.pkl"
    decoder_file = f"{config.MODEL_DIR}decoder-1.pkl"
    
    assert os.path.exists(encoder_file), f"Encoder model: '{encoder_file}' doesn't not exist."
    assert os.path.exists(decoder_file), f"Decoder model: '{decoder_file}' doesn't not exist."

    # TODO #3: Select appropriate values for the Python variables below.
    embed_size = config.IMG_EMBED_SIZE
    hidden_size = config.HIDDEN_SIZE
    vocab_size = len(test_data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Load the trained weights.
    # map location helps in save and load accross devices (gpu/cpu)
    encoder.load_state_dict(torch.load(encoder_file, map_location=device), strict=False)
    decoder.load_state_dict(torch.load(decoder_file, map_location=device), strict=False)
    encoder.eval()
    decoder.eval()
    
    print("Model loaded...")

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)
    
    df_test = get_training_data(config.IMAGE_ID_FILE_TEST, config.CAPTION_FILE)
    
    image_ids = []
    true_captions = []
    pred_captions = []
    for i in range(10):
        image_id, caption = pick_random_test_image(df_test)
        copy_file_to_correct_folder(image_id)
        image_file = f"asset/test_image/{image_id}"
        pred_caption = predict_image_caption(image_file, 
                                             transform_image=transform_test, 
                                             model_encoder=encoder, 
                                             model_decoder=decoder, 
                                             device=device)
        image_ids.append(image_id)
        true_captions.append(caption)
        pred_captions.append(pred_caption)
        
    df_pred = pd.DataFrame({"IMAGE_ID": image_ids, 
                            "TRUE_CAPTION": true_captions, 
                            "PRED_CAPTION": pred_captions})
    
    df_pred.to_csv("model/predictions.csv", index=False)
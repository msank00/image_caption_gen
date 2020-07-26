from torchvision import transforms
from src.data_loader import get_data_loader
from src.utils import Config

from PIL import Image
import numpy as np

import os
import torch
from src.model import EncoderCNN, DecoderRNN

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
    encoder_file = f"{config.MODEL_DIR}encoder-10.pkl"
    decoder_file = f"{config.MODEL_DIR}decoder-10.pkl"
    
    assert os.path.exists(encoder_file), f"Encoder model: '{encoder_file}' doesn't not exist."
    assert os.path.exists(decoder_file), f"Decoder model: '{decoder_file}' doesn't not exist."

    # TODO #3: Select appropriate values for the Python variables below.
    embed_size = config.IMG_EMBED_SIZE
    hidden_size = config.HIDDEN_SIZE
    vocab_size = len(test_data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights.
    # map location helps in save and load accross devices (gpu/cpu)
    encoder.load_state_dict(torch.load(encoder_file, map_location=device))
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))
    print("Model loaded...")

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)
    
    test_image_file = "asset/test_image/train_track.jpg"
    # test_image_file = "asset/test_image/3234115903_f4dfc8fc75.jpg"
    # test_image_file = "asset/test_image/241347760_d44c8d3a01.jpg"
    pred_caption = predict_image_caption(test_image_file, 
                                         transform_image=transform_test, 
                                         model_encoder=encoder, 
                                         model_decoder=decoder, 
                                         device=device)
    
    print(f"Predicted caption: {pred_caption}")
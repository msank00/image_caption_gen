import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


from src.data_loader import get_data_loader
from src.model import DecoderRNN, DecoderRNNUpdated, EncoderCNN
from src.utils import Config, get_training_data, set_timezone, tag_date_time, sentence_similarity
import nltk
from src.utils import pick_random_test_image, copy_file_to_correct_folder, predict_image_caption, find_bleu_score, process_predicted_tokens

set_timezone()


if __name__ == "__main__":

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    config = Config("config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data_loader = get_data_loader(
        transform=transform_test,
        caption_file=config.CAPTION_FILE,
        image_id_file=config.IMAGE_ID_FILE_TEST,
        image_folder=config.IMAGE_DATA_DIR,
        config=config,
        vocab_file=config.VOCAB_FILE,
        mode="test",
    )

    # TODO #2: Specify the saved models to load.
    print(f"DEV MODE: {config.DEV_MODE}")
    if config.DEV_MODE:
        print("Loading dev model....")
        encoder_file = f"{config.MODEL_DIR}encoder-checkpoint-dev.pt"
        decoder_file = f"{config.MODEL_DIR}decoder-checkpoint-dev.pt"
    else:
        encoder_file = f"{config.MODEL_DIR}encoder-checkpoint.pt"
        decoder_file = f"{config.MODEL_DIR}decoder-checkpoint.pt"

    assert os.path.exists(
        encoder_file
    ), f"Encoder model: '{encoder_file}' doesn't not exist."
    assert os.path.exists(
        decoder_file
    ), f"Decoder model: '{decoder_file}' doesn't not exist."

    # TODO #3: Select appropriate values for the Python variables below.
    embed_size = config.IMG_EMBED_SIZE
    hidden_size = config.HIDDEN_SIZE
    vocab_size = len(test_data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    # decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # DecoderRNNUpdated
    decoder = DecoderRNNUpdated(
        embed_size, hidden_size, vocab_size, device=device
    )

    # Load the trained weights.
    # map location helps in save and load accross devices (gpu/cpu)
    encoder.load_state_dict(
        torch.load(encoder_file, map_location=device), strict=False
    )
    decoder.load_state_dict(
        torch.load(decoder_file, map_location=device), strict=False
    )
    encoder.eval()
    decoder.eval()

    print("Model loaded...")

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    df_test = get_training_data(config.IMAGE_ID_FILE_TEST, config.CAPTION_FILE)

    n = len(df_test)

    image_ids = []
    true_captions = []
    pred_captions = []
    bleu_scores = []
    sent_similarity = []
    
    for i in range(n):
        print(i)
        # image_id, caption = pick_random_test_image(df_test)

        image_id = df_test.iloc[i]["IMAGE_ID"]
        caption = df_test.iloc[i]["CAPTION"]

        copy_file_to_correct_folder(image_id)
        image_file = f"asset/test_image/{image_id}"
        pred_caption = predict_image_caption(
            image_file,
            transform_image=transform_test,
            model_encoder=encoder,
            model_decoder=decoder,
            test_data_loader=test_data_loader,
            device=device,
        )
        image_ids.append(image_id)
        true_captions.append(caption)
        pred_captions.append(pred_caption)
        bleu_scores.append(find_bleu_score(pred_caption, [caption]))    
        sent_similarity.append(sentence_similarity(pred_caption, caption))

    df_pred = pd.DataFrame(
        {
            "IMAGE_ID": image_ids,
            "TRUE_CAPTION": true_captions,
            "PRED_CAPTION": pred_captions,
            "BLEU_SCORE": bleu_scores,
            "COSINE_SIMILARITY": sent_similarity
        }
    )
    df_pred = df_pred.sort_values(by='COSINE_SIMILARITY', ascending=False).reset_index(drop=True)
                   
    df_pred.to_csv(f"model/predictions_{tag_date_time()}.csv", index=False)

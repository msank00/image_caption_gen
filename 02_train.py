import torch
import torch.nn as nn
from torchvision import transforms
from src.data_loader import get_data_loader
from src.model import EncoderCNN, DecoderRNN
from src.loss import get_loss_function
from src.optimizer import get_optimizer
import math
from src.utils import Config
import torch.utils.data as data
import numpy as np
import os
import time
from tqdm import tqdm

if __name__ == "__main__":
    

    config = Config("config.yaml")

    # (Optional) TODO #2: Amend the image transform below.
    transform_train = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))])

    # Build data loader.
    train_data_loader = get_data_loader(transform=transform_train, 
                                caption_file=config.CAPTION_FILE, 
                                image_id_file=config.IMAGE_ID_FILE_TRAIN, 
                                image_folder=config.IMAGE_DATA_DIR, 
                                config=config,
                                mode="train",
                                batch_size=config.BATCH_SIZE,
                                vocab_threshold=config.VOCAB_THRESHOLD, 
                                vocab_file=config.VOCAB_FILE,
                                vocab_from_file=config.VOCAB_FROM_FILE)

    # Build data loader.
    val_data_loader = get_data_loader(transform=transform_train, 
                                caption_file=config.CAPTION_FILE, 
                                image_id_file=config.IMAGE_ID_FILE_VAL, 
                                image_folder=config.IMAGE_DATA_DIR, 
                                config=config,
                                mode="validation",
                                batch_size=config.BATCH_SIZE,
                                vocab_threshold=config.VOCAB_THRESHOLD, 
                                vocab_file=config.VOCAB_FILE,
                                vocab_from_file=True) # validation data should use vocab generated from train


    # The size of the vocabulary.
    vocab_size = len(train_data_loader.dataset.vocab)

    # Initialize the encoder and decoder. 
    encoder = EncoderCNN(config.WORD_EMBED_SIZE)
    decoder = DecoderRNN(config.WORD_EMBED_SIZE, 
                        config.HIDDEN_SIZE, 
                        vocab_size)

    # Move models to GPU if CUDA is available. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    criterion = get_loss_function(loss_type="cross_entropy_loss")

    params = list(decoder.parameters()) + list(encoder.embed.parameters())

    optimizer = get_optimizer(params, optim_type="adam", learning_rate=0.001) 

    # Set the total number of training steps per epoch.
    total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)

    old_time = time.time()

    for epoch in tqdm(range(1, config.NUM_EPOCHS+1)):
        
        tqdm_steps = tqdm(range(1, total_step+1), desc='Bar desc', leave=True)
        
        #for i_step in tqdm(range(1, total_step+1)):
        for i_step in tqdm_steps:
                
            if time.time() - old_time > 60:
                old_time = time.time()
            
            # Randomly sample a caption length, and sample indices with that length.
            indices = train_data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            images, captions = next(iter(train_data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # Backward pass.
            loss.backward()
            
            # Update the parameters in the optimizer.
            optimizer.step()
                
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, config.NUM_EPOCHS, i_step, total_step, loss.item(), np.exp(loss.item()))
            
            tqdm_steps.set_description(f"Bar desc train loss: {np.round(loss.item(),4)}, train perplexityt: {np.round(np.exp(loss.item()),4)}")
            
            # Print training statistics (on same line).
            # print(stats)
            
            # Print training statistics (on different line).
            #if i_step % config.PRINT_EVERY == 0:
            #    print(stats)
                
        # Save the weights.
        if epoch % config.SAVE_EVERY == 0:
            torch.save(decoder.state_dict(), os.path.join(config.MODEL_DIR, f"decoder-{epoch}.pkl"))
            torch.save(encoder.state_dict(), os.path.join(config.MODEL_DIR, f"encoder-{epoch}.pkl"))


    print("done")

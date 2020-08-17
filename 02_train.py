from comet_ml import Experiment

import math
import os
import time
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

from src.data_loader import get_data_loader
from src.evaluation import performance_plot
from src.loss import get_loss_function
from src.model import DecoderRNN, DecoderRNNUpdated, EncoderCNN
from src.optimizer import get_optimizer
from src.utils import Config, seed_everything



COMMET_ML_API_KEY = os.environ.get("COMMET_ML_API_KEY")
experiment = Experiment(
    api_key=COMMET_ML_API_KEY, project_name="image_caption_generation"
)

print("Seed everything. Ensure reproducibility...")
seed_everything(seed=42)

if __name__ == "__main__":

    config = Config("config.yaml")
    if config.DEV_MODE:
        warnings.warn(f"Running in dev_mode: {config.DEV_MODE}")

    # Move models to GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (Optional) TODO #2: Amend the image transform below.
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.RandomCrop(
                224
            ),  # get 224x224 crop from random location
            transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize(
                (0.485, 0.456, 0.406),  # normalize image for pre-trained model
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    # Build data loader.
    train_data_loader = get_data_loader(
        transform=transform_train,
        caption_file=config.CAPTION_FILE,
        image_id_file=config.IMAGE_ID_FILE_TRAIN,
        image_folder=config.IMAGE_DATA_DIR,
        config=config,
        mode="train",
        batch_size=config.BATCH_SIZE,
        vocab_threshold=config.VOCAB_THRESHOLD,
        vocab_file=config.VOCAB_FILE,
        vocab_from_file=config.VOCAB_FROM_FILE,
    )

    # Build data loader.
    val_data_loader = get_data_loader(
        transform=transform_train,
        caption_file=config.CAPTION_FILE,
        image_id_file=config.IMAGE_ID_FILE_VAL,
        image_folder=config.IMAGE_DATA_DIR,
        config=config,
        mode="validation",
        batch_size=config.BATCH_SIZE,
        vocab_threshold=config.VOCAB_THRESHOLD,
        vocab_file=config.VOCAB_FILE,
        vocab_from_file=True,
    )  # validation data should use vocab generated from train

    # The size of the vocabulary.
    vocab_size = len(train_data_loader.dataset.vocab)

    hyper_params = {
        "device": str(device),
        "epochs": config.NUM_EPOCHS,
        "learning_rate": 1e-4,
        "batch_size": config.BATCH_SIZE,
        "vocab_threshold": config.VOCAB_THRESHOLD,
        "vocab_size": vocab_size,
        "image_embed_soze": config.IMG_EMBED_SIZE,
        "word_embed_size": config.WORD_EMBED_SIZE,
        "hidden_zise": config.HIDDEN_SIZE,
        "dev_mode": config.DEV_MODE,
    }
    experiment.log_parameters(hyper_params)


    # Initialize the encoder and decoder.
    encoder = EncoderCNN(config.WORD_EMBED_SIZE)
    decoder = DecoderRNNUpdated(
        config.WORD_EMBED_SIZE,
        config.HIDDEN_SIZE,
        vocab_size,
        num_layers=2,
        device=device,
    )

    encoder.to(device)
    decoder.to(device)

    criterion = get_loss_function(loss_type="cross_entropy_loss")

    params = list(decoder.parameters()) + list(encoder.embed.parameters())

    optimizer = get_optimizer(params, optim_type="adam", learning_rate=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3, verbose=True
    )

    # Set the total number of training steps per epoch.
    total_train_step = math.ceil(
        len(train_data_loader.dataset.caption_lengths)
        / train_data_loader.batch_sampler.batch_size
    )

    # Set the total number of validation steps per epoch.
    total_validation_step = math.ceil(
        len(val_data_loader.dataset.caption_lengths)
        / val_data_loader.batch_sampler.batch_size
    )

    old_time = time.time()
    # tqdm_epochs = tqdm(range(1, config.NUM_EPOCHS+1), desc="EPOCH:", leave=True)

    train_loss_list = []
    val_loss_list = []

    train_ppl_list = []
    val_ppl_list = []

    best_val_loss = math.inf

    print("\n Start Training \n")

    step = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):

        # tqdm_train_steps = tqdm(range(1, total_train_step+1), desc='TRAIN BATCH:', leave=True)

        # TRAINING
        encoder.train()
        decoder.train()
        total_train_loss = 0.0
        total_train_perplexity = 0.0
        for i_step in range(1, total_train_step + 1):

            if time.time() - old_time > 60:
                old_time = time.time()

            # Randomly sample a caption length, and sample indices with that length.
            indices = train_data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(train_data_loader))

            # make the captions for targets and teacher forcer
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, : captions.shape[1] - 1].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions_train)

            # Calculate the batch loss.
            loss = criterion(
                outputs.view(-1, vocab_size),
                captions_target.contiguous().view(-1),
            )

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            # stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, config.NUM_EPOCHS, i_step, total_step, loss.item(), np.exp(loss.item()))
            train_loss = loss.item()
            total_train_loss += train_loss

            train_perplexity = np.exp(train_loss)
            total_train_perplexity += train_perplexity

            train_batch_msg = f"\tEpoch: [{epoch}/{config.NUM_EPOCHS}] | TRAIN BATCH steps: [{i_step}/{total_train_step}] |  Loss: {np.round(train_loss,4)}, PPL: {np.round(train_perplexity, 4)}"
            # tqdm_train_steps.set_description(train_batch_msg)
            print(train_batch_msg)

            if config.DEV_MODE:
                if i_step == 5:
                    break

        avg_train_loss = np.round(total_train_loss / total_train_step, 4)
        avg_train_perplexity = np.round(
            total_train_perplexity / total_train_step, 4
        )

        train_loss_list.append(avg_train_loss)
        train_ppl_list.append(avg_train_perplexity)

        # VALIDATION
        encoder.eval()
        decoder.eval()
        total_val_loss = 0.0
        total_val_perplexity = 0.0
        # tqdm_val_steps = tqdm(range(1, total_validation_step+1), desc='VAL BATCH:', leave=True)
        with torch.no_grad():
            for i_step in range(1, total_validation_step + 1):
                # Randomly sample a caption length, and sample indices with that length.
                indices = val_data_loader.dataset.get_train_indices()
                # Create and assign a batch sampler to retrieve a batch with the sampled indices.
                new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
                val_data_loader.batch_sampler.sampler = new_sampler

                # Obtain the batch.
                images, captions = next(iter(val_data_loader))

                # make the captions for targets and teacher forcer
                captions_target = captions[:, 1:].to(device)
                captions_train = captions[:, : captions.shape[1] - 1].to(
                    device
                )

                # Move batch of images and captions to GPU if CUDA is available.
                images = images.to(device)

                # Pass the inputs through the CNN-RNN model.
                features = encoder(images)
                outputs = decoder(features, captions_train)

                # Calculate the batch loss.
                loss = criterion(
                    outputs.view(-1, vocab_size),
                    captions_target.contiguous().view(-1),
                )
                val_loss = loss.item()
                total_val_loss += val_loss

                val_perplexity = np.exp(val_loss)
                total_val_perplexity += val_perplexity

                val_batch_msg = f"\tEpoch: [{epoch}/{config.NUM_EPOCHS}] | VAL BATCH steps: [{i_step}/{total_validation_step}] | Loss: {np.round(val_loss,4)}, PPL: {np.round(val_perplexity, 4)}"
                # tqdm_val_steps.set_description()
                print(val_batch_msg)

                if config.DEV_MODE:
                    if i_step == 5:
                        break

        # performance per epoch
        avg_val_loss = np.round(total_val_loss / total_validation_step, 4)
        avg_val_perplexity = np.round(
            total_val_perplexity / total_validation_step, 4
        )

        scheduler.step(avg_val_loss)

        val_loss_list.append(avg_val_loss)
        val_ppl_list.append(avg_val_perplexity)

        epoch_msg = f"Epoch: [{epoch}/{config.NUM_EPOCHS}] | Train_loss: {np.round(avg_train_loss,4)}, Train_ppl: {np.round(avg_train_perplexity,4)}, Val_loss: {np.round(avg_val_loss,4)}, Val_ppl: {np.round(avg_val_perplexity, 4)}"
        print(epoch_msg)
        # tqdm_epochs.set_description(epoch_msg)

        # Save the weights.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(
                f"****** Saving Best model so far. Best val loss/epoch: {avg_val_loss} ******"
            )
            if config.DEV_MODE:
                file_suffix = f"-checkpoint-dev"
            else:
                file_suffix = f"-checkpoint"

            torch.save(
                decoder.state_dict(),
                os.path.join(config.MODEL_DIR, f"decoder{file_suffix}.pt"),
            )
            torch.save(
                encoder.state_dict(),
                os.path.join(config.MODEL_DIR, f"encoder{file_suffix}.pt"),
            )

        step += 1
        experiment.log_metric("train loss", avg_train_loss, step=step)
        experiment.log_metric(
            "train perplexity", avg_train_perplexity, step=step
        )
        experiment.log_metric("val loss", avg_val_loss, step=step)
        experiment.log_metric("val perplexity", avg_val_perplexity, step=step)

    outfile_loss_plot = "model/loss_plot.png"
    performance_plot(
        train_loss_list,
        val_loss_list,
        outfile=outfile_loss_plot,
        title="Loss vs Epoch",
        ylab="Avg. Batch Loss",
    )

    experiment.log_image(outfile_loss_plot)

    outfile_ppl_plot = "model/ppl_plot.png"
    performance_plot(
        train_ppl_list,
        val_ppl_list,
        outfile=outfile_ppl_plot,
        title="Perplpexity vs Epoch",
        ylab="Avg. Batch Perplexity",
    )

    experiment.log_image(outfile_ppl_plot)

    print("done")

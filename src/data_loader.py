import nltk 
import os
import torch
from torch.utils.data import Dataset, DataLoader, sampler 
import torchvision as tv 
from .vocabulary import Vocabulary
from .utils import Config, get_training_data
from PIL import Image
import numpy as np
from tqdm import tqdm 
import random
import json
import pandas as pd


class FlickrDataset(Dataset):
    
    def __init__(self, 
                 transform: tv.transforms, 
                 mode:str,
                 batch_size:int,
                 vocab_threshold:int,
                 vocab_file:str,
                 start_word:str,
                 end_word:int,
                 unk_word:str,
                 caption_file:str,
                 image_id_file:str,
                 vocab_from_file: bool,
                 image_folder:str):
        self.transform = transform
        self.mode = mode
        self.caption_file = caption_file
        self.image_id_file = image_id_file
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold,
                                vocab_file, 
                                caption_file, 
                                image_id_file, 
                                vocab_from_file,
                                start_word, 
                                end_word, 
                                unk_word)
        self.image_folder = image_folder
        self.df_data = get_training_data(self.image_id_file, 
                                         self.caption_file)
        
        if self.mode in ["train", "validation"]:
            all_captions = self.df_data.CAPTION.values.tolist()
            all_tokens = [nltk.tokenize.word_tokenize(str(caption).lower()) for caption in tqdm(all_captions)]
            self.caption_lengths = [len(token) for token in all_tokens]
            
        
    def __getitem__(self, index):
        
        if self.mode in ["train", "validation"]:
            item = self.df_data.iloc[index]
            image_id = item["IMAGE_ID"]
            caption = item["CAPTION"]
            
            image_file = os.path.join(self.image_folder, image_id)
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
            
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            
            caption = torch.Tensor(caption).long()
            
            # return pre-processed image and caption
            return image, caption 
        else:
            item = self.df_data.iloc[index]
            image_id = item["IMAGE_ID"]
            image_file = os.path.join(self.image_folder, image_id)
            PIL_image = Image.open(image_file).convert("RGB")
            
            original_image = np.array(PIL_image)
            transformed_image = self.transform(PIL_image)
            
            return original_image, transformed_image 
             
    
    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices
    
    def __len__(self):
        return len(df)
    

def get_data_loader(transform:tv.transforms, 
                       caption_file:str,
                       image_id_file:str,
                       image_folder:str,
                       config:Config,
                       mode:str="train", 
                       batch_size:int = 1,
                       vocab_threshold = None,
                       vocab_file: str = "output/vocab.pkl",
                       start_word :str = "<start>",
                       end_word:str = "<end>",
                       unk_word:str = "<unk>",
                       vocab_from_file:bool = True, 
                       num_workers:int = 0):
    """Returns the data loader

    :param transform: [description]
    :type transform: tv.transforms
    :param mode: [description], defaults to "train"
    :type mode: str, optional
    :param batch_size: [description], defaults to 1
    :type batch_size: int, optional
    :param vocab_threshold: [description], defaults to None
    :type vocab_threshold: [type], optional
    :param vocab_file: [description], defaults to "output/vocab.pkl"
    :type vocab_file: str, optional
    :param start_word: [description], defaults to "<start>"
    :type start_word: str, optional
    :param end_word: [description], defaults to "<end>"
    :type end_word: str, optional
    :param unk_word: [description], defaults to "<unk>"
    :type unk_word: str, optional
    :param vocab_from_file: [description], defaults to True
    :type vocab_from_file: bool, optional
    :param num_workers: [description], defaults to 0
    :type num_workers: int, optional
    
    """
    
    assert mode in ["train", "validation", "test"], f"mode: '{mode}' must be one of ['train','validation','test']"
    if vocab_from_file == False: assert mode=="train", f"mode: '{mode}', but to generate vocab from caption file, mode must be 'train' "
    
    if mode == "train":
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        assert image_id_file.find("train"), f"double check image_id_file: {image_id_file}. File name should have the substring 'train'"
        assert os.path.exists(image_id_file), f"image id file: {image_id_file} doesn't not exist."
        assert os.path.exists(caption_file), f"caption file: {caption_file} doesn't not exist."
        assert os.path.isdir(config.IMAGE_DATA_DIR), f"{config.IMAGE_DATA_DIR} not a directory"
        assert len(os.listdir(config.IMAGE_DATA_DIR))!=0, f"{config.IMAGE_DATA_DIR} is empty."
    
    if mode == "validation":
        assert image_id_file.find("dev"), f"double check image_id_file: {image_id_file}. File name should have the substring 'dev' "
        assert os.path.exists(image_id_file), f"image id file: {image_id_file} doesn't not exist."
        assert os.path.exists(caption_file), f"caption file: {caption_file} doesn't not exist."
        assert os.path.isdir(config.IMAGE_DATA_DIR), f"{config.IMAGE_DATA_DIR} not a directory"
        assert len(os.listdir(config.IMAGE_DATA_DIR))!=0, f"{config.IMAGE_DATA_DIR} is empty."
        assert os.path.exists(vocab_file), f"Must first generate {vocab_file} from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
    
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert image_id_file.find("test"), f"double check image_id_file: {image_id_file}. File name should have the substring 'test'"
        assert os.path.exists(vocab_file), f"Must first generate {vocab_file} from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."


    img_folder = config.IMAGE_DATA_DIR
    annotations_file = caption_file    
    
    # image caption dataset
    dataset = FlickrDataset(transform, 
                            mode, 
                            batch_size, 
                            vocab_threshold, 
                            vocab_file, 
                            start_word, 
                            end_word, 
                            unk_word, 
                            caption_file, 
                            image_id_file, 
                            vocab_from_file, 
                            image_folder)
    
    if mode in ["train", "validation"]:
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

    return data_loader
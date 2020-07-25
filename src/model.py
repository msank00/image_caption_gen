import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size:int):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images: torch.Tensor):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        return features


class DecoderRNN(nn.Module):
    def __init__(self, 
                 embed_size:int, 
                 hidden_size:int, 
                 vocab_size:int, 
                 num_layers:int = 1):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        captions = captions[:,:-1] # TODO: why -1?
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim= 1)
        
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)
        
        return out
    
    def predict_token_ids(self, 
                inputs:torch.Tensor,
                states=None, 
                max_len=20):
        """Given a preprocessed image tensor,
        returns predicted image caption sentence as
        list of tensor ids of length max_len. These 
        list of token ids need further mapping using 
        vocabulary dict:idx2word to get the final 
        sentence.

        :param inputs: [description]
        :type inputs: torch.Tensor
        :param states: [description], defaults to None
        :type states: [type], optional
        :param max_len: [description], defaults to 20
        :type max_len: int, optional
        """
        
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(dim=1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)
        
        return output_sentence
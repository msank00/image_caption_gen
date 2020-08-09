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
                 num_layers:int = 1, 
                 dropout:float = 0):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True, 
                            dropout = dropout)
        
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
    
    
class DecoderRNNUpdated(nn.Module):
    # https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3
    def __init__(self, 
                 embed_size:int, 
                 hidden_size:int, 
                 vocab_size:int, 
                 device, 
                 num_layers:int = 1, 
                 dropout:float = 0):
        super().__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, 
                                            embedding_dim=self.embed_size)
        
        self.lstm_cell = nn.LSTMCell(input_size = embed_size, 
                            hidden_size = hidden_size)
        
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=self.vocab_size)
        
        # activations
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features, captions):
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell_state = torch.zeros((batch_size,self.hidden_size)).to(self.device)
        
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).to(self.device)  #.cuda()
        
        # embed the captions
        
        captions_embed = self.embedding_layer(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):
            
            # for the first time step the input is the feature vector
            # features dimension: [batch_size, 300]
            # hidden_state dimension: [batch_size, hidden_dim]
            # cell_state dimension: [batch_size, hidden_dim]
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))
            
            # output of the attention mechanism
            # out dimension: [batch_size, vocab_size]
            out = self.linear(hidden_state)
        
            # build the output tensor
            # outputs dimension: [batch_size, caption_len, vocab_size]
            outputs[:, t, :] = out
            #outputs.append(out)
        
        return outputs    
        # return torch.cat(outputs, dim=1)
    
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
        batch_size = 1
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell_state = torch.zeros((batch_size,self.hidden_size)).to(self.device)
        
        output_sentence = []
        inputs = inputs.squeeze(dim=0)
        for i in range(max_len):
            hidden_state, cell_state = self.lstm_cell(inputs, (hidden_state, cell_state))
            out = self.linear(hidden_state)
            last_pick = out.max(dim=1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick)
        
        return output_sentence
import argparse
import torch
from torch import nn
from wav2vec import Wav2VecWrapper

class SentimentLSTM(nn.Module):
    def __init__(self,args,hidden_dim=512, output_dim=7,embedding_dim=256,device ='cuda',drop_prob=0.5):
        super(SentimentLSTM,self).__init__()
        self.device=device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
       
        # self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.wav2vec= Wav2VecWrapper(args)

        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,num_layers =1,
                            batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        
                 
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self,x):
        batch_size = x.size(0)
        # embeddings and lstm_out
        #embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        x = self.wav2vec(x)
        lstm_out, hidden = self.lstm(x)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.out_layer(out)
        
        # sigmoid function
        # sig_out = self.sig(out)
        
        # # reshape to be batch_size first
        # sig_out = sig_out.view(batch_size, -1)

        # sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return  out
        
        
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        hidden = (h0,c0)
        return hidden
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--finetune_method', 
        default='none',
        type=str, 
        help='finetune method: adapter, embedding prompt, input prompt'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim', 
        default=128,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--embedding_prompt_dim', 
        default=5,
        type=int, 
        help='adapter dimension'
    )
    
    args = parser.parse_args()
    model = SentimentLSTM(args=args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(model.eval())
    print(output[0].shape)
#
#CUDA_VISIBLE_DEVICES=0, taskset -c 1-60 python3 finetune_emotion.py --pretrain_model wav2vec --dataset crema_d --learning_rate 0.0005 --num_epochs 30 --finetune_method finetune
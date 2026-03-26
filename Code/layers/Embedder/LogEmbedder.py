import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class LogEmbeddingAtt_Text(nn.Module):
    def __init__(self, embedding_layer, configs):
        super(LogEmbeddingAtt_Text, self).__init__()
        
        self.embedding_layer = embedding_layer
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        if configs.muti_gpu:
            self.device = 'cuda'
        else:
            self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=1024,
                dropout=self.dropout,
                activation='gelu',  
                layer_norm_eps=1e-5,  
                batch_first=True,
            ),
            num_layers=2
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.embedding_layer.requires_grad = False
        
    def forward(self, x, _):
        bs, sq_len, total_dim = x.shape
        split_size = total_dim // 2
        
        x_id, x_mask = torch.split(x, split_size, dim=2)  
        emb = self.embedding_layer(x_id)  

        x_mask_flat = x_mask.reshape(bs*sq_len,split_size).bool()
        emb_flat = emb.view(bs*sq_len,split_size, self.d_model)

        cls_tokens = self.cls_token.expand(emb_flat.size(0), -1, -1).to(self.device)  
        
        input_with_cls = torch.cat([cls_tokens, emb_flat], dim=1)  

        x_mask_flat = torch.cat([torch.ones(bs*sq_len, 1).bool().to(self.device), x_mask_flat], dim=1)

        encoded = self.transformer_encoder(input_with_cls, src_key_padding_mask=~x_mask_flat)

        encoded = encoded[:,0,:]
        encoded = encoded.view(bs, sq_len, self.d_model)

        return encoded
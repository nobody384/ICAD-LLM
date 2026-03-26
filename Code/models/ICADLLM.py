from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import logging


from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from einops import rearrange

from layers.Embedder.TimeSeriesEmbedder import DataEmbedding, DataEmbeddingAtt
from layers.Embedder.TabularDataEmbedder import TabularDataEmbeddingAtt
from layers.Embedder.LogEmbedder import LogEmbeddingAtt_Text

SPECIAL_TOKEN_NUM = 5
EXAMPLE_TOKEN = 0
TARGET_TOKEN = 1
DATATYPE_SPLIT_TOKEN = 2
EXAMPLE_SPLIT_TOKEN = 3
TARGET_SPLIT_TOKEN = 4

INSTRUCTION_PROMPT = "Determine if the sample exhibits any significant discrepancies or anomalies by comparing it to the reference set."

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.data_types = configs.data_type.split(',')
        self.normal_data_num = configs.normal_data_num

        model_name = 'Qwen/Qwen2.5-0.5B'
        self.instruction_tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        llm = Qwen2Model.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

        if "TimeSeries" in self.data_types:
            self.time_series_embedder = DataEmbeddingAtt(configs.enc_in, configs.d_model)
        if 'TabularData' in self.data_types:
            self.tabular_data_embedder = TabularDataEmbeddingAtt(configs.enc_in, configs.d_model)
        if 'Log' in self.data_types:
            self.log_embedder = LogEmbeddingAtt_Text(llm.embed_tokens , configs)

        self.anomaly_llm = llm
        for i, (name, param) in enumerate(self.anomaly_llm.named_parameters()):
            param.requires_grad = True

        self.new_tokens = nn.Embedding(SPECIAL_TOKEN_NUM, configs.d_model)

        self.sim = nn.CosineSimilarity()

    def forward(self, x, x_examples=None, data_type=None, x_mask=None, x_examples_mask=None):
        x_example_embeddings = []

        x_embedding = self.embedding(x, data_type)

        x_examples = x_examples.transpose(1, 0)
        for x_example in x_examples:
            x_example_embedding = self.embedding(x_example, data_type)
            x_example_embeddings.append(x_example_embedding)
        input_embedding, example_begin_idx, examples_token_idxs, examples_summary_idx, target_token_idx, position_ids = self.concat_input(x_example_embeddings, x_embedding, data_type)
        attention_mask = self.get_mask(
            input_embedding, example_begin_idx, examples_token_idxs, examples_summary_idx, target_token_idx)
        output_embedding = self.anomaly_llm(
            inputs_embeds=input_embedding, 
            position_ids=position_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        if data_type == 'TabularData':
            anomaly_output = self.sim(
                output_embedding[:, examples_summary_idx],
                output_embedding[:, target_token_idx]
            ).arccos()
        elif data_type == 'TimeSeries':
            anomaly_output = self.sim(
                output_embedding[:, examples_summary_idx],
                output_embedding[:, target_token_idx]
            ).arccos()
        elif data_type == 'Log':
            anomaly_output = self.sim(
                output_embedding[:, examples_summary_idx], 
                output_embedding[:, target_token_idx]
            ).arccos()
        else:
            raise NotImplementedError()
        return anomaly_output
    
    def get_mask(self, inputs_embeds, example_begin_idx, examples_token_idxs, examples_summary_idx, target_token_idx):
        dtype = torch.float32
        device = inputs_embeds.device
        batch_size, sequence_length, _ = inputs_embeds.shape
        target_length = sequence_length + 1
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=device
        )
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        last_examples_token_idx = example_begin_idx - 1
        for examples_token_idx in examples_token_idxs:
            if last_examples_token_idx is not None:
                causal_mask[last_examples_token_idx + 1 : examples_token_idx + 1, 
                            example_begin_idx : last_examples_token_idx + 1] = min_dtype
                causal_mask[examples_summary_idx, last_examples_token_idx + 1 : examples_token_idx] = min_dtype
            last_examples_token_idx = examples_token_idx
        causal_mask[examples_summary_idx + 1 : target_token_idx + 1, 
                    example_begin_idx : examples_summary_idx + 1] = min_dtype

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        return causal_mask


    def norm(self, x):
        means = x.mean(1, keepdim=True).detach()
        x_ = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_ /= stdev
        return x_, means, stdev
    
    def embedding(self, x, data_type):
        if data_type == 'TimeSeries':
            x, _, _ = self.norm(x)
            x_embedding = self.time_series_embedder(x, None)
            x_embedding, _, _ = self.norm(x_embedding)
        elif data_type == 'TabularData':
            x_embedding = self.tabular_data_embedder(x, None)
        elif data_type == 'Log':
            x_embedding =  self.log_embedder(x, None)
        else:
            raise NotImplementedError(f'Unknown data type {data_type}')
        return x_embedding

    def instruction_to_embedding(self, instruction):
        instruction_ids = torch.tensor(self.instruction_tokenizer(instruction)['input_ids']).to(self.anomaly_llm.device)
        instruction_embedding = self.anomaly_llm.embed_tokens(instruction_ids)
        return instruction_embedding
    
    def concat_input(self, x_example_embeddings, x_embedding, data_type):
        bs, _, _ = x_embedding.shape
        embeddings = []
        position_ids = []
        examples_token_idxs = []
        embeddings.append(self.new_tokens(torch.LongTensor([[DATATYPE_SPLIT_TOKEN]]).to(self.anomaly_llm.device)).repeat(bs, 1, 1))
        embeddings.append(self.instruction_to_embedding(INSTRUCTION_PROMPT).repeat(bs, 1, 1))
        prefix_length = sum([_.shape[1] for _ in embeddings])
        example_length = x_embedding.shape[1] + 1
        position_ids.append(torch.arange(0, prefix_length, dtype=torch.long, device=x_embedding.device).repeat(bs, 1))
        example_begin_idx = sum([_.shape[1] for _ in embeddings])
        for idx in range(self.normal_data_num):
            embeddings.append(x_example_embeddings[idx])
            examples_token_idxs.append(sum([_.shape[1] for _ in embeddings]))
            embeddings.append(self.new_tokens(torch.LongTensor([[EXAMPLE_SPLIT_TOKEN]]).to(self.anomaly_llm.device)).repeat(bs, 1, 1))
            position_ids.append(torch.arange(prefix_length, prefix_length + example_length, dtype=torch.long, device=x_embedding.device).repeat(bs, 1))

        examples_summary_idx = sum([_.shape[1] for _ in embeddings])
        embeddings.append(self.new_tokens(torch.LongTensor([[EXAMPLE_TOKEN]]).to(self.anomaly_llm.device)).repeat(bs, 1, 1))

        embeddings.append(self.new_tokens(torch.LongTensor([[TARGET_SPLIT_TOKEN]]).to(self.anomaly_llm.device)).repeat(bs, 1, 1))
        embeddings.append(x_embedding)

        target_token_idx = sum([_.shape[1] for _ in embeddings])
        embeddings.append(self.new_tokens(torch.LongTensor([[TARGET_TOKEN]]).to(self.anomaly_llm.device)).repeat(bs, 1, 1))

        position_ids.append(torch.arange(prefix_length + example_length, prefix_length + example_length * 2 + 2, dtype=torch.long, device=x_embedding.device).repeat(bs, 1))
        
        out_embeddings = torch.concatenate(embeddings, dim=1)
        out_position_ids = torch.concatenate(position_ids, dim=1)
        return out_embeddings, example_begin_idx, examples_token_idxs, examples_summary_idx, target_token_idx, out_position_ids


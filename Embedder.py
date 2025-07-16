#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gracedavenport
"""
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
#%%
class Embedder():
    
    def __init__(self, path):
        """
        Initializes an instance of a mpnet summarization model and tokenizer from a specified path.
    
        Parameters:
            path (str): The path to the pretrained mpnet model directory.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        self.model = AutoModel.from_pretrained(path)
        
        
    def _get_chunks(self, string, max_length_input, overlap):
        """
        Splits a long string into overlapping chunks based on the tokenizer's input limits.
    
        If the tokenized input exceeds `max_length_input`, the string is split using
        RecursiveCharacterTextSplitter to ensure compatibility with model input length.
    
        Parameters:
            string (str): The input text to be chunked.
            max_length_input (int): Maximum number of tokens allowed per chunk.
            overlap (int): Number of tokens to overlap between consecutive chunks.
    
        Returns:
            chunks (list): A list of text chunks.
        """
        input_ids = self.tokenizer(string)["input_ids"]
        
        if len(input_ids) > max_length_input:
            
            token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=self.tokenizer, chunk_size=max_length_input, chunk_overlap=overlap)
            
            return token_splitter.split_text(string)
        
        else:
            
            return [string]
        
        
    def _mean_pooling(self, output, attention_mask):
        """
        Performs mean pooling on token embeddings.
    
        Parameters:
            output (Tuple[torch.Tensor]): The output from the model's forward pass. 
            attention_mask (torch.Tensor): The attention mask tensor with shape 
                (batch_size, sequence_length), where 1 indicates valid tokens and 0 indicates padding.
    
        Returns:
            embeddings (torch.Tensor): The mean-pooled embeddings with shape (batch_size, embedding_dim).
        """        
        token_embeddings = output[0]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
    def get_embeddings(self, string, max_length_input, overlap):
        """
        Generates an embedding vector for the input string by splitting it into chunks (if needed),
        encoding each chunk, and then averaging the chunk embeddings.
    
        Parameters:
            string (str): The input text to embed.
            max_length_input (int): The maximum token length for each chunk before splitting.
            overlap (int): Number of tokens to overlap between chunks when splitting.
    
        Returns:
            embeddings (torch.Tensor): A single embedding tensor representing the input string,
                          averaged over chunk embeddings if multiple chunks exist.
        """
        chunks = self._get_chunks(string, max_length_input, overlap)
        
        embeddings = []
        
        for chunk in chunks:
            
            inputs = self.tokenizer(chunk, padding=True, truncation=True, 
                                    return_tensors="pt")
            
            with torch.no_grad():
                
                output = self.model(**inputs)
                
            embed = self._mean_pooling(output, inputs["attention_mask"])
            
            embeddings.append(embed)
            
        if len(embeddings) > 1:
            
            return torch.stack(embeddings).mean(dim=0)
        
        else:
            
            return embeddings[0]


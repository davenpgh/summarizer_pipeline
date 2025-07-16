#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gracedavenport
"""

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, LEDForConditionalGeneration

#%%
class Summarizer:
    
    def summarize(self, string, max_length_input, truncation, min_length_output, max_length_output):
        """
        Generates a summary of the input text using a pretrained transformer model.
        
        Parameters:
            string (str): The input text to summarize.
            max_length_input (int): The maximum number of tokens allowed in the input.
            truncation (bool): Whether to truncate the input if it exceeds max_length_input.
            min_length_output (int): The minimum number of tokens in the generated summary.
            max_length_output (int): The maximum number of tokens in the generated summary.
        
        Returns:
            summary (str): The summarized version of the input text.
        """
                
        
        inputs = self.tokenizer.encode(f"summerize: {string}", return_tensors="pt",
                                       mx_length=max_length_input, truncation=truncation)
        
        output = self.model.generate(inputs, min_length=min_length_output, 
                                     max_length=max_length_output, length_penalty=2.0,
                                     num_beams=4, early_stopping=True)
        
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return summary
    
    
class BartSummarizer(Summarizer):
    
    def __init__(self, path):
        """
        Initializes an instance of a BART summarization model and tokenizer from a specified path.
    
        Parameters:
            path (str): The path to the pretrained BART model directory.
        """
        self.tokenizer = BartTokenizer.from_pretrained(path)
        
        self.model = BartForConditionalGeneration.from_pretrained(
            path, return_dict=True)
        
        
class LedSummarizer(Summarizer):
    
    def __init__(self, path):
        """
        Initializes an instance of a LED summarization model and tokenizer from a specified path.
    
        Parameters:
            path (str): The path to the pretrained LED model directory.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        self.model = LEDForConditionalGeneration.from_pretrained(
            path, return_dict=True)
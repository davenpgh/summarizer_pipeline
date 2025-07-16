#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gracedavenport
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
#%%
class SummarizerPipeline():
    
    def __init__(self, short_suummarizer, long_summarizer, embedder):
        """
        Initializes the hybrid summarization system with separate short and long document summarizers 
        and an embedder for semantic similarity checking.
        
        Parameters:
            short_summarizer: A BartSummarizer class object for handling shorter texts.
            long_summarizer: A LedSummarizer class object optimized for longer texts.
            embedder: An Embedder class object used to generate vector embeddings 
                      for semantic similarity calculations.
        """
        self.short_summarizer = short_suummarizer
        
        self.long_summarizer = long_summarizer
        
        self.embedder = embedder
        
        
    def run(self, string, max_length_input_short = 1024, max_length_input_long = 16000,
            truncation = True, min_length_output = 80, max_length_output = 300, 
            overlap = 20):
        """
        Runs the hybrid summarization process on the input string. It first generates a short summary,
        then checks the semantic similarity between the summary and the original text. If similarity is 
        low, it applies a long-document hybrid summarization approach.
    
        Parameters:
            string (str): The input text to summarize.
            max_length_input_short (int, optional): Maximum input length for the short summarizer. Default is 1024.
            max_length_input_long (int, optional): Maximum input length for the long summarizer. Default is 16000.
            truncation (bool, optional): Whether to truncate inputs exceeding max length. Default is True.
            min_length_output (int, optional): Minimum length of generated summaries. Default is 80.
            max_length_output (int, optional): Maximum length of generated summaries. Default is 300.
            overlap (int, optional): Token overlap between chunks when splitting long inputs. Default is 20.
    
        Returns:
            summary (str): The generated summary.
        """
        summary = self.short_summarizer.summarize(string, max_length_input = max_length_input_short,
                                                  truncation = truncation, min_length_output = min_length_output,
                                                  max_length_output = max_length_output)
        
        cosine_sim = self._cos_similarity(summary, string)
        
        if cosine_sim < 0.7:
            
            return self._summarize_hybrid(string, max_length_input_short = max_length_input_short,
                                          max_length_input_long = max_length_input_long,
                                          truncation = truncation, min_length_output = min_length_output,
                                          max_length_output = max_length_output, overlap = overlap)
        else:
            
            return summary
        
        
    def _cos_similarity(self, summary, document, max_length_input = 512, overlap = 20):
        """
        Computes the cosine similarity between the embeddings of the summary and the original document.
    
        Parameters:
            summary (str): The summary text.
            document (str): The original document text.
            max_length_input (int, optional): Maximum token length when generating embeddings. Default is 512.
            overlap (int, optional): Token overlap when chunking text for embeddings. Default is 20.
    
        Returns:
            cosine_sim (float): Cosine similarity score between summary and document embeddings.
        """
        vector_summary = self.embedder.get_embeddings(summary, max_length_input, overlap)
        
        vector_document = self.embedder.get_embeddings(document, max_length_input, overlap)
        
        return cosine_similarity(vector_summary, vector_document)
    
    
    def _summarize_hybrid(self, string, max_length_input_short, max_length_input_long,
                          truncation, min_length_output, max_length_output, overlap):
        """
        Performs hybrid summarization on long documents by recursively chunking and summarizing the text 
        until it fits within the short summarizer's input size limits.
    
        Parameters:
            string (str): The input text to summarize.
            max_length_input_short (int): Maximum input length for the short summarizer.
            max_length_input_long (int): Maximum input length for the long summarizer.
            truncation (bool): Whether to truncate inputs exceeding max length.
            min_length_output (int): Minimum length of generated summaries.
            max_length_output (int): Maximum length of generated summaries.
            overlap (int): Token overlap between chunks when splitting long inputs.
    
        Returns:
            summary (str): The final summarized text after hybrid summarization.
        """
        input_ids = self.long_summarizer.tokenizer(string)["input_ids"]
        
        while len(input_ids) > 1024:
            
            token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer = self.long_summarizer.tokenizer, chunk_size = max_length_input_long,
                chunk_overlap = overlap)
            
            chunks = token_splitter.split_text(string)
            
            string = ""
            
            for chunk in chunks:
                
                summary_chunk = self.long_summarizer.summarize(chunk, max_length_input = max_length_input_long,
                                                               truncation = truncation,
                                                               min_length_output = min_length_output,
                                                               max_length_output = max_length_output)
                
                string += summary_chunk
                
            input_ids = self.long_summarizer.tokenizer(string)["input_ids"]
            
        else:
            
            summary = self.short_summarizer.summarize(string, max_length_input = max_length_input_short,
                                                      truncation = truncation, 
                                                      min_length_output = min_length_output,
                                                      max_length_output = max_length_output)
            
        return summary
        
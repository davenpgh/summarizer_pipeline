# summarizer_pipeline

A text summarization system that intelligently combines short and long document summarizers with semantic similarity checking to produce high-quality summaries for texts of varying lengths.

The SummarizerPipline() class first generates a summary using a short-text summarizer and compares its semantic similarity to the original text using embeddings. If the similarity is below a threshold, it applies a recursive hybrid summarization process designed for long documents. This process chunks the text, summarizes each chunk using a long-document summarizer, concatenates chunk summaries, and repeats until the text is short enough to summarize with the short-text summarizer.

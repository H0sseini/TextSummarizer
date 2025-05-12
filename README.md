# TextSummarizer
Using AI to summarize the texts
This tool will help you to summarize texts (.pdf, .docx, .txt, .md, or input texts). The summarization is done using bart-large-cnn model. The language is preferably English but I also used it for a Farsi text and the result was neither promising nor that bad. Two methods are considered for summarization:

  •	<b>Abstractive:</b> which abstracts the input text.
  
  •	<b>Extractive:</b> which extracts key sentences.
  
There are also three options for the length of the summarized text. For the extractive method, the model returns up to 5 sentences if you select the short version and up to 10 sentences otherwise. For the abstractive type, the model returns summarized text with up to 150, 300, and 600 words for the summary lengths short, medium, and long, respectively.

The summarization is better done when the input text is around 10,000 words. 

Of course, it is worth noting the AI model makes mistakes so, don’t rely solely on them. 
 

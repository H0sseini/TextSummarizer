# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:30:07 2025

@author: H0sseini
"""
import re
import os
import torch
import fitz  # PyMuPDF
from docx import Document
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from io import BytesIO
from huggingface_hub import snapshot_download




def download_bart_large_cnn(local_dir="./app/models/bart-large-cnn"):
    print(f"Preparing to download BART-large-CNN model to: {local_dir}")

    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        # Check if model files already exist
        expected_files = [
            "config.json", "generation_config.json", "model.safetensors",
            "tokenizer_config.json", "tokenizer.json", "vocab.json",
            "merges.txt", "special_tokens_map.json"
        ]
        if all(os.path.isfile(os.path.join(local_dir, f)) for f in expected_files):
            print("✔ Model files already present. Skipping download.")
            return True
        else:
            print("⚠ Some model files are missing. Redownloading...")

    # Download the full snapshot (model and tokenizer files)
    try:
        snapshot_download(
            repo_id="facebook/bart-large-cnn",
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Make sure files are copied directly
        )
    
        print("✅ Download complete. Model is ready for use.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download the model: {e}")
        return False
    


try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    nltk.download("punkt_tab")
except (LookupError, ImportError):
    nltk.download("punkt")

class SummarizationTool:
    def __init__(self, model_path="./app/models/bart-large-cnn"):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {'cuda:0' if self.device == 0 else 'cpu'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                       model_max_length=512)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if self.device == 0 else "cpu")

        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            framework="pt",
            truncation=True,
            max_length=200,
            min_length=50,
            do_sample=False,           # prevents random hallucination
            repetition_penalty=2.0,    # discourages repeating phrases
            length_penalty=1.0,        # balances between long/short
            early_stopping=True
        )

        self.MAX_TOKENS = 512
        self.MAX_VALID_LENGTH = 700
        self.OVERLAP = 100
        self.mode_lengths = {
            "short": 150,
            "medium": 300,
            "detailed": 600
        }

    def find_char(self, text, character):
        # finding punctuation marks to make the output cleaner
        indices = []
        for index, letter in enumerate(text):
            if letter == character:
                indices.append(index)
        return indices
    
    def add_space_after_punctuation(self, text):
        # Pattern to find punctuation marks not followed by a space,
        # excluding cases where it's followed by another punctuation, digit, or letter (likely URL or number)
        pattern = r'(?<!\w)(?<!\.\w)([.,!?;:])(?=[^\s\d.,!?;:\w])'
        
        # This version adds space only after punctuation if the next char is a letter or other symbol (but not digit or . or / etc.)
        def replacer(match):
            punct = match.group(1)
            return punct + ' '

        # Temporarily protect known patterns like URLs and numbers
        protected_patterns = []

        # Save and replace URLs and domains
        def protect(match):
            protected_patterns.append(match.group(0))
            return f"__PROTECTED{len(protected_patterns)-1}__"

        text = re.sub(r'https?://\S+|www\.\S+|\b\w+\.(com|org|net|edu|gov)\b', protect, text)
        text = re.sub(r'\b\d+[.,]\d+\b', protect, text)  # Protect decimal numbers like 3.14

        # Add space after punctuation if needed
        text = re.sub(r'([.,!?;:])(?=\S)', lambda m: m.group(1) + ' ', text)

        # Restore protected patterns
        for i, original in enumerate(protected_patterns):
            text = text.replace(f"__PROTECTED{i}__", original)

        return text
        
    def clean_text(self, text):
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text

    def split_text(self, text, max_tokens=None, overlap=None):
       max_tokens = max_tokens or self.MAX_TOKENS
       overlap = overlap or self.OVERLAP

       tokens = self.tokenizer(text, return_offsets_mapping=True, return_attention_mask=False)
       input_ids = tokens["input_ids"]
       offsets = tokens["offset_mapping"]
    
       chunks = []
       start = 0
    
       while start < len(input_ids):
            end = min(start + max_tokens, len(input_ids))
            chunk_offsets = offsets[start:end]
            if chunk_offsets:
                start_char = chunk_offsets[0][0]
                end_char = chunk_offsets[-1][1]
                chunk = text[start_char:end_char].strip()
                chunks.append(chunk)
            start += max_tokens - overlap
    
       return chunks

    def summarize_chunks(self, chunks, min_length=None, max_length=None):
        summaries = []
    
        if self.device == 0:  # Using CUDA
            try:
                def summarize_input(texts):
                    return self.summarizer(
                        texts,
                        min_length=50 if min_length is None else min_length,
                        max_length=200 if max_length is None else max_length,
                        truncation=True
                    )
    
                dataset = Dataset.from_dict({"text": chunks})
                results = dataset.map(
                    lambda batch: {"summary": [s["summary_text"] for s in summarize_input(batch["text"])]},
                    batched=True,
                    batch_size=8
                )
    
                for entry in results:
                    summaries.extend(entry["summary"])
    
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                # Fallback to sequential processing if batch fails
                for i, chunk in enumerate(chunks):
                    try:
                        summary_output = self.summarizer(
                            chunk,
                            min_length=50 if min_length is None else min_length,
                            max_length=200 if max_length is None else max_length,
                            truncation=True
                        )
                        summaries.append(summary_output[0]["summary_text"])
                    except Exception as e:
                        print(f"[Chunk {i}] Error:", e)
                        summaries.append("")
        else:
            for i, chunk in enumerate(chunks):
                try:
                    summary_output = self.summarizer(
                        chunk,
                        min_length=50 if min_length is None else min_length,
                        max_length=200 if max_length is None else max_length,
                        truncation=True
                    )
                    summaries.append(summary_output[0]["summary_text"])
                except Exception as e:
                    print(f"[Chunk {i}] Error:", e)
                    summaries.append("")
    
         
        return summaries
    
    def remove_junks(self, text):
        # Removing junks created by the model
        keywords = ['CNN.com', 'iReporter', 'CNN.com/Travel',   'Samaritans', 
                    'www.samaritans.org', 'For confidential support']
        #full_text = self.smart_join(text)
        full_text = ''.join(text)
        if (full_text.find(keywords[1]) - full_text.find(keywords[0]) == 21 and
            full_text.find(keywords[2]) - full_text.find(keywords[0]) == 139):
            full_text = full_text[:full_text.find(keywords[0])]
        if (full_text.find(keywords[4]) - full_text.find(keywords[3]) == 69):
            full_text = full_text[:full_text.find(keywords['For confidential support'])]
        return full_text

    def summarize_first_level(self, text):
        text = self.clean_text(text)
        chunks = self.split_text(text)
        summaries = self.summarize_chunks(chunks)  # No min/max lengths here
        
        return self.remove_junks(summaries)
        

    def summarize_second_level(self, text, max_words):
        chunks = self.split_text(text)
        summaries = self.summarize_chunks(
            chunks,
            min_length=int(max_words * 0.6),
            max_length=max_words
        )
        return self.remove_junks(summaries)
        

    def extractive_summarize(self, text, num_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)

    def summarize(self, text, mode="medium", summary_type="abstractive"):
        if summary_type == "extractive":
            return self.extractive_summarize(text, num_sentences=5 if mode == "short" else 10)

        try:
            first_summary = self.summarize_first_level(text)
            word_count = len(first_summary.split())
            max_words = self.mode_lengths.get(mode, 300)

            if word_count <= max_words:
                
                return self.add_space_after_punctuation(first_summary)
            else:
                while word_count > max_words:
                    try:
                        final_summary = self.summarize_second_level(first_summary, max_words)
                        word_count = len(final_summary.split())
                        first_summary = final_summary
                    
                    except Exception as e:
                        print(f"[Final Summary] Error: {e}")
                        return self.add_space_after_punctuation(first_summary)
                
                return self.add_space_after_punctuation(final_summary)
        except Exception as e:
            print(f"[Summarization] Error: {e}")
            # Fallback to extractive summarization if abstractive fails
            return self.extractive_summarize(text, num_sentences=5 if mode == "short" else 10)

    def extract_text_from_pdf(self, file_bytes):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_text_from_docx(self, file_bytes):
        doc = Document(BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])

    def extract_text_from_txt(self, file_bytes):
        return file_bytes.decode("utf-8", errors="ignore")

    def extract_text_from_md(self, file_bytes):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            text = re.sub(r'[^\S\r\n]+', ' ', text)
            text = re.sub(r'#.*', '', text)
            text = re.sub(r'[*_~>-]', '', text)
            return text.strip()
        except Exception as e:
            return f"[Error reading markdown]: {str(e)}"

    def extract_text_from_bytes(self, file_bytes, filename):
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(file_bytes)
        elif ext == ".docx":
            return self.extract_text_from_docx(file_bytes)
        elif ext == ".txt":
            return self.extract_text_from_txt(file_bytes)
        elif ext == ".md":
            return self.extract_text_from_md(file_bytes)
        else:
            raise ValueError("Unsupported file format")

    def extract_text_from_string(self, text_input):
        return self.clean_text(text_input)


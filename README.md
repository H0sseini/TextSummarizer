# TextSummarizer
Using AI to summarize the texts.

A local document summarization tool built with FastAPI and Hugging Face's BART model. The language is preferably English but I also used it for a Farsi text and the result was neither promising nor that bad. 

---

## Features

* Upload or paste text content
* Summarize using Abstractive or Extractive methods
* Three length options: Short, Medium, Detailed
* Supports `.pdf`, `.docx`, `.txt`, and `.md` files
* Detects and uses GPU if available

---

## Requirements

* Python 3.10+
* Git
* Windows (for `.bat` launcher) or any platform for manual run

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/H0sseini/TextSummarizer.git
cd TextSummarizer
```

### 2. Create a virtual environment (optional but recommended)

```
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download model (optional and only required on the first run)

Place the required BART model files in the following folder:

```
./app/models/bart-large-cnn/
```

If files are missing or you prefer the app handles the download, the backend will automatically download necessary files and put them in the above folder.

---

## How to Run

### 📦 Option A: Use the launcher (Windows only)

Double-click the `run_app.bat` and a webpage will be opened that loads the app after a while for you. 


### 🖥️ Option B: Manual run

From the project directory:

```
uvicorn frontend.app:app --reload
```

Or if uvicorn's path is not added:

```
python -m uvicorn frontend.app:app --reload
```

Then open your browser and go to:

```
http://127.0.0.1:8000
```

---

## Usage

1. Choose a file **or** paste text into the box.
2. Select summary type and length.
3. Press **Summarize**.
4. Copy result with the **Copy to Clipboard** button.

---

## Notes

* Model files (e.g., `model.safetensors`) are excluded from Git.
* GPU support is automatic.
* Extractive summarization uses `sumy` (LexRank).

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---
## Limitations

📖 The summarization is better done when the input text is around 10,000 words. 

⚠️ The model requires 1.6 GB of data, which will be downloaded the first time. For the next uses, it won’t download data unless it is a necessary update. If your system does not have Python or the repo's dependencies (such as torch, datasets, fastapi, etc.), the installation process may require downloading an additional 5.0 GB of data for the first time.

The model may take some time when using the CPU, so it is highly recommended to use a GPU, if available.
       

Of course, it is worth noting the AI model makes mistakes so, don’t rely solely on it.

Have fun! 🚀


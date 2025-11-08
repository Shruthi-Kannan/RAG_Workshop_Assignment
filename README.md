# My RAG System (Retrieval-Augmented Generation)

## Topic: Deep Learning Research
**Topic Choice Explanation:** I chose Deep Learning because I am studying machine learning algorithms and wanted to create a tool capable of accurately retrieving and synthesizing complex technical definitions and architectural details from various research papers. The need for precise technical answers makes it an excellent domain for RAG validation.

---

## 🛠️ Assignment Modifications

### 1. Documents (Part 1 Submission)

- **Documents Included:** 5 PDF files related to Deep Learning (e.g., Transformers, CNNs, GANs, etc.).
- **Location:** All 5 documents are placed in the `./data/` folder.

### 2. Text Chunking (Part 2 Submission)

- **Original Settings:** `chunk_size=800`, `chunk_overlap=150`
- **New Settings Used:**
  ```python
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1200,
      chunk_overlap=200,
      separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
  )

Note: I chose to increase the chunk_size to 1200 and the chunk_overlap to 200. This was chosen because Deep Learning research papers often contain long, complex sentences, formulas, and arguments. The larger chunk size helps keep complete definitions and technical descriptions together, ensuring the LLM has full context for any given technical point.

### 3. Retrieval Configuration (Part 3 Submission)

- **Original MMR Settings:** `k=5`, `fetch_k=10`, `lambda_mult=0.7`
- **New Settings Used:**

	```Python
	retriever = vectorstore.as_retriever(
    	search_type="mmr",
    	search_kwargs={
        	"k": 7,            # Increased from 5
        	"fetch_k": 15,     # Increased from 10
        	"lambda_mult": 0.3 # Decreased from 0.7
    		}
	)

Note: I decreased lambda_mult to 0.3 to prioritize diversity (Maximum Marginal Relevance - MMR). This is critical for the Deep Learning topic when answering synthesis questions (e.g., comparing two different loss functions from two different papers) as it ensures the retriever pulls relevant chunks from all necessary sources, even if they aren't the absolute highest-similarity chunks. I also increased k to 7 to provide a slightly larger context window for the LLM.

## 🚀 How to Run
Clone the repository.

Install Ollama: Install and run the local LLM server from https://ollama.ai/.

Pull the model: ollama pull phi3:mini

Install Python requirements: pip install -r requirements.txt

Run the script: python RAG_Assignment_DL.py

## ✅ Test Questions:
1. "What are some of the challenges in neural network optimization?" 
2. "Explain Adversarial Training in simple terms."

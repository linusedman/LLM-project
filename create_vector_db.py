# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 08:53:44 2025

@author: Lovisa
"""

import os
from langchain_community.vectorstores import FAISS  # "db" to store and retrieve embeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter  # split long documents
from pdfminer.high_level import extract_text  # extract text from pdfs
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import logging

dir_path = Path("./scraped_pdfs")

# Silence pdfminer logs below ERROR
logging.getLogger("pdfminer").setLevel(logging.ERROR)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
doc_lst = []
# Extract text
#text = extract_text(DOC_PATH, page_numbers=[0])
for i, file in enumerate(dir_path.iterdir()):
    # if i >= 20:
    #     break

    if file.name.endswith(".pdf") and not file.name.startswith("SFS"):
        try:
            print(f"Processing {file.name}")        
            text = extract_text(file)
    
            doc = Document(page_content=text, metadata={"source": file.name})
            chunks = splitter.split_documents([doc])
            doc_lst.extend(chunks) # avoid nested lists
        except Exception as e:
            print(f"Failed with {file.name}")

    else:
        print(f"Skipping {file.name}")
        
print(len(doc_lst))

# Local embedding model
embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

# Database
db = FAISS.from_documents(doc_lst, embeddings)

# Save the database
db.save_local("faiss_index_all_sv")
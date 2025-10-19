from time import perf_counter

from chatbot import response_stream

from langchain_community.vectorstores import FAISS  # "db" to store and retrieve embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

def get_response(text_input):
    t_start = perf_counter()
    final = ""
    for chunk in response_stream({'text': text_input, 'files': []}, history=[]):
        final = chunk
    t_end = perf_counter()
    latency = t_end - t_start
    return final, latency

print(get_response("hej"))
import statistics
from time import perf_counter

from chatbot import response_stream

from langchain_community.vectorstores import FAISS  # "db" to store and retrieve embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

def get_response(text_input):
    t_start = perf_counter()
    final = ""
    for chunk in response_stream({'text': text_input, 'files': []}, history=[]):
        response = chunk
    t_end = perf_counter()
    latency = t_end - t_start
    return response, latency

def get_n_responses(text_input, n):
    responses, latencies = [], []
    for _ in range(n):
        response, latency = get_response(text_input)
        responses.append(response)
        latencies.append(latency)
    return responses, latencies

def calc_latency_stats(latencies):
    if not latencies:
        return "—"
    # Convert seconds to microseconds
    lat_us = [x * 1e6 for x in latencies]
    median = statistics.median(lat_us)
    avg = statistics.mean(lat_us)
    return f"Average={avg:.3f}µs | Median={median:.3f}µs"

responses, latencies = get_n_responses("hej", 5)
print(responses)
print(latencies)
print(calc_latency_stats(latencies))
import statistics
from time import perf_counter
from itertools import combinations
from chatbot import response_stream

from sentence_transformers import SentenceTransformer, util
embedding_model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

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

def embdding_cos_similarity(text_1, text_2):
    embedding_1 = embedding_model.encode(text_1, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(text_2, convert_to_tensor=True)
    return float(util.cos_sim(embedding_1, embedding_2).item())

def pairwise_avg_similarity(responses):
    if len(responses) < 2:
        return 1.0
    similarities = [embdding_cos_similarity(text_1, text_2) for text_1, text_2 in combinations(responses, 2)]
    return sum(similarities) / len(similarities)

responses, latencies = get_n_responses("hej", 5)
print(responses)
print(latencies)
print(pairwise_avg_similarity(responses))
print(calc_latency_stats(latencies))
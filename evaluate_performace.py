import statistics
from time import perf_counter, sleep
from itertools import combinations
from chatbot import response_stream
from model_no_RAG import response_stream_no_RAG
import pandas as pd

from sentence_transformers import SentenceTransformer, util
embedding_model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

def get_response(text_input, func=response_stream):
    t_start = perf_counter()
    response = ""
    for chunk in func({'text': text_input, 'files': []}, history=[]):
        response = chunk
    t_end = perf_counter()
    latency = t_end - t_start
    sleep(5)
    return response, latency

def get_n_responses(text_input, n, use_RAG=True):
    func = response_stream
    if not use_RAG:
        response_stream_no_RAG    # Use model without RAG
    responses, latencies = [], []
    for _ in range(n):
        response, latency = get_response(text_input, func)
        responses.append(response)
        latencies.append(latency)
    return responses, latencies

def calc_latency_stats(latencies):
    if not latencies:
        return "—", "-"
    median = statistics.median(latencies)
    avg = statistics.mean(latencies)
    return avg, median

def embdding_cos_similarity(text_1, text_2):
    embedding_1 = embedding_model.encode(text_1, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(text_2, convert_to_tensor=True)
    return float(util.cos_sim(embedding_1, embedding_2).item())

def pairwise_similarity_summary(responses):
    if len(responses) < 2:
        return 1.0, None, None
    similarities = []
    for text_1, text_2 in combinations(responses, 2):
        sim = embdding_cos_similarity(text_1, text_2)
        similarities.append((sim, (text_1, text_2)))

    avg_similarity = sum(s for s, _ in similarities) / len(similarities)
    most_similar_pair = max(similarities, key=lambda x: x[0])[1]
    least_similar_pair = min(similarities, key=lambda x: x[0])[1]

    return avg_similarity, most_similar_pair, least_similar_pair


def calculate_stats(use_RAG=True):
    stats_df = pd.DataFrame(columns=["item", "prompt", "cos_sim", "latency_avg", "latency_median", "most_similar_prompts", "least_similar_prompts"])

    # items = ["räkor", "kött", "tomater"]
    # items = ["räkor"]
    items = ["nötkött"]


    open_question_prompts = [f"Jag ska sälja {item}, vad ska är viktigt att tänka på?" for item in items]
    short_answer_prompts = [f"Jag ska sälja {item}, gör en lista av det jag MÅSTE inkludera på etiketten och ENDAST det!" for item in items]

    num_responses = 10
    print("Begin loop")
    for item, open_question, short_answer in zip(items, open_question_prompts, short_answer_prompts):
        print("-----------------------------------------------------")
        print(f"Processing item: {item}")
        print("Getting open question responses")
        open_question_responses, open_question_latencies = get_n_responses(open_question, num_responses, use_RAG)
        print("Getting short answer responses")
        short_answer_responses, short_answer_latencies = get_n_responses(short_answer, num_responses, use_RAG)

        print("Start calculating pairwise avg similarities")
        open_question_sim_avg, open_question_most, open_question_least = pairwise_similarity_summary(open_question_responses)
        short_answer_sim_avg, short_answer_most, short_answer_least = pairwise_similarity_summary(short_answer_responses)
        print("Start calculating latensy stats")
        open_question_latency_avg, open_question_latency_median = calc_latency_stats(open_question_latencies)
        short_answer_latency_avg, short_answer_latency_median = calc_latency_stats(short_answer_latencies)

        stats_df.loc[len(stats_df)] = [item, "open_question", open_question_sim_avg, open_question_latency_avg, open_question_latency_median, open_question_most, open_question_least]
        stats_df.loc[len(stats_df)] = [item, "short_answer", short_answer_sim_avg, short_answer_latency_avg, short_answer_latency_median, short_answer_most, short_answer_least]
        print(f"Done processing item: {item}\n\n")
    print("Done with loop")

    filename = f"statistics/stats_{item[0]}"
    if not use_RAG:
        filename += "_no_RAG"
    print("Saving df")
    stats_df.to_csv(f"{filename}.csv", index=False)
    print("df saved")

if __name__ == "__main__":
    # calculate_stats(True)
    calculate_stats(False)

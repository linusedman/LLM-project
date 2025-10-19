import statistics
from time import perf_counter
from itertools import combinations
from chatbot import response_stream
import pandas as pd

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
        return "—", "-"
    median = statistics.median(latencies)
    avg = statistics.mean(latencies)
    return avg, median

def embdding_cos_similarity(text_1, text_2):
    embedding_1 = embedding_model.encode(text_1, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(text_2, convert_to_tensor=True)
    return float(util.cos_sim(embedding_1, embedding_2).item())

def pairwise_avg_similarity(responses):
    if len(responses) < 2:
        return 1.0
    similarities = [embdding_cos_similarity(text_1, text_2) for text_1, text_2 in combinations(responses, 2)]
    return sum(similarities) / len(similarities)


def main():
    stats_df = pd.DataFrame(columns=["item", "prompt", "cos_sim", "latency_avg", "latency_median"])

    # items = ["räkor", "hasselnötter", "griskött", "ekologiska tomater", "kantareller", "nötkött", "kött från utrotningshotade djur"]
    items = ["räkor"]


    open_question_prompts = [f"Jag ska sälja {item}, vad ska är viktigt att tänka på?" for item in items]
    short_answer_prompts = [f"Jag ska sälja {item}, gör en lista av det jag MÅSTE inkludera på etiketten och ENDAST det!" for item in items]

    num_responses = 10
    print("Begin loop")
    for item, open_question, short_answer in zip(items, open_question_prompts, short_answer_prompts):
        print(f"Processing item: {item}")
        print("Getting open question responses")
        open_question_responses, open_question_latencies = get_n_responses(open_question, num_responses)
        print("Getting short answer responses")
        short_answer_responses, short_answer_latencies = get_n_responses(short_answer, num_responses)

        print(f"Open question responses: {open_question_responses}")
        print(f"Short answer responses: {short_answer_responses}")

        print("Start calculating pairwise avg similarities")
        open_question_sim = pairwise_avg_similarity(open_question_responses)
        short_answer_sim = pairwise_avg_similarity(short_answer_responses)
        print("Start calculating latensy stats")
        open_question_latency_avg, open_question_latency_median = calc_latency_stats(open_question_latencies)
        short_answer_latency_avg, short_answer_latency_median = calc_latency_stats(short_answer_latencies)

        stats_df.loc[len(stats_df)] = [item, "open_question", open_question_sim, open_question_latency_avg, open_question_latency_median]
        stats_df.loc[len(stats_df)] = [item, "short_answer", short_answer_sim, short_answer_latency_avg, short_answer_latency_median]
    print("Done with loop")
    print(stats_df)
if __name__ == "__main__":
    main()
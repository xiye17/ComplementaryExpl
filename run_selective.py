import os
import argparse
import itertools
from random import choices

from tqdm import tqdm
from math import ceil

import numpy as np
from scipy.special import softmax

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
    DEFAULT_TRAIN_SEP
)

from task_helper import TaskHelper, load_train_test_set
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results

def register_selecive_args(parser):
    parser.add_argument('--run_scoring', default=False, action='store_true')

    parser.add_argument('--score_strategy', type=str, default="api-question", choices=["api-question", "esnli-promptquestion"])
    parser.add_argument('--score_engine', type=str, default=None, choices=['opt', "davinci", "text-davinci-001", "text-davinci-002", "text-curie-001", "code-davinci-002"])
    parser.add_argument('--score_batch_size', type=int, default=-1)
    parser.add_argument('--style_template', type=str, default="default")

    parser.add_argument('--do_random_scoring', default=False, action="store_true")
    parser.add_argument('--do_fixedrand_scoring', default=False, action="store_true")
    parser.add_argument('--do_negative_control', default=False, action="store_true")

    parser.add_argument('--random_score_seed', default=0, type=int)
    parser.add_argument('--do_bertscore_scoring', default=False, action="store_true")
    parser.add_argument('--do_clsscore_scoring', default=False, action="store_true")

    parser.add_argument('--selection_strategy', type=str, default="nn", help="shot selection strategy",
                            choices=["nn", "furthest", "mmr",])
    parser.add_argument('--mmr_lambda', type=float, default=0.5, help="weigts for diversity in mmr")

class ScoringHelper:
    def __init__(self, strategy):
        self.strategy = strategy

    @classmethod
    def from_strategy(cls, strategy):
        if strategy == "api-question":
            return QuestionScoringHelper(strategy)
        elif strategy == "esnli-promptquestion":
            return ESNLIPromptQuestion(strategy)
        else:
            raise RuntimeError("Not Implemented Yet")

    def score_by_last_segment(self, response, agg_style):
        completion_offset = response["prompt"].rfind(DEFAULT_TRAIN_SEP) + 2
        tokens = response["logprobs"]["tokens"]
        logprobs = response["logprobs"]["token_logprobs"]
        token_offset = response["logprobs"]["text_offset"]

        completion_start_tok_idx = token_offset.index(completion_offset)
        # print(tokens[completion_start_tok_idx:])
        if agg_style == "sum":
            return sum(logprobs[completion_start_tok_idx:])
        elif agg_style == "avg":
            return sum(logprobs[completion_start_tok_idx:]) / (len(logprobs) - completion_start_tok_idx)
        else:
            raise RuntimeError("Not Implemented Yet")


    @staticmethod
    def pair_for_bertscore(strategy, test_ex, train_ex):
        if strategy == "api-question":
            return (train_ex["question"], test_ex["question"])
        # dirty implementtion
        elif strategy == "esnli-promptquestion":
            train_q = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?'.format(
                train_ex["premise"], train_ex["hypothesis"])
            test_q = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?'.format(
                test_ex["premise"], test_ex["hypothesis"])
            return (train_q, test_q)
        else:
            raise RuntimeError("Not Implemented Yet")

class QuestionScoringHelper(ScoringHelper):
    def get_prompt(self, test_ex, train_ex):
        return "Q: {}{}Q: {}".format(train_ex["question"], DEFAULT_TRAIN_SEP, test_ex["question"])

    def score_response(self, response, agg_style):
        return self.score_by_last_segment(response, agg_style)

class ESNLIPromptQuestion(ScoringHelper):
    def get_prompt(self, test_ex, train_ex):
        train_q = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?'.format(
                train_ex["premise"], train_ex["hypothesis"])
        test_q = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?'.format(
            test_ex["premise"], test_ex["hypothesis"])
        return "{}{}{}".format(train_q, DEFAULT_TRAIN_SEP, test_q)

    def score_response(self, response, agg_style):
        return self.score_by_last_segment(response, agg_style)


def selective_train_test_pairwise_score_filename_func(args):
    if args.do_random_scoring:
        raise RuntimeError("No cache for random scores")
    elif args.do_fixedrand_scoring:
        raise RuntimeError("No cache for random scores")
    elif args.do_bertscore_scoring:
        return "misc/{}--tr{}-{}-dv{}-{}--bertscore{}--scores.json".format(args.task,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy)
    elif args.do_clsscore_scoring:
        return "misc/{}--tr{}-{}-dv{}-{}--clsscore{}--scores.json".format(args.task,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy)
    else:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--str{}--scores.json".format(args.task, args.score_engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy)


def selective_train_train_pairwise_score_filename_func(args):
    if args.do_random_scoring:
        raise RuntimeError("No cache for random scores")
    elif args.do_fixedrand_scoring:
        raise RuntimeError("No cache for random scores")
    elif args.do_bertscore_scoring:
        return "misc/{}--tr{}-{}-tr{}-{}--bertscore{}--scores.json".format(args.task,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_train, args.slice_train + args.num_train,
            args.score_strategy)
    elif args.do_clsscore_scoring:
        return "misc/{}--tr{}-{}-tr{}-{}--clsscore{}--scores.json".format(args.task,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_train, args.slice_train + args.num_train,
            args.score_strategy)
    else:
        return "misc/{}--eng{}--tr{}-{}-tr{}-{}--str{}--scores.json".format(args.task, args.score_engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_train, args.slice_train + args.num_train,
            args.score_strategy)

def selective_query_result_filename_func(args):
    if args.selection_strategy == "nn":
        leading_str = args.task
    elif args.selection_strategy == "furthest":
        leading_str = args.task + "--" + args.selection_strategy
    elif args.selection_strategy == "mmr":
        leading_str = args.task + "--" + args.selection_strategy + str(args.mmr_lambda)
    else:
        raise RuntimeError("Not yet implemented")
    if args.do_random_scoring:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--randscore{}--shot{}--numsamp{}--temp{}--sty{}--predictions.json".format(leading_str, args.engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.random_score_seed,
            args.num_shots,
            args.num_samples,
            args.temperature,
            args.style_template)
    if args.do_fixedrand_scoring:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--fixedrandscore{}--shot{}--numsamp{}--temp{}--sty{}--predictions.json".format(leading_str, args.engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.random_score_seed,
            args.num_shots,
            args.num_samples,
            args.temperature,
            args.style_template)
    elif args.do_bertscore_scoring:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--bertscore{}--shot{}--numsamp{}--temp{}--sty{}--predictions.json".format(leading_str, args.engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy,
            args.num_shots,
            args.num_samples,
            args.temperature,
            args.style_template)
    elif args.do_clsscore_scoring:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--clsscore{}--shot{}--numsamp{}--temp{}--sty{}--predictions.json".format(leading_str, args.engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy,
            args.num_shots,
            args.num_samples,
            args.temperature,
            args.style_template)
    else:
        return "misc/{}--eng{}--tr{}-{}-dv{}-{}--str{}--scoreeng{}--shot{}--numsamp{}--temp{}--sty{}--predictions.json".format(leading_str, args.engine,
            args.slice_train, args.slice_train + args.num_train,
            args.slice_dev, args.slice_dev + args.num_dev,
            args.score_strategy,
            args.score_engine,
            args.num_shots,
            args.num_samples,
            args.temperature,
            args.style_template)


def determine_llm_score_aggregation_style(args):
    if args.selection_strategy == "nn":
        return "avg"
    if args.selection_strategy == "mmr":
        return "avg"
    else:
        raise RuntimeError("Not Implemented Yet")

def process_llm_scores(args, scores, make_symetric=False):
    scores = np.array(scores)
    scores = softmax(scores, axis=1)
    if make_symetric:
        scores_t = np.transpose(scores)
        scores = (scores + scores_t) / 2

    return scores

# should returns n_test * n_train matrix
# scores must be: the larger, the better, it s score not distance
def score_train_test_pairs(args, test_data, train_data):
    if args.do_random_scoring:
        np.random.seed(args.random_score_seed)
        return np.random.rand(len(test_data), len(train_data))
    if args.do_fixedrand_scoring:
        np.random.seed(args.random_score_seed)
        return np.random.rand(len(train_data)) * np.ones((len(test_data), 1))
    elif args.do_bertscore_scoring:
        return score_data_pairs_by_bertscore(args, test_data, train_data, selective_train_test_pairwise_score_filename_func)
    elif args.do_clsscore_scoring:
        return score_data_pairs_by_clsscore(args, test_data, train_data, selective_train_test_pairwise_score_filename_func)
    else:
        agg_style = determine_llm_score_aggregation_style(args)
        print("Train Test Score AGG style:", agg_style)
        scores = score_data_pairs_by_llm(args, test_data, train_data, selective_train_test_pairwise_score_filename_func, agg_style)
        scores = process_llm_scores(args, scores, False)
        return scores

def score_train_train_pairs(args, train_data):
    if args.do_random_scoring:
        np.random.seed(args.random_score_seed)
        return np.random.rand(len(train_data), len(train_data))
    if args.do_fixedrand_scoring:
        np.random.seed(args.random_score_seed)
        return np.random.rand(len(train_data)) * np.ones((len(train_data), 1))
    elif args.do_bertscore_scoring:
        return score_data_pairs_by_bertscore(args, train_data, train_data, selective_train_train_pairwise_score_filename_func)
    elif args.do_clsscore_scoring:
        return score_data_pairs_by_clsscore(args, train_data, train_data, selective_train_train_pairwise_score_filename_func)
    else:
        agg_style = determine_llm_score_aggregation_style(args)
        print("Train Internal Score AGG style:", agg_style)
        scores = score_data_pairs_by_llm(args, train_data, train_data, selective_train_train_pairwise_score_filename_func, agg_style)
        scores = process_llm_scores(args, scores, True)
        return scores

def score_data_pairs_by_bertscore(args, test_data, train_data, cache_filename_func):
    cache_filename = cache_filename_func(args)
    if os.path.exists(cache_filename):
        print("Cached Predictions Detected:", cache_filename)
        return np.array(read_json(cache_filename))

    from evaluate import load
    bertscore = load("bertscore")

    scores = []
    for test_ex in tqdm(test_data, desc="bertscore"):
        predictions = []
        references = []
        for train_ex in train_data:
            train_ref, test_pred = ScoringHelper.pair_for_bertscore(args.score_strategy, test_ex, train_ex)
            predictions.append(test_pred)
            references.append(train_ref)

        results = bertscore.compute(predictions=predictions, references=references, lang="en")
        scores.append(results["f1"])

    dump_json(scores, cache_filename)
    return np.array(scores)


def score_data_pairs_by_clsscore(args, test_data, train_data, cache_filename_func):
    cache_filename = cache_filename_func(args)
    if os.path.exists(cache_filename):
        print("Cached Predictions Detected:", cache_filename)
        if args.force_override:
            print("Force Overriding Previous Predictions")
        else:
            return np.array(read_json(cache_filename))

    from cls_score import ClsEmbeddingScore
    scorer = ClsEmbeddingScore()

    test_sents = []
    train_sents = []
    for test_ex in test_data:
        _, test_pred = ScoringHelper.pair_for_bertscore(args.score_strategy, test_ex, train_data[0])
        test_sents.append(test_pred)

    for train_ex in train_data:
        train_ref, _ = ScoringHelper.pair_for_bertscore(args.score_strategy, test_data[0], train_ex)
        train_sents.append(train_ref)

    scores = scorer.compute_pairwise(test_sents, train_sents)

    dump_json(scores, cache_filename)
    return np.array(scores)

def score_data_pairs_by_llm(args, test_data, train_data, cache_filename_func, agg_style):
    score_helper = ScoringHelper.from_strategy(args.score_strategy)
    # scoring
    prompts_to_score = []
    for test_ex in test_data:
        prompts_for_ex = []
        for train_ex in train_data:
            prompts_for_ex.append(score_helper.get_prompt(test_ex, train_ex))
        prompts_to_score.append(prompts_for_ex)

    cache_filename = cache_filename_func(args)

    if args.run_scoring:
        _saved_num_samples = args.num_samples
        _saved_temp = args.temperature
        _saved_engine = args.engine
        _saved_batch_size = args.batch_size
        args.num_samples = 1
        args.temperature = 0.0
        args.engine = args.score_engine
        args.batch_size = args.score_batch_size
        responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_score)
        args.num_samples = _saved_num_samples
        args.temperature = _saved_temp
        args.engine = _saved_engine
        args.batch_size = _saved_batch_size
    else:
        responses = read_json(cache_filename)

    # responses = read_json(selective_pairwise_score_filename_func(args))
    scores = []
    for resps_by_example in responses:
        resps_by_example = flatten_nested_list(resps_by_example)
        scores_by_example = [score_helper.score_response(x, agg_style) for x in resps_by_example]
        scores.append(scores_by_example)

    return scores

def mmr_style_selection(args, scores_by_ex, train_internal_scores):
    if args.selection_strategy != "mmr":
        assert (not args.do_bertscore_scoring and not args.do_clsscore_scoring)
    assert train_internal_scores is not None
    selected_idx = []
    num_train = len(train_internal_scores)
    if args.mmr_lambda == 0.0:
        np.random.seed(args.random_score_seed)
    for _ in range(args.num_shots):
        if not selected_idx and args.mmr_lambda == 0.0:
            next_idx = np.random.randint(num_train)
            selected_idx.append(next_idx)
            continue
        score_idx_pairs = []
        for attempted_idx in range(num_train):
            if attempted_idx in selected_idx:
                continue
            sim_to_ex = scores_by_ex[attempted_idx]
            if selected_idx:
                max_sim_to_shots = max([train_internal_scores[shot_id][attempted_idx] for shot_id in selected_idx])
            else:
                max_sim_to_shots = 0
            score = args.mmr_lambda * sim_to_ex - (1 - args.mmr_lambda) * max_sim_to_shots
            score_idx_pairs.append((score, attempted_idx))
        next_idx = sorted(score_idx_pairs, key=lambda x: x[0], reverse=True)[0][1]
        selected_idx.append(next_idx)
    return selected_idx

def shot_selection(args, scores_by_ex, train_internal_scores):
    if args.selection_strategy == "nn":
        selected_idx = sorted(enumerate(scores_by_ex), key=lambda x: x[1], reverse=True)[:args.num_shots]
        selected_idx = [x[0] for x in selected_idx]
    elif args.selection_strategy == "furthest":
        selected_idx = sorted(enumerate(scores_by_ex), key=lambda x: x[1])[:args.num_shots]
        selected_idx = [x[0] for x in selected_idx]
    elif args.selection_strategy == "mmr":
        selected_idx = mmr_style_selection(args, scores_by_ex, train_internal_scores)
    else:
        raise RuntimeError("Not implemented yet")
    return selected_idx

def predict_framework(args):
    train_data, test_data = load_train_test_set(args)

    pairwise_scores = score_train_test_pairs(args, test_data, train_data)
    if args.selection_strategy == "mmr":
        train_internal_scores = score_train_train_pairs(args, train_data)
    elif args.selection_strategy == "nn":
        train_internal_scores = score_train_train_pairs(args, train_data)
    else:
        train_internal_scores = None
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)

    prompts_to_complete = []
    for test_ex, scores_by_ex in zip(test_data, pairwise_scores):
        selected_idx = shot_selection(args, scores_by_ex, train_internal_scores)
        selected_shots = [train_data[i] for i in selected_idx]
        prompts_to_complete.append(
            [task_helper.prompt_func(test_ex, selected_shots)]
        )

    task_max_tokens = task_helper.get_completion_length()
    cache_filename = selective_query_result_filename_func(args)
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def score_of_completion(response):
    completion_offset = len(response["prompt"])
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    if completion_offset in token_offset:
        completion_start_tok_idx = token_offset.index(completion_offset)
    else:
        completion_start_tok_idx = next(filter(lambda x: token_offset[x] >= completion_offset, range(len(token_offset))), len(token_offset))

    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>", completion_start_tok_idx)
    elif "\n" in tokens[completion_start_tok_idx:]:
        completion_end_tok_idx = tokens.index("\n", completion_start_tok_idx)
    else:
        completion_end_tok_idx = len(tokens)
    # completion_end_tok_idx = tokens.index("<|endoftext|>")
    # return len(tokens) - completion_start_tok_idx

    tok_scores = response["logprobs"]["token_logprobs"][completion_start_tok_idx:completion_end_tok_idx + 1]
    toks = response["logprobs"]["tokens"][completion_start_tok_idx:completion_end_tok_idx + 1]

    tok_scores = np.array(tok_scores)
    return tok_scores.sum(), tok_scores.mean()

def confidence_of_completion(response, answer_hint):
    completion_offset = len(response["prompt"])
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    # answer_offset = response["text"]
    lower_text =  response["text"].lower()
    lower_hint = answer_hint.lower()
    if lower_hint in lower_text:
        answer_offset = completion_offset + lower_text.index(lower_hint) + len(lower_hint)
    else:
        answer_offset = completion_offset

    if answer_offset in token_offset:
        answer_start_tok_idx = token_offset.index(answer_offset)
    elif answer_offset >= token_offset[-1]:
        return 0.
    else: 
        answer_start_tok_idx = next(filter(lambda x: token_offset[x] >= answer_offset, range(len(token_offset))))

    if "<|endoftext|>" in tokens:
        answer_end_tok_idx = tokens.index("<|endoftext|>", answer_start_tok_idx)
    elif "\n" in tokens[answer_start_tok_idx:]:
        answer_end_tok_idx = tokens.index("\n", answer_start_tok_idx)
    else:
        answer_end_tok_idx = len(tokens)
    if tokens[answer_end_tok_idx - 1].strip() == '.':
        answer_end_tok_idx = answer_end_tok_idx - 1

    # completion_end_tok_idx = tokens.index("<|endoftext|>")
    # return len(tokens) - completion_start_tok_idx

    tok_scores = response["logprobs"]["token_logprobs"][answer_start_tok_idx:answer_end_tok_idx ]
    toks = response["logprobs"]["tokens"][answer_start_tok_idx:answer_end_tok_idx ]
    tok_scores = np.array(tok_scores)
    conf = np.exp(np.sum(tok_scores))
    # print("".join(toks), conf)

    return conf

def run_evaluation(args, test_data, responses, print_perplexity=True, return_verbose=False):
    evaluator = get_task_evaluator(args.task)

    prompting_style = "eandp"

    max_sample_num = max([len(x) for x in responses]) if responses else 0
    num_eval_samples = args.num_eval_samples if args.num_eval_samples > 0 else max_sample_num

    predictions = [
        [Prediction(x["text"], x["prompt"], *score_of_completion(x)) for x in completions[:num_eval_samples]] for completions in responses
    ]

    if args.do_print:
        TaskEvaluator.do_printing = True

    sums = np.array([[x.logprob for x in preds] for preds in predictions])
    norms = np.array([[x.norm_logprob for x in preds] for preds in predictions])
    avg_sum = sums.mean(axis=1).mean(axis=0)
    avg_norm = norms.mean(axis=1).mean(axis=0)

    if print_perplexity:
        print("AVG Logprob: {:.4f}".format(avg_sum))
        print("AVG Norm Logprob: {:.4f}".format(avg_norm))

    eval_results = evaluator.evaluate(predictions, test_data, prompting_style, return_verbose=return_verbose)
    eval_results["avg_logprob"] = sums.mean(axis=1).mean(axis=0)
    eval_results["avg_normlogprob"] = norms.mean(axis=1).mean(axis=0)
    if return_verbose:
        confidences = [
            [confidence_of_completion(x, evaluator.ANSWER_HINT) for x in completions[:num_eval_samples]] for completions in responses
        ]
        avg_conf = np.array(confidences).mean(axis=1).mean(axis=0)
        eval_results["avg_confidence"] = avg_conf

    return eval_results


def eval_framework(args):
    _, test_data = load_train_test_set(args)
    responses = read_json(selective_query_result_filename_func(args))
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_selecive_args(parser)
    args = parser.parse_args()
    if args.score_engine is None:
        args.score_engine = args.engine
    if args.score_batch_size == -1:
        args.score_batch_size = args.batch_size

    assert args.task is not None
    config_args_and_api(args)
    if args.run_prediction:
        predict_framework(args)
    else:
        eval_framework(args)

if __name__=="__main__":
    main()

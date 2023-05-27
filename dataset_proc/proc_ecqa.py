## ecqa
import json
import random
import sys
from os.path import join

def read_jsonline(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(x) for x in lines]

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)
IDXLISTS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

def encode_question_and_choices(ex):
    question_line = "{}".format(ex["question"])
    choice_lines = ["Answer Choices:"] + [ idx + " " + c for (idx, c) in zip(IDXLISTS, ex["choices"])]
    return "\n".join([question_line] + choice_lines)

def answer_index(ex):
    assert ex["answer"] in ex["choices"]
    return IDXLISTS[ex["choices"].index(ex["answer"])]

def get_standard_prompt_and_completion(shots, ex):
    showcase_examples = [
            "{}\nA: The answer is {}.\n".format(encode_question_and_choices(s), answer_index(s)) for s in shots
    ]
    input_example = "{}\nA:".format(encode_question_and_choices(ex))
    
    prompt = "\n".join(showcase_examples + [input_example])
    completion = "The answer is {}.".format(answer_index(ex))
    return prompt, completion

def get_eandp_prompt(shots, ex):
    showcase_examples = [
            "{}\nA: {} So the answer is {}.\n".format(encode_question_and_choices(s), s["explanation"], answer_index(s)) for s in shots
        ]
    input_example = "{}\nA:".format(encode_question_and_choices(ex))

    prompt = "\n".join(showcase_examples + [input_example])
    completion = "{} So the answer is {}.".format(ex["explanation"], answer_index(ex))
    return prompt, completion


def make_ecqa_dataset(split):
    # print(sys.version)
    # control seed
    assert sys.version_info.major == 3 and  sys.version_info.minor == 8

    lines = read_jsonline(f'raw_data/ecqa_{split}.jsonl')

    idx_ex_pairs = list(enumerate(lines))

    random.seed(123)
    random.shuffle(idx_ex_pairs)

    proced_examples = []
    for idx, ex in idx_ex_pairs:
        pure_question = ex["question"]
        choices = ex["choices"]
        answer = answer_index(ex)
        text_answer = ex["answer"]
        explanation = ex["explanation"]

        question = encode_question_and_choices(ex)
        completion = "{} So the answer is {}.".format(ex["explanation"], answer_index(ex))

        question 
        proced_examples.append(
            {
                "id": f"{split}-{idx}",
                "base_question": pure_question,
                "choices": choices,
                "answer": answer,
                "base_answer": text_answer,
                "explanation": explanation,
                "question": question,
                "completion": completion,
            }
        )


    print(len(proced_examples))
    with open(f"../data/ecqa_{split}.json", "w") as f:
        json.dump(proced_examples, f)


make_ecqa_dataset("train")
make_ecqa_dataset("test")

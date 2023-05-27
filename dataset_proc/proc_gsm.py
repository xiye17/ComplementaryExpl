import json
import sys
import random
import re

def read_jsonline(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if (not x.startswith("#")) and len(x) > 0]
    return [json.loads(x) for x in lines]

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    try:
        with open(fname, 'w', encoding='utf-8') as f:
            return json.dump(obj, f, indent=indent)
    except KeyboardInterrupt:
        print("Not allowed to stop while writing json")

def canonicalize_line(l):
    if not l.endswith("."):
        l = l + "."

    # remove computation
    l = re.sub('\<\<.+\>\>', '', l)

    l = l.replace("=", " = ")
    l = l.replace("+", " + ")
    l = l.replace("*", " * ")

    new_cs = []
    for i, c in enumerate(l):
        if c == "-" and l[i - 1].isdigit():
            new_cs.append(" - ")
        else:
            new_cs.append(c)
    l = "".join(new_cs)

    lst = re.findall(r'[0-9\.]+/[0-9\.]+', l)
    for eq in lst:
        a_str, b_str = eq.split("/")
        if b_str.endswith("."):
            b_str = b_str.rstrip(".")
        try:
            a, b = float(a_str), float(b_str)
        except:
            break
        if a >= b:
            l = l.replace(eq, a_str + " / " + b_str)

    l = " ".join([x for x in l.split() if x])
    return l

def sort_answer(answer_line):
    lines = answer_line.split("\n")
    assert "####" in lines[-1]
    answer = lines[-1].replace("####", "").strip()
    # print(answer)

    expl_lines = lines[:-1]

    expl_lines = [canonicalize_line(l) for l in expl_lines]
    expl = " ".join(expl_lines)

    comp_lines = expl_lines + ["The answer is " + answer + "."]
    comp = " ".join(comp_lines)

    return answer, expl, comp


def main(split):
    lines = read_jsonline(f"raw_data/gsm_{split}.jsonl")
    assert sys.version_info.major == 3 and  sys.version_info.minor == 8

    idx_ex_pairs = list(enumerate(lines))

    random.seed(123)
    random.shuffle(idx_ex_pairs)

    proced_examples = []
    for idx, ex in idx_ex_pairs:
        question = ex["question"]
        question = " ".join([x for x in question.split() if x])
        answer_line = ex["answer"]
        ans, expl, comp = sort_answer(answer_line)
        proced_examples.append(
            {
                "id": f"{split}-{idx}",
                "question": question,
                "answer": ans,
                "explanation": expl,
                "completion": comp,
            }
        )

    dump_json(proced_examples, f"../data/gsm_{split}.json")


main("train")
main("test")

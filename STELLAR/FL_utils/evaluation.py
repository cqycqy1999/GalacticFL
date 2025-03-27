import os
import json
import random

import torch
import datasets
import numpy as np

from transformers import GenerationConfig

model_type = 'llama'
datasets.utils.logging.set_verbosity_error()
device_map = "auto"
max_new_token: int = 32
verbose: bool = False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def global_evaluation(model, tokenizer, prompter, dev_data_path, output_dir_path):
    # TODO 需要补全在humaneval(codealpaca)和gsm8k(gsm8k-test)上的写法、MMLU benchmark上的检测
    if "Dolly" in output_dir_path:
        return evaluate_mmlu(model, tokenizer, prompter, dev_data_path)
    elif "CodeAlpaca" in output_dir_path:
        return evaluate_humaneval(model, tokenizer, prompter, dev_data_path)
    else:
        return evaluate_gsm8k_test(model, tokenizer, prompter, dev_data_path)
    

def evaluate_mmlu(model, tokenizer, prompter, dev_data_path):
    data_class = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 
                  'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
                  'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 
                  'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 
                  'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
                  'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 
                  'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 
                  'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 
                  'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 
                  'machine_learning', 'management', 'marketing', 'medical_genetics', 
                  'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
                  'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 
                  'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 
                  'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    right_count_dict = dict.fromkeys(data_class, 0)
    total_count_dict = dict.fromkeys(data_class, 0)
    acc_count_dict = dict.fromkeys(data_class, 0)
    with open(dev_data_path, 'r') as f:
        test_set = json.load(f)
    count = 0

    if model_type == 'llama':
        sampling = GenerationConfig(
            do_sample=True,
            temperature=0.2,
            top_p=0.6,
            top_k=30,
            num_beams=1,
            max_new_tokens=max_new_token,
            early_stopping=True,
        )

    if model_type == 'gpt2':
        sampling = GenerationConfig(
            bos_token_id=50256,
            eos_token_id=50256,
            _from_model_config=True,
        )

    for data_point in test_set:
        count += 1
        target = data_point["output"]
        class_test_set = data_point["class"]

        tgt_ans_idx = target.replace('The answer is: ', '').split('. ')[0]
        tgt_ans = target.replace('The answer is: ', '').split('. ')[1]

        test_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            'The answer is: ',
        )

        with torch.autocast("cuda"):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            input = inputs["input_ids"].to('cuda')
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input,
                    generation_config=sampling,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_token,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
            split = prompter.template["response_split"]
            ans = generation_output_decoded.split(split)[-1].strip()
            if verbose:
                print('-------------------')
                print(test_prompt)
                print(tgt_ans)
                print(tgt_ans_idx)
                print(ans)
            if tgt_ans_idx + '.' in ans or tgt_ans in ans:
                right_count_dict[class_test_set] += 1
            total_count_dict[class_test_set] += 1

    mean_acc = 0

    for key in acc_count_dict.keys():
        tmp = right_count_dict[key] / total_count_dict[key]
        mean_acc += tmp
        acc_count_dict[key] = tmp
    mean_acc /= len(acc_count_dict.keys())
    csv_data = [right_count_dict, total_count_dict, acc_count_dict]

    if verbose:
        print(right_count_dict)
    print('Acc: ', acc_count_dict)
    print('========== Accuracy ==========')
    print(mean_acc)

    return mean_acc

def evaluate_humaneval(model, tokenizer, prompter, dev_data_path):
    pass

def evaluate_gsm8k_test(model, tokenizer, prompter, dev_data_path):
    pass
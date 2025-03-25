import os
import copy

import fire
from typing import List
from tqdm import tqdm

import torch
from utils.prompter import Prompter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def STELLAR(
        # model/data argparse
        global_model: str = "decapoda-research/llama-7b-hf",
        data_path: str = "./data", #TODO
        output_dir_path: str = "./", #TODO
        # FL server argparse
        client_selection_strategy: str = "random",
        client_selection_ratio: float = 1,
        num_rounds: int = 5,
        num_clients: int = 10,
        niid: float = 0.3,
        # FL client argparse
        local_batch_size: int = 128,
        local_micro_batch_size: int = 16, # TODO
        local_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_setsize: int = 9,
        local_save_steps: int = 3, # TODO
        cutoff_len: int = 512, # TODO 这个值可能也是一个因素
        # LoRA argparse
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # TODO lora_target_layer ？是否要对entire model的每一层加索引；不然怎么选择哪些层？
        # LLM argparse
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,
        prompt_template_name: str = "alpaca",
        # aggragation argparse
        stacking: bool = False, # if True, is FloRA;Flase and uniform rank level is FedIT
        # evaluation argparse
        dev_data_path: str = './mmlu_test_1444.jsonl', # TODO
        # heterogeneous case argparse
        heter_adapter: bool = False,
        local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4], 
        zero_padding: bool = False,
        Adalora: bool = False,
        full_finetuning: bool = False, # TODO 这个还不知道啥意思
):     
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated LoRA Finetuning LLM:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir_path: {output_dir_path}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_ratio: {client_selection_ratio}\n"
            f"num_rounds: {num_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"niid: {niid}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_epochs: {local_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_setsize: {local_val_setsize}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
            f"prompt_template_name: {prompt_template_name}\n"
            f"stacking: {stacking}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"heter_adapter: {heter_adapter}\n"
            f"local_ranks: {local_ranks}\n"
            f"zero_padding: {zero_padding}\n"
            f"Adalora: {Adalora}\n"
            f"full: {full_finetuning}\n"
        )

    data_path = os.path.join(data_path, "Dolly", str(num_clients), str(niid).replace(".","")) # TODO
    assert (os.path.exists(data_path)), f"Please generate the data files for each client."

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter

    # TODO 分布式进程下的处理，fl？
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        load_in_8bit = False,
        torch_dtype = torch.float32,
        device_map = device_map,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(global_model, token='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')
    # 当需要padding时， 请使用token ID = 0
    tokenizer.pad_token_id = (
        0
    )

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, 
            truncation=True, # 截断，和max_length一起使用
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,)
        if(
            result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point): # TODO 先这样
        if "Dolly" in data_path:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
                data_point["response"],
                data_point["category"],
            )
        elif "CodeAlpaca" in data_path:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            # if GSM8K in data_path
            full_prompt = prompter.generate_prompt(
                data_point["question"],
                data_point["answer"],
            )
        
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    # lora adapterrs aggregation
    if full_finetuning == False:
        if stacking == False:
            if zero_padding:
                config = LoraConfig(
                    base_model_name_or_path = global_model,
                    r = max(local_ranks),
                    lora_alpha = lora_alpha * max(local_ranks),
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
            else:
                config = LoraConfig(
                    base_model_name_or_path = global_model,
                    r = lora_r,
                    lora_alpha = lora_alpha,
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
        else:
            config = LoraConfig(
                base_model_name_or_path = global_model,
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules = lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )

    
if __name__ == "__main__":
    fire.Fire(STELLAR)

import os
import copy

import json
import fire
from typing import List
from tqdm import tqdm

import torch
from utils.prompter import Prompter
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, AdaLoraConfig, PeftModel ,get_peft_model
from FL_utils import (GeneralClient, FedAvg, client_selection,
                      global_evaluation)


# 添加内存监控函数
def log_gpu_memory_usage(tag, client_id = None, epoch = None, output_dir_path = None):
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024) 

    log_dict = {
        "timestamp": time.time(),
        "tag": tag,
        "client_id": client_id,
        "epoch": epoch,
        "allocated_MB": allocated,
        "reserved_MB": reserved,
        "max_allocated_MB": max_allocated,
    }

    # 创建日志目录并追加数据
    if output_dir_path:
        os.makedirs(os.path.join(output_dir_path, "memory_logs"), exist_ok=True)
        log_path = os.path.join(output_dir_path, "memory_logs", "gpu_memory_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_dict) + "\n")

    return log_dict
   

def STELLAR(
        # model/data argparse
        global_model: str = "decapoda-research/llama-7b-hf",
        data_path: str = "./data", #TODO
        output_dir_path: str = "./output", #TODO
        # FL server argparse
        client_selection_strategy: str = "random",
        client_selection_ratio: float = 1,
        num_communication_rounds: int = 3, 
        num_clients: int = 10,
        niid: float = 0.1,
        # FL client argparse
        local_batch_size: int = 128,
        local_micro_batch_size: int = 16, # TODO
        local_epochs: int = 1,
        local_learning_rate: float = 3e-4,
        local_val_setsize: int = 0,
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
            f"num_communication_rounds: {num_communication_rounds}\n"
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
            f"is_stacking: {stacking}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"heter_adapter: {heter_adapter}\n"
            f"local_ranks: {local_ranks}\n"
            f"is_zero_padding: {zero_padding}\n"
            f"is_Adalora: {Adalora}\n"
            f"is_FFT: {full_finetuning}\n"
        )

    # data_path = os.path.join(data_path, "Dolly", str(num_clients), str(niid).replace(".","")) # TODO
    assert (os.path.exists(data_path)), f"Please generate the data files for each client."

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)

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
    
    def generate_and_tokenize_prompt(data_point): # TODO 先这样;在prompter.py的generate_prompt函数中，有三个参数，分别是instruction，input和label（其实就是输出啦）
        if "Dolly" in data_path:
            instruction = f"{data_point['instruction']} {data_point['category']}"
            full_prompt = prompter.generate_prompt(
                instruction,
                data_point["context"],
                data_point["response"],
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
                None,
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

    # lora adapterrs aggregation Server
    if not full_finetuning:
        if not stacking:
            if zero_padding:
                config_ori = LoraConfig(
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
            config_ori = LoraConfig(
                base_model_name_or_path = global_model,
                # r = lora_r * num_clients,
                r = sum(local_ranks) if heter_adapter else lora_r * num_clients, # TODO
                lora_alpha = lora_alpha * num_clients,
                target_modules = lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )

    # TODO 这个地方可能会导致模型被拆分 会和分裂联邦相关
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    if "Dolly" in data_path:
        dataset_name = "Dolly"
    elif "CodeAlpaca" in data_path:
        dataset_name = "CodeAlpaca"
    else:
        dataset_name = "GSM8K"
    output_dir_path = os.path.join(output_dir_path, dataset_name, str(num_clients), str(niid).replace(".",""))
    
    acc_list = []

    for epoch in tqdm(range(num_communication_rounds)):
        print("Conducting the client selection...")
        selected_clients_set = client_selection(num_clients, client_selection_ratio, client_selection_strategy, epoch)

        for client_id in selected_clients_set:
            if not full_finetuning:
                if Adalora:
                    config = AdaLoraConfig(
                        r = local_ranks[client_id],
                        lora_alpha = 2 * local_ranks[client_id],
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                        base_model_name_or_path = global_model 
                    )
                    model_client = copy.deepcopy(model)
                    model_client = get_peft_model(model_client, config)
                else:
                    if stacking:
                        if heter_adapter:
                            config = LoraConfig(
                                r = local_ranks[client_id],
                                lora_alpha = 2 * local_ranks[client_id],
                                target_modules = lora_target_modules,
                                lora_dropout = lora_dropout,
                                bias = "none",
                                task_type = "CAUSAL_LM",
                                base_model_name_or_path = global_model
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                        else:
                            config = LoraConfig(
                                r = lora_r,
                                lora_alpha = lora_alpha,
                                target_modules = lora_target_modules,
                                lora_dropout = lora_dropout,
                                bias = "none",
                                task_type = "CAUSAL_LM",
                                base_model_name_or_path = global_model
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                    else:
                        if heter_adapter:
                            config = LoraConfig(
                                r = local_ranks[client_id],
                                lora_alpha = 2 * local_ranks[client_id],
                                target_modules = lora_target_modules,
                                lora_dropout = lora_dropout,
                                bias = "none",
                                task_type = "CAUSAL_LM",
                                base_model_name_or_path = global_model
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                        else:
                            model_client = model
            else:   
                model_client = model

            client = GeneralClient(client_id, model_client, data_path, output_dir_path)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.prepare_local_dataset(generate_and_tokenize_prompt, local_val_setsize)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_epochs,
                local_learning_rate,
                group_by_length,
                ddp,
            )

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            # TODO 训练前记录mem
            log_gpu_memory_usage(f"client_before_training", client_id, epoch, output_dir_path)
            with profile(
                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True
            ) as prof:
                print("Local training starts ... ")
                client.train()
            
            # TODO 训练后记录mem
            log_gpu_memory_usage(f"client_after_training", client_id, epoch, output_dir_path)

            # print("Local training starts ... ")
            # client.train()
            # TODO 保存本轮训练的性能分析结果
            os.makedirs(os.path.join(output_dir_path, "memory_logs", "profiles"), exist_ok=True)
            prof.export_chrome_trace(os.path.join(output_dir_path, "memory_logs", "profiles", f"client_{client_id}_epoch_{epoch}_trace.json"))

            print("\nTerminating the local training of Client_{}".format(client_id))
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            del client

        print("\nCollecting the weights of clients and Conducting the global model aggregation...")

        # TODO 聚合前记录全局模型的mem
        log_gpu_memory_usage(f"server_before_aggregation", None, epoch, output_dir_path)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True
        ) as prof:
            model = FedAvg(model, selected_clients_set, 
                           output_dir_path, local_dataset_len_dict,
                           epoch, stacking, lora_r, heter_adapter, local_ranks,
                           zero_padding, full_finetuning)
            
        # model = FedAvg(model, selected_clients_set, 
        #                output_dir_path, local_dataset_len_dict,
        #                epoch, stacking, lora_r, heter_adapter, local_ranks,
        #                zero_padding, full_finetuning)
        # TODO 聚合后记录全局模型的mem
        log_gpu_memory_usage(f"server_after_aggregation", None, epoch, output_dir_path)

        # TODO 保存聚合过程性能分析
        prof.export_chrome_trace(os.path.join(output_dir_path, "memory_logs", "profiles", f"server_epoch_{epoch}_trace.json"))

        if not full_finetuning:
            if stacking:
                config_ori.save_pretrained(
                    os.path.join(output_dir_path, str(epoch)), load_in_8bit = False, 
                    torch_dtype = torch.float16, device_map = device_map
                )
                model = PeftModel.from_pretrained(
                    model, os.path.join(output_dir_path, str(epoch))
                )
            else:
                torch.save(model.state_dict(), os.path.join(output_dir_path, str(epoch), "adapter_model.bin"))
                config.save_pretrained(
                    os.path.join(output_dir_path, str(epoch)), load_in_8bit = False, 
                    torch_dtype = torch.float16, device_map = device_map
                )
        else:
            config = AutoConfig.from_pretrained(global_model)
            tokenizer.save_pretrained(
                os.path.join(output_dir_path, str(epoch)), 
                load_in_8bit = False, torch_dtype = torch.float32, device_map = device_map
            )
            config.save_pretrained(
                os.path.join(output_dir_path, str(epoch)), 
                load_in_8bit = False, torch_dtype = torch.float32, device_map = device_map
            )

            print("save model...")

        acc = global_evaluation(model, tokenizer, prompter, dev_data_path, output_dir_path)
        print('Acc of Epoch', str(epoch), "is:", acc)
        acc_list.append(acc)

        if stacking:
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(output_dir_path, str(epoch)+'/final'),
                                    load_in_8bit = False,
                                    torch_dtype = torch.float32,
                                    device_map = device_map)
                
        if epoch < (num_communication_rounds-1):
            rm_dir = os.path.join(output_dir_path, str(epoch))
            os.system("rm -rf {xxxxx}".format(xxxxx = rm_dir))

    print(acc_list)
    filename = output_dir_path + "log.txt"
    file = open(filename, 'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Log Saved")
                

if __name__ == "__main__":
    fire.Fire(STELLAR)

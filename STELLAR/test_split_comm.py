import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import gc

def load_model(model_name):
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_hidden_states(model, input_ids, attention_mask, split_at_layer):
    """Get the hidden states from the model."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True
        )

    hidden_states = outputs.hidden_states[split_at_layer]

    return hidden_states

def calculate_communication_size(tensor):
    """计算张量通信量大小"""
    return tensor.element_size() * tensor.nelement()  # GB

def measure_split_communication(model, tokenizer, input_texts, split_layers):
    """测量不同分裂点的通信量."""
    total_layers = len(model.transformer.h)
    print(f"Total layers: {total_layers}")

    # 确保split_layers在有效范围内
    valid_split_layers = [layer for layer in split_layers if 0 <= layer < total_layers]

    # 使用gpu如果可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"use device: {device}")

    results = {}

    for input_text in input_texts:
        print(f"\n测试输入: '{input_text[:50]}...'")
        inputs = tokenizer(input_text, return_tensors="pt")
        token_count = inputs.input_ids.size(1)
        print(f"Token count: {token_count}")

        layer_comm_sizes = []

        for layer in valid_split_layers:
            """清理GPU缓存"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 获取中间激活值
            with torch.no_grad():
                hidden_states = get_hidden_states(
                    model, inputs.input_ids, inputs.attention_mask, layer)
                
            # 计算通信大小
            comm_size = calculate_communication_size(hidden_states)
            layer_comm_sizes.append(comm_size)

            print(f"底层节点持有 {layer}/{total_layers} 层时:")
            print(f"  - 隐藏状态形状: {hidden_states.shape}")
            print(f"  - 通信量: {comm_size/1024/1024:.4f} MB")
            
            # 清理内存
            del hidden_states
            gc.collect()
        
        results[token_count] = {
            'layers': valid_split_layers,
            'comm_sizes': layer_comm_sizes
        }
    
    return results, total_layers

def plot_results(results, total_layers, model_name):
    """绘制通信量结果图"""
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', 'D', '^', '*']
    
    for i, (token_count, data) in enumerate(results.items()):
        layers = data['layers']
        comm_sizes = [size/1024/1024 for size in data['comm_sizes']]  # 转为MB
        
        marker = markers[i % len(markers)]
        plt.plot(layers, comm_sizes, marker=marker, linewidth=2, label=f'{token_count} tokens')
    
    plt.xlabel('底层节点层数', fontsize=12)
    plt.ylabel('通信量 (MB)', fontsize=12)
    plt.title(f'{model_name} 在不同分裂点的通信开销', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    plt.savefig('split_communication_analysis.png')
    print(f"结果图已保存为 'split_communication_analysis.png'")
    
    try:
        plt.show()
    except:
        pass

def main():
    # 模型配置
    model_name = "bigscience/bloomz-560m"
    model, tokenizer = load_model(model_name)
    
    # 测试输入 - 不同长度的文本
    test_inputs = [
        "Hello, how are you today?",  # 短文本
        "This is a medium length text for testing the communication overhead in split neural networks.",  # 中等长度
        "A longer text sample " * 20  # 长文本
    ]
    
    # 测试不同的分裂点：1层、10层、20层（如果模型有足够多层）
    total_layers = len(model.transformer.h)
    split_layers = [1]
    
    if total_layers >= 10:
        split_layers.append(10)
    if total_layers >= 20:
        split_layers.append(20)
    
    # 添加四分位点
    quarter = total_layers // 4
    split_layers.extend([quarter, quarter*2, quarter*3, total_layers-1])
    split_layers = sorted(list(set(split_layers)))  # 去重并排序
    
    # 执行测量
    results, total_layers = measure_split_communication(model, tokenizer, test_inputs, split_layers)
    
    # 绘制结果
    plot_results(results, total_layers, model_name)
    
    # 输出表格摘要
    print("\n通信量摘要 (MB):")
    for token_count, data in results.items():
        layers = data['layers']
        comm_sizes = [size/1024/1024 for size in data['comm_sizes']]
        
        print(f"\n{token_count} tokens:")
        print(f"{'分裂层':<10} | {'通信量 (MB)':<15}")
        print("-" * 30)
        for layer, size in zip(layers, comm_sizes):
            print(f"{layer:<10} | {size:<15.4f}")

if __name__ == "__main__":
    main()
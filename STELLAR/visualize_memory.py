import os
import json

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def visualize_memory_usage(output_dir_path):
    log_path = os.path.join(output_dir_path, "memory_logs", "gpu_memory_log.jsonl")
    if not os.path.exists(log_path):
        raise ValueError(f"Can't read {log_path}")
        return
    
    print(f"Reading memory logs from {log_path}")
    print(f"12345")

    data = []
    with open(log_path, 'r') as fp:
        for line in fp:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invaild line: {line}")

    df = pd.DataFrame(data)
    print(f"Read {len(df)} memory logs")

    save_dir = os.path.join(output_dir_path, "memory_logs", 'plots')
    os.makedirs(save_dir, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    if  'tag' in df.columns and 'client_id' in df.columns:
        # 1 每个客户端的训练时间
        client_times = []
        for epoch in df['epoch'].unique():
            for client in sorted(df[df['client_id'].notna()]['client_id'].unique()):
                before = df[(df['tag'] == 'client_before_training') & 
                           (df['client_id'] == client) & 
                           (df['epoch'] == epoch)]['timestamp'].values
                after = df[(df['tag'] == 'client_after_training') & 
                          (df['client_id'] == client) & 
                          (df['epoch'] == epoch)]['timestamp'].values
                
                if len(before) > 0 or len(after) > 0:
                    training_time = after[0] - before[0]
                    client_times.append({
                        'client_id': f"Client {int(client)}",
                        'epoch': f"Epoch {int(epoch)}",
                        'training_time': training_time
                    })

        if client_times:
            time_df = pd.DataFrame(client_times)
            plt.figure(figsize=(12, 6))
            sns.barplot(x='client_id', y='training_time', hue='epoch', data=time_df)
            plt.title('Training Time per Client and Epoch')
            plt.xlabel('Client ID')
            plt.ylabel('Training Time (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'client_training_times.png'), dpi=300)
            plt.close()
        
        # 2 训练前后内存差异对比
        memory_changes = []
        for tag_type in ['allocated_MB', 'reserved_MB']:
            before_avg = df[df['tag'] == 'client_before_training'][tag_type].mean()
            after_avg = df[df['tag'] == 'client_after_training'][tag_type].mean()
            memory_changes.append({
                'memory_type': 'Before Training' if 'allocated' in tag_type else 'Before Training (Reserved)',
                'memory_value': before_avg
            })
            memory_changes.append({
                'memory_type': 'After Training' if 'allocated' in tag_type else 'After Training (Reserved)',
                'memory_value': after_avg
            })
        
        mem_change_df = pd.DataFrame(memory_changes)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='memory_type', y='memory_value', data=mem_change_df)
        plt.title('Memory Usage Before vs After Training')
        plt.xlabel('Phase')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for i, p in enumerate(plt.gca().patches):
            plt.text(p.get_x() + p.get_width()/2., p.get_height() + 100,
                    f'{p.get_height():.1f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'before_after_memory.png'), dpi=300)
        plt.close()
        
        # 3. 服务器聚合时间分析
        server_times = []
        for epoch in df['epoch'].unique():
            before = df[(df['tag'] == 'server_before_aggregation') & 
                       (df['epoch'] == epoch)]['timestamp'].values
            after = df[(df['tag'] == 'server_after_aggregation') & 
                      (df['epoch'] == epoch)]['timestamp'].values
            
            if len(before) > 0 and len(after) > 0:
                agg_time = after[0] - before[0]
                server_times.append({
                    'epoch': f"Epoch {int(epoch)}",
                    'aggregation_time': agg_time
                })
        
        if server_times:
            server_df = pd.DataFrame(server_times)
            plt.figure(figsize=(8, 5))
            sns.barplot(x='epoch', y='aggregation_time', data=server_df)
            plt.title('Server Aggregation Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Aggregation Time (seconds)')
            
            # 添加数值标签
            for i, p in enumerate(plt.gca().patches):
                plt.text(p.get_x() + p.get_width()/2., p.get_height() + 0.1,
                        f'{p.get_height():.2f}s', ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'server_aggregation_time.png'), dpi=300)
            plt.close()
            
        # 4. 内存随时间变化的热力图
        pivot_df = df.pivot_table(
            index=['epoch', 'client_id'], 
            columns='tag', 
            values='allocated_MB',
            aggfunc='first'
        ).reset_index()
        
        if not pivot_df.empty and 'client_before_training' in pivot_df.columns:
            plt.figure(figsize=(12, 8))
            pivot_df = pivot_df.sort_values(['epoch', 'client_id'])
            pivot_long = pd.melt(
                pivot_df, 
                id_vars=['epoch', 'client_id'],
                value_vars=['client_before_training', 'client_after_training'],
                var_name='phase', value_name='memory'
            )
            pivot_long = pivot_long.dropna()
            pivot_long['client_epoch'] = pivot_long.apply(
                lambda x: f"Client {int(x['client_id']) if pd.notna(x['client_id']) else 'Server'}, Epoch {int(x['epoch'])}", 
                axis=1
            )
            
            if not pivot_long.empty:
                g = sns.catplot(
                    data=pivot_long,
                    kind="bar",
                    x="client_id", y="memory",
                    hue="phase", col="epoch",
                    height=5, aspect=1.5
                )
                g.set_axis_labels("Client ID", "Memory (MB)")
                g.set_titles("Epoch {col_name}")
                g.fig.suptitle('Memory Usage Before vs After Training by Client', y=1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'client_memory_comparison.png'), dpi=300)
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize memory usage")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the memory_logs folder")
    args = parser.parse_args()

    visualize_memory_usage(args.output_dir)
import os
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch

def load_snapshot(filepath):
    """加载内存快照文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_snapshot(snapshot):
    """分析内存快照数据"""
    # 提取关键信息
    stats = {
        'total_size': 0,
        'allocation_counts': {},
        'memory_by_type': {},
        'tensor_sizes': [],
        'peak_memory': 0
    }
    
    # 分析分配和释放
    for event in snapshot:
        if hasattr(event, 'bytes'):
            stats['total_size'] += event.bytes
            
        if hasattr(event, 'type'):
            event_type = str(event.type)
            if event_type not in stats['memory_by_type']:
                stats['memory_by_type'][event_type] = 0
            stats['memory_by_type'][event_type] += getattr(event, 'bytes', 0)
            
        # 统计张量大小
        if hasattr(event, 'tensor_size'):
            if hasattr(event, 'bytes') and event.bytes > 0:
                stats['tensor_sizes'].append((event.tensor_size, event.bytes))
            
    # 计算峰值内存
    stats['peak_memory'] = max(stats['memory_by_type'].values()) if stats['memory_by_type'] else 0
    
    return stats

def plot_memory_usage(snapshot, output_path, title):
    """绘制内存使用情况图表"""
    plt.figure(figsize=(12, 8))
    
    # 提取时间和内存数据
    times = []
    memory_used = []
    current_memory = 0
    
    for event in snapshot:
        if hasattr(event, 'time') and hasattr(event, 'bytes'):
            times.append(event.time)
            current_memory += event.bytes  # 假设正数是分配，负数是释放
            memory_used.append(current_memory / (1024 * 1024))  # 转MB
    
    # 绘制内存使用曲线
    plt.plot(times, memory_used, label='Memory Usage (MB)')
    plt.title(f'Memory Usage Over Time: {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    
    # 添加峰值标记
    peak_memory = max(memory_used) if memory_used else 0
    plt.axhline(y=peak_memory, color='r', linestyle='--', 
                label=f'Peak Memory: {peak_memory:.2f} MB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize PyTorch Memory Snapshots')
    parser.add_argument('--snapshot_dir', type=str, required=True, 
                        help='Directory containing memory snapshots')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (default: same as snapshot_dir)')
    
    args = parser.parse_args()
    
    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir) if args.output_dir else snapshot_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 处理所有快照文件
    snapshot_files = list(snapshot_dir.glob('*.pickle'))
    print(f"Found {len(snapshot_files)} snapshot files")
    
    for snapshot_file in snapshot_files:
        try:
            print(f"Processing {snapshot_file.name}...")
            snapshot = load_snapshot(str(snapshot_file))
            
            # 分析快照
            stats = analyze_snapshot(snapshot)
            
            # 保存分析结果
            stats_file = output_dir / f"{snapshot_file.stem}_stats.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Snapshot Analysis: {snapshot_file.name}\n")
                f.write(f"Total Size: {stats['total_size'] / (1024*1024):.2f} MB\n")
                f.write(f"Peak Memory: {stats['peak_memory'] / (1024*1024):.2f} MB\n")
                f.write("\nMemory by Type:\n")
                for t, size in stats['memory_by_type'].items():
                    f.write(f"  {t}: {size / (1024*1024):.2f} MB\n")
            
            # 绘制内存使用图
            plot_file = output_dir / f"{snapshot_file.stem}_plot.png"
            plot_memory_usage(snapshot, plot_file, snapshot_file.stem)
            
            print(f"  Analysis saved to {stats_file}")
            print(f"  Plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error processing {snapshot_file.name}: {e}")
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np

# Time breakdown data
time_labels = ['Forward', 'Backward', 'Communication', 'Aggregation & Idle']
time_colors = ['#AD0B08', '#237B9F', '#71BFB2', '#EC817E',]
original_time_data = {
    'Full': [87.27, 182.69, 127.22, 119.30+1.06],
    'LoRA': [88.64, 70.37, 1.03, 13.68+0.16],
    'Adapter': [89.44, 71.56, 1.12, 14.32+0.17],
    'Pruning': [47.99, 89.52, 67.43, 72.86+0.44]
}

total_times = {key: sum(values) for key, values in original_time_data.items()}

time_data = original_time_data.copy()
# Normalize the time data to percentages
for key in time_data:
    total = sum(time_data[key])
    time_data[key] = [value / total * 100 for value in time_data[key]]

# Create the figure for the bar chart
fig, bar_ax = plt.subplots(figsize=(20, 8))

time_label_offsets = {
    # 格式：'方法': (x偏移, y偏移)
    'Full': (0, 0),  # Idle
    'LoRA': (0, 0),  # Communication
    'Adapter': (0, 0),  # Communication
    'Pruning': (0, 0)  # Communication
}


text_offsets = {
    # 格式：'方法': {部分索引: (x偏移, y偏移)}
    'Full': {3: (0, 0), },  # Idle和Aggregation
    'LoRA': {2: (0, 0), 3: (0, 0), },  # Communication, Idle和Aggregation
    'Adapter': {2: (0, 0), 3: (0, 0), },  # Communication, Idle和Aggregation
    'Pruning': {}  # 这个方法的所有比例都足够大，不需要偏移
}


# Set smaller bar height and adjust y_positions to bring bars closer
bar_height = 0.3  # Slightly thinner bars
y_positions = np.linspace(0, len(time_data)-1, len(time_data)) * 0.55  # Reduced spacing between bars

left_positions = np.zeros(len(time_data))

# Plot the stacked horizontal bars
for i, (label, color) in enumerate(zip(time_labels, time_colors)):
    component_times = [time_data[title][i] for title in time_data]
    bar_ax.barh(y_positions, component_times, height=bar_height, 
                left=left_positions, color=color, label=label, edgecolor='black')
    left_positions += component_times

# Add percentage labels on each segment (remove the "if value > 5" condition)
left_positions = np.zeros(len(time_data))
for i, component in enumerate(time_labels):
    component_times = [time_data[title][i] for title in time_data]
    for j, (title, value) in enumerate(zip(time_data.keys(), component_times)):
        # Display the text regardless of the value size
        x_pos = left_positions[j] + value / 2
        y_pos = y_positions[j]

        if title in text_offsets and i in text_offsets[title]:
            x_offset, y_offset = text_offsets[title][i]
            x_pos += x_offset
            y_pos += y_offset
        # bar_ax.text(left_positions[j] + value / 2, y_positions[j], 
        #             f"{value:.1f}%", ha='center', va='center', 
        #             fontsize=22, fontweight='bold')

        if value > 0.000001:
            bar_ax.text(x_pos, y_pos,
                        f"{value:.1f}", ha='center', va='center',
                        fontsize=22, fontweight='bold')  # Only add text if segment is wide enough
    left_positions += component_times

# 总标签
for i, (title, total_time) in enumerate(total_times.items()):
    x_offset, y_offset = time_label_offsets[title]
    bar_ax.text(101 + x_offset, y_positions[i] + y_offset,
                f"{total_time:.1f}mins", ha='left', va='center',
                fontsize=26, fontweight='bold', color='black')  # Adjusted position for total time label

# Configure the time breakdown chart
bar_ax.set_yticks(y_positions)
bar_ax.set_yticklabels(list(time_data.keys()), fontsize=26, fontweight='bold')
bar_ax.set_xlabel('Percentage of Time Spent on Each Step in the Wall-Clock Time (%)', fontsize=26, fontweight='bold')

# Set x-axis limits and labels
bar_ax.set_xlim(0, 100)
bar_ax.set_xticks(range(0, 101, 20))
bar_ax.set_xticklabels([f'{x}%' for x in range(0, 101, 20)], fontsize=26, fontweight='bold')

# Invert the y-axis to match the pie chart order
bar_ax.invert_yaxis()

# Title for the chart
# bar_ax.set_title('Time Breakdown', fontsize=26, fontweight='bold')

# Adjust legend position
bar_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), 
              ncol=5, fontsize=26, frameon=True, prop={'weight': 'bold','size': 26})

# Adjust layout and make room for the legend
plt.subplots_adjust(bottom=0.25)

# Save the figure
plt.savefig('moti_cause_2_bar.png', dpi=300, bbox_inches='tight')
plt.savefig('moti_cause_2_bar.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Data for the four methods
# labels = ['Parameters', 'Gradients & Optimizer', 'Activations', 'Temp Buffers & Framework Overhead']
# colors = ['#1999B2', '#95BCE5', '#E84445', '#F39DA0']

# # Data values in GB
# fft_values = [14.2, 70, 17.19, 8.03]
# lora_values = [14.2, 0.1, 17.19, 1.03]
# adapter_values = [14.2, 0.12, 17.19, 1.05]
# pruning_values = [8.4, 42.2, 10.3, 4.72]

# # Data percentages
# fft_perc = [12.97751782123926, 63.973679400475234, 15.710107841345275, 7.33869493694023]
# lora_perc = [43.66543665436654, 0.3075030750307503, 52.859778597785976, 3.167281672816728]
# adapter_perc = [43.611793611793615, 0.3685503685503686, 52.794840294840306, 3.224815724815725]
# pruning_perc = [12.800975312404756, 64.3096616885096, 15.696434014020117, 7.192928985065528]

# # Titles for each method
# titles = ['Full FT', 'LoRA FT', 'Adapter FT', 'Pruning-then-FT']

# #TODO 增加的time breakdown图；
# time_labels = ['Forward', 'Backward', 'Communication', 'Idle', 'Aggregation & others']
# time_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0']
# time_data = {
#     'Full FT': [30, 50, 10, 5, 5],
#     'LoRA FT': [20, 30, 15, 10, 5],
#     'Adapter FT': [25, 35, 20, 10, 10],
#     'Pruning-then-FT': [15, 25, 30, 20, 10]
# }

# for key in time_data:
#     total = sum(time_data[key])
#     time_data[key] = [value / total * 100 for value in time_data[key]]
# # Plot with a 1x4 layout instead of 2x2
# fig, axes = plt.subplots(2, 1, figsize=(20, 15), 
#                          gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})

# pie_axes = fig.subplots(1, 4)

# # Add a legend at the top
# first_wedges = None

# # Define text distance adjustments - can be modified to adjust text positioning
# # Format: [param_dist, grad_dist, act_dist, temp_dist]
# text_positions = {
#     'Full FT': [
#         [0.0, 0.0],     # Parameters
#         [0.3, 0.2],     # Gradients
#         [0.05, 0.25],     # Activations
#         [0.5, 0.3]      # Temp Buffers
#     ],
#     'LoRA FT': [
#         [0.0, 0.0],     # Parameters
#         [0.0, 0.0],     # Gradients
#         [0.1, 0.0],     # Activations
#         [0.0, 0.1]      # Temp Buffers
#     ],
#     'Adapter FT': [
#         [0.0, 0.0],     # Parameters
#         [0.0, 0.0],     # Gradients
#         [0.1, 0.0],     # Activations
#         [0.0, 0.1]      # Temp Buffers
#     ],
#     'Pruning-then-FT': [
#         [0.1, 0.0],     # Parameters
#         [0.2, 0.2],     # Gradients
#         [0.05, 0.25],     # Activations
#         [0.5, 0.3]      # Temp Buffers
#     ]
# }

# memory_fontsize = 20

# # Create the pie charts
# for ax, values, perc, title in zip(pie_axes, [fft_values, lora_values, adapter_values, pruning_values], 
#                                   [fft_perc, lora_perc, adapter_perc, pruning_perc], titles):
#     # Create the pie chart without autopct
#     wedges, _ = ax.pie(values, colors=colors, startangle=90)

#     ax.set_frame_on(False)
    
#     # Store the first set of wedges for the legend
#     if first_wedges is None:
#         first_wedges = wedges
    
#     # Get text distances for this chart
#     positions = text_positions[title]
    
#     # Add custom text annotations with adjustable positions
#     for i, wedge in enumerate(wedges):
#         # Get angle of the wedge center
#         theta = (wedge.theta2 + wedge.theta1) / 2
#         theta_rad = np.deg2rad(theta)
        
#         # theta_adjusted = theta + positions[i][1]
#         # theta_rad = np.deg2rad(theta_adjusted)
        
#         # Calculate text position
#         base_radius = 0.7   
#         base_x = base_radius * np.cos(theta_rad)
#         base_y = base_radius * np.sin(theta_rad)

#         vert_offset = positions[i][0]
#         horiz_offset = positions[i][1]
    
#         # Calculate final position
#         x = base_x + horiz_offset
#         y = base_y + vert_offset
        
#         # Add the text with black color
#         ax.text(x, y, f"{values[i]:.1f} GB\n({perc[i]:.1f}%)", 
#                 ha='center', va='center', 
#                 fontsize=memory_fontsize, fontweight='bold', color='black')
    
#     # Add title below the pie chart
#     ax.text(0, -1.3, title, ha='center', va='center', fontsize=26, fontweight='bold')
    
#     # Equal aspect ratio ensures that pie is drawn as a circle
#     ax.set_aspect('equal')

# bar_ax = axes[1]
# bar_height = 0.6
# y_positions = np.arange(len(titles))

# left_positions = np.zeros(len(titles))

# for i, (label, color) in enumerate(zip(time_labels, time_colors)):
#     # Extract this component's time for each method
#     component_times = [time_data[title][i] for title in titles]
    
#     # Plot this segment for all methods
#     bar_ax.barh(y_positions, component_times, height=bar_height, 
#                 left=left_positions, color=color, label=label, edgecolor='black')
    
#     # Update the left position for the next segment
#     left_positions += component_times

# # Add percentage labels on each segment
# left_positions = np.zeros(len(titles))
# for i, component in enumerate(time_labels):
#     component_times = [time_data[title][i] for title in titles]
    
#     for j, value in enumerate(component_times):
#         if value > 5:  # Only add text if segment is wide enough
#             bar_ax.text(left_positions[j] + value/2, y_positions[j], 
#                       f"{value:.1f}%", ha='center', va='center', 
#                       fontsize=14, fontweight='bold')
    
#     left_positions += component_times

# # Configure the time breakdown chart
# bar_ax.set_yticks(y_positions)
# bar_ax.set_yticklabels(titles, fontsize=18, fontweight='bold')
# bar_ax.set_xlabel('Percentage of Time (%)', fontsize=20, fontweight='bold')
# bar_ax.set_xlim(0, 100)
# bar_ax.set_xticks(range(0, 101, 20))
# bar_ax.set_xticklabels([f'{x}%' for x in range(0, 101, 20)], fontsize=16)
# bar_ax.invert_yaxis()  # To match the order of the pie charts above
# bar_ax.set_title('Time Breakdown', fontsize=24, fontweight='bold')
# bar_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
#               ncol=5, fontsize=18, frameon=True)


# fig.legend(first_wedges, labels, loc='upper center', ncol=2, 
#            bbox_to_anchor=(0.5, 1.15), fontsize = 26, prop={'weight': 'bold', 'size': 26})

# # Adjust layout
# plt.tight_layout(pad=1)
# plt.subplots_adjust(top=1.1, bottom=0.15)  # Make room for legend at top

# # Add a main title below the legend
# # fig.suptitle('Memory Usage Comparison Across Different Fine-tuning Methods', 
# #              fontsize=18, fontweight='bold', y=0.90)

# # Save the figure
# plt.savefig('moti_cause_2.png', dpi=300, bbox_inches='tight')
# # plt.savefig('memory_usage_comparison.pdf', format='pdf', bbox_inches='tight')

# # Show the plot
# plt.show()
# plot_pie_charts.py

import matplotlib.pyplot as plt
import numpy as np

# Data for the four methods
labels = ['Parameters', 'Gradients & Optimizer', 'Activations', 'Temp Buffers & Framework Overhead']
colors = ['#1999B2', '#95BCE5', '#E84445', '#F39DA0']

# Data values in GB
fft_values = [14, 70, 17.19, 8.03]
lora_values = [14, 0.1, 17.19, 1.03]
adapter_values = [14, 0.12, 17.19, 1.05]
pruning_values = [8.2, 42.2, 10.3, 4.72]

# Data percentages
fft_perc = [12.97751782123926, 63.973679400475234, 15.710107841345275, 7.33869493694023]
lora_perc = [43.66543665436654, 0.3075030750307503, 52.859778597785976, 3.167281672816728]
adapter_perc = [43.611793611793615, 0.3685503685503686, 52.794840294840306, 3.224815724815725]
pruning_perc = [12.800975312404756, 64.3096616885096, 15.696434014020117, 7.192928985065528]

# Titles for each method
titles = ['Full FT', 'LoRA FT', 'Adapter FT', 'Pruning-then-FT']

# Define text distance adjustments for each method's pie chart
text_positions = {
    'Full FT': [
        [0.0, 0.0],     # Parameters
        [0.3, 0.2],     # Gradients
        [0.05, 0.25],   # Activations
        [0.5, 0.3]      # Temp Buffers
    ],
    'LoRA FT': [
        [0.0, 0.0],     # Parameters
        [0.0, 0.0],     # Gradients
        [0.1, 0.0],     # Activations
        [0.0, 0.1]      # Temp Buffers
    ],
    'Adapter FT': [
        [0.0, 0.0],     # Parameters
        [0.0, 0.0],     # Gradients
        [0.1, 0.0],     # Activations
        [0.0, 0.1]      # Temp Buffers
    ],
    'Pruning-then-FT': [
        [0.1, 0.0],     # Parameters
        [0.2, 0.2],     # Gradients
        [0.05, 0.25],   # Activations
        [0.5, 0.3]      # Temp Buffers
    ]
}

memory_fontsize = 20

# Create the figure and a grid for pie charts (1 row, 4 columns)
# fig, axes = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
# pie_axes = fig.subplots(1, 4)
fig, pie_axes = plt.subplots(1, 4, figsize=(16, 8))

first_wedges = None

# Create the pie charts
for ax, values, perc, title in zip(pie_axes, [fft_values, lora_values, adapter_values, pruning_values], 
                                  [fft_perc, lora_perc, adapter_perc, pruning_perc], titles):
    wedges, _ = ax.pie(values, colors=colors, startangle=90)
    ax.set_frame_on(False)

    if first_wedges is None:
        first_wedges = wedges

    positions = text_positions[title]

    for i, wedge in enumerate(wedges):
        theta = (wedge.theta2 + wedge.theta1) / 2
        theta_rad = np.deg2rad(theta)

        base_radius = 0.7
        base_x = base_radius * np.cos(theta_rad)
        base_y = base_radius * np.sin(theta_rad)

        vert_offset = positions[i][0]
        horiz_offset = positions[i][1]

        x = base_x + horiz_offset
        y = base_y + vert_offset

        ax.text(x, y, f"{values[i]:.1f} GB\n({perc[i]:.1f}%)", 
                ha='center', va='center', 
                fontsize=memory_fontsize, fontweight='bold', color='black')

    ax.text(0, -1.2, title, ha='center', va='center', fontsize=26, fontweight='bold')
    ax.set_aspect('equal')

fig.legend(first_wedges, labels, loc='upper center', ncol=2, 
           bbox_to_anchor=(0.5, 1.05), fontsize=26, prop={'weight': 'bold', 'size': 26})

plt.tight_layout(pad=1)
plt.subplots_adjust(top=1.1, bottom=0.15)

# Save the figure
plt.savefig('moti_cause_2_pie.png', dpi=300, bbox_inches='tight')
plt.savefig('moti_cause_2_pie.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

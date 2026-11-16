"""
Script to create BiLSTM architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors
color_input = '#E8F4F8'
color_embedding = '#B8E6F0'
color_lstm = '#7EC8E3'
color_dropout = '#FFE5B4'
color_dense = '#FFA07A'
color_output = '#FF6B6B'

# Helper function to draw layer box
def draw_layer(ax, x, y, width, height, label, sublabel, color):
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', 
                          facecolor=color, 
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2 + 0.2, label, 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x + width/2, y + height/2 - 0.3, sublabel, 
            ha='center', va='center', fontsize=9, style='italic')

# Helper function to draw arrow
def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=30, 
                           linewidth=2,
                           color='black')
    ax.add_patch(arrow)

# Title
ax.text(5, 13.2, 'BiLSTM Architecture for Emotion Classification', 
        ha='center', fontsize=16, fontweight='bold')

# Layer positions (x, y, width, height)
layers = [
    # (x, y, width, height, label, sublabel, color)
    (2.5, 11.5, 5, 0.8, 'Input Layer', '(batch_size, seq_length)', color_input),
    (2.5, 10.0, 5, 0.8, 'Embedding Layer', 'vocab_size → 128 dim', color_embedding),
    (2.5, 8.5, 5, 0.8, 'Bidirectional LSTM', '2 layers, hidden_dim=64', color_lstm),
    (2.5, 7.0, 5, 0.8, 'Hidden States Concatenation', 'forward + backward → 128 dim', color_lstm),
    (2.5, 5.5, 5, 0.8, 'Dropout Layer', 'p = 0.3', color_dropout),
    (2.5, 4.0, 5, 0.8, 'Fully Connected Layer', '128 → 6 classes', color_dense),
    (2.5, 2.5, 5, 0.8, 'Output (Logits)', 'Softmax applied in loss', color_output),
]

# Draw all layers
for x, y, w, h, label, sublabel, color in layers:
    draw_layer(ax, x, y, w, h, label, sublabel, color)

# Draw arrows between layers
arrow_positions = [
    (5, 11.5, 5, 10.8),  # Input to Embedding
    (5, 10.0, 5, 9.3),   # Embedding to BiLSTM
    (5, 8.5, 5, 7.8),    # BiLSTM to Concatenation
    (5, 7.0, 5, 6.3),    # Concatenation to Dropout
    (5, 5.5, 5, 4.8),    # Dropout to FC
    (5, 4.0, 5, 3.3),    # FC to Output
]

for x1, y1, x2, y2 in arrow_positions:
    draw_arrow(ax, x1, y1, x2, y2)

# Add dimension annotations on the right
annotations = [
    (8.2, 11.9, '(B, L)'),
    (8.2, 10.4, '(B, L, 128)'),
    (8.2, 8.9, '(B, L, 128)'),
    (8.2, 7.4, '(B, 128)'),
    (8.2, 5.9, '(B, 128)'),
    (8.2, 4.4, '(B, 6)'),
    (8.2, 2.9, '(B, 6)'),
]

for x, y, text in annotations:
    ax.text(x, y, text, ha='left', va='center', 
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend
legend_y = 1.2
ax.text(1, legend_y, 'Legend:', fontsize=10, fontweight='bold')
ax.text(1, legend_y - 0.3, 'B = batch_size', fontsize=9)
ax.text(1, legend_y - 0.6, 'L = sequence_length (max 100)', fontsize=9)
ax.text(5, legend_y - 0.3, '128 = 2 × hidden_dim (64)', fontsize=9)
ax.text(5, legend_y - 0.6, 'Bidirectional doubles output', fontsize=9)

# Add model details box
details_box = FancyBboxPatch((0.5, 0.1), 9, 0.7,
                            boxstyle="round,pad=0.05",
                            edgecolor='gray',
                            facecolor='#F5F5F5',
                            linewidth=1)
ax.add_patch(details_box)

details_text = (
    "Model Details: vocab_size ≈ 10,000 | Total params ≈ 1.5M | "
    "Loss: CrossEntropyLoss (with class weights) | Optimizer: Adam (lr=0.001)"
)
ax.text(5, 0.45, details_text, ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/bilstm_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Architecture diagram saved to figures/bilstm_architecture.png")
plt.close()

# Also create a simple layer dimension table diagram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'BiLSTM Layer Dimensions', 
        ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

# Create table data
table_data = [
    ['Layer', 'Input Shape', 'Output Shape', 'Parameters'],
    ['Embedding', '(B, L)', '(B, L, 128)', 'vocab_size × 128'],
    ['BiLSTM Layer 1', '(B, L, 128)', '(B, L, 128)', '~132K'],
    ['BiLSTM Layer 2', '(B, L, 128)', '(B, L, 128)', '~132K'],
    ['Concatenate', '(B, L, 128)', '(B, 128)', '0 (operation)'],
    ['Dropout', '(B, 128)', '(B, 128)', '0 (p=0.3)'],
    ['Linear', '(B, 128)', '(B, 6)', '128 × 6 + 6 = 774'],
    ['', '', '', ''],
    ['Total', '', '', '≈ 1.5M parameters'],
]

# Create table
table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#4A90E2')
    cell.set_text_props(weight='bold', color='white')

# Style total row
for i in range(4):
    cell = table[(8, i)]
    cell.set_facecolor('#E8F4F8')
    cell.set_text_props(weight='bold')

# Alternate row colors
for i in range(1, 7):
    for j in range(4):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F9F9F9')

plt.savefig('figures/bilstm_dimensions_table.png', dpi=300, bbox_inches='tight')
print("✓ Dimension table saved to figures/bilstm_dimensions_table.png")
plt.close()

print("\n✓ All architecture diagrams created successfully!")

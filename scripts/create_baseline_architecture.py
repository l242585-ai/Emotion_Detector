"""
Baseline Architecture Diagram Generator
Creates architecture diagram for TF-IDF + Logistic Regression model
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Define color schemes
COLORS = {
    'input': '#E8F4F8',
    'embedding': '#B8E6F0',
    'dropout': '#FFE5B4',
    'dense': '#FFA07A',
    'output': '#FF6B6B',
    'feature': '#98D8C8',
}

def draw_layer(ax, x, y, width, height, label, sublabel, color, linewidth=2):
    """Helper function to draw a layer box"""
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', 
                          facecolor=color, 
                          linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2 + 0.2, label, 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(x + width/2, y + height/2 - 0.25, sublabel, 
            ha='center', va='center', fontsize=8, style='italic')

def draw_arrow(ax, x1, y1, x2, y2, linewidth=2, color='black'):
    """Helper function to draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=25, 
                           linewidth=linewidth,
                           color=color)
    ax.add_patch(arrow)

def add_dimension_box(ax, x, y, text):
    """Add dimension annotation box"""
    ax.text(x, y, text, ha='left', va='center', 
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.9))

def create_baseline_architecture():
    """Create TF-IDF + Logistic Regression architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # Title
    ax.text(5, 12.3, 'Baseline Model: TF-IDF + Logistic Regression', 
            ha='center', fontsize=16, fontweight='bold')

    # Layer definitions
    layers = [
        (2.5, 10.5, 5, 0.8, 'Input Text', '"i feel so happy today"', COLORS['input']),
        (2.5, 9.2, 5, 0.8, 'Text Preprocessing', 'Lowercase, remove special chars', COLORS['feature']),
        (2.5, 7.9, 5, 0.8, 'TF-IDF Vectorization', 'max_features=5000, ngrams=(1,2)', COLORS['feature']),
        (2.5, 6.6, 5, 0.8, 'Feature Vector', 'Sparse vector (5000 dimensions)', COLORS['embedding']),
        (2.5, 5.3, 5, 0.8, 'Logistic Regression', 'Multinomial, L2 regularization', COLORS['dense']),
        (2.5, 4.0, 5, 0.8, 'Class Weights', 'Balanced (handles imbalance)', COLORS['dropout']),
        (2.5, 2.7, 5, 0.8, 'Output Probabilities', '6 emotion classes', COLORS['output']),
    ]

    # Draw layers
    for x, y, w, h, label, sublabel, color in layers:
        draw_layer(ax, x, y, w, h, label, sublabel, color)

    # Draw arrows
    arrow_positions = [
        (5, 10.5, 5, 10.0),
        (5, 9.2, 5, 8.7),
        (5, 7.9, 5, 7.4),
        (5, 6.6, 5, 6.1),
        (5, 5.3, 5, 4.8),
        (5, 4.0, 5, 3.5),
    ]

    for x1, y1, x2, y2 in arrow_positions:
        draw_arrow(ax, x1, y1, x2, y2)

    # Add dimension annotations
    annotations = [
        (8.0, 10.9, 'Text'),
        (8.0, 9.6, 'Text'),
        (8.0, 8.3, '(1, 5000)'),
        (8.0, 7.0, '(1, 5000)'),
        (8.0, 5.7, '(1, 6)'),
        (8.0, 4.4, '(1, 6)'),
        (8.0, 3.1, '(1, 6)'),
    ]

    for x, y, text in annotations:
        add_dimension_box(ax, x, y, text)

    # Model details box
    details_box = FancyBboxPatch((0.5, 1.3), 9, 1.0,
                                boxstyle="round,pad=0.05",
                                edgecolor='gray',
                                facecolor='#F5F5F5',
                                linewidth=1)
    ax.add_patch(details_box)

    details = [
        "Model Details:",
        "• Parameters: ~30K (5000 features × 6 classes)",
        "• Training Time: Seconds (CPU)",
        "• Advantages: Fast, interpretable, strong baseline",
        "• Disadvantages: No sequential context, limited negation handling"
    ]
    
    for i, text in enumerate(details):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(5, 2.05 - i*0.15, text, ha='center', va='center', 
                fontsize=9, fontweight=weight)

    # Key features box
    features_box = FancyBboxPatch((0.5, 0.1), 9, 0.9,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#3498db',
                                 facecolor='#EBF5FB',
                                 linewidth=2)
    ax.add_patch(features_box)

    ax.text(5, 0.75, 'Key TF-IDF Features', ha='center', fontweight='bold', fontsize=10)
    ax.text(2, 0.45, '• Term Frequency: Word importance', ha='left', fontsize=8)
    ax.text(2, 0.25, '• Inverse Document Freq: Down-weight common words', ha='left', fontsize=8)
    ax.text(6, 0.45, '• Bigrams: Capture "not happy", "very sad"', ha='left', fontsize=8)
    ax.text(6, 0.25, '• Sparse representation: Efficient storage', ha='left', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/baseline_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Baseline architecture saved to figures/baseline_architecture.png")
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Creating Baseline Architecture Diagram")
    print("="*60 + "\n")
    
    create_baseline_architecture()
    
    print("\n" + "="*60)
    print("✓ Baseline architecture diagram created successfully!")
    print("="*60)
    print("\nGenerated file: figures/baseline_architecture.png")

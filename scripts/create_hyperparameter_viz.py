"""
Create hyperparameter tuning visualization
Shows the impact of each hyperparameter on validation accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the hyperparameter log
df = pd.read_csv('experiments/hyperparameter_log.csv')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

# Create figure with 2 rows, 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Hyperparameter Tuning Results - Sequential 3-Phase Search', 
             fontsize=16, fontweight='bold', y=0.995)

# BiLSTM results
bilstm_df = df[df['model'] == 'BiLSTM'].copy()

# Phase 1: Learning Rate
phase1 = bilstm_df[bilstm_df['phase'] == '1_learning_rate']
axes[0, 0].plot(phase1['learning_rate'], phase1['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#2E86AB')
axes[0, 0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('BiLSTM Phase 1: Learning Rate\n(batch=32, dropout=0.3)', 
                      fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')
for x, y in zip(phase1['learning_rate'], phase1['val_acc']):
    axes[0, 0].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

# Phase 2: Batch Size
phase2 = bilstm_df[bilstm_df['phase'] == '2_batch_size']
axes[0, 1].plot(phase2['batch_size'], phase2['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#A23B72')
axes[0, 1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('BiLSTM Phase 2: Batch Size\n(lr=0.01, dropout=0.3)', 
                      fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
for x, y in zip(phase2['batch_size'], phase2['val_acc']):
    axes[0, 1].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

# Phase 3: Dropout
phase3 = bilstm_df[bilstm_df['phase'] == '3_dropout']
axes[0, 2].plot(phase3['dropout'], phase3['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#F18F01')
axes[0, 2].set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 2].set_title('BiLSTM Phase 3: Dropout\n(lr=0.01, batch=64)', 
                      fontsize=13, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
for x, y in zip(phase3['dropout'], phase3['val_acc']):
    axes[0, 2].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

# DistilBERT results
distilbert_df = df[df['model'] == 'DistilBERT'].copy()

# Phase 1: Learning Rate
phase1_d = distilbert_df[distilbert_df['phase'] == '1_learning_rate']
axes[1, 0].plot(phase1_d['learning_rate'], phase1_d['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#2E86AB')
axes[1, 0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_title('DistilBERT Phase 1: Learning Rate\n(batch=16, dropout=0.3)', 
                      fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xscale('log')
for x, y in zip(phase1_d['learning_rate'], phase1_d['val_acc']):
    axes[1, 0].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

# Phase 2: Batch Size
phase2_d = distilbert_df[distilbert_df['phase'] == '2_batch_size']
axes[1, 1].plot(phase2_d['batch_size'], phase2_d['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#A23B72')
axes[1, 1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_title('DistilBERT Phase 2: Batch Size\n(lr=2e-5, dropout=0.3)', 
                      fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
for x, y in zip(phase2_d['batch_size'], phase2_d['val_acc']):
    axes[1, 1].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

# Phase 3: Dropout
phase3_d = distilbert_df[distilbert_df['phase'] == '3_dropout']
axes[1, 2].plot(phase3_d['dropout'], phase3_d['val_acc'], 'o-', 
                linewidth=2, markersize=10, color='#F18F01')
axes[1, 2].set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 2].set_title('DistilBERT Phase 3: Dropout\n(lr=2e-5, batch=8)', 
                      fontsize=13, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
for x, y in zip(phase3_d['dropout'], phase3_d['val_acc']):
    axes[1, 2].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/hyperparameter_tuning_phases.png', dpi=300, bbox_inches='tight')
print('✅ Saved: results/hyperparameter_tuning_phases.png')

# Create summary comparison bar chart
fig2, ax = plt.subplots(figsize=(12, 6))

# Get best results for each model
bilstm_best = bilstm_df.loc[bilstm_df['val_acc'].idxmax()]
distilbert_best = distilbert_df.loc[distilbert_df['val_acc'].idxmax()]

models = ['BiLSTM\n(lr=0.01, batch=64, dropout=0.2)', 
          'DistilBERT\n(lr=2e-5, batch=8, dropout=0.2)']
accuracies = [bilstm_best['val_acc'], distilbert_best['val_acc']]
colors = ['#2E86AB', '#F18F01']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Best Hyperparameter Configuration Comparison', fontsize=15, fontweight='bold')
ax.set_ylim([0.88, 0.96])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/best_hyperparameters_comparison.png', dpi=300, bbox_inches='tight')
print('✅ Saved: results/best_hyperparameters_comparison.png')

print('\n✅ All hyperparameter tuning visualizations created!')

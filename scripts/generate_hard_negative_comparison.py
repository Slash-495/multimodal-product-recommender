import json
from pathlib import Path

# Authoritative Baselines (Epoch 1)
base = {
    'model': 'Original Baseline',
    'epoch': 1,
    'recall@10': 0.0115,
    'recall@50': 0.0515,
    'ndcg@10': 0.0050,
    'mrr@10': 0.0031
}

with open('outputs/movielens/content_checkpoint_comparison.json', 'r') as f:
    concat_data = json.load(f)['checkpoints']

# Load Hard Negative Checkpoint Results
with open('outputs/movielens/hard_negative_diagnostics.json', 'r') as f:
    diag_data = json.load(f)['diagnostics']

hard_data = [
    {'epoch': 1, 'train_loss': 5.514312, 'valid_loss': 5.497309, 'recall@10': 0.0119, 'recall@50': 0.0533, 'ndcg@10': 0.0052, 'mrr@10': 0.0033},
    {'epoch': 2, 'train_loss': 5.503958, 'valid_loss': 5.524134, 'recall@10': 0.0094, 'recall@50': 0.0495, 'ndcg@10': 0.0040, 'mrr@10': 0.0024},
    {'epoch': 3, 'train_loss': 5.485753, 'valid_loss': 5.547693, 'recall@10': 0.0088, 'recall@50': 0.0438, 'ndcg@10': 0.0040, 'mrr@10': 0.0026},
    {'epoch': 4, 'train_loss': 5.463252, 'valid_loss': 5.588549, 'recall@10': 0.0064, 'recall@50': 0.0341, 'ndcg@10': 0.0028, 'mrr@10': 0.0017},
    {'epoch': 5, 'train_loss': 5.435115, 'valid_loss': 5.701356, 'recall@10': 0.0073, 'recall@50': 0.0334, 'ndcg@10': 0.0033, 'mrr@10': 0.0021}
]

best_hard = hard_data[0] # Epoch 1
best_concat = concat_data[0] # Epoch 1

metrics = ['recall@10', 'recall@50', 'ndcg@10', 'mrr@10']

# Hard Neg vs Concat
hard_vs_concat = {}
for m in metrics:
    h_val = best_hard[m]
    c_val = best_concat[m]
    abs_d = h_val - c_val
    rel_p = (abs_d / c_val) * 100.0
    hard_vs_concat[m] = {
        'concat': c_val,
        'hard_negative': h_val,
        'abs_improvement': round(abs_d, 6),
        'rel_improvement_pct': round(rel_p, 2)
    }

# Hard Neg vs Original Baseline
hard_vs_baseline = {}
for m in metrics:
    h_val = best_hard[m]
    b_val = base[m]
    abs_d = h_val - b_val
    rel_p = (abs_d / b_val) * 100.0
    hard_vs_baseline[m] = {
        'baseline': b_val,
        'hard_negative': h_val,
        'abs_improvement': round(abs_d, 6),
        'rel_improvement_pct': round(rel_p, 2)
    }

output_dict = {
    'experiment': 'hard_negative_mining_comparison',
    'models': {
        'baseline_epoch_1': base,
        'content_concat_epochs': concat_data,
        'hard_negative_concat_epochs': hard_data
    },
    'best_hard_negative_checkpoint': best_hard,
    'diagnostics': diag_data,
    'comparisons': {
        'hard_negative_vs_concat': hard_vs_concat,
        'hard_negative_vs_baseline': hard_vs_baseline
    }
}

output_file = Path('outputs/movielens/hard_negative_comparison.json')
with open(output_file, 'w') as f:
    json.dump(output_dict, f, indent=2)

print('=== HARD NEGATIVE COMPARISON GENERATED ===')
print('Hard-Negative vs Content-Concat (Epoch 1):')
for m, v in hard_vs_concat.items():
    print(f"  {m}: Hard-Neg={v['hard_negative']:.6f} | Concat={v['concat']:.6f} | Abs={v['abs_improvement']:+.6f} | Rel={v['rel_improvement_pct']:+.2f}%")

print('\nHard-Negative vs Original Baseline (Epoch 1):')
for m, v in hard_vs_baseline.items():
    print(f"  {m}: Hard-Neg={v['hard_negative']:.6f} | Baseline={v['baseline']:.6f} | Abs={v['abs_improvement']:+.6f} | Rel={v['rel_improvement_pct']:+.2f}%")

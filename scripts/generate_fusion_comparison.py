import json
from pathlib import Path

base = {
    'model': 'Baseline',
    'epoch': 1,
    'recall@10': 0.0115,
    'recall@50': 0.0515,
    'ndcg@10': 0.0050,
    'mrr@10': 0.0031
}

with open('outputs/movielens/content_checkpoint_comparison.json', 'r') as f:
    concat_data = json.load(f)['checkpoints']

with open('outputs/movielens/gated_checkpoint_comparison.json', 'r') as f:
    gated_json = json.load(f)
    gated_data = gated_json['checkpoints']
    gate_diag = gated_json['gate_diagnostics']

best_gated = gated_data[0]
best_concat = concat_data[0]

metrics = ['recall@10', 'recall@50', 'ndcg@10', 'mrr@10']

gated_vs_baseline = {}
for m in metrics:
    g_val = best_gated[m]
    b_val = base[m]
    abs_d = g_val - b_val
    rel_p = (abs_d / b_val) * 100.0
    gated_vs_baseline[m] = {
        'baseline': b_val,
        'gated': g_val,
        'abs_improvement': round(abs_d, 6),
        'rel_improvement_pct': round(rel_p, 2)
    }

gated_vs_concat = {}
for m in metrics:
    g_val = best_gated[m]
    c_val = best_concat[m]
    abs_d = g_val - c_val
    rel_p = (abs_d / c_val) * 100.0
    gated_vs_concat[m] = {
        'concat': c_val,
        'gated': g_val,
        'abs_improvement': round(abs_d, 6),
        'rel_improvement_pct': round(rel_p, 2)
    }

fusion_comp = {
    'experiment': 'three_way_fusion_comparison',
    'models': {
        'baseline_epoch_1': base,
        'content_concat_epochs': concat_data,
        'content_gated_epochs': gated_data
    },
    'best_gated_checkpoint': best_gated,
    'gate_diagnostics': gate_diag,
    'comparisons': {
        'gated_vs_baseline': gated_vs_baseline,
        'gated_vs_concat': gated_vs_concat
    }
}

output_file = Path('outputs/movielens/fusion_comparison.json')
with open(output_file, 'w') as f:
    json.dump(fusion_comp, f, indent=2)

print('=== THREE-WAY COMPARISON CALCULATIONS COMPLETE ===')
print('Gated vs Baseline (Epoch 1):')
for m, v in gated_vs_baseline.items():
    print(f"  {m}: Gated={v['gated']:.6f} | Baseline={v['baseline']:.6f} | Abs={v['abs_improvement']:+.6f} | Rel={v['rel_improvement_pct']:+.2f}%")

print('\nGated vs Concat (Epoch 1):')
for m, v in gated_vs_concat.items():
    print(f"  {m}: Gated={v['gated']:.6f} | Concat={v['concat']:.6f} | Abs={v['abs_improvement']:+.6f} | Rel={v['rel_improvement_pct']:+.2f}%")

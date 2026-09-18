"""Seeded MRF restoration on thick shapes and thin details; no external images."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from mrf import energy, restore, graph_cut


def clean_image(thin=False):
    image = np.full((48, 48), -1.)
    if thin:
        image[8:40, 12] = 1
        image[24, 8:40] = 1
    else:
        image[8:40, 8:14] = 1
        image[8:14, 8:35] = 1
        image[22:28, 8:30] = 1
    return image


def scores(estimate, clean):
    foreground, recovered = clean == 1, estimate == 1
    union = np.sum(foreground | recovered)
    return {'pixel_accuracy': float(np.mean(estimate == clean)),
            'foreground_iou': float(np.sum(foreground & recovered) / max(union, 1)),
            'balanced_accuracy': float((np.mean(estimate[foreground] == 1) + np.mean(estimate[~foreground] == -1))/2)}


def run(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    figure, axes = plt.subplots(2, 4, figsize=(10, 5.3), constrained_layout=True)
    for thin in (False, True):
        clean = clean_image(thin)
        for noise in (.1, .2, .3):
            for seed in (7, 19, 41):
                rng = np.random.default_rng(seed)
                observed = clean.copy()
                observed[rng.random(clean.shape) < noise] *= -1
                local, diagnostic = restore(observed, restarts=4, seed=seed)
                exact = graph_cut(observed)
                for method, image in [('noisy', observed), ('icm', local), ('graph_cut', exact)]:
                    records.append({'image': 'thin_lines' if thin else 'thick_shape', 'noise_probability': noise,
                                    'seed': seed, 'method': method, 'energy': energy(image, observed), **scores(image, clean)})
                if noise == .2 and seed == 7:
                    for ax, title, image in zip(axes[int(thin)], ['Clean', '20% flips', 'ICM, 4 restarts', 'Exact graph cut'], [clean, observed, local, exact]):
                        ax.imshow(image, cmap='gray', vmin=-1, vmax=1)
                        ax.set_title(title, fontsize=11)
                        ax.axis('off')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'weights': {'h': 0., 'beta': .8, 'eta': 1.},
              'selection': 'minimum energy; clean images are used only for final scoring',
              'icm_restarts': 4, 'max_sweeps': 50, 'records': records}
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    figure.savefig(output/'restoration.png', dpi=180)
    plt.close(figure)
    for kind in ('thick_shape', 'thin_lines'):
        for method in ('noisy', 'icm', 'graph_cut'):
            rows = [r for r in records if r['image'] == kind and r['method'] == method and r['noise_probability'] == .2]
            print(kind, method, {key: round(float(np.mean([r[key] for r in rows])), 4) for key in ['pixel_accuracy','foreground_iou','balanced_accuracy']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results')
    run(parser.parse_args().output)

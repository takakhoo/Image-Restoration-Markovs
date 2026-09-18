"""Binary attractive-grid MRF: local search and exact s-t minimum cut."""
import numpy as np


def validate(x, y, h, beta, eta):
    if x.ndim != 2 or x.shape != y.shape or not x.size:
        raise ValueError('nonempty 2-D images of identical shape required')
    if not np.isin(x, [-1, 1]).all() or not np.isin(y, [-1, 1]).all():
        raise ValueError('binary images must contain only -1 and +1')
    if not np.isfinite([h, beta, eta]).all() or beta < 0 or eta < 0:
        raise ValueError('finite weights and nonnegative beta/eta required')


def energy(x, y, h=0., beta=.8, eta=1.):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    validate(x, y, h, beta, eta)
    pairs = np.sum(x[1:] * x[:-1]) + np.sum(x[:, 1:] * x[:, :-1])
    return float(h*x.sum() - beta*pairs - eta*np.sum(x*y))


def delta_energy(x, y, i, j, h=0., beta=.8, eta=1.):
    neighbors = ((x[i-1, j] if i > 0 else 0) +
                 (x[i+1, j] if i+1 < x.shape[0] else 0) +
                 (x[i, j-1] if j > 0 else 0) +
                 (x[i, j+1] if j+1 < x.shape[1] else 0))
    return float(2*x[i, j]*(-h + beta*neighbors + eta*y[i, j]))


def restore(y, h=0., beta=.8, eta=1., restarts=1, seed=7, max_sweeps=50):
    """Select restarts by observable energy only. No clean-target argument."""
    y = np.asarray(y, dtype=float)
    validate(y, y, h, beta, eta)
    if restarts < 1 or max_sweeps < 1:
        raise ValueError('positive restart and sweep budgets required')
    rng = np.random.default_rng(seed)
    best, best_energy, traces = None, float('inf'), []
    for restart in range(restarts):
        x = y.copy()
        if restart:
            x[rng.random(y.shape) < .05] *= -1
        trace = [energy(x, y, h, beta, eta)]
        for _ in range(max_sweeps):
            changed = False
            for index in rng.permutation(x.size):
                i, j = divmod(int(index), x.shape[1])
                if delta_energy(x, y, i, j, h, beta, eta) < 0:
                    x[i, j] *= -1
                    changed = True
            trace.append(energy(x, y, h, beta, eta))
            if not changed:
                break
        traces.append(trace)
        if trace[-1] < best_energy:
            best, best_energy = x.copy(), trace[-1]
    return best, {'energy': best_energy, 'traces': traces}


def graph_cut(y, h=0., beta=.8, eta=1.):
    """Global energy minimizer for binary labels and attractive (beta >= 0) edges."""
    import networkx as nx
    y = np.asarray(y, dtype=float)
    validate(y, y, h, beta, eta)
    graph = nx.DiGraph()
    source, sink = 'source', 'sink'
    graph.add_nodes_from([source, sink])
    for i, j in np.ndindex(y.shape):
        node = (i, j)
        minus, plus = -h + eta*y[i, j], h - eta*y[i, j]
        shift = min(minus, plus)
        graph.add_edge(source, node, capacity=float(minus-shift))
        graph.add_edge(node, sink, capacity=float(plus-shift))
        for other in [(i+1, j), (i, j+1)]:
            if other[0] < y.shape[0] and other[1] < y.shape[1]:
                graph.add_edge(node, other, capacity=2*beta)
                graph.add_edge(other, node, capacity=2*beta)
    _, (positive, _) = nx.minimum_cut(graph, source, sink)
    result = np.full(y.shape, -1.)
    for index in np.ndindex(y.shape):
        if index in positive:
            result[index] = 1
    return result

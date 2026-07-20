import ast
from pathlib import Path

from lovellm import othermodel as om_mod
from lovellm.metrics import accuracy
from lovellm.personas import make_pool


def test_schema_independence_structural():
    """PROTOCOL rule: othermodel.py never imports personas.py."""
    src = (Path(om_mod.__file__)).read_text()
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [a.name for a in node.names]
    assert not any("personas" in name for name in imported)


def test_observed_beats_prior():
    """C3 direction: accumulated observation beats the empty-R prior,
    scored via the frozen scoring map."""
    pool = make_pool(4, 48, 21)
    observed_pairs, prior_pairs = [], []
    for p in pool:
        om = om_mod.prior()
        for t in range(48):
            om_mod.observe(om, p.signal(t))
            if (t + 1) % 8 == 0:
                gt = p.hidden_state(t)
                observed_pairs.append((om_mod.estimate(om), gt))
                prior_pairs.append((om_mod.estimate(om_mod.prior()), gt))
    obs = accuracy(observed_pairs)
    pri = accuracy(prior_pairs)
    combined_obs = (obs["valence_acc"] + obs["need_acc"]) / 2
    combined_pri = (pri["valence_acc"] + pri["need_acc"]) / 2
    assert combined_obs > combined_pri + 0.10


def test_confidence_grows():
    om = om_mod.prior()
    assert om_mod.estimate(om).conf == 0.0
    pool = make_pool(1, 20, 5)
    for t in range(20):
        om_mod.observe(om, pool[0].signal(t))
    assert om_mod.estimate(om).conf > 0.8

"""Regression checks added 2026-10-06 after a code/manuscript reconciliation.

Each test pins a property of the pipeline AS RUN for every reported grid, so
that documentation (PROTOCOL.md, the paper) cannot drift from the code again:

1. divergence comparator — the stranger response that enters D is the
   UNPADDED one (context_tokens=[]); the length-padded stranger response is
   generated but feeds only the padding-sensitivity quality row.
2. conditioning normalization — the mixture is renormalized to unit sum, and
   because the need-topic vector sums to 2 the EFFECTIVE weights differ from
   the nominal constants (R1: 0.75/0.15/0.10 nominal -> 0.6818/0.1364/0.1818).
3. estimator-input consistency — offline re-estimation
   (scripts/analyze_specificity.py) reads exactly the stream the pipeline
   feeds the estimator, including the session-opening shared-event marker.
4. initiative channel — the known-condition initiative rate is the allocated
   slot count / turns_per_session, entered directly; no proactive message is
   generated.
"""

from __future__ import annotations

import numpy as np

from lovellm import othermodel as om_mod
from lovellm import rstore
from lovellm.allocation import Budget
from lovellm.arms import FULL
from lovellm.basemodel import SyntheticBaseModel
from lovellm.features import STYLE_DIM, response_depth
from lovellm.metrics import SCORING_MAP_NEED, score_valence
from lovellm.personas import make_pool
from lovellm.pipeline import (
    PROBES,
    R1_MECH,
    V1_MECH,
    _conditioning,
    estimator_input,
    run_relationship,
)

N_SESSIONS = 3
TURNS = 8
SEED = 4242


class RecordingBaseModel(SyntheticBaseModel):
    """Synthetic stand-in that logs the context length of every call and the
    text it returned, AND makes the padded and unpadded responses
    distinguishable: a call that receives context tokens appends one extra
    on-topic word per context token (bounded), so a padded stranger response
    is strictly longer than the unpadded one generated from the same seed.
    Without this, the plain synthetic responder ignores context and the
    comparator test could pass with the wrong comparator."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def respond(self, conditioning, memory_markers, depth_hint, temperature, rng, context_tokens=None):
        out = super().respond(conditioning, memory_markers, depth_hint, temperature, rng, context_tokens)
        if context_tokens:
            extra = min(len(context_tokens), 6)
            out = out[:-1] + " " + " ".join(["topicword"] * extra) + "."
        self.calls.append((len(context_tokens or []), out))
        return out


def _one_full_run(bm):
    pool = make_pool(2, N_SESSIONS * TURNS, SEED)
    return run_relationship(
        FULL, bm, pool[0], pool[1], Budget(), N_SESSIONS, TURNS, SEED, n_gen=2, mech=R1_MECH,
    )


def test_divergence_comparator_is_unpadded_stranger():
    bm = RecordingBaseModel()
    run = _one_full_run(bm)
    by_text = {}
    for ctx_len, text in bm.calls:
        by_text.setdefault(text, set()).add(ctx_len)
    saw_padded = False
    for ck in run.checkpoints:
        # Every stranger response used for D came from a zero-context call,
        # and, once memory exists, a DIFFERENT (longer) padded response was
        # also generated for the same pair and is NOT among the comparators.
        for s in ck.responses_stranger:
            assert by_text[s] == {0}, "divergence used a stranger response generated with context"
        if ck.parity_len > 0:
            padded = [t for (l, t) in bm.calls if l == ck.parity_len]
            assert padded, "no length-padded stranger generation found"
            assert not set(padded) & set(ck.responses_stranger)
            saw_padded = True
            # The stored stranger channels are those of the UNPADDED texts:
            # recomputing response_depth from responses_stranger reproduces
            # the stored channel, whereas the padded texts would not.
            unpadded_depth = float(np.mean([response_depth(t) for t in ck.responses_stranger]))
            padded_depth = float(np.mean([response_depth(t) for t in padded]))
            assert abs(ck.channels_stranger["response_depth"] - unpadded_depth) < 1e-9
            assert padded_depth > unpadded_depth
    assert saw_padded


def test_conditioning_effective_weights_differ_from_nominal():
    probe = PROBES[0]
    r = rstore.empty()
    om = om_mod.prior()
    # Distinct, recognisable component vectors.
    rng = np.random.default_rng(1)
    r.pref_est = rng.dirichlet(np.ones(r.pref_est.shape[0]))
    om.need_belief = np.array([0.7, 0.1, 0.1, 0.1])
    need_vec = om_mod.need_topics(om)
    assert abs(need_vec.sum() - 2.0) < 1e-9, "need-topic vector no longer sums to 2; update the paper"
    for mech, eff in ((R1_MECH, (0.75 / 1.10, 0.15 / 1.10, 0.20 / 1.10)), (V1_MECH, (0.5 / 1.2, 0.3 / 1.2, 0.4 / 1.2))):
        cond = _conditioning(FULL, probe, r, om, mech)[STYLE_DIM:]
        expected = eff[0] * probe[STYLE_DIM:] + eff[1] * r.pref_est + eff[2] * (need_vec / need_vec.sum())
        assert np.allclose(cond, expected, atol=1e-9)
        assert abs(cond.sum() - 1.0) < 1e-9
    # Pin the numbers the paper reports.
    assert np.allclose([0.75 / 1.10, 0.15 / 1.10, 0.20 / 1.10], [0.6818, 0.1364, 0.1818], atol=5e-4)
    assert np.allclose([0.5 / 1.2, 0.3 / 1.2, 0.4 / 1.2], [0.4167, 0.25, 0.3333], atol=5e-4)


def test_offline_estimator_stream_matches_pipeline():
    bm = SyntheticBaseModel()
    pool = make_pool(2, N_SESSIONS * TURNS, SEED)
    run = run_relationship(FULL, bm, pool[0], pool[1], Budget(), N_SESSIONS, TURNS, SEED, n_gen=1, mech=R1_MECH)
    # Offline replay with the shared helper (what analyze_specificity.py uses).
    om = om_mod.prior()
    for t in range(N_SESSIONS * TURNS):
        om_mod.observe(om, estimator_input(pool[0], t, TURNS))
    offline = om_mod.estimate(om)
    pipeline_est = run.checkpoints[-1].accuracy_pair[0]
    assert offline.valence_bin == pipeline_est.valence_bin
    assert offline.need_label == pipeline_est.need_label
    assert abs(offline.conf - pipeline_est.conf) < 1e-12
    # And the signal-only stream (the legacy script) is a DIFFERENT stream.
    assert any(estimator_input(pool[0], t, TURNS) != pool[0].signal(t) for t in range(N_SESSIONS * TURNS))


def test_initiative_channel_is_allocation_not_generation():
    bm = RecordingBaseModel()
    run = _one_full_run(bm)
    for ck in run.checkpoints:
        assert ck.channels_known["initiative_rate"] == ck.alloc_initiative / TURNS
        assert ck.channels_stranger["initiative_rate"] == 0.0
    # Calls per checkpoint: 3 generations (known, stranger, padded stranger)
    # per probe per paired generation — and nothing else, i.e. no proactive
    # message generations exist.
    n_probes = len(PROBES)
    assert len(bm.calls) == N_SESSIONS * n_probes * 2 * 3


def test_scoring_map_covers_estimator_labels():
    est = om_mod.estimate(om_mod.prior())
    assert est.need_label in SCORING_MAP_NEED
    assert score_valence(0.0) == "neu"


def test_original_renderer_restored_for_initial_grid():
    """PROTOCOL A9: the initial grid's runner must use the ORIGINAL renderer
    (fixed 6 words per topic, 60-token context cap), not the R1 renderer."""
    import re
    from lovellm import realmodel, realmodel_v1

    assert realmodel_v1.WORDS_PER_TOPIC_IN_PROMPT == 6
    assert realmodel_v1.CONTEXT_EXCERPT_V1 == 60
    assert realmodel.CONTEXT_EXCERPT == 12 and realmodel.LIST_WORDS_TOTAL == 16
    # Word-list behavior without loading the model.
    v1 = realmodel_v1.MLXBaseModelV1.__new__(realmodel_v1.MLXBaseModelV1)
    r1 = realmodel.MLXBaseModel.__new__(realmodel.MLXBaseModel)
    cond = PROBES[0]
    assert len(v1._vocab_from_conditioning(cond)) == 18          # 3 topics x 6 words, weight-blind
    assert len(r1._vocab_from_conditioning(cond)) == 16          # 16 words apportioned by weight
    src = open(__file__.replace("tests/test_consistency.py", "scripts/reproduce_real.py")).read()
    assert "MLXBaseModelV1()" in src and not re.search(r"\bbm = MLXBaseModel\(\)", src)


def test_context_share_inert_under_r1_renderer():
    """PROTOCOL A10: with the 12-token excerpt cap and marker cap 1, extra
    retrieval slots change neither the rendered excerpt nor the marker list."""
    from lovellm.pipeline import _memory_markers
    from lovellm.realmodel import CONTEXT_EXCERPT

    pool = make_pool(4, 48, 777_001)
    r = rstore.empty()
    for t in range(48):
        rstore.update(r, estimator_input(pool[0], t, 8), t // 8, t)
    for probe in PROBES:
        a, b = rstore.retrieve(r, probe, 4), rstore.retrieve(r, probe, 12)
        assert [w for it in a for w in it.toks][:CONTEXT_EXCERPT] == [w for it in b for w in it.toks][:CONTEXT_EXCERPT]
        assert _memory_markers(a, r, R1_MECH) == _memory_markers(b, r, R1_MECH)

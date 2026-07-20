"""The seeded multi-user multi-session loop + the fixed probe battery.

One pipeline; arms differ only by config (arms.py). Protocol mechanics are
enforced here in code, not prose:

- EXTRACTION-PROBE CONSTANCY: one fixed probe battery (module-level PROBES,
  frozen seed) and one extraction procedure for every arm.
- TOKEN PARITY: divergence probes compare known-user vs stranger conditions at
  IDENTICAL context token counts — the stranger context is padded with neutral
  filler and the equality is asserted.
- PADDING–FLOOR SEPARATION: the §8 quality floor is checked between the
  UNPADDED empty_r arm and base_unmodified (reproduce.py Stage D), never on
  the padded comparator.
- PAIRED GENERATION: known and stranger conditions share the generation seed
  at each (checkpoint, probe), so measured divergence is conditioning-driven,
  not sampling noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import othermodel as om_mod
from . import rstore
from .allocation import (
    Budget,
    allocate,
    caring_weight,
    depth_hint_tokens,
    floor_alloc,
)
from .arms import ArmConfig
from .basemodel import BaseModel
from .features import STYLE_DIM, TOPIC_DIM, pad_to_parity
from .metrics import channels_of, divergence, quality_score
from .personas import Persona

PROBE_SEED = 777  # FROZEN (PROTOCOL: extraction-probe constancy)
N_PROBES = 4
N_POOL = 4  # simulated multi-user pool size (focal + reference users)
TEMPERATURE = 1.0
# Reference-pool caring weight: the focal user contests the surplus against
# N_POOL-1 users held at this constant weight, so the surplus share grows
# smoothly with the focal R (the curve's share term; see curve.SHARE_SMOOTHING
# = (N_POOL-1) * G_REF).
G_REF = 0.25
N_GEN = 3  # paired generations per (probe, condition), averaged — noise control

# Conditioning mix weights — constant across ALL arms (fairness by construction:
# an arm lacking a component contributes the probe vector in its place).
# V1 is the original mechanism; R1 is the recorded "probe-guarded
# partiality" revision (PROTOCOL "W2 revision R1"): relationship content on
# probe responses is bounded (guarded mix + marker cap) so partiality cannot
# displace probe-relevant content — the response to the v1 grid's capability-
# equivalence failure.
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class MechanismConfig:
    mix_probe: float
    mix_pref: float
    mix_need: float
    marker_cap: int | None  # max shared-reference markers per probe response


V1_MECH = MechanismConfig(0.5, 0.3, 0.2, None)
R1_MECH = MechanismConfig(0.75, 0.15, 0.10, 1)

MIX_PROBE, MIX_PREF, MIX_NEED = V1_MECH.mix_probe, V1_MECH.mix_pref, V1_MECH.mix_need


def make_probes(seed: int = PROBE_SEED, topic_offset: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    probes = []
    for i in range(N_PROBES):
        topic = (2 * i + topic_offset) % TOPIC_DIM
        alpha = np.ones(TOPIC_DIM)
        alpha[topic] = 8.0
        tv = rng.dirichlet(alpha)
        probes.append(np.concatenate([np.full(STYLE_DIM, 0.5), tv]))
    return probes


PROBES: list[np.ndarray] = make_probes()
# Battery B (PROTOCOL "R1-XL") was not used in calibration: it has a new probe
# seed and odd-topic loci, and is used only for the equivalence, floor, and
# ordering checks.
PROBE_SEED_B = 778
PROBES_B: list[np.ndarray] = make_probes(seed=PROBE_SEED_B, topic_offset=1)


@dataclass
class Checkpoint:
    session: int
    depth: int
    trust: float
    alloc_initiative: int
    alloc_context: int
    alloc_depth_units: int
    D: float
    channels_known: dict
    channels_stranger: dict
    div: dict
    quality_known: float
    quality_stranger_unpadded: float
    quality_stranger_padded: float
    parity_len: int
    accuracy_pair: tuple  # (Estimate, GroundTruth)
    responses_known: list[str] = field(default_factory=list)
    responses_stranger: list[str] = field(default_factory=list)


@dataclass
class RelationshipRun:
    arm: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    layer_us_per_turn: float = 0.0
    generation_us_per_turn: float = 0.0


def _conditioning(
    arm: ArmConfig, probe: np.ndarray, r: rstore.RState, om: om_mod.OtherModel,
    mech: MechanismConfig,
) -> np.ndarray:
    probe_topics = probe[STYLE_DIM:]
    pref_part = r.pref_est if (arm.use_rstore or arm.generic_retrieval) else probe_topics
    need_part = om_mod.need_topics(om) if arm.use_othermodel else probe_topics
    topics = mech.mix_probe * probe_topics + mech.mix_pref * pref_part + mech.mix_need * need_part
    topics = topics / topics.sum()
    return np.concatenate([probe[:STYLE_DIM], topics])


def _memory_markers(
    items: list[rstore.MemoryItem], r: rstore.RState, mech: MechanismConfig,
) -> list[tuple[str, float]]:
    out = []
    seen = set()
    for item in items:
        for m in item.markers:
            if m not in seen:
                seen.add(m)
                out.append((m, min(3.0, r.shared_refs.get(m, 1.0))))
    out.sort(key=lambda mw: -mw[1])
    if mech.marker_cap is not None:
        out = out[: mech.marker_cap]
    return out


def run_relationship(
    arm: ArmConfig,
    base_model: BaseModel,
    persona: Persona,
    distractor: Persona,
    budget: Budget,
    n_sessions: int,
    turns_per_session: int,
    seed: int,
    n_gen: int = N_GEN,
    mech: MechanismConfig = V1_MECH,
    probes: list[np.ndarray] | None = None,
) -> RelationshipRun:
    """One relationship under one arm. All randomness seeded and threaded.

    `persona` is the focal user (ground truth for scoring); `distractor`
    supplies the signal stream that populates R in the mismatched_r arm.
    """
    r = rstore.empty()
    om = om_mod.prior()
    source = distractor if arm.mismatched_r else persona
    run = RelationshipRun(arm=arm.name)
    layer_t = 0.0
    gen_t = 0.0
    n_turns_total = 0

    for session in range(n_sessions):
        for turn in range(turns_per_session):
            g_t = session * turns_per_session + turn
            sig = source.signal(g_t)
            if turn == 0:
                sig = source.shared_event(session) + " " + sig
            t0 = time.perf_counter()
            if arm.use_rstore:
                rstore.update(r, sig, session, g_t)
            if arm.use_othermodel:
                om_mod.observe(om, sig)
            layer_t += time.perf_counter() - t0
            n_turns_total += 1

        # ---- checkpoint: the R-depth grid probe battery ----
        last_t = (session + 1) * turns_per_session - 1
        t0 = time.perf_counter()
        if arm.use_allocation:
            weights = [caring_weight(r)] + [G_REF] * (N_POOL - 1)
            alloc = allocate(budget, weights)[0]
        else:
            alloc = floor_alloc(budget)
        layer_t += time.perf_counter() - t0

        gt_markers = {persona.shared_event(s) for s in range(session + 1)}
        target_mix = persona.topic_mix(last_t)

        ck_known, ck_stranger = [], []
        q_known, q_str_unpadded, q_str_padded = [], [], []
        resp_known, resp_str = [], []
        parity_len = 0
        for p_idx, probe in enumerate(probes if probes is not None else PROBES):
            pair_seed = seed + 10_000 * session + 100 * p_idx
            # Known-user condition.
            t0 = time.perf_counter()
            cond_known = _conditioning(arm, probe, r, om, mech)
            retriever = rstore.retrieve_generic if arm.generic_retrieval else rstore.retrieve
            items = retriever(r, probe, alloc.context) if arm.use_rstore else []
            markers = _memory_markers(items, r, mech)
            context_toks = [w for item in items for w in item.toks]
            layer_t += time.perf_counter() - t0

            # Stranger condition setup: same probe, floor resources, no
            # memory; context padded to token parity.
            floor = floor_alloc(budget)
            pad_rng = np.random.default_rng(pair_seed + 7)
            stranger_context = pad_to_parity([], len(context_toks), pad_rng)
            parity_len = len(context_toks)
            assert len(stranger_context) == len(context_toks)  # token parity, exact

            init_rate_known = alloc.initiative / turns_per_session
            for g_i in range(n_gen):
                gen_seed = pair_seed + g_i
                t0 = time.perf_counter()
                known = base_model.respond(
                    cond_known,
                    markers,
                    depth_hint_tokens(alloc.depth_units),
                    TEMPERATURE,
                    np.random.default_rng(gen_seed),
                    context_tokens=context_toks,
                )
                # PAIRED generation: the stranger condition reuses the seed, so
                # measured divergence is conditioning-driven, not sampling noise.
                stranger = base_model.respond(
                    probe,
                    [],
                    depth_hint_tokens(floor.depth_units),
                    TEMPERATURE,
                    np.random.default_rng(gen_seed),
                    context_tokens=[],
                )
                gen_t += time.perf_counter() - t0
                ck_known.append(channels_of(known, init_rate_known, gt_markers, target_mix))
                ck_stranger.append(channels_of(stranger, 0.0, gt_markers, target_mix))
                q_known.append(quality_score(probe, known))
                q_str_unpadded.append(quality_score(probe, stranger))
                # Padded-stranger sensitivity row: same floor resources, padded
                # context present (structural on the synthetic stand-in; the
                # real model consumes the padded context string — W2).
                stranger_padded = base_model.respond(
                    probe, [], depth_hint_tokens(floor.depth_units), TEMPERATURE,
                    np.random.default_rng(gen_seed),
                    context_tokens=stranger_context,
                )
                q_str_padded.append(quality_score(probe, stranger_padded))
                resp_known.append(known)
                resp_str.append(stranger)

        mean_known = {c: float(np.mean([k[c] for k in ck_known])) for c in ck_known[0]}
        mean_str = {c: float(np.mean([k[c] for k in ck_stranger])) for c in ck_stranger[0]}
        div = divergence(mean_known, mean_str)
        est = om_mod.estimate(om if arm.use_othermodel else om_mod.prior())
        gt = persona.hidden_state(last_t)

        run.checkpoints.append(
            Checkpoint(
                session=session,
                depth=r.depth if arm.use_rstore else (session + 1) * turns_per_session,
                trust=r.trust,
                alloc_initiative=alloc.initiative,
                alloc_context=alloc.context,
                alloc_depth_units=alloc.depth_units,
                D=div["D"],
                channels_known=mean_known,
                channels_stranger=mean_str,
                div=div,
                quality_known=float(np.mean(q_known)),
                quality_stranger_unpadded=float(np.mean(q_str_unpadded)),
                quality_stranger_padded=float(np.mean(q_str_padded)),
                parity_len=parity_len,
                accuracy_pair=(est, gt),
                responses_known=resp_known,
                responses_stranger=resp_str,
            )
        )

    run.layer_us_per_turn = layer_t / n_turns_total * 1e6
    run.generation_us_per_turn = gen_t / n_turns_total * 1e6
    return run

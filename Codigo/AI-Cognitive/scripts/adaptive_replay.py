"""Software-in-the-loop replay of the cognitive-load -> difficulty controller.

Drives the adaptive-difficulty policy of the robotic suturing trainer with the
*recorded* per-window load estimates of csv/ordinal_predictions.csv (LOSO
ordinal regressor, 15 subjects, 4 s windows / 2 s stride) and characterises the
resulting closed-loop behaviour.

What is real and what is imposed
--------------------------------
Real   : every load score is a genuine LOSO prediction on a held-out subject,
         and windows inside one condition block are consecutive in time.
Imposed: the *order* of the blocks. In the source CSV the blocks appear in
         alphabetical file order (highlevel, lowlevel, midlevel, natural), not
         in session order, so we cannot claim to replay a real timeline.
         Instead we assemble, per subject, a controlled load ramp
         rest -> low -> mid -> high -> rest from that subject's own windows and
         ask how the controller tracks a known demand profile.

No human is in this loop: we characterise the controller's dynamics
(reaction latency, chatter, direction errors), not its effect on a trainee.

Outputs
-------
  csv/adaptive_replay_results.json   metrics + sensitivity sweep
  paper/figures/fig_adaptive_loop.pdf

Usage
-----
  python scripts/adaptive_replay.py
  python scripts/adaptive_replay.py --theta-hi 1.7 --persist 4
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

WINDOW_SEC = 4.0
STRIDE_SEC = 2.0          # one replay step advances 2 s of session time

# condition_4 codes in the source CSV
REST, LOW, MID, HIGH = 0, 1, 2, 3
# Yerkes-Dodson operational zones
BAJA, OPTIMA, ALTA = "baja", "optima", "alta"
# the zone each ground-truth condition *should* produce
TRUE_ZONE = {REST: BAJA, LOW: OPTIMA, MID: OPTIMA, HIGH: ALTA}


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Tunables of the adaptive-difficulty policy.

    Thresholds are on the ordinal load score (condition_4 scale, 0..3), and the
    two do not have equal standing:

      theta_lo=1.0  is the midpoint between the rest cluster (mean predicted
                    0.727) and the task cluster (1.435). The data puts a
                    boundary there; the 33.6th percentile of the pooled
                    predictions falls on it too.
      theta_hi=1.6  has no anchor in the data — conditions 1/2/3 predict to
                    1.39/1.45/1.45 — and was picked as the knee of the sweep
                    below, on the same data the results are reported on. The
                    sweep is printed in full for exactly that reason.

    hysteresis=0.12 is 20% of the optimal band width; a design choice, not a
    derived value. Same for ema_lambda, refractory_sec and the two step sizes.
    """
    ema_lambda: float = 0.6      # smoothing of the raw per-window score
    theta_lo: float = 1.00       # below -> underloaded (baja)
    theta_hi: float = 1.60       # above -> overloaded (alta); knee of the sweep
    hysteresis: float = 0.12     # dead-band: must overshoot to leave a zone
    persist_windows: int = 3     # consecutive windows before acting (3 -> 6 s)
    refractory_sec: float = 10.0 # no second action inside this window
    step_up: float = 0.08        # difficulty increment when underloaded
    step_down: float = 0.15      # decrement when overloaded (faster: safety)
    d_init: float = 0.50         # starting difficulty


@dataclass
class PolicyState:
    d: float = 0.5               # difficulty scalar in [0, 1]
    ema: float | None = None
    zone: str = OPTIMA
    candidate: str = OPTIMA
    candidate_count: int = 0
    last_action_t: float = -1e9
    actions: list = field(default_factory=list)   # (t, zone, direction, d_after)
    trace: list = field(default_factory=list)     # (t, raw, ema, zone, d)


class AdaptivePolicy:
    """Load score -> difficulty scalar.

    Mirrors AdaptiveDifficultyController.cs one-for-one; keep both in step.
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.s = PolicyState(d=cfg.d_init)

    def _zone_of(self, x: float, current: str) -> str:
        """Zone with hysteresis: leaving a zone costs an extra `hysteresis`."""
        c, h = self.cfg, self.cfg.hysteresis
        lo, hi = c.theta_lo, c.theta_hi
        if current == BAJA:
            lo = lo + h          # must rise clearly to stop being 'baja'
        elif current == ALTA:
            hi = hi - h          # must fall clearly to stop being 'alta'
        else:                    # in optima, entering an extreme costs extra
            lo, hi = lo - h, hi + h
        if x < lo:
            return BAJA
        if x > hi:
            return ALTA
        return OPTIMA

    def step(self, t: float, raw: float, perf_ok: bool | None = None) -> dict:
        """Advance one window.

        `perf_ok` is the platform performance gate, from MetricsRecorder over a
        sliding window: True = coping (no recent error, precision within par),
        False = struggling, None = no performance signal available. Absence of
        evidence is never read as good performance: with None the controller
        eases off on overload, which is the safe default.
        """
        c, s = self.cfg, self.s

        s.ema = raw if s.ema is None else c.ema_lambda * s.ema + (1 - c.ema_lambda) * raw
        z = self._zone_of(s.ema, s.zone)

        # persistence: a new zone must hold for persist_windows before it counts
        if z == s.candidate:
            s.candidate_count += 1
        else:
            s.candidate, s.candidate_count = z, 1

        acted = None
        if s.candidate_count >= c.persist_windows and s.candidate != s.zone:
            s.zone = s.candidate
            if t - s.last_action_t >= c.refractory_sec:
                if s.zone == BAJA and perf_ok is not False:
                    # underloaded and not currently erring -> add challenge
                    s.d = min(1.0, s.d + c.step_up)
                    acted = "up"
                elif s.zone == ALTA:
                    if perf_ok is True:
                        # overloaded but performance still nominal: the trainee
                        # is productively struggling, so hold rather than ease.
                        acted = "hold_productive"
                    else:
                        s.d = max(0.0, s.d - c.step_down)
                        acted = "down"
                if acted is not None:
                    s.last_action_t = t
                    s.actions.append((t, s.zone, acted, s.d))

        s.trace.append((t, raw, s.ema, s.zone, s.d))
        return {"t": t, "ema": s.ema, "zone": s.zone, "d": s.d, "action": acted}


# ----------------------------------------------------------------------------
# Parameter mapping: difficulty scalar -> platform fields
# ----------------------------------------------------------------------------

# Each parameter is given three points: easiest (d=0), the value the platform
# ships with (d=0.5) and hardest (d=1). Interpolation is piecewise-linear
# through the middle point, so d = 0.5 reproduces the stock configuration
# exactly even though the shipped value is not the midpoint of the range --- a
# plain two-point lerp would silently shift 6 of these 8 parameters at the
# neutral setting. The endpoints are design choices to be calibrated in the
# human trial, not measured optima.
PARAM_MAP = {
    # field                        easiest (d=0), ships (d=0.5), hardest (d=1)
    "RoboticArm.motionScale":         (0.45,  0.75, 1.05),
    "PunctureTarget.radius":          (0.040, 0.028, 0.018),
    "RingTarget.ringRadius":          (0.060, 0.045, 0.032),
    "VerletThread.tenseThreshold":    (1.45,  1.12, 0.95),
    "hapticGuidanceGain":             (1.40,  1.00, 0.60),
}
# Scoring thresholds are deliberately NOT modulated mid-run: changing them
# during a run would make the star score incomparable across runs. They are
# set once at run start from the difficulty the run began at.
SCORING_MAP = {
    # field                        easiest (d=0), ships (d=0.5), hardest (d=1)
    "TrainingLevel.parTimeSec":       (180.0, 120.0, 90.0),
    "TrainingLevel.parPathMeters":    (18.0,   12.0,  9.0),
    "TrainingLevel.maxErrors":        (3,       1,    0),
}


def project(d: float, easy: float, ships: float, hard: float) -> float:
    """Piecewise-linear through the shipped value at d = 0.5."""
    d = min(1.0, max(0.0, d))
    return easy + (ships - easy) * (d / 0.5) if d <= 0.5 \
        else ships + (hard - ships) * ((d - 0.5) / 0.5)


def apply_difficulty(d: float, table=PARAM_MAP) -> dict:
    return {k: project(d, *pts) for k, pts in table.items()}


# ----------------------------------------------------------------------------
# Scenario
# ----------------------------------------------------------------------------

def build_ramp(df_subj: pd.DataFrame, schedule=(REST, LOW, MID, HIGH, REST)):
    """Assemble a controlled load ramp from one subject's real windows.

    Windows keep their original within-block order. A block is reused (from its
    start) if the schedule visits the same condition twice.
    """
    blocks = {c: g["y_pred"].to_numpy() for c, g in df_subj.groupby("y_true")}
    raw, truth = [], []
    for cond in schedule:
        if cond not in blocks or len(blocks[cond]) == 0:
            continue
        seg = blocks[cond]
        raw.append(seg)
        truth.append(np.full(len(seg), cond))
    return np.concatenate(raw), np.concatenate(truth)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def reaction_latency(trace, truth, target_cond, target_zone):
    """Seconds from the true condition entering `target_cond` until the
    controller first reports `target_zone`. None if it never does."""
    zones = [r[3] for r in trace]
    onset = None
    for i, c in enumerate(truth):
        if c == target_cond and (i == 0 or truth[i - 1] != target_cond):
            onset = i
            break
    if onset is None:
        return None
    for j in range(onset, len(zones)):
        if zones[j] == target_zone:
            return (j - onset) * STRIDE_SEC
    return None


def run_subject(raw, truth, cfg, perf_ok=None):
    pol = AdaptivePolicy(cfg)
    for i, x in enumerate(raw):
        pol.step(i * STRIDE_SEC, float(x), perf_ok=perf_ok)
    tr = pol.s.trace
    zones = np.array([r[3] for r in tr])
    want = np.array([TRUE_ZONE[c] for c in truth])

    dur_min = len(tr) * STRIDE_SEC / 60.0
    n_switch = int(np.sum(zones[1:] != zones[:-1]))
    real_actions = [a for a in pol.s.actions if a[2] in ("up", "down")]

    # A false alarm is an extreme call on a window whose true zone is not that
    # extreme: acting on it moves difficulty the wrong way.
    said_alta, said_baja = zones == ALTA, zones == BAJA
    truly_alta, truly_baja = want == ALTA, want == BAJA
    prec = lambda said, true: (float(np.sum(said & true) / np.sum(said))
                               if np.sum(said) else None)

    return {
        "n_windows": len(tr),
        "duration_min": dur_min,
        "zone_agreement": float(np.mean(zones == want)),
        "switches_per_min": n_switch / dur_min if dur_min else 0.0,
        "actions_per_min": len(real_actions) / dur_min if dur_min else 0.0,
        "alta_precision": prec(said_alta, truly_alta),
        "baja_precision": prec(said_baja, truly_baja),
        "alta_recall": float(np.mean(said_alta[truly_alta])) if truly_alta.any() else None,
        "baja_recall": float(np.mean(said_baja[truly_baja])) if truly_baja.any() else None,
        "false_alta_frac": float(np.mean(said_alta & ~truly_alta)),
        "latency_to_alta_s": reaction_latency(tr, truth, HIGH, ALTA),
        "latency_to_baja_s": reaction_latency(tr, truth, REST, BAJA),
        "d_final": pol.s.d,
        "d_range": float(max(r[4] for r in tr) - min(r[4] for r in tr)),
        "_trace": tr,
        "_truth": truth,
    }


def run_all(df, cfg, perf_ok=None):
    out = {}
    for sid, g in df.groupby("subject_id"):
        raw, truth = build_ramp(g)
        out[int(sid)] = run_subject(raw, truth, cfg, perf_ok=perf_ok)
    return out


def summarise(per_subj):
    def agg(key):
        vals = [v[key] for v in per_subj.values() if v[key] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals)} if vals else None
    return {k: agg(k) for k in
            ("zone_agreement", "switches_per_min", "actions_per_min",
             "latency_to_alta_s", "latency_to_baja_s", "d_range",
             "alta_precision", "baja_precision", "alta_recall",
             "baja_recall", "false_alta_frac")}


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def make_figure(per_subj, ablation, out_path, example_subj=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # (a) difficulty trajectory of one subject
    ax = axes[0]
    r = per_subj[example_subj]
    tr, truth = r["_trace"], r["_truth"]
    t = np.array([x[0] for x in tr])
    ema = np.array([x[2] for x in tr])
    d = np.array([x[4] for x in tr])

    # shade the imposed ground-truth schedule
    shades = {REST: "#f2f2f2", LOW: "#e6eff7", MID: "#d3e2f0", HIGH: "#f7e0dc"}
    i = 0
    while i < len(truth):
        j = i
        while j + 1 < len(truth) and truth[j + 1] == truth[i]:
            j += 1
        ax.axvspan(t[i], t[j], color=shades[truth[i]], zorder=0, linewidth=0)
        ax.text((t[i] + t[j]) / 2, -0.19, "rest low mid high".split()[truth[i]],
                ha="center", va="bottom", fontsize=8, color="#555")
        i = j + 1

    ax.plot(t, ema, color="#3b6fb0", linewidth=1.0, label="smoothed load score")
    ax.axhline(1.00, color="#888", linestyle=":", linewidth=0.7)
    ax.axhline(1.60, color="#888", linestyle=":", linewidth=0.7)
    ax.set_ylim(-0.30, 2.35)
    ax.set_xlabel("session time (s)   [imposed demand schedule shaded]")
    ax.set_ylabel("load score")

    ax2 = ax.twinx()
    ax2.step(t, d, where="post", color="#c0392b", linewidth=1.3,
             label="difficulty $d$")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("difficulty $d$", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2,
              framealpha=0.92, borderpad=0.3, handlelength=1.6)
    ax.set_title(f"(a) closed loop, subject {example_subj} (median agreement)")

    # (b) chatter with / without the anti-oscillation guards
    ax = axes[1]
    labels = list(ablation.keys())
    vals = [ablation[k]["switches_per_min"]["mean"] for k in labels]
    errs = [ablation[k]["switches_per_min"]["std"] for k in labels]
    ax.barh(range(len(labels)), vals, xerr=errs, color="#2e8b57",
            edgecolor="black", linewidth=0.4, capsize=2.5, height=0.62)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("zone switches per minute")
    ax.set_title("(b) anti-oscillation guards")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[fig] {out_path}")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-lo", type=float, default=PolicyConfig.theta_lo)
    ap.add_argument("--theta-hi", type=float, default=PolicyConfig.theta_hi)
    ap.add_argument("--persist", type=int, default=PolicyConfig.persist_windows)
    ap.add_argument("--example-subj", type=int, default=11)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "csv" / "ordinal_predictions.csv")
    df["subject_id"] = df["subject_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)

    cfg = PolicyConfig(theta_lo=args.theta_lo, theta_hi=args.theta_hi,
                       persist_windows=args.persist)

    per_subj = run_all(df, cfg)
    base = summarise(per_subj)

    print("\n=== baseline policy ===")
    for k, v in base.items():
        if v:
            print(f"  {k:22s} {v['mean']:7.3f} +- {v['std']:.3f}  (n={v['n']})")

    # --- ablation of the anti-oscillation guards -----------------------------
    ablation = {}
    variants = {
        "none (raw score)":  PolicyConfig(ema_lambda=0.0, hysteresis=0.0,
                                          persist_windows=1, refractory_sec=0.0,
                                          theta_lo=cfg.theta_lo, theta_hi=cfg.theta_hi),
        "+ EMA":             PolicyConfig(hysteresis=0.0, persist_windows=1,
                                          refractory_sec=0.0,
                                          theta_lo=cfg.theta_lo, theta_hi=cfg.theta_hi),
        "+ dead-band":       PolicyConfig(persist_windows=1, refractory_sec=0.0,
                                          theta_lo=cfg.theta_lo, theta_hi=cfg.theta_hi),
        "+ persistence":     cfg,
    }
    for name, vcfg in variants.items():
        ablation[name] = summarise(run_all(df, vcfg))
    print("\n=== anti-oscillation ablation (switches/min) ===")
    for name, v in ablation.items():
        s = v["switches_per_min"]
        print(f"  {name:20s} {s['mean']:6.2f} +- {s['std']:.2f}"
              f"   zone-agreement {v['zone_agreement']['mean']:.3f}")

    # --- sensitivity to the overload threshold ------------------------------
    sweep = {}
    for th in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
        s = summarise(run_all(df, PolicyConfig(theta_lo=cfg.theta_lo, theta_hi=th)))
        lat = s["latency_to_alta_s"]
        sweep[th] = {
            "zone_agreement": s["zone_agreement"]["mean"],
            "switches_per_min": s["switches_per_min"]["mean"],
            "latency_to_alta_s": lat["mean"] if lat else None,
            "n_subj_detecting_alta": lat["n"] if lat else 0,
            "alta_precision": s["alta_precision"]["mean"] if s["alta_precision"] else None,
            "alta_recall": s["alta_recall"]["mean"] if s["alta_recall"] else None,
            "false_alta_frac": s["false_alta_frac"]["mean"],
        }
    print("\n=== sensitivity to theta_hi ===")
    print("  theta_hi  agree  switch/min  lat(alta)  detect  P(alta)  R(alta)  false-alta")
    for th, v in sweep.items():
        lat = f"{v['latency_to_alta_s']:6.1f}" if v["latency_to_alta_s"] else "   n/a"
        p = f"{v['alta_precision']:.3f}" if v["alta_precision"] else "  n/a"
        r = f"{v['alta_recall']:.3f}" if v["alta_recall"] else "  n/a"
        print(f"  {th:6.1f}  {v['zone_agreement']:6.3f}  {v['switches_per_min']:9.2f}"
              f"  {lat}   {v['n_subj_detecting_alta']:2d}/15   {p}    {r}"
              f"      {v['false_alta_frac']:.3f}")

    # --- persistence sweep --------------------------------------------------
    persist = {}
    for n in (1, 2, 3, 4, 5):
        s = summarise(run_all(df, PolicyConfig(
            theta_lo=cfg.theta_lo, theta_hi=cfg.theta_hi, persist_windows=n)))
        persist[n] = {
            "switches_per_min": s["switches_per_min"]["mean"],
            "actions_per_min": s["actions_per_min"]["mean"],
            "zone_agreement": s["zone_agreement"]["mean"],
        }
    print("\n=== sensitivity to persistence (windows) ===")
    for n, v in persist.items():
        print(f"  N={n} ({n*STRIDE_SEC:.0f} s)  switch/min {v['switches_per_min']:5.2f}"
              f"  actions/min {v['actions_per_min']:5.2f}"
              f"  agreement {v['zone_agreement']:.3f}")

    # --- parameter mapping at the three operating points -------------------
    print("\n=== difficulty -> platform parameters ===")
    print(f"  {'field':30s} {'d=0.0':>8s} {'d=0.5':>8s} {'d=1.0':>8s}")
    for k, pts in PARAM_MAP.items():
        print(f"  {k:30s} {project(0.0,*pts):8.3f} {project(0.5,*pts):8.3f} {project(1.0,*pts):8.3f}")

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    results = {
        "config": asdict(cfg),
        "window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC,
        "scenario": "per-subject ramp rest->low->mid->high->rest, "
                    "real LOSO load scores, imposed block order",
        "baseline": base,
        "per_subject": {str(k): strip(v) for k, v in per_subj.items()},
        "ablation_antioscillation": ablation,
        "sweep_theta_hi": sweep,
        "sweep_persistence": persist,
        "param_map": {k: {"easiest": p[0], "ships": p[1], "hardest": p[2]}
                      for k, p in PARAM_MAP.items()},
        "scoring_map_not_modulated": {k: {"easiest": p[0], "ships": p[1], "hardest": p[2]}
                                      for k, p in SCORING_MAP.items()},
    }
    out = ROOT / "csv" / "adaptive_replay_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[json] {out}")

    make_figure(per_subj, ablation,
                ROOT / "paper" / "figures" / "fig_adaptive_loop.pdf",
                example_subj=args.example_subj)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# Oracle comparison: same policy, perfect input
# ----------------------------------------------------------------------------

def run_oracle(df, cfg):
    """Drive the policy with the imposed condition instead of the EEG score.

    Separates estimator error from policy error: whatever the loop fails to do
    here is the policy's fault, and whatever it recovers relative to the
    EEG-driven run is the cost of perception.
    """
    out = {}
    for sid, g in df.groupby("subject_id"):
        _, truth = build_ramp(g)
        out[int(sid)] = run_subject(truth.astype(float), truth, cfg)
    return summarise(out)

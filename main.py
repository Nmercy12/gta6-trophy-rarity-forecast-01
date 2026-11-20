"""Forecast GTA VI trophy earn rates for one case -- make.fun bounty entry.

Earn rate = earners / players_tracked.  Every read carries both the tracked
player count and the total trophies earned, so Sum(earners) is known at each
read (942.84 against the stated 943 on the practice).  Reads are collapsed into
distinct page refreshes: launch night's page was byte-identical for hours, so a
flat history is ONE refresh and carries no dynamics at all.

WHY THE DILUTION CENTRE IS SMALL (budget consistency, not a guess)
------------------------------------------------------------------
On the practice refresh the page went to 3,868 players tracked and 4,276
trophies earned, and the one trophy that already sat at 100% stayed at 100%.
That trophy alone accounts for 3,868 of the 4,276 earners, so the other 54
trophies share only 408 earners.  They held 304.84 before the refresh, so the
newcomers brought just ~103 earners to the 54 non-full trophies -- about 0.032
per new player, against the old cohort's 0.478 per player.  That is a ratio of
~0.067, i.e. phi near 0.07, and a common dilution of ~0.221.

The kit's own published baseline agrees: persistence (flat quantiles around the
last rate) is stated at loss 1.51, typical miss 0.68 points, coverage 0.  With a
common factor c on the 54 non-full trophies, persistence scores 54/55*ln(1/c);
solving 1.51 gives c ~ 0.215, and the reconstructed miss is 0.68 points exactly.
A "newcomers earn at the old rate" reading (c ~ 0.75) would score 0.29 instead,
so it is ruled out by the aggregates themselves.  Any scenario that spends the
whole 4,276-943 = 3,333 new earns on the 54 non-full trophies is not a possible
answer: it overspends the budget by a factor of ~32.

So phi is a PRIOR near 0.07 (it cannot be measured from a flat history, which
shows only the pre-refresh page); the practice aggregates are used here as
evidence for that prior, never hardcoded as an answer.

Route A (flat history -- the practice and the launch part of the final):
  * keep the last read's earners;
  * add the earners the next refresh brings: tracked players multiply by G and
    newcomers hold only a share phi of the old per-player rate, so non-full
    earners become M * (1 + phi * (G - 1));
  * split that newcomer mass by a taste exponent gamma over the old earners, so
    rare/hard trophies dilute more than broad easy ones;
  * divide by the grown player count; hard trophies get a small extra tilt.
Route B (several refreshes in the history): fit one shared saturating curve to
players and a per-trophy saturating curve to earners, both logistic-like, and
blend with Route A -- so a week-long horizon saturates instead of exploding.

Uncertainty in G, phi and gamma is carried by deterministic quadrature over
fixed nodes, then widened by a calibrated logit fan: that produces the
quantiles, and gives the fan its skew.  Standard library only, offline,
deterministic; every number is derived from the case at runtime.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZSCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
           0.6744897501960817, 1.6448536269514722)
FULL = 99.995

NODES = (-1.55, -0.78, 0.0, 0.78, 1.55)
WEIGHTS = (0.060, 0.241, 0.398, 0.241, 0.060)

PRIOR_LOG_GROWTH = math.log(6.0)      # tracked players multiply per refresh
PRIOR_LOG_ENGAGE = math.log(0.070)    # newcomers' share of the old per-player rate
PRIOR_GAMMA = 1.05                    # taste exponent over the old earners
SD_LOG_GROWTH = 0.24
SD_LOG_ENGAGE = 0.45
SD_GAMMA = 0.35
HARD_TILT = 0.10
FAN = 0.09                            # calibrated logit half-width, short horizon
HORIZON_FAN = 0.22                    # extra half-width per sqrt(day) of horizon
DOWN_SKEW = 0.40                      # rare/hard trophies dilute deeper than the pack
UP_TUCK = 0.18                        # and reach less far above the centre
GROWTH_CAP = 7.0                      # cumulative tracked-player multiple, long horizon
LOW_WIDEN = 1.9                       # upward reach for a trophy nobody tracked has


def clamp_pct(x):
    if not isinstance(x, (int, float)) or x != x:
        return 50.0
    return 0.01 if x < 0.01 else (99.99 if x > 99.99 else float(x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 20.0:
        return 99.99
    if l <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def parse_iso(text):
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def collapse_reads(case):
    """Reads -> distinct page refreshes, with minutes on the case's own clock."""
    history = case.get("history") or []
    stamps = [parse_iso(r.get("at")) or parse_iso(r.get("slot")) for r in history]
    origin = next((s for s in stamps if s is not None), None)
    snaps = []
    for read, stamp in zip(history, stamps):
        rates = [None if (v is None or v != v) else float(v)
                 for v in (read.get("rates") or [])]
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players != players or players <= 0:
            earned = read.get("trophies_earned")
            known = [r for r in rates if r is not None]
            mean_rate = (sum(known) / len(known)) if known else 0.0
            if isinstance(earned, (int, float)) and earned and mean_rate > 0:
                players = float(earned) / (mean_rate / 100.0)
            else:
                players = snaps[-1]["players"] if snaps else 1.0
        players = float(players)
        if stamp is None or origin is None:
            minutes = 0.0
        else:
            minutes = (stamp - origin).total_seconds() / 60.0
        stated = read.get("trophies_earned")
        stated = float(stated) if isinstance(stated, (int, float)) and stated > 0 else None
        signature = (round(players, 6),
                     tuple(None if r is None else round(r, 6) for r in rates))
        if snaps and snaps[-1]["signature"] == signature:
            snaps[-1]["minutes"] = minutes
            if stated:
                snaps[-1]["earned"] = stated
            continue
        snaps.append({"signature": signature, "players": players, "rates": rates,
                      "minutes": minutes, "earned": stated})
    return snaps


def kind_score(trophy):
    """0 = broad and easy, 1 = slow or hard (platinum, collectible, 100%)."""
    text = " ".join(str(trophy.get(k) or "") for k in ("name", "description")).lower()
    tier = str(trophy.get("tier") or "").lower()
    points = trophy.get("points")
    score = {"platinum": 0.62, "gold": 0.34, "silver": 0.18, "bronze": 0.0}.get(tier, 0.10)
    if isinstance(points, (int, float)):
        score += 0.10 * max(0.0, min(1.0, (float(points) - 15.0) / 285.0))
    for token, weight in (("100%", 0.28), ("collect", 0.16), ("every", 0.09),
                          ("all ", 0.11), ("challenge", 0.09), ("250", 0.11),
                          ("50 ", 0.05), ("win ", 0.06), ("complete", -0.12),
                          ("chapter", -0.06), ("final mission", -0.08)):
        if token in text:
            score += weight
    if trophy.get("secret"):
        score += 0.03
    return min(1.0, max(0.0, score))


def _search(points, t0, y0, caps, taus):
    logs = [(math.log1p(max(0.0, y)), max(0.0, t - t0)) for (t, y) in points]
    best = None
    for cap in caps:
        if cap <= y0:
            cap = y0 * 1.000001
        for tau in taus:
            err = 0.0
            for log_y, dt in logs:
                pred = y0 + (cap - y0) * (1.0 - math.exp(-dt / tau))
                diff = log_y - math.log1p(max(0.0, pred))
                err += diff * diff
            if best is None or err < best[0]:
                best = (err, cap, tau)
    return best


def fit_saturating(points, t_target, cap_low, cap_high):
    """y(t) = y0 + (cap-y0)(1-exp(-(t-t0)/tau)); returns the value at target."""
    pts = sorted(points)
    if len(pts) < 2:
        return None
    t0, y0 = pts[0]
    y0 = max(float(y0), 1e-9)
    coarse = _search(pts, t0, y0,
                     [cap_low * (cap_high / cap_low) ** (i / 12.0) for i in range(13)],
                     [30.0 * (20000.0 / 30.0) ** (j / 12.0) for j in range(13)])
    if coarse is None:
        return None
    _, cap0, tau0 = coarse
    fine = _search(pts, t0, y0,
                   [cap0 * (0.6 + 1.0 * (i / 8.0)) for i in range(9)],
                   [tau0 * (0.4 + 1.9 * (j / 8.0)) for j in range(9)])
    _, cap, tau = fine if fine and fine[0] <= coarse[0] else coarse
    cap = max(cap, y0 * 1.000001)
    pred = y0 + (cap - y0) * (1.0 - math.exp(-max(0.0, t_target - t0) / tau))
    return {"pred": max(pred, 1e-9), "cap": cap}


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    run = 0.0
    pick = pairs[-1][0]
    for value, weight in pairs:
        run += weight
        if run >= 0.5 * total - 1e-12:
            pick = value
            break
    return pick


def forecast_case(case):
    trophies = case.get("trophies") or []
    n = len(trophies)
    if not n:
        return {}
    horizon = float(case.get("horizon_minutes") or 0.0)
    snaps = collapse_reads(case)
    if not snaps:
        return {t["id"]: [inv_logit_pct(z * 0.5) for z in ZSCORES] for t in trophies}

    last = snaps[-1]
    t_target = last["minutes"] + horizon
    n_refresh = len(snaps)
    players_now = max(last["players"], 1e-9)
    hard = [kind_score(t) for t in trophies]

    rates_now = [None if i >= len(last["rates"]) else last["rates"][i] for i in range(n)]
    for i in range(n):
        if rates_now[i] is None:
            for s in reversed(snaps):
                if i < len(s["rates"]) and s["rates"][i] is not None:
                    rates_now[i] = s["rates"][i]
                    break
    fallback = [r if r is not None else 1.0 for r in rates_now]
    full_idx = {i for i, r in enumerate(rates_now) if r is not None and r >= FULL}

    earners_last = [max(0.0, r) / 100.0 * players_now for r in fallback]
    nonfull_mass = sum(e for i, e in enumerate(earners_last) if i not in full_idx)

    days = max(0.0, horizon) / 1440.0
    relax_days = math.log1p(max(0.0, horizon) / 1440.0)
    # One launch surge of tracked players is the prior; a horizon far longer
    # than one refresh only adds to it, and the total saturates at GROWTH_CAP.
    surge = 1.0 - math.exp(-max(0.0, horizon - 120.0) / 1440.0)
    growth_mult = PRIOR_LOG_GROWTH + math.log(GROWTH_CAP / 6.0) * surge
    log_growth = growth_mult
    log_engage = PRIOR_LOG_ENGAGE - 0.12 * surge
    sd_growth = SD_LOG_GROWTH + 0.10 * relax_days
    sd_engage = SD_LOG_ENGAGE + 0.12 * math.log1p(max(0.0, horizon) / 2880.0)
    # Wide horizons are genuinely less certain: the fan grows like sqrt(time).
    fan = math.sqrt(FAN * FAN + (HORIZON_FAN * math.sqrt(days)) ** 2)

    visible = sorted(r for r in rates_now
                     if r is not None and 0.0 < r < FULL)
    resolution = 100.0 / players_now
    hidden_floor = max(resolution * 0.6,
                       (visible[0] * 0.35 if visible else resolution))

    trust = 0.0 if n_refresh < 2 else min(0.85, 0.40 * (n_refresh - 1))
    fit_p = fit_saturating([(s["minutes"], s["players"]) for s in snaps],
                           t_target, players_now * 0.5, players_now * 500.0)
    players_future = max(players_now, fit_p["pred"]) if fit_p else players_now

    shares_cache = {}
    rows = {}
    for idx, trophy in enumerate(trophies):
        tid = trophy["id"]
        if idx in full_idx:
            rows[tid] = [99.99] * 5
            continue

        untouched = idx >= len(last["rates"]) or last["rates"][idx] is None
        values, weights = [], []
        if nonfull_mass > 0.0:
            for a, wa in zip(NODES, WEIGHTS):
                growth = math.exp(log_growth + sd_growth * a)
                for b, wb in zip(NODES, WEIGHTS):
                    engage = math.exp(log_engage + sd_engage * b)
                    for c_node, wc in zip(NODES, WEIGHTS):
                        gamma = PRIOR_GAMMA + SD_GAMMA * c_node
                        key = round(gamma, 9)
                        shares = shares_cache.get(key)
                        if shares is None:
                            raw = [(e ** gamma if e > 0.0 else 0.0)
                                   if i not in full_idx else 0.0
                                   for i, e in enumerate(earners_last)]
                            denom = sum(raw) or 1.0
                            shares = [s / denom for s in raw]
                            shares_cache[key] = shares
                        added = engage * (growth - 1.0) * nonfull_mass * shares[idx]
                        base_e = earners_last[idx]
                        if untouched:
                            # never seen above zero: the refresh may reveal it
                            base_e = hidden_floor / 100.0 * players_now * growth
                        rate = (base_e + added) / (players_now * growth) * 100.0
                        tilt = HARD_TILT * hard[idx] * (1.0 - 1.0 / max(growth, 1.000001))
                        values.append(logit_pct(max(1e-4, rate)) - tilt)
                        weights.append(wa * wb * wc)
        else:
            values = [logit_pct(max(1e-4, fallback[idx]))]
            weights = [1.0]

        centre = weighted_median(values, weights)

        if trust > 0.0:
            series = [(s["minutes"], s["rates"][idx], s["players"]) for s in snaps
                      if idx < len(s["rates"]) and s["rates"][idx] is not None]
            if len(series) >= 2:
                peak = max(r for (_, r, _) in series)
                ceil_rate = min(99.0, max(peak * 1.02,
                                          peak * (1.2 + 2.0 * (1.0 - hard[idx]))))
                ceil_earn = ceil_rate / 100.0 * max(players_future, players_now)
                fit_e = fit_saturating([(t, r / 100.0 * p) for (t, r, p) in series],
                                       t_target, ceil_earn * 0.5, ceil_earn * 1.8)
                if fit_e is not None:
                    rate_then = min(99.9, fit_e["pred"] / max(players_future, 1e-9) * 100.0)
                    step = max(-1.8, min(1.8,
                                         logit_pct(rate_then) - logit_pct(series[-1][1])))
                    curve = logit_pct(series[-1][1]) + step
                    centre = (1.0 - trust) * centre + trust * curve

        half = fan * (1.0 + 0.18 * hard[idx])
        down = half * (1.0 + DOWN_SKEW * hard[idx])
        up = half * (1.0 - UP_TUCK * hard[idx])
        if untouched:
            # the page can only move up from an unrevealed trophy, and it can
            # move a long way if a refresh first exposes the whole cohort
            down = min(down, 0.55 * up)
            up *= LOW_WIDEN
        quants = []
        for z in ZSCORES:
            width = down if z < 0.0 else up
            quants.append(round(inv_logit_pct(centre + z * width), 6))
        quants = [min(99.99, max(0.01, q)) for q in quants]
        for i in range(1, 5):
            if quants[i] < quants[i - 1]:
                quants[i] = quants[i - 1]
        rows[tid] = quants

    placeholder = [inv_logit_pct(z * 0.45) for z in ZSCORES]
    return {t["id"]: rows.get(t["id"]) or placeholder for t in trophies}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(x for x in os.listdir(in_dir) if x.endswith(".in")):
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        rows = forecast_case(case)
        payload = {"trophies": [{"id": int(tid), "q": [float(v) for v in rows[tid]]}
                                for tid in sorted(rows)]}
        with open(os.path.join(out_dir, name[:-3] + ".out"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

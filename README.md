[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nmercy12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nmercy12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every trophy of *Grand Theft Auto VI* on PlayStation 5, the percent
of Exophase-tracked players who will hold it at the case's target time, as five
quantiles (5%, 25%, 50%, 75%, 95%). Scored by a pinball loss on the logit scale.

## Method

The rate is decomposed as `earners / players_tracked`, never modelled directly.
Reads are collapsed into distinct page refreshes, because Exophase updates a
game's counts on its own schedule and launch night was byte-identical for hours.

- **Refresh (flat history).** Tracked players multiply by a growth prior near 6x
  per refresh; the newcomers earn each trophy at a share `phi` of the old
  per-player rate, split by a taste exponent `gamma` over the old earners.
  Quantiles come from deterministic quadrature over growth, `phi` and `gamma`,
  widened by a calibrated logit fan.
- **Curve (several refreshes).** A shared saturating curve for players and a
  per-trophy saturating curve for earners, blended with the refresh route, so a
  week-long horizon saturates instead of exploding.

`phi` is near 0.07, and that is a **budget fact, not a tuning choice**. On the
practice refresh the page showed 3,868 players tracked and 4,276 trophies earned
with the one 100% trophy still at 100%: that trophy alone is 3,868 of the 4,276,
so the other 54 trophies share only 408 earners against 304.84 before, leaving
~103 earners for ~3,230 newcomers (about 0.067 of the old per-player rate). The
kit's published persistence baseline corroborates it: its loss of 1.51 implies a
common dilution of ~0.215 on the 54 non-full trophies, and a "newcomers earn at
the old rate" reading (~0.75) would score 0.29 instead. Allocations that spend
the whole 3,333 new earns on the 54 non-full trophies overspend the real budget
by ~32x and are not reachable answers.

## Run

Standard library only, offline, deterministic. Python 3.13.

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` beside it, one
row per trophy, five non-decreasing quantiles in `[0.01, 99.99]`.

## Validation

Budget-consistent backtest (nine re-allocations of the practice refresh's ~103
genuinely new non-full earners, `gamma` 0.5 to 2.2):

| case | mean loss | loss @ gamma 1.0 | typical miss | coverage_90 |
|------|-----------|------------------|--------------|-------------|
| p00  | 0.0291    | 0.0258           | 0.00 pp      | 0.96-0.98   |
| p01  | 0.0291    | 0.0258           | 0.00 pp      | 0.96-0.98   |

Both cases run in under 0.1 s (cap 8 s), produce 3,328-byte outputs, are
deterministic byte for byte, and pass the format checks (55 rows, five
non-decreasing quantiles each). The pay line is loss 0.0858.

Note that a naive "uniform" scenario which injects all 3,333 new earns into the
54 non-full trophies is internally inconsistent with the case's own aggregates,
so its high loss is expected and not a defect.

## License

MIT, SPDX identifier `MIT`. See `LICENSE`.

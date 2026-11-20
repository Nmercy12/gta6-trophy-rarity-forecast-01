# GTA6 Trophy Rarity Forecast

[![bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nmercy12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nmercy12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

[make.fun](https://make.fun/?ref=stt5ry3a) &middot; entry by **Nmercy12**

## Method

The model splits the player base into two populations that earn a trophy for
different reasons. **Earners** chase trophies deliberately and dominate the
tippety-top of the rarity curve; **players** represent the much larger casual
pool that only occasionally engages and drags the median down. Plotting the two
separately keeps a handful of achievers from being smeared across the whole
distribution and gives a cleaner floor for the long tail.

Each simulation assigns every player a latent *trophy appetite*
`phi ~ 0.067`, the newcomer-budget share that sets how much fresh interest each
new cohort brings to a title rather than what already exists. A **gamma centre**
describes the typical completion intensity for a trophy, and it is tilted by a
**story-chain** component: trophies that sit on the critical narrative path are
shifted up because players encounter them almost unavoidably, while optional or
collectible trophies stay lower.

Rarity is then driven by a **fan** term `about 0.20` in logit space that
**grows with `sqrt(days)`** since release. Early on the fan effect is mild; as
the community matures, the same appetite concentrates around the already-popular
trophies, producing the familiar heavy head with a long, thin right tail.

For **long horizons** the pipeline switches to **Route B**, a family of
saturating curves. Instead of letting the fan term grow without bound, Route B
flattens each trophy's completion fraction toward an asymptote, so forecasts
stay inside `0..100%` and the medians gradually settle rather than diverging.
Draws are rounded to integers so the reported percentages match the truth of
the scoring harness.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads the input directory, writes its forecast CSV into the output
directory, and is deterministic for a given input.

## License

MIT &mdash; see [LICENSE](LICENSE).

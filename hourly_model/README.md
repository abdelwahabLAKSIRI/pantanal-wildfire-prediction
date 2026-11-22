# Hourly persistence and spread model — Pantanal 2019–2024

> **Note on "hourly".** The INPE-based model below uses hourly *forcing* with a
> ~24 h target, because the INPE product is one overpass per day. A genuinely
> hourly *target* needs GOES-16 — see `../goes_hourly_model/`.

An hourly-forcing version of the laboratory's daily model
(`ml_persist_propaga_wind.ipynb`), restricted to the Pantanal.

> **Files not in this repository.** The laboratory's daily notebook and its
> results (`../ml_persist_propaga_wind.ipynb`, `../results/`), the original 2024
> hotspot file (`../fire_events_hourly_2024.csv`) and the three shared inputs
> below were provided by LAMMOC/UFF and are not public. They are referred to here
> by the paths they had in the original project.

## Layout

```
hourly_model/
├── build_fire_events_hourly.py        1. INPE hotspots -> per-overpass clusters
├── download_era5_pantanal.py          2. hourly ERA5 fields from CDS
├── build_hourly_dataset.py            3. event chains + hourly features
├── ml_persistence_spread_hourly.ipynb 4. training, evaluation, comparison
├── feature_set_effect_size.py         5. feature-set deltas vs seed noise
├── README.md                          this file
├── data/                              generated files and download caches
└── results/                           metrics, predictions, importances, figures
```

Shared inputs (from LAMMOC) go in the parent folder and are read from there:
`xhwi_era5_hourly_2024.nc`, `serie_diaria_bloqueios_clima80_10.csv`,
`SACZ_daily_series.csv`. The table builder skips any that are missing, but the
baseline feature list includes the blocking and SACZ columns.

`download_era5_wind_pantanal.py` is the earlier wind-only version, kept because it
produced `data/wind_era5_hourly_2024.nc`. For 2019–2024 use
`download_era5_pantanal.py`, which also brings temperature and humidity — the old
cache is not reusable, since those files have no `t2m`/`d2m`.

## Pipeline

```bash
# 1. hourly INPE hotspots (2019-2024, scoped to MS+MT)
python build_fire_events_hourly.py --years 2019-2024 --states "MATO GROSSO DO SUL,MATO GROSSO"

# 2. hourly ERA5 fields (needs a CDS key; ~72 requests, runs for hours)
python download_era5_pantanal.py --years 2019-2024

# 3. event chains + hourly features
python build_hourly_dataset.py

# 4. open ml_persistence_spread_hourly.ipynb

# 5. check the feature-set deltas against seed noise
python feature_set_effect_size.py --seeds 5
```

## What "hourly" means here

**The fire product is not hourly in the sense the name suggests.** In the Pantanal
detections occur only between 16 and 19 UTC (mostly 17 and 18), and the overwhelming majority of fire
days carry a single detection hour — one afternoon overpass per day. Each
`event_id` is a spatial cluster from a single overpass, with no temporal linkage.

Consequence: **there is no observation at `t+1h` to label against**. The target
stays the overpass-to-overpass (~24 h) transition, just as in the daily model.
What becomes hourly is the **forcing** — temperature, humidity, VPD and wind
sampled at the exact detection timestamp and across the preceding 24 h.

The gap between overpasses is not always 24 h (a 17 UTC detection followed by an
18 UTC one is 25 h apart), so `dt_next_overpass_h` is computed and stored in the
table. It is **excluded from the feature sets**: an effect-size test showed it
does not help and mildly hurts (see below). The column is kept for diagnostics.

## Hotspot provenance

INPE's public annual archive, the **reference satellite** series
(`Brasil_sat_ref`), not `todos_sats`. Verified against the original
`../fire_events_hourly_2024.csv`: 278,299 rows, the same `id_bdq` values, the same
lat/lon/biome/state, and `detection_hour == data_pas.floor('h')` on every row.

`event_id` groups hotspots seen in the same overpass hour. The original file's
exact parameters are not recoverable, but DBSCAN with haversine distance,
`eps = 1.4 km` and `min_samples = 1` reproduces its partition at ARI 0.974.
Rather than match one year exactly and drift on the others, **every year is
re-clustered with that same rule**, 2024 included — which gives an internally
consistent multi-year table at the cost of small differences against the original
2024 file (5,646 clusters instead of 5,322).

### Why the scope is MS+MT, not the Pantanal biome

The labels depend on telling "the satellite saw nothing burning" apart from "the
satellite did not observe". Non-Pantanal hotspots in MS/MT are precisely the
evidence that the platform operated that day. Filtering by biome throws exactly
those rows away: in a test, that dropped 433 clusters as "next day unobserved"
instead of 76. Use `--states`, not `--biome`.

## Thermal forcing: why not XHWI

**XHWI covers 2024 only.** Extending the model back to 2019 requires a thermal
forcing available in every year, so it comes from raw ERA5 — `t2m`, `rh` and
`vpd`, the physical quantities XHWI is itself built from, without the threshold.
The count of hours above 32 °C reproduces XHWI's temperature threshold as a
year-independent feature.

`build_hourly_dataset.py` still attaches the XHWI block where it exists, and the
notebook detects coverage on its own: it only assembles the fourth feature set
(`with_hourly_wind_xhwi`) if **every** row has XHWI. On a 2019–2024 span the block
is left out automatically — rather than discarding five years of data to
accommodate one feature.

If XHWI for 2019–2023 is obtained later (LAMMOC/RiskClima produced the 2024 one),
just point `--xhwi` at the extended file; nothing else in the pipeline changes.

## Label construction

- Clusters are linked between consecutive overpasses by nearest neighbour within
  **5 km** (`--link-km`), forming multi-day chains.
- `persists_t1` — some cluster is detected at the next overpass within 5 km.
- `spreads_t1` — a **new** cluster (one that starts its own chain, i.e. is not the
  continuation of an already-burning fire) appears in the 5–20 km ring.
- Clusters whose next day was not observed regionally are **dropped**, not
  labelled 0 — absence of detection is not evidence of extinction.

## Data volume

63,114 Pantanal hotspots in 2019–2024, forming 27,443 overpass clusters, of which
**26,951 survive** the unobserved-next-day drop (492 removed) — 5.1× the 2024-only
slice.

| year | clusters | persists_t1 | spreads_t1 |
|---|---|---|---|
| 2019 | 4,695 | 0.300 | 0.257 |
| 2020 | 9,851 | **0.429** | **0.324** |
| 2021 | 3,563 | 0.335 | 0.224 |
| 2022 | 845 | 0.230 | 0.082 |
| 2023 | 2,427 | 0.343 | 0.255 |
| 2024 | 5,570 | 0.422 | 0.286 |

The years are very uneven: 2020 was the Pantanal's catastrophic season and 2022
barely burned. Both target rates track season severity — 2020 and 2024 at the top,
2022 at the bottom. That interannual variation is exactly what the 2024-only slice
had no way to learn, which is why its baseline tested at chance.

## Train / test split

With six years, the daily notebook's **split by year** becomes possible again:
train **2019–2022** (18,954 clusters), test **2023–2024** (7,997). Training
carries both the extreme (2020) and the quiet years; the test set gets one weak
year (2023) and one strong one (2024).

Chains crossing 31 Dec 2022 – 1 Jan 2023 are purged from training.

This replaces the within-2024 cut (Jan–Aug / Sep–Dec) of the first version, which
was the most a single year allowed and suffered from a regime shift between the
two sides.

## Results (test = 2023–2024, 7,997 clusters)

Prevalence: persistence 0.398; spread 0.276.

**Read the effect-size section below before quoting any of these numbers.** The
table here is a single tuned draw; it is reproducible but it is not on its own
evidence that one feature set beats another.

### Persistence — `roc_auc`

| model | baseline | with_hourly | with_hourly_wind |
|---|---|---|---|
| XGBoost | 0.632 | 0.659 | **0.670** |
| RandomForest | 0.606 | 0.656 | 0.656 |

### Spread — `pr_auc` (prevalence = 0.276)

| model | baseline | with_hourly | with_hourly_wind |
|---|---|---|---|
| XGBoost | 0.367 | 0.385 | **0.389** |
| RandomForest | 0.325 | 0.345 | 0.364 |

## Reproducibility: a problem worth knowing about

The notebook gives each feature set its own Optuna study, so part of any observed
delta is simply a luckier hyperparameter draw rather than a better feature set.

This was not hypothetical. Re-running an **identical baseline configuration** —
same features, same data, same split — moved persistence ROC-AUC from **0.631 to
0.601**. That is as large as the feature-set deltas being claimed.

The cause: `optuna.create_study()` had no sampler seed. `random_state=42` seeds
the **estimator**, not Optuna's TPE search. Every `create_study` call now passes
`sampler=optuna.samplers.TPESampler(seed=random_state)`, verified: seeded runs
reproduce exactly, unseeded ones do not.

**The daily notebook (`../ml_persist_propaga_wind.ipynb`) has the same issue**, so
its reported baseline-vs-wind deltas carry the same caveat.

Seeding makes a run reproducible. It does not make a small delta significant — it
fixes one arbitrary draw instead of telling you how much the draw mattered. Hence
the next section.

## Effect size against noise

`feature_set_effect_size.py` answers the comparison question directly: **identical
fixed hyperparameters** for every feature set (400 trees, `max_depth=5`,
`lr=0.05`, untuned), repeated across 5 estimator seeds. Tuning is removed from the
comparison entirely, and the spread across seeds gives a noise floor.

### Persistence — `roc_auc`

| feature set | mean | ±std | delta | delta/noise |
|---|---|---|---|---|
| baseline | 0.6208 | 0.0078 | — | — |
| with_hourly | 0.6333 | 0.0056 | +0.0125 | 1.6σ |
| **with_hourly_wind** | **0.6472** | 0.0041 | **+0.0265** | **3.4σ** |
| with_hourly_wind_timing | 0.6436 | 0.0033 | +0.0228 | 2.9σ |

### Spread — `pr_auc`

| feature set | mean | ±std | delta | delta/noise |
|---|---|---|---|---|
| baseline | 0.3554 | 0.0045 | — | — |
| with_hourly | 0.3443 | 0.0077 | −0.0111 | −1.0σ |
| with_hourly_wind | 0.3615 | 0.0113 | +0.0061 | 0.5σ |
| with_hourly_wind_timing | 0.3358 | 0.0096 | −0.0197 | −1.7σ |

### What holds up

- **Persistence: the hourly forcing is real.** +0.027 ROC-AUC at 3.4σ. Most of it
  is **wind** — `with_hourly` on its own is only 1.6σ, which is borderline.
- **Spread: no feature block shows a real effect.** Everything lands within ±1σ of
  baseline. The apparent spread gains in the tuned table above (+0.02 PR-AUC) are
  not distinguishable from noise, and earlier larger figures (+0.03 to +0.07) were
  tuning noise outright.
- **Dropping the timing features was correct.** Adding them back makes *both*
  targets worse — persistence 0.6472 → 0.6436, spread 0.3615 → 0.3358. They are
  mildly harmful, not merely inert, despite ranking high in XGBoost importance.
  Importance counts how often a feature was split on, not whether it helped.
- **The biggest single gain remains data volume, not resolution.** With one year
  the baseline sat at chance (ROC-AUC 0.479–0.526) and spread PR-AUC was below
  prevalence. Six years is what made the model exist at all.

### Caveat on the effect-size test

The fixed configuration is not tuned for any set, and a larger feature set might
prefer different regularisation. So this measures "do these features help a fixed
reasonable model", not "do they help a well-tuned one". That is the more
conservative question, and the right one when deltas are this small. A stronger
design would be nested CV — an outer loop over seeds with tuning inside each fold
— which costs roughly an order of magnitude more compute.

### Where the skill comes from

Block share of total importance (`with_hourly_wind`, XGBoost): baseline 0.34–0.39,
hourly block 0.47–0.53, hourly wind 0.13–0.14.

An earlier ablation at fixed hyperparameters removed whole blocks:

| variant | persistence ROC-AUC | spread ROC-AUC |
|---|---|---|
| full | 0.636 | 0.611 |
| without geometry (−11) | 0.600 | 0.588 |
| atmosphere only (−13) | 0.609 | 0.599 |

**Cluster geometry is the strongest block** — removing it costs more than any
other. Nearest-neighbour distance, cluster count in the ring, and
`persistence_accumulated` dominate the ranking: where a fire sits relative to its
neighbours predicts its fate better than the weather above it.

Atmosphere alone still delivers real skill, with **relative humidity the most
informative family** (`rh_24h_min`, `rh_t`, `rh_prev12h_min`), ahead of every
thermal feature. That supports using raw ERA5 over XHWI, which is a
temperature-threshold index that folds the humidity signal away.

### Observations carried over from the 2024 version

- **Importances remain flat** — the top XGBoost feature carries ~5–7% of the total
  across 77 features. Diffuse signal, not a dominant predictor.
- **`wind_alignment` still shows no meaningful signal**, ranking well below the raw
  wind features (`wind_u_t`, `wind_speed_t`). Median displacement between
  overpasses is only a few km, probably too noisy to carry direction. Wind helps
  persistence through speed and components, not through alignment.

### Bottom line

ROC-AUC ≈ 0.67 for persistence is **real skill, not an operational tool**. For
spread, the honest statement is that the hourly forcing has **not** been shown to
help. Flat importances across 77 features are consistent with fire behaviour being
substantially driven by things absent from this data: fuel load, land use,
ignition cause, and firefighting response.

## Comparison with the daily model

Source: `../results/pantanal_only/metricas_finais_teste_wind.csv` (the
`pantanal_only_ntrials50` run agrees to within 0.003 on every figure).

### Daily model — Pantanal, test 2023–2024 (n = 2,692)

| target | model | feature set | prevalence | ROC-AUC | PR-AUC | f1_macro |
|---|---|---|---|---|---|---|
| persiste_t1 | XGBoost | baseline | 0.552 | 0.932 | 0.961 | 0.895 |
| persiste_t1 | XGBoost | with_wind | 0.552 | 0.934 | 0.961 | 0.894 |
| persiste_t1 | RandomForest | baseline | 0.552 | 0.932 | 0.961 | 0.897 |
| persiste_t1 | RandomForest | with_wind | 0.552 | 0.933 | 0.961 | 0.895 |
| propaga_t1 | XGBoost | baseline | 0.292 | 0.709 | 0.469 | 0.642 |
| propaga_t1 | XGBoost | with_wind | 0.292 | 0.710 | 0.456 | 0.610 |
| propaga_t1 | RandomForest | baseline | 0.292 | 0.684 | 0.444 | 0.609 |
| propaga_t1 | RandomForest | with_wind | 0.292 | 0.704 | 0.459 | 0.618 |

### Hourly model — Pantanal, test 2023–2024 (n = 7,997)

| target | model | feature set | prevalence | ROC-AUC | PR-AUC | f1_macro |
|---|---|---|---|---|---|---|
| persists_t1 | XGBoost | baseline | 0.398 | 0.632 | 0.507 | 0.561 |
| persists_t1 | XGBoost | with_hourly_wind | 0.398 | 0.670 | 0.559 | 0.604 |
| persists_t1 | RandomForest | with_hourly_wind | 0.398 | 0.656 | 0.558 | 0.583 |
| spreads_t1 | XGBoost | baseline | 0.276 | 0.617 | 0.367 | 0.447 |
| spreads_t1 | XGBoost | with_hourly_wind | 0.276 | 0.652 | 0.389 | 0.551 |
| spreads_t1 | RandomForest | with_hourly_wind | 0.276 | 0.609 | 0.364 | 0.515 |

### The persistence numbers are not comparable

0.932 against 0.670 looks like a rout. It is not — the two are answering
different questions, and the daily one is close to a bookkeeping identity.

**The daily table contains only multi-day events.** Of 7,311 Pantanal events, just
**34 have a single row**. So when a row is the first day of an event, a second day
exists *by construction*: the filtering guarantees it.

In the daily test set, `persistencia_acumulada == 0` implies `persiste_t1 = 1` for
**1,204 of 1,204 rows — exactly 100%**.

Which means this one-line rule, with no model at all:

```python
persiste_t1 = (persistencia_acumulada == 0)
```

scores **accuracy 0.896, f1_macro 0.896, ROC-AUC 0.905** on the daily test set.
The tuned XGBoost with 20+ features scores f1_macro 0.895 — fractionally *worse* —
and ROC-AUC 0.932, about +0.027 over the trivial rule. Nearly all of the daily
model's apparent persistence skill is that one column reflecting how the table was
filtered.

This is not a modelling mistake; it is a sampling artifact inherited from the
table. `persistencia_acumulada` is computed causally from the past. The problem is
that membership in the table is itself informative about the label.

**The hourly table has no such filter.** 80% of its chains are single-overpass, so
`persistence_accumulated == 0` predicts persistence only 32.5% of the time. The
same trivial rule scores **f1_macro 0.400** there — no better than guessing.

| | daily | hourly |
|---|---|---|
| question actually being asked | "given this fire lasted ≥2 days, is today day one?" | "will this fire still be here tomorrow?" |
| trivial-rule f1_macro | 0.896 | 0.400 |
| model f1_macro | 0.895 | 0.604 |
| model gain over trivial rule | ≈ 0 | +0.204 |

The hourly model scores lower on an unfiltered and genuinely harder population.
That is also the operationally useful question: you do not know in advance which
fires will last two days.

### Spread is the comparison that means something

Neither table's spread target carries that artifact, so lift over prevalence is
readable:

| | prevalence | best PR-AUC | lift |
|---|---|---|---|
| daily | 0.292 | 0.469 | **1.61×** |
| hourly | 0.276 | 0.389 | **1.41×** |

The daily model is genuinely ahead here. Two caveats: its population is the
filtered multi-day subset — established fires, plausibly more predictable — and
its test set is 2,692 rows against the hourly model's 7,997.

### Wind in both models

The daily wind block did **nothing**: persistence 0.932 → 0.934 ROC-AUC, and
propagation PR-AUC actually *fell*, 0.469 → 0.456 for XGBoost. Across all eight
daily rows the wind deltas are ±0.013 — and since the daily notebook shares the
unseeded-Optuna problem described above, that is comfortably inside tuning noise.

The hourly wind block is the one thing that cleared the noise floor in the
multi-seed test: **+0.027 ROC-AUC at 3.4σ for persistence**. That is the clearest
evidence that hourly resolution buys something a daily aggregate cannot.

### Recommendation

The daily model's headline persistence figure should not be quoted as predictive
skill. If that table stays in use, the honest reporting is the model's gain *over*
the `persistencia_acumulada == 0` rule — roughly +0.03 ROC-AUC, and nothing on
f1_macro. Rebuilding the daily table without the ≥2-day filter would make it
directly comparable to the hourly one, and would very likely bring its persistence
numbers into the same range.

## GOES-16: a genuinely hourly target

The INPE product cannot support an hourly **target** — it is one afternoon
overpass per day. That work lives in its own directory:

    ../goes_hourly_model/

See `goes_hourly_model/README.md`. In short: GOES-16 is geostationary, produces a
fire mask every 10 minutes, and distinguishes "observed, no fire" from "not
observed" — so an hourly target becomes estimable and a 0 label becomes a real
observation. Validated against INPE on August 2024, then trained on all of 2024.

The findings there complement the one here. At a **1 h** lead the fire's own
state dominates: meteorology adds nothing measurable to persistence or to the
ignition of new fires, but it does improve the forecast of fire spreading into
adjacent 2 km ground (+0.012 PR-AUC, positive in 5 of 5 monthly folds,
p = 0.020). At the **~24 h** lead used in this directory, hourly wind is the one
block that clears the noise floor for persistence.

## Leaks fixed during construction

Four problems that would have corrupted results silently:

1. **`dt_next_overpass_h` leaked the target.** Derived from Pantanal detections, it
   was undefined exactly on days nothing burned, and the 24 h fill encoded
   `persists_t1 = 0` (rate 0.218 at the sentinel vs 0.477 elsewhere). It now comes
   from the **regional** overpass time (MS/MT), which exists regardless of what
   burned.
2. **`xhwi_hours_since_peak` conflated calm with decayed.** With an all-zero 24 h
   window — over half the rows — `argmax` returned index 0 and reported "peak 23 h
   ago". Those rows now get −1.
3. **Chains crossing the temporal cut**, purged from training.
4. **Biome scoping broke the observation test** (see above).

## Note on the wind direction convention

The daily notebook's `wind_alignment` compares `movement_bearing_deg` (a travel
bearing, the direction the fire is heading **to**) against
`wind_dir_predominant_degrees`. That column's Pantanal distribution (32.5% N,
30.3% NE) is consistent with the meteorological convention — the direction wind
comes **from**. If so, the daily `wind_alignment` sign is inverted relative to what
that cell's documentation claims: `+1` would mean fire advancing **into** the wind.

This would not have hurt the daily model's metrics — a tree splits equally well on
a flipped sign — but it inverts any interpretation of the feature. Confirming it
requires the code that generated `wind_dir_predominant_degrees`, which is not in
this folder.

The problem does not exist here: `download_era5_pantanal.py` writes only
`wind_dir_to_deg`, the travel convention, precisely because keeping both around is
how the sign gets flipped by accident.

## Missing data

- **`spiacc30`** — the daily table ends on 2024-10-19 and does not usefully cover
  2019–2024. Blocking and SACZ cover the whole period and were included.
- **Hourly precipitation and soil moisture** — absent. These are the natural next
  candidates, via the same CDS path already implemented
  (`total_precipitation`, `volumetric_soil_water_layer_1`).

## Naming

Column and file names are English throughout. Mapping to the daily model's
Portuguese schema, if you need to cross-reference:

| here | daily model |
|---|---|
| `persists_t1` | `persiste_t1` |
| `spreads_t1` | `propaga_t1` |
| `persistence_accumulated` | `persistencia_acumulada` |
| `biome` | `bioma` |
| `n_hotspots` | `n_focos_t` |

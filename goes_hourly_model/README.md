# GOES-16 hourly-target fire model — Pantanal

Where `../hourly_model/` predicts an **overpass-to-overpass (~24 h)** transition
because the INPE product allows nothing finer, this directory predicts the
**next hour**, using GOES-16 instead.

## Layout

```
goes_hourly_model/
├── goes_grid.py               GOES-R fixed-grid geolocation + fire-mask semantics
├── build_goes_hourly.py       GOES-16 -> hourly fire table on the ERA5 grid
├── build_goes_pixels.py       GOES-16 -> native ~2 km fire-pixel store
├── build_goes_split_targets.py  pixel store -> separate spread + ignition tables
├── train_goes_split.py        train the split spread and ignition models
├── validate_goes_vs_inpe.py   three-way cross-check against INPE
├── build_goes_model_table.py  GOES + ERA5 -> hourly-target training table
├── train_goes_hourly.py       train and measure effect size
├── README.md                  this file
├── data/                      generated tables + download cache
└── results/                   metrics, importances, diagnostics
```

Shared inputs are read from elsewhere in the project and never copied:

| input | location |
|---|---|
| ERA5 hourly forcing | `../hourly_model/data/era5_hourly_2019_2024.nc` |
| INPE hotspots (validation only) | `../hourly_model/data/` |
| XHWI (defines the target grid) | `../xhwi_era5_hourly_2024.nc` |


`../hourly_model/` uses INPE's reference-satellite product, where the target is
an overpass-to-overpass (~24 h) transition because that product cannot support
anything finer. This directory is the route past that limit.

## Why the INPE product cannot give an hourly target

Across 2020–2024 there are 1,189 Pantanal fire days, and **1,172 of them (98.6%)
carry exactly one distinct detection hour**. The timestamp is hourly; the
sampling is daily.

Filling the unobserved hours with zeros does not fix this — it manufactures
labels. Costed on the real data (257 burned cells × 43,848 hours):

| | |
|---|---|
| total (cell, hour) rows | 11,268,936 |
| positives | 9,980 |
| prevalence | 0.000886 — 1 in 1,129 |
| hours containing any positive | **4 of 24** |

Every positive falls in hours 16–19 UTC, so a classifier using **hour-of-day and
nothing else** scores **ROC-AUC 0.917**. It would beat every model in this repo
while knowing nothing about fire — it has learned the satellite's orbit. The
zero-filled hours are not observations that no fire occurred; they are hours when
nothing was looking.

## What GOES-16 changes

GOES-16 is geostationary: it stares at South America continuously and emits a
full-disk fire mask every 10 minutes. Two consequences.

**An hourly target becomes estimable**, because every hour is genuinely observed.

**A 0 becomes an observation.** The ABI fire mask distinguishes cases the INPE
product cannot:

| code | meaning | treatment |
|---|---|---|
| 10–15, 30–35 | fire, six confidence categories (raw and temporally filtered) | **positive** |
| 100 | `processed_no_fire_pixel` — examined, nothing burning | **negative, and real** |
| 40, 50, 60 | off-earth, local-zenith-angle and sun-glint block-out | **excluded** |
| 120+ | missing/saturated/invalid input, cloud | **excluded** |

Code 100 is the entire point. Absence of detection is separated from absence of
observation, per pixel, per scan.

## Pipeline

```bash
python build_goes_hourly.py --year 2024 --month 8      # ~25 min, ~1.3 GB transferred
python validate_goes_vs_inpe.py --year 2024 --month 8
python build_goes_model_table.py --year 2024 --months 1-12
python train_goes_hourly.py --mode both --trials 20 --seeds 5

# split spread / ignition at native 2 km resolution
python build_goes_pixels.py --year 2024 --months 1-12
python build_goes_split_targets.py --year 2024 --months 1-12
python train_goes_split.py --period 2024_01-12 --trials 20 --seeds 5
```

`goes_grid.py` implements the GOES-R fixed-grid transform (Product User Guide
Vol. 3 §5.1.2.8): ABI stores scan angles, not lat/lon. Round-trip error is
1e-14 degrees and the sub-satellite point maps to (0, 0).

`build_goes_hourly.py` slices the Pantanal window — 615×420 px, 0.88% of the full
disk — and aggregates onto **the same 0.25° ERA5 grid the existing forcing uses**,
so the join with `era5_hourly_2019_2024.nc` is free. 211,324 ABI pixels map onto
1,421 cells (median 149 px/cell). Raw files are deleted after extraction, so a
month costs ~1.3 GB of transfer and 4 MB on disk.

Output is one row per (cell, hour): `fire_t`, `n_fire_px`, `obs_frac`, `max_frp`,
`sum_frp`, `max_area`, `max_temp`. Cell-hours below `--min-obs-frac` (default
0.5) are dropped rather than labelled 0.

## Validation month: August 2024

919,743 cell-hours, all **744 of 744** hours present, 1,420 cells,
**prevalence 0.0399** — 45× the zero-filled scheme's 0.000886.

**1. Geolocation.** During INPE's own overpass hours, of 517 cells INPE called
burning, GOES independently flagged **403 (77.9%)**. Chance agreement, if GOES
fire cells were placed at random *at those same afternoon hours*, would be
10.1% — a **7.7× lift**. (`validate_goes_vs_inpe.py` prints 3.8% and 20.4×:
it averages the GOES fire rate over all hours of the month, including the
night, which is too easy a baseline for an afternoon overpass.) Two
instruments with different detection physics will never agree perfectly; what
this rules out is a projection or grid-indexing error, which would have destroyed
the agreement while leaving every other diagnostic looking healthy.

**2. Diurnal cycle** (local time, UTC−4):

| local hour | prevalence | share of fire cell-hours |
|---|---|---|
| 06:00 | 0.0116 | 1.2% |
| 10:00 | 0.0439 | 4.6% |
| **14:00** | **0.0900** | **9.5%** |
| 18:00 | 0.0475 | 5.1% |
| 22:00 | 0.0257 | 2.7% |
| 02:00 | 0.0169 | 1.8% |

Peak at 14:00 local, trough at 06:00, **ratio 7.8×**. That is the shape fire
physically has — rising through the afternoon as fuels dry and winds strengthen,
minimum at dawn when humidity peaks. It is also a shape no amount of modelling
could have recovered from the INPE product.

**10.1% of fire activity occurs between midnight and 06:00 local.** INPE reports
exactly 0% there, not because nothing burns overnight but because nothing is
looking.

**3. Coverage.** Median `obs_frac` 1.000; 90.8% of cell-hours at ≥0.9 observed;
and coverage is near-flat across the day (0.961 at 15:00 local to 0.978 at
10:00). So the zeros are genuine observations, and they are not biased by time of
day — which would have re-introduced the orbit artifact through the back door.

## The hourly-target model (full year 2024)

With GOES the target itself becomes hourly. `build_goes_model_table.py` joins the
fire mask to the ERA5 forcing on the shared 49x29 grid and conditions both
targets on a cell that is burning at hour t:

| target | definition | prevalence |
|---|---|---|
| `persists_t1h` | still burning in the same cell at t+1h | 0.674 |
| `spreads_t1h` | a *neighbouring* cell burning at t+1h that was not at t | 0.207 |

**130,550 rows**, 66 columns, all twelve months of 2024. Every split embargoes
24 h before the test period: consecutive hours in the same cell are
near-duplicates, so without a gap the last training hour and the first test hour
would be the same fire.

### Seasonality — why one holdout number is not enough

| month | rows | persists | spreads | | month | rows | persists | spreads |
|---|---|---|---|---|---|---|---|---|
| Jan | 6,134 | 0.513 | 0.154 | | Jul | 9,629 | 0.680 | 0.173 |
| Feb | 4,500 | 0.446 | 0.230 | | **Aug** | **32,622** | **0.747** | 0.237 |
| Mar | 7,617 | 0.495 | 0.263 | | **Sep** | **30,787** | **0.753** | 0.223 |
| Apr | 2,885 | 0.531 | 0.094 | | Oct | 12,003 | 0.657 | 0.217 |
| May | 5,398 | 0.627 | 0.095 | | Nov | 3,989 | 0.537 | 0.180 |
| Jun | 12,726 | 0.697 | 0.184 | | Dec | 2,260 | 0.477 | 0.145 |

Persistence runs from **0.446 in February to 0.753 in September**. A fire burning
in the wet season has a coin-flip chance of surviving the hour; one burning in
September has three in four. A single holdout averages over that, which is why
the rolling-origin evaluation exists alongside it.

Note also that GOES sees the wet season in a way INPE cannot: Jan-May is 21% of
rows here against 6% of INPE's annual hotspots.

### Hour-of-day is a legitimate feature here

In the INPE product it was pure orbit geometry -- all detections in 4 of 24
hours. GOES observes every hour with near-flat coverage, so the diurnal cycle is
fire physics, and it is strong: over the full year persistence runs **0.792 at
09:00 local against 0.585 at 13:00** (1.35x). Note the shape differs from the
August-only slice, where the peak was 0.878 at 10:00 and the trough 0.640 at
05:00 — in the fire season the trough is at dawn, but pooled over the year it
moves to midday, when many short-lived fires start and do not survive the hour.
Fire *activity* peaks in the afternoon regardless.

### Holdout: train Jan-Aug (80,012 rows) -> test Sep-Dec (49,039)

Fixed hyperparameters, 5 estimator seeds:

| target | feature set | mean | ±std | delta | significance |
|---|---|---|---|---|---|
| `persists_t1h` (roc_auc, prev. 0.699) | context | 0.6415 | 0.0011 | — | reference |
| | **+ fire state** | **0.8302** | 0.0002 | **+0.1887** | **+164σ** |
| | + met | 0.8308 | 0.0002 | +0.1893 | +165σ |
| `spreads_t1h` (pr_auc, prev. 0.214) | context | 0.3355 | 0.0020 | — | reference |
| | **+ fire state** | **0.3908** | 0.0036 | **+0.0553** | **+15.2σ** |
| | + met | 0.3926 | 0.0019 | +0.0570 | +15.7σ |

Tuned XGBoost reaches 0.830 roc_auc on persistence and 0.394 pr_auc on spread,
matching the fixed-hyperparameter numbers closely -- for once the tuning adds
essentially nothing, because the fire-state signal is far too strong to be
sensitive to it.

### Rolling origin: five in-season folds

Train through month m, test month m+1.

| test month | context | + fire | + met | | context | + fire | + met |
|---|---|---|---|---|---|---|---|
| | **persistence (roc_auc)** | | | | **spread (pr_auc)** | | |
| Jun | 0.6334 | 0.7913 | 0.7939 | | 0.2452 | 0.2817 | 0.2814 |
| Jul | 0.6780 | 0.8084 | 0.8090 | | 0.2452 | 0.2937 | 0.3047 |
| Aug | 0.6358 | 0.8065 | 0.8002 | | 0.3464 | 0.3987 | 0.4030 |
| Sep | 0.6309 | 0.8198 | 0.8204 | | 0.3646 | 0.4184 | 0.4189 |
| Oct | 0.6321 | 0.8280 | 0.8303 | | 0.3319 | 0.3671 | 0.3781 |
| **mean** | 0.6420 | **0.8108** | 0.8108 | | 0.3066 | 0.3519 | **0.3572** |

Fire state helps in **5 of 5 folds on both targets**: +0.1688 persistence
(t = 14.4) and +0.0453 spread (t = 11.5).

### Does meteorology add anything?

The two feature sets share a baseline within each fold, so this is a **paired**
comparison -- and pairing is what makes the answer visible.

| target | met adds on top of fire | sd | t | folds positive |
|---|---|---|---|---|
| `persists_t1h` | **−0.0000** | 0.0036 | −0.01 | 4/5 |
| `spreads_t1h` | **+0.0053** | 0.0055 | **+2.14** | 4/5 |

**Persistence: exactly zero.** Not small -- zero to four decimal places, with
folds scattering either side. Whether a fire survives the next hour is fully
determined by what it is doing now.

**Spread: +0.005 pr_auc, t = 2.14.** Real but marginal, and an order of magnitude
smaller than fire state's +0.045. Physically sensible: starting a *new* fire in a
neighbouring cell depends on how receptive that ground is, which is a weather
question, while continuing an existing one is not.

Treat this as **suggestive, not established**. t = 2.14 on five folds is around
p ≈ 0.10.

Note that the unpaired comparison hides this entirely: 0.3519 ± 0.0615 against
0.3572 ± 0.0609 looks like nothing, because fold-to-fold spread is ten times the
effect. Same numbers, and only the paired reading is correct.

This also **corrects the August-only result**, which reported that meteorology
contributed nothing and mildly hurt. One month could not resolve an effect this
size; twelve months and a paired test can.

### Feature decisions applied

Carried over from the feature review, all implemented in
`build_goes_model_table.py`:

* **Retrieval failures are NaN, not 0.** GOES detects fire but often cannot
  retrieve its power, area or temperature (`has_frp` 79.8%, `has_area_temp`
  73.9%). Mapping those to 0 conflated "not measured" with "zero energy", and it
  was not inert -- persistence was 0.55 where `max_frp` was 0 against 0.77 where
  positive, so the model was reading a retrieval-failure flag dressed as a
  physical measurement. The values now stay NaN, which XGBoost handles natively,
  and the flags carry retrieval status explicitly.
* **`day_of_month` removed** -- an index, not a physical quantity. Replaced by
  cyclic `doy_sin`/`doy_cos`.
* **`obs_frac` kept in the table but out of the feature list** -- it is a quality
  gate, and leaving it as a predictor invites the model to learn cloud artifacts.
* **`fire_area_km2` added.** ABI pixels stretch off-nadir, so a raw pixel count
  mixes fire size with viewing geometry. Cell area divided by that cell's pixel
  count gives the true footprint. Median fire 5.1 km², max 749 km², against a
  ~733 km² cell.
* Unused wind-component lags no longer written.

### Two bugs found and fixed while scaling to the full year

1. **April went missing silently.** A transient `RemoteDisconnected` during the
   S3 listing aborted the month, and `list_day` had no retry (unlike `fetch`).
   The loop moved on and the year quietly had an eleven-month hole, visible only
   as a gap in the per-month table. Both functions now retry with exponential
   backoff, and a permanent failure prints `DOWNLOAD FAILED` in capitals.
2. **Lags were misaligned across missing hours.** The time axis was built from
   *observed* timestamps, so where an hour was absent entirely -- a failed scan,
   or every cell below the `obs_frac` gate -- `shift_hours(1)` reached back 2 h
   and labelled it lag-1. The axis is now a complete `date_range`, so gaps stay
   gaps. About 1.6% of hours are affected; it would never have shown up in a
   metric.

### Caveats

One year, one biome. The rolling folds cover only the fire season (June-October),
so nothing here is tested on wet-season-only conditions. The spread target is
sensitive to the neighbourhood definition (currently the 8 adjacent 0.25° cells,
roughly 25-28 km). And the marginal meteorology result needs more years to settle.

## Native-pixel store and split targets

### The old spread target was not measuring spread

`spreads_t1h` -- "a neighbouring 0.25 deg cell burns at t+1h" -- failed three
physical checks:

| check | result | verdict |
|---|---|---|
| implied front speed | 28 km in 1 h = **7.7 m/s** | impossible (crown fires reach 1-3 m/s) |
| directional preference | 25.4% within 45 deg of downwind | **25% = random**; no signal at all |
| dependence on source fire power | 0.14 -> 0.23 across FRP quintiles, levelling off (persistence ran 0.45 -> 0.97) | weak; an earlier version of the table gave a flat 0.20 / 0.17 / 0.23 / 0.25 / 0.24 |

It was measuring "some new fire appeared within 28 km", largely independent of
the fire it was conditioned on. The 0.25 deg aggregation -- chosen to match the
ERA5 grid -- discarded a factor of ~14 in exactly the dimension where spread
happens.

### The fix: store fire at native resolution, then split the target

`build_goes_pixels.py` writes fire detections at native ABI resolution (~2 km).
Fire pixels are sparse, so **all of 2024 is 581,419 pixels in 9.1 MB**; five
years would be ~45 MB. Any future grid or target choice is now a few minutes of
computation instead of a 52-minute re-download.

`build_goes_split_targets.py` then separates two processes the old target had
merged:

| table | unit | rows (2024) | prevalence |
|---|---|---|---|
| `goes_spread_2km` | fire pixel with an unburnt neighbour | 478,640 | 0.194 |
| `goes_ignition` | cell with no fire in or beside it | 599,471 | 0.047 * |

\* ignition negatives are subsampled 20:1, so its prevalence is artificial.
ROC-AUC is invariant to class balance and transfers to the real rate; PR-AUC does
not, and is comparable only between models scored on this same table.

Spread excludes pixels ringed by fire on all eight sides -- they have nowhere to
go, and including them made the target partly about being *interior*.

### The same three checks now pass

| check | old (28 km) | new (2 km) |
|---|---|---|
| implied front speed | 7.7 m/s | **0.56 m/s** -- physical |
| within 45 deg of downwind | 25.4% (= random) | **28.6%, z = 25.6** (n = 93,017) |
| spread rate by source FRP quintile | 0.14 -> 0.23, levels off | **0.109 -> 0.273**, monotonic |
| spread rate by wind speed quintile | — | **0.133 -> 0.263** |

### Results (test Sep-Dec 2024)

**Spread** (pr_auc, prevalence 0.196):

| feature set | mean | ±std | delta | significance |
|---|---|---|---|---|
| context | 0.3421 | 0.0023 | — | reference |
| + fire state | 0.4065 | 0.0018 | +0.0645 | +28.2σ |
| **+ met** | **0.4105** | 0.0013 | **+0.0684** | **+29.9σ** |

Best tuned model: XGBoost, pr_auc 0.4072, **lift 2.08x** over prevalence -- up
from 1.83x on the old target, so the physically correct target is also the better
predicted one.

**Ignition** (roc_auc, prevalence 0.047):

| feature set | mean | ±std | delta | significance |
|---|---|---|---|---|
| context | 0.6573 | 0.0008 | — | reference |
| **+ activity** | **0.8167** | 0.0015 | **+0.1594** | **+103.5σ** |
| + met | 0.8166 | 0.0013 | +0.1593 | +103.4σ |

Best tuned model: XGBoost, roc_auc 0.8189, pr_auc 0.3473, lift 7.47x.

### Where meteorology matters, settled

Paired per-fold comparisons across the five rolling-origin folds:

| target | met adds on top of the fire/activity block | t | folds positive |
|---|---|---|---|
| persistence (t+1h) | −0.0000 | −0.01 | 4/5 |
| **spread (2 km)** | **+0.0118** | **+3.76** | **5/5** |
| ignition | +0.0002 | +0.15 | 3/5 |

This is the question the project kept circling, and splitting the target answered
it. Meteorology matters for **spread** and for nothing else:

* **Persistence** -- whether a burning cell is still burning in an hour -- is
  determined by what the fire is already doing. Weather does not change enough in
  an hour to matter.
* **Spread** -- a front advancing into unburnt fuel -- is exactly the process
  wind and dryness govern, and the effect is now unambiguous: +0.0118 pr_auc at
  t = 3.76, positive in every fold. On the old 28 km target the same test gave
  +0.0053 at t = 2.14 in 4/5 folds -- suggestive at best. Fixing the resolution
  roughly doubled the effect and made it clear.
* **Ignition** -- a fire starting on quiet ground -- is not a weather event here
  at all. What predicts it is proximity to existing fire activity:
  `dist_to_fire_km` drives the ignition rate from 0.083 at 30-60 km down to
  0.028 beyond 250 km, and 53% of ignitions occur in cells that burned in the
  previous 24 h (rate 0.26 there against 0.024 elsewhere), so part of
  "ignition" is fire re-appearing rather than a new start. That is the signature of human ignition following the same
  access routes and land-use fronts, not of weather creating fires.

The activity block is worth +0.177 roc_auc (t = 7.84, 5/5 folds) while
meteorology contributes nothing measurable. For a biome where most fires are
human-set, that is the expected result -- and it was completely invisible while
ignition and spread were merged into one target.

### Rolling-origin detail

| test month | spread (pr_auc) | | | ignition (roc_auc) | | |
|---|---|---|---|---|---|---|
| | context | + fire | + met | context | + activity | + met |
| Jun | 0.2979 | 0.3563 | 0.3779 | 0.6734 | 0.8231 | 0.8266 |
| Jul | 0.3041 | 0.3963 | 0.4093 | 0.6997 | 0.8414 | 0.8419 |
| Aug | 0.3778 | 0.4257 | 0.4400 | 0.6124 | 0.8697 | 0.8712 |
| Sep | 0.3742 | 0.4361 | 0.4394 | 0.6826 | 0.8777 | 0.8767 |
| Oct | 0.3391 | 0.4052 | 0.4122 | 0.6543 | 0.7941 | 0.7906 |

### Leakage guard

`spread_bearing`, `spreads_downwind` and `wind_spread_angle` are computed *from*
the outcome -- they describe where the fire went, which is only known once it has
gone. They are excluded from every feature set and asserted against at load, so
they cannot creep back in.

## Scaling to 2020–2024

At 1 scan/hour: 60 months × 744 files ≈ **44,600 files, ~76 GB transferred**,
~4 MB/month retained. At the observed rate that is roughly 25 hours of wall
clock, and it parallelises cleanly by month.

Using all 6 scans/hour would catch fires shorter than an hour, at 6× the transfer
(~455 GB). Worth measuring on one month before committing: rerun with
`--scans-per-hour 6` and compare prevalence against the 1/hour table to see how
much is being missed.

## Open design questions for the hourly model

The target is now `fire_t` per (cell, hour), which is a different modelling
problem from the current one — grid-cell occupancy rather than event-level
transition. Before building it:

- **Target definition.** Fire *presence* at t+1h, or *new ignition* (fire at t+1h
  where there was none at t), or *persistence* (fire at t+1h given fire at t)?
  These are three different questions and the third is closest to the current
  model.
- **Negative sampling.** At 0.0399 prevalence, 96% of rows are negative and most
  are cells that never burn all month. Restricting to cells with any fire in a
  surrounding window, or to a fire-season subset, would concentrate the signal.
- **Autocorrelation.** Fire at t and t+1h in the same cell is strongly correlated,
  so a naive split would leak badly. Splitting must be by time block, and probably
  with spatial blocking too.
- **Confidence categories.** `--high-confidence-only` restricts to good/saturated
  pixels; the low/medium-probability categories are noisier but more numerous.

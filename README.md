# ExperimentOS

ExperimentOS is a local research control centre for synthetic datasets, experiment configurations, metric runs, checkpoints, hypotheses, and findings. It uses only the Python standard library at runtime. The pytest suite is the only development dependency.

## Quickstart

Use Python 3.10 or newer. From a clone, install the local command and test dependency:

```powershell
python -m pip install -e . pytest
exo datasets register --name demo --seed 17 --rows 50 --features 3
```

Save this as `demo.json`:

```json
{
  "name": "demo-decay",
  "dataset": "demo",
  "simulator": {
    "profile": "exponential_decay",
    "initial_loss": 1.0,
    "noise_floor": 0.08,
    "decay": 0.05,
    "noise_std": 0.01
  },
  "steps": 40,
  "seed": 23
}
```

Then run the lifecycle:

```powershell
exo run --config demo.json
exo runs
exo checkpoints save RUN_ID --step 20
exo run --resume CHECKPOINT_ID
exo hypothesize --text "loss falls below 0.2" --predicts "loss <= 0.2" --run RUN_ID
exo findings add --text "The simulated curve crossed the threshold." --run RUN_ID
exo compare RUN_ID RESUMED_RUN_ID
exo report --format html
```

The CLI prints the run ID and metric log path. The default workspace is `.experimentos` in the current directory. Set `EXPERIMENTOS_HOME` or pass `--home PATH` before the command to use another workspace. The `exo` command can also be invoked without installation as `python -m experimentos` from the repository root.

## Architecture

```text
JSON config -----> config validation -----> seeded curve simulator
                                             |             |
Synthetic CSV -> dataset registry            |             +--> RNG state
                                             +--> metrics.jsonl
                                                          |
                                   checkpoint <-----------+
                                      |                   |
                                      +---- resume --------+
                                                          |
                     hypotheses / findings <--- run evidence
                                      |
                             compare and reports
```

Files live under `.experimentos/`: synthetic CSVs and their registry are in `datasets/` and `datasets.json`; each run has its normalized config, metadata, and `metrics.jsonl` in `runs/`; checkpoints store the config, metric prefix, step, and serialized RNG state in `checkpoints/`; hypotheses and findings append to JSONL ledgers; generated reports go in `reports/`.

## HONESTY BOUNDARY

ExperimentOS computes seeded synthetic metric curves from closed-form functions. It does not train models, load model weights, evaluate real systems, or use machine learning frameworks. Registered datasets are generated synthetic CSV files, not collected data. Run outputs and reports are simulator outputs, not empirical research results. The HTML report displays this boundary in a prominent banner.

## Simulator curve families

Steps are one-based. The simulator computes a base value and adds a seeded Gaussian noise draw with mean zero and standard deviation `noise_std` on every step. The seed initializes Python's `random.Random` generator. Setting `noise_std` to `0` gives the exact closed-form curve. A checkpoint includes the generator state, including its Gaussian cache, so continuation is identical to an uninterrupted run in the same Python runtime.

Let `i` be the step, `L0` be `initial_loss`, `F` be `noise_floor`, `d` be `decay`, `P` be `plateau_steps`, and `m` be `slope`.

| Profile | Base loss before noise |
| --- | --- |
| `exponential_decay` | `F + (L0 - F) * exp(-d * (i - 1))` |
| `plateau_then_drop` | `L0` when `i <= P`; otherwise `F + (L0 - F) * exp(-d * (i - P))` |
| `linear` | `L0 + m * (i - 1)` |

All formulas are deterministic. Noise is computed as `random.Random(seed).gauss(0, noise_std)` in step order. Re-running the same normalized config under the same Python version produces byte-identical `metrics.jsonl` files. Run IDs and metadata are unique and are not part of the metric log.

## Config reference

Config is a JSON object with the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset` | string | yes | Registered dataset ID, normally the normalized dataset name. |
| `simulator` | string or object | yes | One of the three profile names, or an object with `profile` and parameters. |
| `steps` | integer | yes | Positive total step count, with the first logged step numbered 1. |
| `seed` | integer | yes | Fixed seed for the simulator noise generator. |
| `name` | string | no | Friendly run name. Defaults to `synthetic-run`. |

Simulator object parameters are numeric `initial_loss`, `noise_floor`, `decay`, `noise_std`, and `slope`, plus integer `plateau_steps`. Defaults are `1.0`, `0.05`, `0.05`, `0.0`, `-0.01`, and one third of the configured steps, respectively. All can be supplied for any profile. `noise_std` must be non-negative. Example profile-only config: `"simulator": "linear"`.

## Datasets

`exo datasets register --name NAME --seed N` creates a CSV with a sample index, seeded uniform feature columns, and a weighted-sum target with seeded uniform noise. `--rows` and `--features` control its size. The registry records schema, row count, generation method and parameters, seed, and SHA-256 checksum. Registering the same name and parameters writes identical CSV bytes.

## Checkpoints and resume

`exo checkpoints save RUN_ID --step N` records the normalized config, the metric prefix through step N, and the exact RNG state after step N. Omitting `--step` selects the final logged step. `exo run --resume CHECKPOINT_ID` resumes toward the total step count in the saved config. To extend a saved run, pass a larger total with `--steps TOTAL`. Resumed metric logs include the saved prefix, followed by the continued curve.

## Hypotheses, findings, and comparison

`exo hypothesize` accepts a predicate in the form `metric <= number`, with `<`, `<=`, `>`, `>=`, `==`, and `!=` supported. The predicate predicts that the condition will hold. ExperimentOS evaluates it against the minimum observed value for that metric in the cited run. The verdict is `match` when the condition holds and `mismatch` otherwise. This is a mechanical comparison against a synthetic run, not evidence about a real model.

Every finding must cite at least one existing run using `--run`; missing citations raise `UncitedFindingError`. `exo compare RUN_ID...` reads the stored JSONL logs and prints per-step values plus a summary. Best-step selection is the minimum loss, with the earliest step breaking ties. Reports include run comparisons, the hypothesis scoreboard, and cited findings.

## Development

Run the complete test suite with:

```powershell
python -m pytest
```

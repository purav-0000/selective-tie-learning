# Selective Tie Learning

This repository reproduces the experiments for **Selective Tie Learning**.
The study asks a simple question: can a small, chosen set of counterfactual
ties reduce a preference model's reliance on a nuisance feature?

> [!IMPORTANT]
> **Reproducibility note.** We reorganized the research code to make the
> experiments easier to run. The repository contains the data generation,
> models, selectors, configurations, and evaluation used for the paper. This
> work may have introduced minor bugs in convenience code, especially when a
> user resumes an unusual partial run or combines outputs from a custom
> directory layout. The paper appendices give a detailed, self-contained
> account of the data, models, objectives, training, selection, curvature
> calculations, and evaluation. If the convenience code contains a bug, those
> appendices provide the information needed to reproduce the experiments
> independently.

We compare three selectors:

- **UTL** samples ties at random.
- **PM** selects ties with large absolute preference margins.
- **Influence** selects ties that should reduce held-out counterfactual energy.

Each tie keeps the preferred alternative's causal content and changes only its
nuisance content. We score both tie orientations and average their gradients.
We train on one fixed random orientation per tie. All selectors use the same
orientation.

## Contents

```text
loglinear.py       log-linear experiment
nonlinear.py       nonlinear MLP experiment
llm.py             hotel and emoji LLM experiments
experiments/       configurations and result assembly
src/stl/           data, models, selection, and evaluation
```

The top-level commands use the paper settings. You do not need to edit the
source code.

## Install

Use Python 3.12 in a clean environment:

```bash
conda create -n stl python=3.12.13
conda activate stl
python -m pip install --upgrade pip
python -m pip install -e .
```

The log-linear experiment uses a CPU. The nonlinear experiment can use a CPU
or GPU. The paper used one CPU thread for log-linear and four for nonlinear.

### LLM requirements

The LLM experiments require Linux, an NVIDIA GPU with BF16 support, CUDA 12.6,
and FlashAttention 2. Install the matching PyTorch build and the LLM packages:

```bash
python -m pip install torch==2.12.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[llm]"
python -m pip install ninja packaging
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

You also need access to the gated
[`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
model. Accept its Hugging Face licence, then log in:

```bash
hf auth login
```

The Qwen control uses the public
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
The emoji experiment downloads `PKU-Alignment/PKU-SafeRLHF`. The configuration
files pin all model and dataset revisions. Set `HF_HOME` if you want to store
the Hugging Face cache elsewhere.

Use a GPU with at least 24 GB of memory. A 1,000-row Llama GGN uses about 23 GB
of system memory, so we recommend at least 64 GB of RAM. We recommend at least
96 GB for the 4B control. Reserve at least 200 GB of free disk for a full LLM
run. Do not run two scoring jobs on one host unless its RAM can hold both GGN
matrices.

## Reproduce the results

These commands run all paper seeds and budgets:

```bash
python loglinear.py
python nonlinear.py
python llm.py
```

Each command saves per-seed results, aggregate CSV files, and PNG/PDF figures
under `results/`. You can rerun a command after an interruption. The program
checks saved phases before it reuses them.

### Log-linear

The main run uses 1,000 strict pairs, 500 candidate ties, five seeds, and five
UTL draws. It also runs the influence and expected-influence checks.

```bash
python loglinear.py
python loglinear.py --continued
```

The second command starts endpoint training from the strict model. It produces
the appendix results.

### Nonlinear MLP

The main run uses 5,000 strict pairs, 2,000 candidate ties, five seeds, and a
width-128 MLP.

```bash
python nonlinear.py
python nonlinear.py --continued
```

Use this shorter command to check a new machine:

```bash
python nonlinear.py --quick
```

Add `--device cuda` to use a GPU. A short run cannot reproduce the paper's
uncertainty bands. In particular, two seeds give very wide 95% Student-*t*
intervals. Use all five seeds for the reported curves.

### Hotel and emoji LLM experiments

```bash
python llm.py
```

The main run uses three seeds, one epoch of BF16 LoRA-DPO, two UTL draws, and a
shared panel of 2,000 ties. Influence uses a 500-row selection probe and a
1,000-row GGN. Counterfactual energy uses 1,000 held-out anchors and four
redraws per anchor.

Use `--gpu` to choose a GPU. Use `--datasets` or `--seeds` to run a subset:

```bash
python llm.py --gpu 1 --datasets hotels --seeds 0
```

Run the appendix continuation and the one-seed 4B control with:

```bash
python llm.py --continued
python llm.py --model qwen3-4b
```

## Measures

- `P` is accuracy in the training environment.
- `Q_clean` removes the training correlation.
- `Q_adv` reverses the correlation. Synthetic files call it `Q_mixed`.
- Counterfactual energy is the mean squared reward change caused by a
  causal-preserving nuisance edit on held-out examples.

The bands show 95% Student-*t* intervals across seeds. We first average UTL
draws within each seed.

## Loss normalization

The paper writes the augmented loss as

```text
mean(strict losses) + sum(tie losses) / N,
```

where `N` is the number of strict pairs and `k` is the number of ties. The code
uses the usual mean over the combined data:

```text
(sum(strict losses) + sum(tie losses)) / (N+k).
```

At a fixed budget, the code's loss is exactly `N/(N+k)` times the paper's loss.
This factor multiplies every term equally. It does not change the balance
between strict pairs and ties.

For the log-linear model, this difference has no effect. The model has no
regularization and trains to the exact optimum. Multiplying its whole loss by
a positive constant leaves that optimum unchanged.

The LLM also has no weight decay. Adam largely removes a common gradient scale
through its moment normalization. Gradient clipping and Adam's numerical
constant could still create a small difference during finite training, so we
checked both loss forms. They gave the same practical results.

The MLP includes L2 regularization and uses finite Adam and L-BFGS training. A
change in the data-loss scale therefore changes the strength of a fixed L2
term and can also change the optimization path. We tested both forms. The
effect was modest and did not change the experimental conclusions. The public
configuration uses the combined-data mean and sets the L2 value (`0.01`) for
that form.

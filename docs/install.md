# Install

Install MARL-BattleGrounds on Linux, a Mac, Windows or a cluster, or try it in Colab, and start playing.

Website: https://marl-battlegrounds.com/#start

## Start Here

| You want to | Do this |
|---|---|
| Look around in your browser, with nothing to install | Open the [Colab notebook](#colab): the API, the speed, training a team and the trained Season 0 teams (no game window) |
| Play the game | The [one command](#one-command): three lines on Linux, a Mac or Windows with WSL2 |
| Use MARL-BattleGrounds in your own project | [pip](#pip) |

You do not need to install JAX first: each route installs the right JAX for your machine.

## Pick Your Route

| Your machine | Route | GPU | Tested |
|---|---|---|---|
| Linux (x86_64 or ARM) | [One command](#one-command) | NVIDIA, picked for you | Tested: x86_64 with an NVIDIA RTX 5090; x86_64 and ARM on the CPU |
| Apple Silicon Mac | [One command](#one-command) | CPU only (JAX has no Apple GPU support) | Tested (macOS 15, CPU) |
| Windows | [WSL2](#windows), then the one command; or [Docker](#docker) | NVIDIA through WSL2 or Docker | Not tested yet; it should work |
| Intel Mac, old Linux, or you prefer containers | [Docker](#docker) | NVIDIA on Linux and Windows hosts | Tested: Linux with Docker (CPU, and NVIDIA with the CUDA 12 image) and with Podman; the ARM image on ARM Linux. Not tested yet; it should work: Docker Desktop on a Mac or Windows |
| Your own Python project | [pip](#pip) | You choose | Tested: Linux x86_64 and ARM and an Apple Silicon Mac, each with Python 3.12, 3.13 and 3.14 |
| Only a browser | [Colab](#colab) | Colab's free T4 | Tested |
| A cluster or a remote GPU machine | [Clusters and remote machines](#clusters-and-remote-machines) | NVIDIA | Tested on one Linux machine: the uv steps with the network cut off, and Apptainer on the CPU. Not tested yet; it should work: a real cluster, Apptainer with a GPU, the SSH tunnel |
| AMD GPU or Google TPU | [GPU builds](#gpu-builds) | AMD or TPU | Not tested yet; it should work |

"Tested" means we ran exactly these instructions on that system for this release.

Python 3.12, 3.13 and 3.14 all work. Python 3.14 is recommended: official leaderboard runs use it, and extending a saved training run needs the exact Python it was saved with (see [Watch Out](#watch-out-saved-runs-need-the-same-python)). The one command and Docker use Python 3.14 for you. Native Windows without WSL2 is not supported.

## One Command

```bash
curl -L https://github.com/arm-2-lab/MARL-BattleGrounds/archive/refs/tags/v1.0.1.tar.gz | tar xz
cd MARL-BattleGrounds-1.0.1
sh start.sh
```

`start.sh` installs [uv](https://docs.astral.sh/uv/) if you do not have it, picks the right JAX build for your machine, installs Python 3.14 and the exact tested packages, then opens the BattleClient in your browser. The first run downloads up to 3.5 GB with a GPU build (about 0.3 GB on the CPU); later runs start in seconds.
By default you play Team A against the trained Recurrent MAPPO team on map 47. Press `?` in the page for How To Play. Each game's replay is saved in `recordings/`.
Without curl, use `wget -qO- <the same link> | tar xz`. With git: `git clone --branch v1.0.1 https://github.com/arm-2-lab/MARL-BattleGrounds.git`, then `cd MARL-BattleGrounds` and `sh start.sh`.

Run everything else from the same folder with `uv run`, for example:

```bash
uv run --no-sync python -m marl_battlegrounds evaluate --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check
```

Open the example notebooks with `uv run --no-sync jupyter lab`, then `examples/quickstart.ipynb`.

| Option | What it does |
|---|---|
| `sh start.sh --team-b qmix --map 12` | Another opponent and map. Teams: `manual` (you), `mappo`, `ippo`, `ff-mappo`, `ff-ippo`, `qmix`, `pqn-vdn`, `tdm-alpha`, `tdm-beta`, `tdm-gamma`, `random` |
| `sh start.sh --team-a ippo --team-b mappo` | Watch two trained teams play |
| `sh start.sh --port 8765 --replay-port 8767 --no-browser` | Fixed ports, and print the link instead of opening a browser |
| `sh start.sh --dry-run` | Print the chosen build and the commands; install nothing |
| `MARL_BGS_ACCELERATOR=cuda12 sh start.sh` | Choose the build yourself: `cpu`, `cuda13`, `cuda12`, `rocm` or `tpu` |
| `sh start.sh --help` | Install, then list `play.py`'s options |

**Watch Out**
- Open the `http://127.0.0.1:...` link that is printed. `localhost` is refused on purpose.
- uv's installer adds `~/.local/bin` to your shell's startup files. Open a new terminal to use `uv` yourself.
- `start.sh` makes this folder's environment exactly "everything users need plus your build". It removes anything else installed there, such as developer tools. Later, use `uv run`, or run `start.sh` again. A plain `uv sync`, even with `--extra training`, removes the GPU build `start.sh` installed, and JAX then runs on the CPU. Contributors use the command in `CONTRIBUTING.md`.

## GPU Builds

`start.sh` reads `nvidia-smi` and picks a build from your oldest NVIDIA driver and GPU. Set `MARL_BGS_ACCELERATOR` to choose yourself. pip users name the build as an extra.

| Build | For | Needs | Tested |
|---|---|---|---|
| `cuda13` | NVIDIA RTX 20-series and newer | Driver 580 or newer, compute capability 7.5 or higher | Tested (an RTX 5090, and Colab's T4) |
| `cuda12` | Older NVIDIA GPUs and older drivers | Driver 525 or newer, compute capability 5.2 or higher | Tested on an RTX 5090 (driver 580) only. Older GPUs: not tested yet; they should work |
| `cpu` | Everything else | Nothing | Tested |
| `rocm` | AMD GPUs on Linux x86_64 | ROCm 7 installed on the machine | Not tested yet; it should work |
| `tpu` | Google Cloud TPU VMs | A TPU VM | Not tested yet; it should work |

Check which device JAX uses: `uv run --no-sync python -c "import jax; print(jax.devices())"`.

## pip

```bash
python3 -m venv .venv && . .venv/bin/activate   # a fresh environment, Python 3.12 to 3.14
pip install "marl-battlegrounds[all]"           # CPU
pip install "marl-battlegrounds[all,cuda13]"    # NVIDIA, driver 580+ (or cuda12, rocm, tpu)
```

You do not need to install JAX first: the extra installs JAX 0.10.1 for your GPU. pip replaces any other JAX version, so use a fresh environment.
`all` adds training, Hydra, notebooks and plots to the core package. For the exact versions we test with, add the release's constraints file for your build (`cpu`, `cuda13`, `cuda12`, `rocm` or `tpu`):

```bash
pip install "marl-battlegrounds[all,cuda13]==1.0.1" -c https://github.com/arm-2-lab/MARL-BattleGrounds/releases/download/v1.0.1/constraints-cuda13.txt
```

No install at all, with uv: `uv run --no-project --python 3.14 --with "marl-battlegrounds[all]" python -m marl_battlegrounds evaluate --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check`.

**Watch Out:** the builds exclude each other. To switch from `cuda13` to `cuda12`, use a fresh environment, or uninstall `jax-cuda13-plugin` and `jax-cuda13-pjrt` first. JAX loads the CUDA 13 plugin before the CUDA 12 one.

## Docker

```bash
docker run --rm -it -p 127.0.0.1:8765:8766 -p 127.0.0.1:8767:8768 -v "${PWD}:/work" ghcr.io/arm-2-lab/marl-battlegrounds:1.0.1
```

This opens the BattleClient: open the `http://127.0.0.1:8765/...` link it prints. The image holds Python 3.14.5, the tested packages and the six trained Season 0 models. Your current folder is `/work` in the container, so replays and results land next to you.
NVIDIA GPU: add `--gpus all` and use the `1.0.1-cuda12` tag (driver 525 or newer; on Linux it needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), on Windows Docker Desktop with WSL2). The CUDA 12 image is for x86_64 computers; on an ARM computer with an NVIDIA GPU, use the one command.
In Windows Command Prompt, write `-v "%cd%:/work"` instead. PowerShell, bash and zsh use the command as shown.

| Command after the image name | What it does |
|---|---|
| (nothing), or `play --team-b qmix --map 12` | The BattleClient, with the same team and map options as `start.sh` |
| `replay recordings/<file>.marlbg-replay.json` | Opens a saved replay in the analytical Replay Viewer |
| `evaluate --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check` | Any MARL-BattleGrounds command, such as `train`, `evaluate`, `tournament` or `models` |
| `python` or `bash` | Any other program, run as given |

JupyterLab, run from the unpacked release folder so `examples/` is there:

```bash
docker run --rm -it -p 127.0.0.1:8888:8888 -v "${PWD}:/work" ghcr.io/arm-2-lab/marl-battlegrounds:1.0.1 jupyter lab --ip 0.0.0.0 --no-browser --allow-root -y
```

**Watch Out**
- Always publish ports as `-p 127.0.0.1:...`. Without `127.0.0.1`, anyone on your network can reach the page.
- The host ports must equal the ports inside, 8765 and 8767 by default. To use others, set both: `-p 127.0.0.1:9000:8766 -p 127.0.0.1:9002:8768 -e MARL_BGS_PORT=9000 -e MARL_BGS_REPLAY_PORT=9002`. A second container running at the same time needs its own two ports.
- A language model server on your computer is `host.docker.internal` from inside the container, not `127.0.0.1`. On Linux, add `--add-host=host.docker.internal:host-gateway`, and make the server listen on more than `127.0.0.1`.
- In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` before `docker run`. On Fedora, RHEL and other SELinux systems, write `-v "${PWD}:/work:z"` (not tested yet; it should work).
- A GPU image on a machine without a usable GPU (for example, run without `--gpus all`) prints a long "Jax plugin configuration error", then runs on the CPU.
- Docker Desktop is free for personal use, education, non-commercial open source and small businesses; larger organizations and government bodies need a paid plan. Podman works the same way: write `podman` for `docker`.

| Limit | What to do |
|---|---|
| No GPU in Docker on a Mac | Use the CPU image; Docker cannot reach Apple GPUs |
| `study` (big multi-run jobs) is not tested in containers and stops when the container stops | Start the container with `-d`; not tested yet |
| You cannot add packages to a running container | Build a two-line Dockerfile: `FROM ghcr.io/arm-2-lab/marl-battlegrounds:1.0.1`, then `RUN uv pip install <package>` |
| Docker Desktop (Mac, Windows) limits the container's memory | Raise the memory limit in Docker Desktop's settings for large training runs |

## Windows

In PowerShell: `wsl --install`, then restart. Open the Ubuntu app and run the [one command](#one-command) in your WSL home folder (`cd ~` first, not `/mnt/c/...`, which is much slower). NVIDIA GPUs should work inside WSL2 with the normal Windows NVIDIA driver (install no driver inside Ubuntu); JAX calls this experimental, and it is not tested yet. Or use [Docker](#docker) Desktop.

**Watch Out:** cloning with Git for Windows and then using the folder from WSL or Docker works, because the repository turns off line-ending conversion.

## Colab

[Open the walkthrough in Colab](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/quickstart.ipynb). It opens on a free T4 GPU. Its first cell installs MARL-BattleGrounds with the JAX build for that GPU and restarts the runtime once by itself (Colab says the session crashed; that is expected), then Runtime > Run all again. It walks through the API with a drawn game, the speed, a short training run with the released curve, and the trained Season 0 teams from Hugging Face.
The browser clients (BattleClient and Replay Viewer) do not run inside Colab, because their pages only accept `127.0.0.1`. Use the one command or Docker on your own machine for them.

## Clusters And Remote Machines

Remote GPU machine: start the client there with `sh start.sh --port 8765 --replay-port 8767 --no-browser`. On your laptop run `ssh -N -L 8765:127.0.0.1:8765 -L 8767:127.0.0.1:8767 <host>`, then open the printed `127.0.0.1` link.
Cluster with Apptainer: on the login node, set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to scratch storage (the image is about 4 GB to download), then, in a scratch folder, run `apptainer pull marl-bgs.sif docker://ghcr.io/arm-2-lab/marl-battlegrounds:1.0.1-cuda12` once (the image file lands in the current folder). On a compute node, in that folder: `apptainer exec --nv marl-bgs.sif python -m marl_battlegrounds evaluate --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check`.
Cluster with uv: unpack the release on scratch storage (the environment is about 6 GB), set `UV_CACHE_DIR` there, run `MARL_BGS_ACCELERATOR=cuda12 sh start.sh --help` on the login node, open a new shell (so `uv` is on your `PATH`), and on a compute node `uv run --no-sync python -m marl_battlegrounds evaluate --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check`. Compute nodes often have no internet: on the login node, download the models first with `uv run --no-sync python -m marl_battlegrounds models download MARL-BattleGrounds/tdm-season-0-recurrent-mappo` (and any others), then set `HF_HUB_OFFLINE=1` on the compute node. Linux needs glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL 8+); older systems use Apptainer.

## Offline Use

Trained models download from the [MARL-BattleGrounds Hugging Face organization](https://huggingface.co/MARL-BattleGrounds) the first time you use them and stay in your Hugging Face cache. After that, set `HF_HUB_OFFLINE=1` to run without a network. The Docker image already holds the six Season 0 models.
Released models work offline too: download them once with `python -m marl_battlegrounds models download <name>`, then set `HF_HUB_OFFLINE=1`; see [Released Models](released_models.md#downloads-cache-and-offline-use).

## Watch Out: Saved Runs Need The Same Python

Extending a saved training run checks the exact Python version that trained it, down to the last number (for example 3.14.5), and the exact versions of `jax`, `jaxlib`, `numpy`, `flax`, `optax` and `orbax-checkpoint` (and `flashbax` for QMIX). `uv python upgrade`, or upgrading any of those packages, stops a run saved before from being extended. The one command and the constraints files keep those packages fixed, but not Python's last number: `start.sh` may pick any Python 3.14. To extend a run saved with another Python, pin it in the folder and run the command again, for example `uv python pin 3.14.5`, then `sh start.sh`. The Docker image already has 3.14.5. Starting a new run from any saved model needs no version match.

## Next

- [Getting started](getting_started.md): make an environment, reset and step.
- [BattleClient](battle_client.md): play against your own models.
- [Training](training.md) and [evaluation](evaluation.md).

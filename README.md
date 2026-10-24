<div align="center">

# CarryOn
### A Dual-Loop Harness for Personalized and Reliable Robot Assistance

[**Project Page**](https://vincent2311.github.io/Projects/CarryOn/) · [**Paper (arXiv)**](https://arxiv.org/abs/2610.21453)


</div>

CarryOn equips a frozen vision-language-action (VLA) policy with two connected loops: a **planning loop** that adapts tasks to the user and scene, and an **execution loop** that checks actions for contact risk and repairs failures. Visual feedback connects both loops throughout the task.

## Content

- [News](#news)
- [Project Video](#project-video)
- [Method](#method)
- [Experiments](#experiments)
- [Getting started](#getting-started)
- [Repository layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## News

- **2026-10-15:** Paper released.

## Project Video



https://github.com/user-attachments/assets/da8d01b8-b8b5-42d4-acec-96ecb626f0ff



## Method

<p align="center">
  <img src="https://vincent2311.github.io/Projects/CarryOn/assets/architecture.png" alt="CarryOn's dual-loop harness: planning with user and scene memory, proactive shielding, reactive recovery, and offline reflection." width="100%">
</p>

- **Planning with persistent, revisable context.** User and scene memory support personalized task planning and are updated through dialogue and observation.
- **Proactive protection and reactive recovery.** The execution loop shields VLA actions from predicted hazardous contact and uses control-based recovery skills to restore task progress after failures.
- **Refining recovery across attempts.** Offline reflection improves reusable recovery skills from unresolved failures, while the VLA remains frozen.


## Experiments

This repository contains CarryOn and its four non-real-robot experiments:

| Setting | What it evaluates |
| --- | --- |
| LIBERO Perturbation | Recovery after object displacement and offline reflection |
| LIBERO-Safety | Proactive shielding across VLA backbones |
| LIBERO-Recovery | Recovery from recorded failure states |
| RoboLab | Multimodal requests and memory revision across scenes |

See the [reproduction guide](REPRODUCING.md) for environment setup, model requirements, and experiment commands.


## Getting started

Run the experiments on Linux. Install the shared package from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[models]"
carryon --version
```

Follow the commands in [REPRODUCING.md](REPRODUCING.md) for [installation](REPRODUCING.md#install), [RoboLab](REPRODUCING.md#robolab), [LIBERO-Safety](REPRODUCING.md#libero-safety), [LIBERO Perturbation](REPRODUCING.md#libero-perturbation), or [LIBERO-Recovery](REPRODUCING.md#libero-recovery). The guide includes small and full runs, recovery with memory generated from the current run, and result analysis.

## Repository layout

| Directory | Contents |
|---|---|
| `src/carryon/` | Shared planning, memory, execution and recovery components |
| `experiments/` | Experiment runners, Shield runtime, configurations and benchmark inputs |
| `requirements/` | Dependency versions, installation sources and model downloads |
| `experiments/*/references/` | Object reference images used by each recovery experiment |
| `prompts/` | Planner, Judge, memory, and Shield prompts |

## Acknowledgements

CarryOn builds on LIBERO, LIBERO-Safety, StarVLA, openpi, OpenVLA-OFT, VLAct, cuRobo, SAM 3 and GraspGen. 


## Citation

```bibtex
@article{wang2026carryon,
  title={CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance},
  author={Wang, Zixuan and Liu, Yuqi and Chen, Yuxin and Yang, Sengqiao and Ye, Jinhui and Chen, Pengguang and Lu, Changsheng and Liu, Shu and Yu, Bei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2610.21453},
  year={2026}
}
```

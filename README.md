<div align="center">

# CarryOn
### A Dual-Loop Harness for Personalized and Reliable Robot Assistance

Zixuan Wang · Yuqi Liu · Yuxin Chen · Senqiao Yang · Jinhui Ye  
Pengguang Chen · Changsheng Lu · Shu Liu · Bei Yu · Jiaya Jia

**HKUST · CUHK · SmartMore**

[**Project Page**](https://vincent2311.github.io/Projects/CarryOn/) · [**Overview Video**](https://vincent2311.github.io/Projects/CarryOn/#overview-video)

</div>

> **TL;DR:** CarryOn equips a frozen vision-language-action policy with persistent, revisable memory, proactive geometric shielding, and recovery skills that improve through reflection. It connects what a robot should do for a particular user with how it can carry the task through safely and recover when execution fails.

## News

- **2026-10-13:** Code will be released within three weeks, **by November 3, 2026**.

## Highlights

- **Personalization across conversations and scenes.** User preferences and scene knowledge persist across interactions and are revised when dialogue or observations change the context.
- **Protection before hazardous contact.** A response-aware geometric shield checks gripper and carried-object motion before releasing actions. On LIBERO-Safety, it reduces the violation rate of an already safety-finetuned policy from **52.6% to 6.1%**.
- **Recovery that improves from experience.** Reusable control-based skills restore interrupted task progress. Two reflection-and-retry rounds raise cumulative completion on 200 perturbed LIBERO-Object episodes to **80.0%**, compared with **13.0%** for the frozen VLA alone.
- **One continuing task around a frozen policy.** Planning, shielding, recovery and visual feedback share the active task context; adaptation is retained in memories and recovery skill versions.

## Why CarryOn?

The same instruction can require different actions for different people. “Serve me food” should respect a remembered vegetarian preference, use scene knowledge to locate an ingredient, and change the plan if a fresh observation reveals meat where a carrot was expected. A suitable plan also needs support during execution: an obstacle may create hazardous contact, and a lost grasp may interrupt delivery.

CarryOn addresses both decisions with a dual-loop harness. The **planning loop** determines what to do for the user and the current scene. The **execution loop** protects and repairs the active subtask. Visual feedback connects the loops while preserving the user's goal.

## Method

<p align="center">
  <img src="https://vincent2311.github.io/Projects/CarryOn/assets/architecture.png" alt="CarryOn architecture: planning with user and scene memories, proactive shielding, visual supervision, recovery, and offline reflection" width="100%">
</p>

### 1. Plan with persistent, revisable context

The planner combines the current text or image request with user profiles, endpoint-specific scene records and available skills. User memory retains enduring preferences across conversations; scene memory guides endpoint selection and the search for hidden objects. Dialogue can revise a preference, and new observations can correct a stale location before subsequent planning proceeds.

### 2. Check motion before contact

The proactive shield distinguishes intended task contact from forbidden contact. It predicts short-horizon end-effector motion from controller response and executed-motion history, propagates gripper and carried-object geometry, and checks the candidate action sequence against forbidden scene geometry. It releases a safe prefix or retreats and reassesses with fresh observations. Only actually executed actions update motion history.

### 3. Repair failures and retain useful revisions

When execution fails, CarryOn pauses the VLA and invokes a reusable recovery skill to re-observe, re-ground, grasp and verify the object before returning control to the original subtask. Restoring a grasp and completing the user's task remain distinct outcomes.

Between evaluation rounds, offline reflection examines selected unresolved failures and proposes parameter and strategy revisions. An admission gate checks their structure and applicable numerical bounds. Accepted changes form a new frozen skill version for the next round, while the VLA weights remain fixed.

## Selected Results

### Personalized planning

In RoboLab, memory supports both named objects and visual references supplied in an earlier conversation. Each condition contains **88 requests** across four endpoints; latency is the median time to dispatch.

| Query | Correct endpoint selections | Median dispatch latency |
|---|---:|---:|
| Text | 77 / 88 | 28.6 s |
| Image | 77 / 88 | 34.6 s |
| Text + memory | **88 / 88** | **7.0 s** |
| Image + memory | **88 / 88** | **7.1 s** |

Real-robot studies additionally examine sandwich preferences, activity-dependent packing, and retrieval and revision of covered-ingredient locations.

### Proactive safety across policy backbones

Each backbone is evaluated on **900 LIBERO-Safety episodes** spanning TSA and FSHOA at three difficulty levels. Violation rate counts episodes with constraint violations; violation-free task success requires completing the task without a violation.

| Policy | Violation rate: policy → + CarryOn ↓ | Violation-free task success: policy → + CarryOn ↑ |
|---|---:|---:|
| π0.5 | 71.1% → **22.7%** | 6.7% → **14.6%** |
| Safety-finetuned π0.5 | 52.6% → **6.1%** | 25.3% → **25.6%** |
| OpenVLA-OFT | 61.4% → **17.2%** | 6.9% → **14.8%** |
| VLAct | 69.3% → **26.1%** | 4.3% → **13.9%** |

### Recovery through reflection and retries

| Evaluation | Frozen VLA alone | CarryOn base skill | After one reflection round | After two reflection rounds |
|---|---:|---:|---:|---:|
| LIBERO-Object: 200 episodes, 12 cm hold-loss perturbation | 26 / 200 (13.0%) | 58 / 200 (29.0%) | 141 / 200 (70.5%) | **160 / 200 (80.0%)** |
| LIBERO-Recovery: 770 recorded failure cases | 247 / 770 (32.08%) | 385 / 770 (50.00%) | 569 / 770 (73.90%) | **586 / 770 (76.10%)** |

Reflection rounds retry only the preceding failures. Results are **cumulative over the original population** and include additional reflection and execution attempts. The recorded-failure evaluation uses 77 scenes with five open-gripper and five closed-gripper initializations per scene, without adding online perturbations.

### Integrated real-robot assistance

In the food-serving demonstration, CarryOn retrieves a vegetarian preference, inspects a covered ingredient, shields transport around an obstacle, and recovers after the object is removed. When a new observation reveals meat instead of the remembered carrot, the planner revises scene context and stops serving it to the vegetarian user.

[Watch the demonstrations on the project page.](https://vincent2311.github.io/Projects/CarryOn/)

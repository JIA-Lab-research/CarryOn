<div align="center">

# CarryOn
### A Dual-Loop Harness for Personalized and Reliable Robot Assistance

[**Project Page**](https://vincent2311.github.io/Projects/CarryOn/)

</div>

CarryOn equips a frozen vision-language-action (VLA) policy with two connected loops: a **planning loop** that adapts tasks to the user and scene, and an **execution loop** that checks actions for contact risk and repairs failures. Visual feedback connects both loops throughout the task.

## Content

- [News](#news)
- [Project Video](#project-video)
- [Method](#method)

## News

- **2026-10-13:** Code will be released soon.

## Project Video



https://github.com/user-attachments/assets/da8d01b8-b8b5-42d4-acec-96ecb626f0ff



## Method

<p align="center">
  <img src="https://vincent2311.github.io/Projects/CarryOn/assets/architecture.png" alt="CarryOn's dual-loop harness: planning with user and scene memory, proactive shielding, reactive recovery, and offline reflection." width="100%">
</p>

- **Plan with persistent, revisable context.** User memory retains preferences and object references; scene memory records where objects are. The planner uses both to interpret requests and select subtasks, updating context through dialogue and observation.
- **Protect and recover during execution.** A proactive geometric shield predicts gripper and carried-object motion and filters actions before hazardous contact. When manipulation fails, control-based recovery skills repair the physical state so the interrupted task can continue.
- **Refine recovery through reflection.** Offline reflection uses unresolved failures to revise recovery parameters and procedures. Successive versions are retained in a skill library for later attempts, while the VLA stays frozen.

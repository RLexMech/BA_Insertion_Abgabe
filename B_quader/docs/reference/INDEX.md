# Index — docs/reference/ (consult before opening any reference doc)

Purpose: reference docs are read per-section only (CLAUDE.md rule). This index is
the map for picking the right section and knowing its dependencies and caveats.
Snapshot: 2026-07-23; valid while the source docs are unchanged (they are frozen
sources — if one is ever edited, update this index in the same commit).

## research_cylindertask.md (771 lines, 15 numbered sections)

Global caveat (from the doc's own provenance note): written against a Franka
baseline and Isaac Lab 2.x-era material. All Franka-specific values (7-DOF
layouts, effort limits, gains) require adaptation to the UR10e (6-DOF) and
validation against Isaac Lab 2.3 (D-001). Inline source markers must be verified
before citing in the thesis.

| § | Title (line) | Project phase | Depends on / caveats |
|---|---|---|---|
| 1 | System Architecture (31) | 1 | Overview diagram; gains/frequencies here are restated in §5/§8 — cite those, not this |
| 2 | Task Definition (88) | 1–2 | Recommends Franka + 10 mm peg; we use UR10e (D-002) and our own geometry (D-006) — take the randomization table and termination criteria, not robot/geometry |
| 3 | Observation Vector (121) | 2 | 26-dim assumes 7-DOF Franka; UR10e → 6-DOF (24-dim). Explicitly non-privileged, consistent with D-005. Layout is open decision 5 |
| 4 | Action Space (158) | 2 | Presupposes the impedance controller of §5 — read both together |
| 5 | Controller (191) | 2 | Franka parameters (effort_limit 87 N, stiffness 800, damping 40) — measure UR10e gains in sim, never copy. Open decision 3 |
| 6 | Reward Function (224) | 2–3 | Lines 265–277 (scaling guidelines + known exploits) are mandatory reading with any reward edit |
| 7 | Curriculum (278) | 4 | Builds on §6 reward; sampling-based progression logic from IndustReal |
| 8 | PPO Configuration (345) | 3 | Tuned for Factory/Franka setup — starting point only. Tuning guidelines at line 395 |
| 9 | Implementation Sequence (409) | all | Phase-by-phase steps with success criteria + debugging decision tree (line 484) + failure-mode table (line 531). Cross-check docs/plan_alignment.md |
| 10 | Most Important Conclusions (576) | — | Summary of §1–§9, no new content |
| 11 | Top 10 Ranked Resources (605) | thesis | Source ranking for literature citations — verify each before citing |
| 12 | Practical Lessons from Engineers (653) | debug | Read only AFTER PROBLEMS.md (log outranks memory and this doc) |
| 13 | Isaac Lab Factory Investigation (681) | 1–2 | What to reuse vs. NOT copy from Factory (privileged obs, keypoint reward, aggressive randomization) |
| 14 | Limitations and Uncertainties (718) | thesis | Supported vs. inferred vs. uncertain claims — for the discussion chapter |
| 15 | Final Practical Advice (748) | planning | Timeline, minimum-viable-thesis fallbacks, stretch goals |

Common pairings: §4+§5 (actions need the controller), §6+§7 (curriculum builds on
reward), §3+open decision 5 (observation layout), any reward edit → §6 exploits
list.

## workflow_tips.txt (37 lines)

Short enough to read whole. Benchmark for the repository structure (D-008);
check every point when auditing the repo layout.

## literature_check_pose_uncertainty_2026-08-14.md (272 lines)

Checks, against actually-retrieved sources, whether the observation design
"EE pose relative to a KNOWN pocket pose" stays defensible once the fixture
pose is randomized per episode and the part can shift in the gripper
(in-hand/grasp error), and how published insertion works handle both. Each
work is examined for three questions: is the (noisy) target pose in the
observation and is noise added during training; is the in-hand error
observed, estimated, or only randomized; are the reward terms computed on
privileged ground-truth state or on the noisy observed pose. FORGE is the
closest template. Directly relevant to D-034/D-036/D-037 and to the suction
gripper of the real task, which adds a further compliance between end
effector and part.

Caveats: claims cite only pages actually fetched, details in appendices or
paywalled versions are marked "NOT RETRIEVED". Bibliographic data (venues,
page numbers) still needs a final check against the arXiv pages before
thesis citation, same as the 2026-08-06 checks. Rescued on 2026-08-16 from
the superseded working copy under `Desktop\Claude\keine 3\`, where it had
never been committed.

## curriculum_strategies_research.md (2026-08-16 web-research pass)

Compares four strategies for scheduling randomization difficulty (manual
rungs / fixed schedule / adaptive threshold-driven / full randomization from
scratch) against primary sources and released code. Read §1 (short answer)
and §4 (project recommendation) first; §2 holds per-source findings with the
verbatim IndustReal SBC update rule and the Isaac Lab 2.3 / rsl_rl resume
facts; §3 the comparative evidence (IndustReal ablation, ADR vs. fixed);
§5 the thesis-citable subset. UNVERIFIED labels mark claims not confirmed in
a primary source. Companion to literature_check_2026-08-06.md (Claim 1).

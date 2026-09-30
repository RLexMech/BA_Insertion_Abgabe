# Design Decision Log

Running log of all thesis-relevant design decisions for the real task.
Every entry is written at the moment the decision is made. Language:
English (the thesis itself is written in German).

Numbering continues from the proxy-task repo (`ur10e-peg-insertion-rl`,
D-001…D-036): a number means exactly one decision across the whole
thesis. **D-037 (25-channel observation space) and D-038 (fixture yaw)
were taken by the old repo on 2026-08-16 — the first entry in this file
is therefore D-039.** Before assigning a new number, check the old
repo's DECISIONS.md for the highest number in use.

## The 2026-08-27 numbering batch (D-090 ... D-116)

On 2026-08-27 the writer session on `main` merged the stream branches
`p2-szene-umbau`, `p2-rl-code` and `p1-konzept-messung` and emptied
`docs/decisions_inbox.md`: 27 candidates became D-090 ... D-116. Each of
them carries a `- Numbered:` line pointing back here.

Order rule, so the sequence stays readable later: by entry date, and
within one date by the order the entries had in the inbox file. The one
exception is 2026-08-27, which follows the block chain instead --
Block 7 (D-113) -> the D-089 overturn it caused (D-114) -> Block 8/N8.1
(D-115) -> Block 8/N8.2 (D-116) -- because file order would have put
N8.2 before N8.1.

Two branches were deliberately NOT merged: `p1-gains` (user decision
2026-08-27; its D-082/D-083 are superseded on the branch and stay
there) and `p1-thesis` (D-090 forbids it).

## Audit 2026-09-13 -- run boundary

On 2026-09-13 the user set the run boundary at RT-180 (OSC + AutoDR + D-178..D-185, code >= `ee97ed36`; `rt_logs/VERDICTS.md` cut there in `a7b67909`), and seven entries got an inline note dated 2026-09-13: D-165, D-166, D-168, D-169 (their number rests on pre-boundary runs, marked UNVERIFIED FOR THIS SETUP) and D-105, D-110, D-113 (pointer to the OSC/AutoDR state); no number was changed.

## Entry template

```
## D-NNN: <title>
- Date: YYYY-MM-DD
- Status: accepted | proposed (pending verification) | superseded by D-MMM
- Context: <problem and constraints>
- Options considered: <option A / option B / ...>
- Decision: <what was chosen>
- Rationale: <why, with technical arguments>
- Sources: <papers, documentation pages, measurements>
```

---

## D-039: Thesis chapter structure per professor feedback
- Date: 2026-08-16
- Status: accepted (written e-mail confirmation waived by the user
  2026-08-16; the meeting notes are the record)
- Context: The submitted table of contents merged fundamentals and
  state of the art into one chapter and contained a chapter
  "1.3 Aufbau der Arbeit". The professor reviewed it in a meeting.
- Options considered: keep the submitted 6-chapter structure / split
  fundamentals and state of the art into two chapters as requested.
- Decision: Restructure to 7 chapters: 1 Einleitung, 2 Grundlagen
  (as short as possible, incl. ~2 pages Isaac Sim/Lab), 3 Stand der
  Technik (may be longer, ends with a short summary/Einordnung),
  4 Konzeption, 5 Training und Evaluation, 6 Diskussion,
  7 Zusammenfassung und Ausblick. "1.3 Aufbau der Arbeit" is replaced
  by a short methodology outline; "wirtschaftliche Folgen" is removed
  from 1.1.
- Rationale: Direct supervisor requirement. Fundamentals only cover
  established knowledge the reader should not need to look up; the
  state-of-the-art chapter is where scientific work is demonstrated.
- Sources: Professor meeting, documented in
  `docs/Zwischenbericht/Professor-Feedback.md`.

## D-040: Thesis scope is a practical engineering task, no research-gap claim
- Date: 2026-08-16
- Status: accepted (written e-mail confirmation waived by the user
  2026-08-16; the meeting notes are the record)
- Context: Open question whether the thesis must close a research gap
  or prove PPO superior to SAC.
- Options considered: position the thesis as closing a research gap /
  as an algorithm comparison (PPO vs. SAC) / as a practical
  engineering task building on the proxy-task work.
- Decision: The thesis is framed as a practical task: train and
  evaluate a PPO policy for the real insertion task. No research-gap
  or algorithm-superiority claim is required.
- Rationale: Explicit supervisor answer: a purely practical task is
  sufficient for a bachelor thesis; extending the prior solution
  counts as a contribution. This narrows the literature work in the
  concept phase to justifying design choices, not proving novelty.
- Sources: Professor meeting, documented in
  `docs/Zwischenbericht/Professor-Feedback.md`.

## D-041: Suction gripper modelled as a rigid chain
- Date: 2026-08-16
- Status: accepted
- Context: The real end effector holds the part with two suction
  cups. The simulation needs a model of the grasp before any
  environment code is written.
- Options considered: none spanned — recorded as a bare entry per
  user decision 2026-08-16 (plan-review session). Revisit if the
  simulation or the professor review makes suction compliance
  relevant.
- Decision: Robot flange, suction gripper and part are modelled as
  one rigid chain (pre-grasped, no detachment) for the first
  implementation.
- Rationale: Starting simplification already fixed in CLAUDE.md at
  repo setup; keeps the proxy-task code base directly reusable.
- Sources: none (no literature pass, bare entry by user choice).

## D-042: Proxy-settled choices get short adoption entries, not full re-grilling
- Date: 2026-08-16
- Status: accepted
- Context: The concept protocol (CLAUDE.md) demands spanned
  alternatives and a literature pass per decision. Several choices
  were already decided and verified in the proxy task (e.g. quaternion
  observation, old-repo D-037). Re-spanning them would repeat settled
  work and delay the professor review.
- Options considered: full re-grill of every block / short adoption
  entries for proxy-settled choices, full protocol only for new
  aspects.
- Decision: Choices carried over from the verified proxy get a
  compact DECISIONS entry citing the proxy decision and its verified
  result. The full protocol applies only to aspects that are new in
  the real task.
- Rationale: The proxy evidence (≥99 % combined generalisation,
  verified on the training machine) is a stronger justification than
  a fresh literature pass; the thesis cites the measurement. Avoids
  duplicated decision records for one thesis (one number = one
  decision).
- Sources: `[old repo]`
  `docs/Sessionbericht_2026-08-16_Neigung_Versatz.pdf` (verification);
  old-repo DECISIONS.md D-037/D-038.

## D-043: All proxy work is pre-study; the thesis task is the real part only
- Date: 2026-08-16
- Status: accepted
- Context: Thesis-outline open question 2: where does the simplified
  pre-study end and the thesis task begin (ch. 4.3)? The old-repo
  handoff recommended counting the square-peg task as stage 1 of the
  thesis work and as fallback main result.
- Options considered: only cylinder as pre-study, square task as
  thesis stage 1 (old-repo recommendation) / all proxy work
  (cylinder + square) as pre-study, thesis task = real part only /
  square task as pre-study with full result reporting.
- Decision: The entire proxy work (cylinder and square peg) is
  pre-study. The thesis task is the real guide-piece insertion with
  the UR5e. Explicitly no fallback main result is planned (user
  choice; the assistant flagged the deadline risk).
- Rationale: Cleanest narrative: the thesis IS the real task; the
  proxy work is methodology groundwork reported as pre-study. The
  user accepts the risk that a stalled real task leaves no fallback
  result.
- Sources: `[old repo]` `docs/thesis_gliederung_proxytask_handoff.md`
  (question 2 and the rejected recommendation); user decision
  2026-08-16.

## D-044: Task variation axes — grasp uncertainty plus fixture offset/yaw/tilt
- Date: 2026-08-16
- Status: accepted (ranges open — set in Block 6, numbers
  `[CAD pending]` / `[measure on site]`)
- Context: Block 1, node N1.2. Real-world variations named by the
  user: the suction grasp sits slightly differently every cycle; the
  fixture can be slightly shifted. The fixture plate is nominally
  parallel to the ground.
- Options considered: fixture offset only / offset + yaw / offset +
  yaw + tilt / any of these plus part-in-gripper pose uncertainty.
- Decision: Four variation axes: (1) part-in-gripper pose
  uncertainty (new vs. proxy), (2) fixture offset in X/Y, (3) small
  fixture yaw, (4) small fixture tilt — although the real plate is
  level.
- Rationale: User argument for including tilt: simulations are not
  trusted, so one trains on uncertainty anyway; a policy that also
  handles tilt is on the safe side. Axes 2–4 reuse the verified
  proxy machinery (offset ±2 cm, tilt 0–10°, yaw ±45° combined at
  ≥99 %); axis 1 is the genuinely new axis of the real task.
- Sources: user statements 2026-08-16; `[old repo]`
  `docs/Sessionbericht_2026-08-16_Neigung_Versatz.pdf`; `[old repo]`
  `docs/reference/literature_check_pose_uncertainty_2026-08-14.md`
  (pose-uncertainty grounding; file still in the superseded copy —
  rescue is an open debt).

## D-045: Insertion strategy is not prescribed — only start and goal are defined
- Date: 2026-08-16
- Status: accepted
- Context: Block 1, node N1.3. The human/robot strategy on the real
  station is corner-to-corner contact, then level the part parallel
  to the plate, then push in. Question: bake this strategy into the
  task design?
- Options considered: prescribe the corner-first strategy (e.g.
  scripted approach or corner-contact start states) / define only
  start distribution and goal state, let the policy discover its
  strategy.
- Decision: The task defines only start distribution and goal state.
  The policy discovers the insertion strategy itself. The
  corner-first observation is kept as a curriculum-design idea for
  Block 6 (e.g. start poses near corner contact), not as a
  constraint.
- Rationale: The proxy policy discovered tilt-handling strategies
  without prescription; hard-coding a human strategy narrows the
  solution space and adds assumptions the robot may not need.
- Sources: user decision 2026-08-16; proxy behaviour documented in
  `[old repo]` `docs/Sessionbericht_2026-08-16_Neigung_Versatz.pdf`.

## D-046: Scope limits of the thesis task (ch. 4.1)
- Date: 2026-08-16
- Status: accepted
- Context: Block 1, node N1.4. Chapter 4.1 must state explicitly
  what is out of scope.
- Options considered: per candidate limit: in scope / out of scope.
- Decision: Out of scope are: (1) sim-to-real transfer — the thesis
  is simulation-only; (2) grasping and releasing — the part is
  pre-grasped, rigid chain (D-041); (3) vision — observations come
  from simulation state, no camera; (4) fixture motion — the fixture
  is rigidly mounted and does not move or comply; (5) deeper stages —
  the task ends when the part is seated in the FIRST stage of the
  pocket, no further push-down.
  [Correction 2026-08-17, D-057: the part seats in the INNER (second)
  stage; limit (5) means "no push-down beyond the seated position".]
- Rationale: Matches the professor-confirmed practical framing
  (D-040) and the pre-grasped rigid-chain model (D-041); limits are
  listed in ch. 6.2 as limitations where self-set.
- Sources: user decisions 2026-08-16 (grill session, Block 1).

## D-047: End state = geometry-derived depth threshold + alignment tolerance, with interpenetration filter
- Date: 2026-08-16
- Status: accepted (concrete numbers `[CAD pending]`)
- Context: Block 1, node N1.1. What counts as "inserted" for the
  guide piece (first pocket stage only, D-046). The user required a
  literature check on how the field defines success before deciding.
- Options considered: depth-only threshold / depth threshold +
  lateral alignment tolerance (field standard) / contact-pattern
  check. Additionally: adopt IndustReal's interpenetration filter
  yes/no.
- Decision: Success = part base within a geometry-derived depth
  threshold of the fully-seated first-stage height (number from CAD,
  `[CAD pending]`) [Correction 2026-08-17, D-057: the seat is the
  INNER (second) stage; guided depth measured ~50.5 mm] AND a
  lateral/keypoint alignment tolerance
  (Isaac Lab convention as starting point). Additionally, evaluation
  adopts IndustReal's interpenetration filter: simulated successes
  with max interpenetration above a threshold do not count.
- Rationale: Unanimous field standard across all six checked primary
  sources (verdict: supported); depth-only is exploitable at the
  part's 0.5–1 mm clearance; contact-pattern checks have no
  precedent as success definition. Thresholds are geometry-derived
  in every quantified source, matching the project's
  numbers-from-CAD rule. The interpenetration filter makes simulated
  success rates physically honest and is directly citable.
- Sources:
  `docs/reference/literature_check_success_definitions_2026-08-16.md`
  (Factory arXiv:2205.03532; IndustReal arXiv:2305.17110, Sec. IV-E;
  AutoMate arXiv:2407.08028; Inoue et al. arXiv:1708.04033; Isaac
  Lab v2.3.0 factory/automate task code; SRSA arXiv:2503.04538).

## D-048: LaTeX conventions fixed in STYLE.md; cleveref and siunitx added to the template
- Date: 2026-08-16
- Status: accepted (verified by a full local build)
- Context: `thesis/STYLE.md` covered language rules only. The LaTeX
  section carried the open item "derive conventions from the
  template". Prose writing had already begun in
  `thesis/chapters/01_Einleitung.tex`, so every further chapter would
  have had to be cleaned up afterwards.
- Options considered: (a) cross-references — add `cleveref` and write
  `\cref` vs. keep plain `\ref` with a hand-written reference word;
  (b) units — add `siunitx` vs. hand-typed `0,2\,mm`; (c) citation
  style — keep the template's numeric biblatex default vs. switch to
  authoryear.
- Decision: (a) `cleveref` added, `\cref` is mandatory, `\ref` only for
  a bare number with a source comment; (b) `siunitx` added, `\qty`,
  `\num`, `\unit`, `\qtyrange` mandatory, decimal point in the source,
  comma in the output; (c) numeric style kept unchanged. In addition,
  the previously unwritten conventions were recorded: label prefix
  scheme, `equation`/`align` with mandatory numbering, `booktabs`
  tables without vertical rules, relative image widths, `\enquote`
  from csquotes, acronyms via `\gls`, no manual spacing commands.
- Rationale: `cleveref` makes the reference word and the number
  structurally inseparable, which removes a whole class of silent
  inconsistencies ("Abb." vs. "Abbildung"). `siunitx` guarantees one
  spacing and one decimal marker across a thesis dominated by
  tolerances, forces and angles. The citation style is a university
  template default and changing it would require the professor's
  approval without a technical benefit.
- Verification: dev laptop, MiKTeX, chain pdflatex → biber →
  pdflatex ×2 on `thesis/main.tex`: exit 0 on every pass, 22 pages, no
  errors or package warnings. `\cref{sec:ziel}` renders as
  "Abschnitt 1.2" (checked in the PDF text layer). siunitx is v3.5.5;
  `\qty{0.2}{\milli\metre}` renders as "0,2 mm".
- Sources: template preamble `thesis/includes/preamble.tex`; cleveref
  and siunitx package documentation (CTAN); user decisions
  2026-08-16.

## D-049: German writing conventions — punctuation, sentence form, terminology, register
- Date: 2026-08-16
- Status: accepted (organizational; binding for every thesis text)
- Context: `thesis/STYLE.md` covered narrative form, tense and evidence
  handling, but left punctuation, sentence form and terminology open.
  The first draft of section 1.1 exposed the gaps: it used a
  label-colon construction, the terminology for the joining operation
  was undecided, and the guidance conflicted with the TH Nuernberg
  guide on register.
- Options considered: (a) punctuation — allow semicolons and dashes as
  in general German usage vs. forbid both; (b) terminology — English
  "Insertion" as the literature uses it vs. German "Fuegen"; (c)
  register in chapter 1 — follow the TH Nuernberg guide, which asks for
  general comprehensibility in the introduction, vs. keep one audience
  for the whole thesis; (d) tense in 1.1 — present throughout, since
  the TH Nuernberg guide lists forward references as a typical error,
  vs. future tense as STYLE.md prescribes for the introduction.
- Decision: (a) No semicolon and no dash as sentence punctuation.
  Hyphens in compounds and the en dash in ranges remain permitted.
  Only complete sentences; label-colon constructions ("Vorteil: …")
  are forbidden in running text. (b) "Fuegen"/"Fuegeprozess"
  throughout; "Insertion", "Einfuehren" and "Einsetzen" are forbidden
  for this operation. General rule: an English term stays English only
  when established and not losslessly translatable (e.g. Reinforcement
  Learning). (c) Audience remains "Maschinenbaustudenten" for the whole
  thesis including the introduction; the TH Nuernberg recommendation is
  deliberately declined. (d) Future tense applies in 1.1 and carries
  the closing demand statement ("In der vorliegenden Arbeit soll …");
  forward references to a following section remain forbidden.
  In addition, `thesis/reference/satzbausteine_wissenschaftliches_arbeiten.pdf`
  is registered as a formulation aid, subordinate to the style
  prohibitions.
- Rationale: (a) and (b) are consistency decisions by the user, who
  will defend the text; a single term per concept and a single
  sentence-ending mark remove a class of reviewer objections at zero
  technical cost. (c) resolves a genuine conflict: the TH Nuernberg
  guide asks the introduction to be understandable to non-specialists,
  which would force explanations of terms every reader of a mechanical
  engineering thesis already knows. The user chose one consistent
  audience over a chapter-local exception. (d) separates two things the
  TH Nuernberg error list conflates: the forbidden element is the
  forward reference, not the future tense, so both rules can hold at
  once.
- Sources: `thesis/STYLE.md`; TH Nuernberg, Fakultaet efi, "Leitfaden
  zum Verfassen wissenschaftlicher Arbeiten", 4th ed. 2023, sec. 3.2
  (required content), p. 30 (typical errors), ch. 5.1 (register), as
  summarised in `docs/reference/einleitung_recherche_2026-08-16.md`
  part A.1; user decisions 2026-08-16.

## D-050: Structure of section 1.1 derived from surveyed thesis conventions
- Date: 2026-08-16
- Status: accepted (structure); the text itself carries open
  placeholders
- Context: The first draft of "1.1 Motivation und Problemstellung" was
  rejected by the user as not reading like an introduction. Instead of
  revising by intuition, the conventions of the genre were surveyed:
  12 German-language theses and 6 university writing guides were
  retrieved as PDFs and read in full.
- Options considered: (a) entry point — general trend statement about
  robot assembly vs. a quantitative technical fact vs. starting from
  the concrete part; (b) second paragraph — a catalogue of classical
  methods vs. a chain of individually named deficits; (c) closing
  sentence — forward reference to section 1.2 vs. a demand statement.
- Decision: (a) Quantitative entry: the UR5e repeat accuracy is set
  against the part clearance, and the apparent contradiction is
  resolved by the relative pose being the governing quantity.
  (b) Three individually named and numbered deficits (unknown relative
  pose, contact behaviour must be specified in advance, a given design
  covers only one geometry and tolerance band), which paragraph 3
  addresses in the same order. (c) Demand statement naming the
  application case, no forward reference. Source density reduced from
  five markers to two.
- Rationale: The survey produced three findings that each contradicted
  the first draft. Seven of the twelve theses carry zero literature
  references in the motivation section, so five source markers in
  345 words made the text read as unsupported assertion rather than as
  established fact. The general trend statement is the one entry type
  that consistently fails in the sample and is in every case left
  unsourced; the entries that carry are a definition, a dated event, or
  a concrete technical state. The forward reference in the closing
  sentence is listed verbatim as a typical error by the university's
  own guide, which also states that quantitative statements are to be
  preferred over qualitative ones. The deficit chain rather than the
  catalogue is the load-bearing figure in the stronger examples of the
  sample, because it lets the chosen method follow formally from the
  named shortcomings.
- Verification: dev laptop, MiKTeX: `thesis/main.tex` compiles without
  errors, 22 pages. `\qty{\pm0.03}{\milli\metre}` renders as
  "±0,03 mm" in the PDF text layer, per D-048.
- Open placeholders in the text: `[ZU PRÜFEN: Datenblatt UR5e]` for the
  repeat accuracy, `[CAD pending]` for the clearance, and
  `[QUELLE ERFORDERLICH]` for the working principle of RL.
- Sources: `docs/reference/einleitung_recherche_2026-08-16.md`, which
  lists all 12 theses and 6 guides with URLs; closest comparable works
  are Malin (FH Vorarlberg, MA 2021, robot-assisted joining) and Lenz
  (TU Darmstadt, MA 2025, peg-in-hole with RL, English).

## D-051: Criterion set for the real task — success rate is the only acceptance criterion
- Date: 2026-08-16
- Status: accepted (concept-level; thresholds are decided separately in
  N2.2, numbers still `[CAD pending]`)
- **CHALLENGED 2026-08-28 by D-137 (status OPEN):** the thesis stream
  recorded a conflict against this entry. Asked for the pass number, the
  user described the requirement as a robust policy that inserts under
  defined conditions including noise — not as a success rate. D-137 does
  NOT resolve it and neither does this marker: either the robustness view
  is expressed as a success rate over a defined disturbance range, in
  which case this entry holds and only the number is missing, or this
  entry needs an amending decision. Until then, do not quote D-051 as
  settled in chapter 1.
- Context: Block 2, node N2.1. The set of evaluation criteria had to be
  fixed before any threshold can be derived. A maximum contact force was
  carried as an explicit candidate from Block 1, and the professor's
  "95 %" is binding as an example of the FORM of an acceptance statement
  only, never as a value. The six sources of
  `literature_check_success_definitions_2026-08-16.md` had already shown
  that success itself is a kinematic depth/pose check throughout, but
  they had not been interrogated on the role of force specifically.
- Options considered: (a) maximum contact force as a second acceptance
  criterion, with its threshold derived from scripted insertions /
  as a reported measurement only; (b) engagement (partial insertion) as
  a second acceptance criterion / as a reported diagnostic / omitted;
  (c) insertion time or step count as an acceptance criterion / as a
  reported figure.
- Decision: The success rate, defined over the end state of D-047, is
  the single pass/fail acceptance criterion. Engagement (partial
  insertion, second and looser depth band on the same height
  measurement), step count to success, and the contact-force
  distribution are reported as diagnostic metrics in the results
  chapter, without a pass/fail statement attached to any of them. The
  contact force additionally retains two non-criterion roles that are
  decided in their own nodes: the force-abort limit (Block 7, N7.3) and
  a possible reward penalty (Block 5, N5.2).
- Rationale: Three arguments, the first from literature and the other
  two specific to this setup. (1) A dedicated literature check found no
  established precedent for force inside the success test. In
  Beltran-Hernandez et al. (2020) success is "the Euclidean distance
  between the robot's end-effector position and the true goal position
  was less than 1 mm" (Sec. 3.3), and force enters only as the collision
  limit F_max and as a reward penalty of -50; in arXiv:2402.18002
  success is a pose threshold and the only force value is "a -0.1
  penalty for forces beyond the [0, 10] N range" (Sec. VII-A). Together
  with the six sources of the earlier check this gives nine sources in
  which force never decides success. The single candidate counterexample
  (Zhang et al. 2022, depth > 30 mm and force < 2 N) could not be read
  in its primary source and is therefore not cited. (2) The gripper is
  modelled as a rigid chain (D-041) and the part as rigid plastic; the
  Block 1 critique already recorded that this simplification distorts
  contact forces. A quantity whose absolute value is known to be
  distorted by a deliberate modelling simplification must not carry a
  pass/fail decision. (3) Without sim-to-real there is no physical
  measurement against which a simulated force threshold could be
  calibrated, so such a threshold would be self-referential. Retaining
  force as a reported distribution preserves the evidence for gentle
  insertion without asserting an unfounded bound. Engagement is kept as
  a diagnostic because it costs only a second threshold on the height
  measurement already needed for success, and it separates the two
  distinct failure modes "the part reaches the pocket but does not
  seat" and "the part misses the pocket" — a distinction the success
  rate alone cannot express; both IndustReal and the Isaac Lab Factory
  task report such a second, looser band.
- Consequences for later blocks: the rigid-chain distortion of contact
  forces is to be stated as a limitation in ch. 6.2; N2.2 derives
  thresholds for the success criterion only; N2.4 writes exactly one
  acceptance statement; N3.2 (F/T observation channel) is not
  constrained by this decision, because a channel the policy sees is
  independent of what the evaluation scores.
- Sources: `docs/reference/literature_check_force_criterion_2026-08-16.md`
  (Beltran-Hernandez et al. 2020, arXiv:2008.10224, Secs. 2.3/3.3/3.4.1;
  arXiv:2402.18002, Secs. VI-B/VII-A; unverified counterexample
  documented in §4);
  `docs/reference/literature_check_success_definitions_2026-08-16.md`
  (six sources, engagement bands in IndustReal and Isaac Lab v2.3.0);
  D-041, D-047; user decisions 2026-08-16.

## D-052: Threshold derivation method for the success criterion
- Date: 2026-08-16
- Status: accepted (methods); all three numbers remain `[CAD pending]`
- Context: Block 2, node N2.2. D-051 leaves the success rate as the only
  acceptance criterion, so exactly three thresholds have to be fixed: the
  insertion-depth threshold, the alignment tolerance, and the
  interpenetration-filter bound of D-047. The Block 1 critique carried
  the open requirement that the alignment tolerance be either derived
  from the clearance or declared self-set in ch. 6.2. The derivation
  METHOD is decided here; the values follow from the CAD in phase 2.
- Options considered: (a) depth threshold as a fraction of the
  first-stage insertion depth, as in Isaac Lab v2.3.0
  (`success_threshold` = 0.04 of asset height) / as an absolute
  millimetre band below the full-seat height, as in IndustReal (3 mm);
  (b) alignment tolerance derived from the nominal clearance / adopted
  from the literature default (2.5 mm lateral centering in Isaac Lab)
  and declared self-set; (c) interpenetration filter reported at a
  single fixed bound / as a sweep over several bounds, as in IndustReal.
- Decision: (a) Absolute millimetre band below the full-seat height of
  the first pocket stage. (b) The alignment tolerance is derived from
  the nominal clearance of part and pocket.
  [Correction 2026-08-21, D-057: "first pocket stage" is wrong. The
  part seats in the INNER (second) stage; the outer first stage is much
  wider and accepts a different component. Read (a) as "below the
  full-seat height of the SEAT, i.e. the inner second stage".]
  (c) The interpenetration filter is reported as a sweep; one bound, tied to the clearance,
  carries the headline success rate, the remaining bounds document the
  sensitivity of that number.
- Rationale: (b) is a geometric necessity rather than a choice: once the
  part is seated at full depth inside the pocket, its remaining lateral
  and angular freedom IS the clearance, because the pocket walls bound
  it. A tolerance derived this way is forced by the geometry and needs
  no calibration argument. This closes the Block 1 open requirement in
  the stronger of the two permitted ways. It is also a genuine
  contribution: the earlier literature check established that no
  surveyed source derives its lateral tolerance from first principles —
  all of them are calibration constants. (c) costs nothing, since it is
  the same evaluation filtered repeatedly, and it converts the
  interpenetration bound from an arbitrary choice into a reported
  sensitivity: a success rate that is stable across bounds is evidence
  that the policy did not exploit solver penetration, whereas a sharp
  drop exposes exactly that exploitation.
- Obligation attached to (a): an absolute band does not carry its own
  justification the way a fraction does. When the CAD number is
  available, the chosen band must be stated in relation to the
  guided insertion depth of the SEAT (for example "3 mm, i.e. 6 % of the
  50.5 mm guided depth"), otherwise the value is unfounded. This
  obligation is a hard item for phase 2.
  [Correction 2026-08-21, D-057: this obligation said "first-stage
  depth" and was rewritten to the inner-stage guided depth, measured at
  about 50.5 mm. Do NOT relate the band to STAGE1_DEPTH = 0.015 --
  that constant is an asset origin offset (rim of stage 1 down to the
  stage-2 opening plane), not an insertion depth. See the correction
  note in D-071.]
- Sources: `docs/reference/literature_check_success_definitions_2026-08-16.md`
  (IndustReal 3 mm band and SAPU sweep over 0.5/1/1.5/2 mm; Isaac Lab
  v2.3.0 fraction-of-height thresholds and 2.5 mm centering; synthesis
  section on tolerances being calibration constants throughout);
  D-047, D-051; Block 1 critique in `docs/Blockberichte/Block-1_Bericht.pdf`;
  user decisions 2026-08-16.

## D-053: Evaluation methodology — seeds, episode counts, dispersion measure
- Date: 2026-08-16
- Status: accepted
- Context: Block 2, node N2.3. With the success rate as the only
  acceptance criterion (D-051) and its thresholds derived (D-052), the
  evaluation protocol had to be fixed: how many training seeds, how many
  evaluation episodes per condition, how dispersion is reported, and
  whether the headline number comes from the best seed or from all
  seeds. The earlier literature check established the field norms:
  1000-5000 trials per condition, 3-5 seeds, success rate as headline
  metric, and no confidence intervals in any of the six sources.
- Options considered: (a) 5 seeds x 1000 evaluation episodes per
  condition, as in IndustReal / 3 seeds x 1000 episodes at the lower
  end of the norm; (b) dispersion as the standard deviation across
  seeds, which is the field standard / additionally binomial confidence
  intervals on the success rate, which exceeds the field standard;
  (c) headline number from the best seed, as in AutoMate and SRSA /
  from all seeds.
- Decision: (a) 5 seeds, 1000 evaluation episodes per condition.
  (b) Dispersion is reported as the standard deviation across seeds
  only. No confidence intervals. (c) The headline number is the mean
  across all five seeds; the best-seed number may be reported in
  addition, with an explicit note that the cited literature
  (AutoMate, SRSA) reports it as its headline figure.
- Rationale: (a) sits in the middle of the surveyed norm and matches
  IndustReal exactly (5 seeds, 1000 trials per seed per condition),
  which makes the protocol directly citable rather than self-invented.
  The cost is dominated by training the five seeds, not by the
  evaluation rollouts, which run in parallel environments. (b) follows
  the field standard deliberately: the user's position is that this
  thesis should not invent methodology where an established convention
  exists, and no surveyed source reports confidence intervals. The raw
  per-episode outcomes are retained, so confidence intervals can be
  computed after the fact if the professor requires them at the review,
  without repeating any run. (c) best-seed reporting inflates the
  headline number by selecting the maximum of five samples; reporting
  the mean across all seeds is the honest statement of expected
  performance, and naming the best seed alongside it preserves
  comparability with AutoMate and SRSA.
- Supersedes: the phase 5 note in HANDOFF.md, which anticipated
  confidence intervals as a deliberate excess over the field standard.
  That note is corrected to match this decision.
- Sources: `docs/reference/literature_check_success_definitions_2026-08-16.md`
  (IndustReal: 5 seeds with 1000 trials per seed, dispersion as +/- std
  across seeds; AutoMate: best-seed selection over 5000 trials; SRSA:
  1000 episodes, 3-5 seeds, std-dev shading; synthesis: no confidence
  intervals in any source); D-051, D-052; user decisions 2026-08-16.

## D-054: Baselines and negative controls for the evaluation
- Date: 2026-08-16
- Status: accepted
- Revision note: an earlier version of this entry, written the same day
  and in the same Block 2 session, listed a third baseline (a scripted
  open-loop insertion representing current industrial practice). The
  user removed it before the node was closed: that program already
  exists as prior work at the company, it solves the task, and the
  thesis deliberately does not refer to it. The entry is corrected in
  place rather than superseded, because the node was still under
  discussion when the change was made.
- Context: Block 2, node N2.5. The node was added to the Block 2 tree at
  the start of the session on user decision, because the tree as built
  in Block 0 contained no comparison condition. An absolute success rate
  is not interpretable on its own: without a reference it cannot be
  judged whether the learned policy solves a hard problem well or an
  easy problem adequately.
- Options considered: (a) which baselines to run — random policy alone /
  random policy plus a scripted open-loop insertion / random policy plus
  the proxy policy applied zero-shot to the real part / all three;
  (b) whether a baseline runs from the nominal pose only or under the
  same randomised start distribution as the policy.
- Decision: Two baselines. (1) An untrained random policy as the
  negative control. (2) The proxy-task policy from the old repo, applied
  zero-shot to the real part without further training. Both are
  evaluated under the same protocol as the learned policy (D-053:
  5 seeds where applicable, 1000 episodes per condition) and under the
  same randomised start distribution, not from the nominal pose.
  A scripted open-loop insertion is NOT run as a baseline.
- Rationale: The random policy is a negative control in the sense of the
  old repo's rule set: if it succeeds at a non-negligible rate, the task
  is too easy and every subsequent result is void. It costs one
  evaluation run and guards the validity of everything else. The proxy
  policy zero-shot quantifies what the pre-study transferred to the real
  part; the checkpoint exists, so the marginal cost is again one
  evaluation run, and it answers in advance the question of what the
  pre-study contributed (D-042, D-043). The scripted insertion is
  excluded because it is not a research question here: a working
  conventional program for this part already exists as prior work, so
  measuring it in simulation would reproduce a known result and would
  additionally invite a comparison between a simulated reimplementation
  and a real program, which the thesis cannot support without
  sim-to-real. Both retained baselines run under the randomised start
  distribution because that is the distribution the policy is scored on;
  a baseline measured under different conditions is not a comparison.
- Consequences for later blocks: the disturbance sweep reuses the
  start-randomisation ranges of N6.2, so the evaluation grid and the
  training ranges must be kept consistent. Because no state-of-the-art
  baseline is measured, the acceptance statement of N2.4 cannot be
  defined relative to one.
- Sources: user decisions 2026-08-16 (node added, scripted baseline
  removed); D-042, D-043 (pre-study status), D-044 (variation axes),
  D-053 (evaluation protocol); old-repo rule on negative controls, to be
  re-adopted with the implementation-phase rules before phase 3.

## D-055: Out-of-distribution probes kept deliberately lean
- Date: 2026-08-16
- Status: accepted
- Context: Block 2, remaining part of node N2.3. The four variation axes
  of D-044 (grasp-pose uncertainty, fixture offset, fixture yaw, fixture
  tilt) are trained over bounded ranges. The question is whether the
  evaluation also probes beyond those ranges, and at what density.
- Options considered: no probes outside the trained ranges / a dense
  ladder of three or more steps beyond the trained range on every axis /
  a lean ladder that is extended only where a result warrants it.
- Decision: Out-of-distribution probes are run, but deliberately lean:
  per axis a short deterministic ladder immediately beyond the trained
  range, evaluated under the D-053 protocol. The ladder is extended, and
  additional axes combined, only towards the end of the evaluation phase
  and only where the lean result shows something worth resolving.
- Rationale: Probes beyond the trained range are the only measurement
  that distinguishes a policy which generalises from one which has
  fitted the training distribution, and they require no additional
  training. A dense grid over four axes, however, multiplies evaluation
  conditions without a prior reason to expect information from most of
  them; the user's position is that exhaustive probing is postponed
  until the results indicate where it pays. The lean ladder preserves
  the qualitative statement (does performance degrade gracefully or
  collapse at the boundary) at a fraction of the cost, and the raw
  outcomes are retained so a denser sweep can be added later without
  repeating earlier runs.
- Sources: D-044 (axes), D-053 (evaluation protocol), D-054 (baselines
  run over the same disturbance grid); user decision 2026-08-16.
## D-056: Two rule sets only — writing plan dissolved, rule conflicts resolved
- Date: 2026-08-16
- Status: accepted (organizational)
- Numbering note: this entry was first written as D-053. A parallel
  session on the same branch assigned D-053 to the evaluation
  methodology at the same time. That entry is referenced from
  DECISIONS.md, HANDOFF.md and KONZEPTPHASE.md, this one from nowhere,
  so this one was renumbered to D-056 on 2026-08-16. Commit message
  `5b3612e` still says D-053; it is the same entry. The collision is
  the reason for the one-session rule now in CLAUDE.md.
- Context: The thesis-writer agent was asked to report its own working
  procedure. It found four hard contradictions and six points at which
  it would have had to guess. The rule material had grown across
  STYLE.md, HOCHSCHULVORGABEN.md, a user-supplied writing plan under
  docs/reference/, and CLAUDE.md, with rules duplicated and in one case
  contradicting each other.
- Options considered: (a) keep the four-level rank order (user
  decision, university requirement, writing plan, own judgement) and
  merely document the conflicts / dissolve the writing plan so only two
  rule sets remain; (b) merge STYLE.md and HOCHSCHULVORGABEN.md into a
  single file / keep them separate.
- Decision: (a) Exactly two rule sets: STYLE.md for how the text is
  written, HOCHSCHULVORGABEN.md for what the university requires. The
  writing plan docs/reference/Einleitung.md was dissolved into
  HOCHSCHULVORGABEN.md section H; its style rules were dropped as
  duplicates or as superseded by D-049. Everything under
  docs/reference/ is evidence and carries no binding rules.
  (b) The two rule sets stay separate.
  Individual points settled at the same time: forward references are
  permitted for floats everywhere and for later sections from chapter 2
  onward, but never in chapter 1; extents are given in pages with a
  working conversion of about 350 words per page; a term not covered by
  the terminology table is reported rather than guessed, and six open
  terms are now listed there; the agent reports a conflict instead of
  resolving it, since writing DECISIONS.md is outside its mandate; the
  check sections carry a table stating which text they apply to, so a
  one-sentence edit no longer triggers the full chapter-1 checklist.
- Rationale: The rank order only works if each level exists in exactly
  one place. The writing plan occupied a middle level that the agent
  never read, because its own instructions did not name it, so a rule
  that looked binding was in practice dead. Dissolving it removes the
  level instead of documenting a workaround. The two rule sets stay
  apart because the university list must be usable in isolation when
  checking before submission, which a merged file would not allow.
- Verification: dev laptop, MiKTeX, chain pdflatex to biber to
  pdflatex twice: exit 0, 22 pages, no errors. Biber parses both .bib
  files; the only warning is that no citation exists yet.
- Side fix found during the review: thesis/sources/citations.bib was
  never loaded via addbibresource, so any citation from it would have
  compiled silently as an unresolved marker. It is now loaded in
  thesis/main.tex.
- Sources: thesis-writer self-report 2026-08-16; TH Nuernberg guide via
  docs/reference/einleitung_recherche_2026-08-16.md part A.1.

## D-057: Insertion target corrected to the inner (second) pocket stage; real geometry measured on site
- Date: 2026-08-17
- Status: accepted (factual correction + first measured numbers)
- Context: On-site measurement at the institute. The fixture turned
  out to be a DEHN-built special-purpose rig, "Testadapter Adhoc-MRK"
  (nameplate: Dokument-Nr. 3011715/3011716, Baujahr 2020) — not a
  catalog product; a web/catalog search on 2026-08-17 found no public
  datasheet. A CAD/drawing request to DEHN via the supervisor is
  pending. D-046 (scope limit 5) and D-047 (depth reference) described
  the part as seating in the FIRST stage of the stepped pocket.
- Options considered: keep the first-stage wording / correct it to
  what the real hardware shows.
- Decision: The part seats in the INNER (second) stage of the pocket;
  the outer first stage is much wider and accepts a different
  component. D-046 limit (5) and D-047's depth reference now read
  "inner (second) stage"; the success-depth threshold is derived from
  the inner-stage geometry. Source ranking for the fixture until DEHN
  answers: the on-site hand measurement leads (no CAD exists); for the
  part, CAD leads and measurement verifies (unchanged).
  Measured on 2026-08-17 (caliper; clearance via paper gauge, ~0.1 mm
  per sheet): part 143.34 x 90.3 mm [CORRECTION 2026-09-11: width superseded by D-121, CAD 90.00 mm, `insertion_tasks_cfg.py` `PART_BODY_X`], total height along the insertion
  axis 50.7 mm (datasheet nominal 144 x 90, 43.5 + 7 = 50.5 mm);
  guided insertion depth ~50.5 mm (part ends just below the highest
  guide edge; edge height varies, low point 36.1 mm); real clearance
  ~0.5 mm along the long axis, ~0.2 mm along the short axis, both
  concentrated on one side (asymmetric) [CORRECTION 2026-08-25:
  per-side clearance now measured, recorded in D-087; these totals
  are superseded]. Part mass still open
  ([weigh at home]; the datasheet's 1.3 kg includes the modules).
- Rationale: Numbers rule — geometry comes from CAD or measurement,
  never guesses. The measured clearance (0.2/0.5 mm) falls below the
  earlier 0.5–1 mm estimate, which triggers the Block-1 reservation:
  the Inoue depth-only regime must be re-examined for the short axis,
  the alignment tolerance can now be derived from real clearance, and
  the interpenetration-filter threshold must be set relative to it
  (all Block 2).
- Partly superseded (2026-08-22, D-073): the height of 50.7 mm was a
  caliper error. DEHN's own part CAD gives 50.506 mm and the datasheet
  43.5 + 7 = 50.5 mm. The measured record is kept above as a report of
  what was measured; the value in use is 50.506 mm. The 143.34 x 90.3
  footprint and the clearances are untouched by this. [CORRECTION 2026-09-11: the width fell later -- D-121, CAD 90.00 mm.]
- Sources: docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.pdf (measured
  values, photos, source per row);
  docs/reference/Datenblaetter/dehnvap-900360.pdf; photos in Bild/.

## D-058: Workcell modelled as two box table plates around a fixed nominal fixture pose
- Date: 2026-08-17
- Status: accepted (first cell measurements taken; some numbers still
  `[measure on site]`)
- Context: The real cell has two tables: table 1 carries the UR5e
  (bolted, final), the lower table 2 carries the test adapter, bolted
  at a position marked as a black rectangle (footprint of the whole
  adapter base plate; the adapter's rail back faces a viewer standing
  in front of table 2). The same setup was previously solved with
  conventional machine learning, so reachability of the position is
  practically proven.
- Options considered: model the tables in full detail (legs,
  profiles) / model only the two table plates as box collision
  surfaces plus the measured relative pose.
- Decision: Table plates as plain box collision surfaces, following
  the proxy precedent D-023 (only the base-to-fixture relative pose
  and touchable surfaces matter physically; the policy has no camera,
  D-046). Legs/visuals are cosmetic and may be added at the very end.
  The black rectangle is the FIXED nominal pose; the D-044 variation
  axes randomise around it in simulation only. Measurement method:
  edge-to-edge with a tape measure (base centre is inaccessible; it
  is reconstructed as edge distance + half the base diameter from the
  official UR5e dimension sheet). Tape accuracy (±5 mm) is
  sufficient: the fine pose is randomised in training and the thesis
  is simulation-only.
  Measured 2026-08-17 (tape/caliper): plate height difference
  95.8 mm (direct, leading; absolute heights 884/787 mm give 97 mm —
  tables are not perfectly level), horizontal gap 6.7 mm, table-2
  side overhangs 447/443 mm, base edge to front/right table-1 edge
  104/220 mm, rectangle 400 x 880 mm, plates 600 x 1000 mm and
  1505 x 750 mm. Still open: rectangle corner distances, pocket
  offset on the base plate, guide-edge height above plate 2, UR5e
  base diameter (UR dimension sheet).
- Rationale: Matches the numbers rule (measurement, no guesses) and
  keeps the environment build minimal; detail modelling would add
  measurement and modelling effort with zero effect on physics or
  observations.
- Sources: docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.pdf (section
  Arbeitszelle, photos Bild/Arbeitszelle_markiert.jpg,
  Bild/Tisch2_Markierung.jpg); old-repo D-023 as precedent; user
  statements 2026-08-17 (grill session).

## D-059: Pocket interior modelled as part-negative plus lugs and first stage, not hand-traced
- Date: 2026-08-17
- Status: accepted (all numbers measured; superseded by the DEHN CAD
  if it arrives)
- Context: The pocket's wave contour follows the part's shape. If
  DEHN sends no CAD, the fixture model must come from measurements.
  Hand-tracing the wave contour is laborious and no more accurate
  than the fit it produces.
- Options considered: hand-measure the full wave contour / model the
  pocket as the part's negative (part CAD contour offset by the
  measured clearance) with straight walls plus only the features the
  user named as functionally relevant.
- Decision: Pocket = part outer contour + measured clearance (0.5 mm
  long axis / 0.2 mm short axis) [CORRECTION 2026-08-25: these were the
  intended input figures; what the finished CAD actually delivers is
  0.5876 mm short axis (D-121; this bracket said 0.2876 until 2026-09-11) and 1.60 mm long axis (D-088), and the real rig's
  per-side play is in D-087. The simulation's clearance is whatever the
  CAD produces, not this pair -- D-087], straight walls. Modelled extras
  (all measured 2026-08-17): (1) two rail-latch lugs on the wave
  side — 108.1 mm centre-to-centre (direct measurement leads over
  the ~12.5 mm-per-side estimate, which would give 101.8), 17 mm
  wide with ~16 mm inner rounding, 7 mm protrusion, full 36 mm
  height; (2) the first stage above — inner 151.7 x 130 mm [CORRECTED
          2026-08-24: 153.0 x 130.0876 mm, measured on the real block
          and matching the CAD — see D-084], located
  175–305 mm in depth (ends 10 mm before the back wall), notches top
  and bottom 47 x 12 mm, corner radii 8 mm everywhere (16 mm circle
  probe = cutter radius); the corner at the end of the lug row is
  the only one running full height to Platte 2, opening 16.8–17 mm —
  flagged as the spot where the policy can gain clearance while
  threading in. A vertical reference photo with a ruler was
  deliberately skipped (user decision).
- Rationale: The fit (clearance) is what the insertion physics
  feels, and the negative-model construction reproduces the measured
  fit exactly — better than summing hand-measurement errors around
  the wave contour. The extra features are exactly those the user
  identified as shaping the insertion process.
- Sources: docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.pdf
  (subsection Tascheninnenkontur); photos
  Bild/Tasche_Nasen_markiert_*.jpg, Bild/Teil_neben_Nasen.jpg,
  Bild/Tasche_Stufe1_Messungen.jpg; user measurements 2026-08-17.


## D-060: Fixture block CAD conventions — separate asset, pocket-centre origin, STEP export, block-only randomisation
- Date: 2026-08-17
- Status: accepted (modelling conventions; the physical-inertness claim
  is literature-backed but UNVERIFIED in our simulation until the scene
  runs on the training machine)
- Context: The user builds the fixture CAD himself (D-059 grob recipe;
  DEHN CAD still pending). Four conventions had to be fixed before
  modelling: part scope, origin, export format, and how the pose
  randomisation (D-044 offset/yaw/tilt) treats the surrounding
  geometry, since a tilted fixture visually clips into the plate
  below it.
- Options considered: (1) separate pocket-block part vs. one combined
  CAD with base plate/tables; (2) randomise only the block vs. move
  the whole adapter stack; (3) origin at pocket-opening centre on the
  block top face vs. at a block corner; (4) STEP vs. STL export.
- Decision: (1) Separate CAD part containing only the block with both
  pocket stages (260 x 210 x 116); base plate, intermediate level,
  back wall and tables remain simulation boxes (D-058, approved
  scene-build plan). (2) Randomisation moves only the kinematic
  block; visual interpenetration with the static plates is accepted
  and collision between block and plates may be disabled. (3) Origin
  = centre of the stage-2 opening on the block top face; X = long
  axis (+X toward the 7.9 mm margin end), Y = depth (+Y toward the
  back wall), Z up. (4) Export as STEP
  (docs/Geometrie/Taschenblock_grob_v1.step). Build steps and all
  coordinates were in docs/Geometrie/CAD_Anleitung_Taschenblock.tex.
  SUPERSEDED 2026-08-24, and the file is DELETED: that recipe planned a
  part-negative to be BUILT. The real fixture was measured on site
  instead and its CAD is the authority (docs/decisions_inbox.md,
  "Fixture CAD authority"; asset CAD/Aufnahme_real_v1.stp). Take the
  geometry from there, not from a recipe the measurement overtook.
- Rationale: The separate block matches the approved scene-build plan
  (fixture asset = pocket block with both stages only) and keeps the
  SDF collision mesh small. Kinematic-vs-static overlap produces no
  contact response in PhysX 5, and fixture-pose randomisation of
  fixed/kinematic sockets without ground-clipping treatment is the
  uniform practice in Factory, IndustReal, FORGE and AutoMate; the
  quantity that is policed is part-pocket interpenetration in
  evaluation (D-047). The pocket-centre origin makes the simulation
  target pose and the D-044 rotation axes coincide with the asset
  origin, removing a constant-offset error source; STEP keeps the
  geometry lossless for USD/SDF conversion on the training machine.
- Sources:
  docs/reference/literature_check_collision_modelling_2026-08-17.md
  (PhysX 5 Rigid Body Dynamics docs; arXiv:2205.03532, 2305.17110,
  2408.04587, 2407.08028); approved scene-build plan 2026-08-17;
  docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.pdf.

## D-061: Scene build pulled forward before the phase-1/phase-2 gates
- Date: 2026-08-17
- Status: accepted (user decision; scope restriction is the safeguard)
- Context: Phase 3 (environment retrofit) is gated on concept blocks
  2-8 and on closing every [CAD pending] placeholder. The user wants
  the simulation scene built now, in parallel with the remaining
  concept blocks.
- Options considered: full phase-3 start (breaking both gates) /
  scene-only pre-work with the gates kept for the rest of phase 3 /
  waiting for the gates.
- Decision: SCENE ONLY is pulled forward: two tables, UR5e, welded
  gripper+part chain, fixture asset, package skeleton. Observation,
  action, reward, curriculum and training code stay untouched on
  proxy state until blocks 3-7 decide them. The gates keep applying
  to the rest of phase 3. Open numbers enter the scene as named
  placeholders only.
- Rationale: The scene needs only geometry, and the geometry is
  measured (D-057/D-058/D-059); no structural decision in the scene
  depends on an open concept block. Building it early de-risks the
  asset pipeline (STEP->USD, SDF) on the training machine while the
  concept work continues.
- Sources: approved scene-build plan 2026-08-17 (phases A-F);
  HANDOFF.md phase gates; user decision in the planning session.

## D-062: Collision modelling via SDF mesh collision, box compound as documented fallback
- Date: 2026-08-17
- Status: accepted (parameters UNVERIFIED until the scene runs on the
  training machine)
- Context: Real clearance is 0.5 mm (long axis) / 0.2 mm (short
  axis) [CORRECTION 2026-08-25: superseded -- real per-side play is
  recorded in D-087 (long 0.8/0.1, short 0.3/0.1); the simulated
  clearance is 1.60 mm long (D-088) and 0.5876 mm short (D-121; 0.2876 until 2026-09-11). The argument
  below held on every figure while the sim short axis was 0.2876 mm. Against 0.5876 mm it fails: 143/256 = 0.559 mm is below that play (the real-rig figures are still exceeded). D-121 names the SDF-resolution justification as gone. Old wording: "0.56 mm voxel spacing still exceeds the short
  axis on every one of these figures"]. The part underside (rail feet, hooks) engages
  counter-contours in the pocket at the end of insertion; both pocket
  stages have functional roundings and a wave contour (photos in
  Bild/, D-059). The proxy used analytic box primitives at ~1 mm
  clearance.
- Options considered: analytic primitive compound (proxy pattern) /
  convex decomposition of the CAD meshes / SDF triangle-mesh
  collision (Factory method) / SDF on cleaned, simplified CAD
  meshes.
- Decision: SDF mesh collision on cleaned CAD meshes for BOTH bodies:
  part mesh from Bild/900360.stp, fixture mesh from the user-built
  CAD (D-059 recipe, D-060 conventions). Fixture kinematic, part
  dynamic (welded into the robot articulation). SDF resolution ~1024
  or sparse subgrid (SDFMeshPropertiesCfg), tuned on the training
  machine. The box compound remains the documented fallback if SDF
  proves unstable or exceeds GPU memory.
- Rationale: Every tight-clearance insertion study in the lineage
  uses SDF collision (Factory, IndustReal 0.5-0.6 mm, FORGE,
  AutoMate; the shipped Isaac Lab factory task runs 0.114 mm
  diametral); convex decomposition is explicitly rejected there for
  tolerance-critical contact. Primitives cannot represent the
  functionally relevant roundings and underside engagement. Caveat
  that forces the resolution choice: SDF voxel spacing = largest
  AABB extent / resolution; at 143 mm and resolution 256 the voxel
  (0.56 mm) would exceed the 0.2 mm clearance. PhysX 5 constraint:
  static actors do not support SDF, hence the kinematic fixture.
- Sources:
  docs/reference/literature_check_collision_modelling_2026-08-17.md;
  arXiv:2205.03532 (Factory), arXiv:2305.17110 (IndustReal),
  arXiv:2408.04587 (FORGE), arXiv:2407.08028 (AutoMate); Isaac Lab
  factory_tasks_cfg.py and schemas_cfg.py (SDFMeshPropertiesCfg);
  PhysX 5.4 Rigid Body Collision docs; D-057 clearance measurement.

## D-063: Real-task code adopted by copying the proxy package into this repo as `insertion`
- Date: 2026-08-17
- Status: accepted
- Context: D-042 fixed "adapt, not rebuild" for the proxy code base,
  but not where the adapted code lives or how it is named. The old
  repo must stay untouched as the citable record of the proxy
  results.
- Options considered: copy the package into this repo under a new
  name / continue on a new branch in the old repo.
- Decision: Copy source/proxytask/... and scripts/ from the old repo
  (clone Generalisierung-Angle-Probe) into THIS repo as
  source/insertion/insertion/tasks/direct/insertion/, package and
  module name `insertion`, gym ID `Ur5e-Insertion-Direct-v0`,
  env-var overrides renamed PROXYTASK_* -> INSERTION_*.
  Proxy-specific derivations (C4 yaw window, square-clearance math,
  analytic inertias) are marked `# PROXY-SPECIFIC, replace`, not
  silently deleted. Robot source: the Isaac Lab built-in UR5e asset
  via .copy() (proxy pattern, ur10e_cfg.py), to be verified against
  the Isaac Lab 2.3 docs; fallback is the official UR URDF import.
  The UR5e home pose is re-solved with UR5e kinematics (the proxy
  home pose is hard-coded UR10e IK).
- Rationale: Keeps the verified proxy repo untouched and citable,
  matches the CLAUDE.md repo separation (this repo = real task), and
  reuses the verified code path per D-042. Explicit marking keeps
  proxy assumptions visible until each is replaced by a measured
  real-task value.
- Sources: D-042; old-repo files proxytask_tasks_cfg.py /
  ur10e_cfg.py / author_peg_ur10e.py (retrofit surfaces identified
  2026-08-17); approved scene-build plan 2026-08-17.

## D-064: Gripper and part enter the scene as a welded chain with named placeholders
- Date: 2026-08-17
- Status: accepted (placeholders open: gripper geometry, suction
  points, part mass)
- Context: The rigid chain (D-041) needs the flange->part transform.
  It depends on the gripper geometry (flange-to-suction-plane length,
  suction-cup positions), which is unmeasured; the user has the
  gripper at home WITHOUT the suction cups mounted. The part mass is
  also still unmeasured (datasheet 1.3 kg includes modules and is
  only an upper bound).
- Options considered: wait for the gripper measurement / build the
  chain now with a placeholder flange->part offset and swap numbers
  later.
- Decision: The welded chain flange -> [gripper placeholder] -> part
  is authored now (FixedJoint pattern from author_peg_ur10e.py). The
  flange->part transform is a single named constant marked
  [measure at home]; the part mass is a named placeholder below the
  1.3 kg datasheet bound. The gripper body itself gets no collision
  geometry until its measurement exists. Suction-cup height comes
  from the cup datasheet or a later measurement.
- Rationale: The structural decision (rigid chain, welded links) is
  independent of the missing numbers; naming the placeholders keeps
  the numbers rule intact (no guessed value enters silently) and the
  swap is one constant per number.
- Sources: D-041; D-057 (part dims); measurement plan: user weighs
  the part and measures the gripper at home; datasheet
  docs/reference/Datenblaetter/dehnvap-900360.pdf (mass upper
  bound).

## D-065: Parallel sessions run in fixed streams (branch + worktree), shared files only on main
- Date: 2026-08-18
- Status: accepted
- Context: The one-session rule was violated in practice (parallel
  measurement/thesis sessions committed while the scene-build session
  ran, 2026-08-17/18). The user wants parallel work to be possible
  without risking the global decision numbering or an overwritten
  shared file (the 2026-08-16 incident).
- Options considered: keep the strict one-session rule / free
  parallel work on main with a writer-only rule for shared files /
  fixed named streams, each on its own branch in its own git
  worktree, plus a writer-only rule and a numberless decisions inbox.
- Decision: Three fixed streams with fixed names, each a long-lived
  branch checked out in its own worktree folder: umbau (source/,
  scripts/), thesis (thesis/), konzept (docs/, Bild/). The four
  shared files (DECISIONS.md, HANDOFF.md, CLAUDE.md, PROBLEMS.md)
  are edited only on main by exactly one writer session per work
  block; other streams file candidate decisions without numbers in
  docs/decisions_inbox.md. Session start = pull + merge main in;
  block end = push + small merge back to main.
- Rationale: One folder can only have one checked-out branch, so
  parallel sessions need worktrees anyway; fixed stream names keep
  the mapping stable across days. Branches alone do not protect the
  decision numbering (numbers are global and unmergeable), hence the
  single-writer rule and the inbox. Small frequent merges keep the
  streams from diverging.
- Sources: CLAUDE.md session rules (2026-08-16 incident); user
  proposal (three named branches) refined in the 2026-08-18 session.
- Addendum 2026-08-18 (same day, user refinement): stream names must
  say WHAT is being worked on, and each worktree/branch is one work
  unit with a hard cut (merge to main only when its verification
  passed). Active pair renamed to phase3-szene (wt-umbau) and
  thesis-schreiben (wt-thesis); the unused konzept stream was
  dropped. Worktrees moved by the user under
  ..\Thesis Workflow Phase 1\. Details in CLAUDE.md.
- Addendum 2 (2026-08-18, user phase model): streams are organised in
  USER PHASES = numbered save points (Phase 1 = scene setup; count
  open-ended). Branch scheme pN-<task>; one folder per phase
  (..\Thesis Workflow Phase N\); a phase cut = merge to main + an
  immutable tag (phase-N-abschluss / phase-N-stand-<topic>) + a
  HANDOFF note recording where the phase ended. Konzept stream
  restored as wt-konzept-messung / p1-konzept-messung (dropping it
  in the first addendum was premature - the measurement session is
  active). Full wording in CLAUDE.md.

## D-066: One handoff per stream, stream instructions in untracked CLAUDE.local.md
- Date: 2026-08-18
- Status: accepted
- Context: With parallel streams (D-065), every session ending its
  block would want to write the single HANDOFF.md, and per-stream
  instructions (e.g. no code rules for thesis writing) would diverge
  CLAUDE.md across branches and overwrite main's version at merge
  time.
- Options considered: keep one shared HANDOFF.md (writer-only) /
  per-stream handoff files with one writer per file; branch-local
  CLAUDE.md with a merge=ours driver / per-worktree untracked
  CLAUDE.local.md.
- Decision: (1) HANDOFF.md stays the global entry point, edited only
  on main. Each stream ends its block by updating its own file:
  HANDOFF-SZENE.md, HANDOFF-THESIS.md, HANDOFF-KONZEPT.md — exactly
  one writer per file. [Extended 2026-08-25: HANDOFF-RL.md added for
  the new stream p2-rl-code, same rule.] (2) Stream-specific instructions live in an
  untracked, gitignored CLAUDE.local.md inside each worktree folder
  (role, read list, no-touch list); the tracked CLAUDE.md stays
  identical everywhere and is edited only on main. (3) The current
  work state reaches a stream by merging main in at session start,
  not by hand-picking files.
- Rationale: One writer per file makes handoff merges conflict-free
  by construction. An untracked local instruction file cannot leak
  into main through a merge, which a branch-divergent CLAUDE.md
  would do without per-clone merge-driver setup. Files present in a
  worktree do not load into a session's context by themselves; the
  read list in CLAUDE.local.md is what controls context, so no file
  curation is needed.
- Sources: D-065 (+addenda); Claude Code memory documentation
  (CLAUDE.local.md loaded per folder, untracked); user requirement
  2026-08-18 (separate handoffs, no code rules in the thesis
  stream).

## D-067: STEP to USD via the Isaac Sim CAD converter, with the proxy repair chain
- Date: 2026-08-21
- Status: ACCEPTED 2026-08-23 -- the route has been walked end to end on
  the training machine for one asset (`greifer_bauteil_asm`): GUI import
  (RT-8), unit repair (RT-10), and an independent re-measurement of the
  repaired file (RT-12). RT-12 reads `metersPerUnit = 1.0`, up-axis Z,
  0 composition arcs, `D-016 invariants: PASS` and `asset contract:
  PASS` on the asset path itself, not on a candidate beside it. The two
  remaining repair steps of this decision -- instanceable flag
  (`fix_fixture_asset.py`) and collision (`add_fixture_collision.py`) --
  are UNVERIFIED; nothing has run them, and `verify_fixture_usd.py` does
  not test for instanceable prims. Accepting the route is not a claim
  about those two scripts.
- Context: CLAUDE.md carried this as the only OPEN asset question:
  (a) the Isaac Sim CAD converter directly on STEP, or (b) an external
  STEP -> STL/OBJ step followed by Isaac Lab's `MeshConverter`. Nothing
  can enter the scene -- gripper, part, real pocket -- until it is
  settled. `scripts/convert_step_asset.py` states it is UNVERIFIED, and
  no `rt_logs/` entry references any conversion.
- Options considered: (a) Isaac Sim CAD converter (`omni.kit.converter.cad`,
  HOOPS backend) directly on the STEP / (b) FreeCAD or similar to
  STL/OBJ, then Isaac Lab `MeshConverter`.
- Decision: (a). GUI import: File > Import in an Isaac Sim started from
  the **native `isaacsim` install folder**, not through the isaaclab
  launcher. Save with `Flatten`/`Export`, never `Save As`. Three repair
  steps follow every import and are ported from the proxy repo rather
  than rewritten: units (`fix_stage_units.py`), instanceable flag
  (`fix_fixture_asset.py`), collision (`add_fixture_collision.py`).
  Prim paths are never hard-coded.
- Rationale: The proxy repo used this exact path and its failure modes
  are already documented with fixes, so the cost is a port, not a
  discovery. Option (b) has no precedent in either repo, and whether
  FreeCAD exists on the training PC is UNKNOWN. Recorded conflict: the
  proxy later *abandoned* the CAD import for the table (D-033) in
  favour of a box composition -- that escape hatch does not exist here,
  because a suction gripper, a terminal-block part and a machined
  pocket are not compositions of boxes.
- Sources: user statement 2026-08-21 (converter used for the proxy);
  proxy `PROBLEMS.md:21,23,24,25,26`; proxy
  `docs/asset_contract_tisch.md:21-34,63-74,125-127`; proxy
  `DECISIONS.md:383-384,1229-1250` (D-033, the abandonment); runs RT-8,
  RT-10, RT-11 and RT-12 on the training machine (`rt_logs/VERDICTS.md`),
  the last of them the one that measures the repaired asset itself.

## D-068: Asset origin convention -- flange mating face, +Z away from the robot
- Date: 2026-08-21
- Status: accepted
- Context: The gripper/part assembly must be welded into the robot USD.
  Where the CAD assembly puts its zero decides whether the weld is an
  identity or a measured offset. `CAD/README.md` marks the origin of
  every STEP file as UNGEPRUEFT.
- Options considered: keep whatever origin the CAD assembly already has
  and measure the offset after import / place the origin on the flange
  mating face / place it at the midpoint between the suction cups.
- Decision: The CAD assembly's origin is placed on the **flange mating
  face** (the surface that contacts the robot flange), +Z pointing away
  from the robot toward the part. The weld transform is then the
  identity. All further geometry is read as plain Z values from that
  zero.
- Rationale: The flange face is the only surface that physically
  contacts the robot, so it is unambiguous and selectable in CAD. The
  cup midpoint is a constructed point, and the bellows and cups are not
  in the CAD at all, so it does not exist there. An identity weld
  removes the whole class of sign and offset errors -- the proxy lost
  time to exactly that (`PROBLEMS.md:25`, missing bore coordinate
  system in the STEP). This convention is **authored, not adopted**:
  Factory, IndustReal and AutoMate state no mesh-origin rule anywhere;
  Factory's is only reverse-engineerable per task from its offset
  arithmetic, and AutoMate's amounts to "plug and socket origins
  coincide when assembled".
- Sources: Isaac Lab 2.3.2 `factory_utils.py:49-102`,
  `factory_env.py:553-585,657-669`, `factory_tasks_cfg.py:14-30`;
  AutoMate `assembly_env.py:134-140,802-806`; proxy
  `docs/asset_contract_tisch.md` (D-029 addendum); proxy
  `PROBLEMS.md:25`.

## D-069: TCP split into a tool frame and a task frame at the part
- Date: 2026-08-21
- Status: accepted (amended 2026-08-21 after the grasp photographs; one
  constant still open, see below)
- Context: The word TCP appears in no Python file in this repo. The
  chain flange -> gripper -> suction plane -> part needs a defined
  reference before any reward or distance term can be written. Industry
  convention puts the TCP on the tool; the RL assembly literature this
  repo follows puts the reward frame on the held part.
- Options considered: a single TCP at the suction plane (tool-fixed,
  survives a part change) / a single frame at the part's mating
  reference / both, as two named constants.
- Decision: Both, as named constants in `insertion_tasks_cfg.py`. Three
  surfaces are involved and they are deliberately named apart, because an
  earlier draft of this entry collapsed "part top" and "suction face" into
  one name and produced an ambiguous number:
  `FLANGE_TO_SUCTION_FACE = 0.1420` m -- the tool TCP, measured **gripped**
  to the INNER chamber floor the cups rest on; and
  `FLANGE_TO_PART_BOTTOM = 0.1520` m -- the task frame, the closed outer
  underside that enters the pocket first. Reward and every distance term use
  the task frame. `FLANGE_TO_SUCTION_TIP_FREE = 0.1486` m (unloaded) is kept
  as documentation only and is never welded with. Derived and documentary:
  bellows compression 0.0066 m, part floor thickness under the cup 0.0100 m.
  Ordering is forced by geometry: part top < suction face < part bottom.
  Geometry confirmed by the user: the two cups reach DOWN INTO two chambers
  of the terminal block, and their midpoint lies on the gripper axis, so the
  chain is a straight line in Z with no lateral offset. `GRASP_POS_NOISE`
  stays OPEN and is listed in `TOOL_CHAIN_PENDING`.
- Rationale: Both numbers are needed to build the weld chain anyway, so
  naming both costs nothing and lets the thesis state the industrial
  convention and the Factory-style task frame without choosing between
  them. Factory names the equivalent frame `held_base` and computes it
  as root pose composed with a constant local offset; the proxy repo
  already follows the same shape with `PEG_TIP_OFFSET` in the flange
  frame. The unloaded 148.6 mm is deliberately **not** the weld value:
  the bellows compress by an unknown amount under load, and the gripped
  measurement removes that unknown instead of estimating it.
- Sources: user measurements 2026-08-21 (calliper: 148.6 mm free,
  142.0 mm gripped to the suction face, 152.0 mm gripped to the part
  bottom) and photographs of the grasp; Isaac Lab 2.3.2
  `factory_utils.py:49-78`; proxy `proxytask_tasks_cfg.py:267-277`.

## D-070: Suction compliance is not modelled; grasp uncertainty enters as reset noise
- Date: 2026-08-21
- Status: accepted
- 2026-09-11: the per-reset offset below was measured impossible on the
  welded body (D-126) and now enters through the observation only (D-183).
- Context: The real gripper holds the part on two bellows suction cups.
  Bellows are visibly compliant, and the part sits slightly differently
  on every grasp. Neither the bellows nor the cups exist in
  `Greifer_Standart_V5.stp`. D-064 left gripper geometry, suction
  points and part mass as open placeholders.
- Options considered: model each cup as a compliant joint with
  stiffness / model no compliance and randomise the flange-to-part
  transform at reset / do neither.
- Decision: No compliance. The gripper stays a rigid welded link with a
  visible mesh and **no** CollisionAPI; the part carries SDF collision.
  Grasp variation is represented by randomising the flange-to-part
  offset at every reset, after Factory's `held_asset_pos_noise`. The
  cups need no geometry for this.
- Rationale: Two separate phenomena were being conflated. "The cups
  grip slightly differently each time" is a per-episode transform
  offset and needs no geometry -- Factory implements exactly this as a
  noise vector. "The bellows yield under contact" is a within-episode
  spring and would require extra joints, changing the articulation and
  therefore the observation and action dimensions; CLAUDE.md requires
  that choice to be made before training, not after. The project is
  explicitly **not** sim-to-real (global CLAUDE.md), so passive
  compliance has no transfer to preserve and modelling it buys realism
  the thesis does not claim. If compliance is ever wanted, the decision
  must be reopened **before** the first real training run.
- Sources: user photographs and statement 2026-08-21; global CLAUDE.md
  (no sim-to-real); Isaac Lab 2.3.2 `factory_tasks_cfg.py:65,123`,
  `factory_env.py:763-775`; D-064.

## D-071: Insertion is a tilted, corner-first entry, learned rather than scripted
- Date: 2026-08-21 (Decision and Context corrected the same evening, see
  the correction note at the end)
- Status: accepted
- Context: The scene work assumed the part drops straight down into the
  pocket. The user corrected this from the real setup: the part cannot
  go in vertically. It is tilted -- tilting to the LEFT is the easier
  side -- the SHORT side is led in first by a few millimetres, and only
  then does the part align to the pocket floor and seat.
  The geometry forces this rather than merely allowing it: the inner
  pocket (stage 2, the seat) does NOT sit centred inside the outer
  pocket (stage 1). On the RIGHT it lies flush against the outer wall;
  on the LEFT there are **7.9 mm** of ledge to the outer wall. The
  clearance to tilt into exists on one side only.
  This supersedes `KONZEPTPHASE.md:344` ("insertion vertical from above
  after corner approach"), which is the older statement.
- Options considered: script the two-stage motion as a waypoint
  sequence / expose the staging only through geometry and reward and
  let the policy discover the motion.
- Decision: The motion is not scripted. The tilt is recorded as a FACT
  about the task (chapter 4.1), not as a prescribed trajectory. This
  re-affirms D-045, which already decided that the task defines only
  the start distribution and the goal state and that corner-first is
  parked as a Block 6 curriculum idea; D-071 adds the corrected
  direction, the 7.9 mm asymmetry that causes it, and the asset
  consequence below. Block 6 still owns any curriculum use of it.
- Rationale: A scripted approach would fix one solution and remove the
  contact-rich search the task is meant to study. Consequence for the
  asset plan: a tilted corner-first entry is not learnable against the
  solid box currently standing in for the fixture, so importing the
  real pocket (`CAD/Aufnahme_real_v1.stp`, Phase D) moves from optional
  to required. It also means a collinear keypoint reward would
  underspecify this part -- IndustReal states collinear keypoints
  underspecify non-axisymmetric parts -- which is consistent with the
  proxy already rejecting keypoints (D-004).
- Sources: user statement and photographs of the grasp and of the
  seated part, 2026-08-21; D-045 (policy discovers the strategy);
  D-057 (the seat is the inner stage); IndustReal (arXiv 2305.17110);
  proxy `docs/factory_mapping.md` (D-004).
- CORRECTION 2026-08-21, same evening: the first version of this entry
  claimed the tilted entry "fits the existing staged constant
  `STAGE1_DEPTH = 0.015` (`insertion_tasks_cfg.py:515`), which already
  encodes a first-stage depth". That was a misreading and is withdrawn.
  `STAGE1_DEPTH` (`insertion_tasks_cfg.py:522`) is an ASSET ORIGIN
  OFFSET: 15 mm from the stage-1 rim down to the stage-2 opening plane,
  where depth counting starts (`INSERTION_PLANE_Z`, `:606`). It is not
  a staging of the insertion motion, and the guided insertion depth is
  about 50.5 mm (D-057). Nothing in the code encodes the tilted entry
  today. The line reference was also wrong (`:515` instead of `:522`).

## D-072: Home-pose screening gates on the leading tool point, not on the flange
- Date: 2026-08-21
- Status: accepted (screener verified offline; the recomputed joint
  angles are NOT yet written into `ur5e_cfg.py` -- see Decision)
- Context: `scripts/tools/screen_home_pose_branches.py` rejected a
  branch unless the flange was the lowest point of the whole arm. The
  gate protects a real property -- "the thing that leads downwards is
  the thing you mean to insert", not merely "nothing collides" -- and
  was encoded here for the first time, because the proxy stated it only
  in prose and, with a single table plate, flange and lowest point
  coincided anyway. With the tool chain measured (D-069, 152 mm from
  flange to part bottom) the gate became unsatisfiable: sweeping the
  standoff shows the last feasible value lies between 0.180 m and
  0.200 m, and at the 0.302 m the cell actually needs **all eight**
  branches are rejected -- although every one of them still has
  positive clearance to every obstacle.
- Options considered: drop the gate and screen on positive clearance
  only / keep the flange wording and cap the standoff at ~0.19 m /
  move the reference point from the flange to the leading tool point.
- Decision: Move the reference point. The gate now reads "no arm part
  hangs below the leading tool point". `HOME_STANDOFF_Z` keeps its
  value of 0.150 m but now measures to the part bottom; the screener
  adds `FLANGE_TO_PART_BOTTOM` back on to place the flange, via a new
  `TOOL_CHAIN_LEN` that is zero for the legacy ur10e path. Recomputed
  ur5e home pose, flange at 0.302 m above the block top, six of eight
  branches clear, D-025 again selects **branch 1**:
  `shoulder_pan -3.140928, shoulder_lift -1.510209, elbow -1.909351,
  wrist_1 -1.292829, wrist_2 +1.570796, wrist_3 -1.570131`.
  These angles are **deliberately not yet written** into
  `ur5e_cfg.py`: the robot USD still carries no gripper and no part, so
  adopting them now would raise a bare robot for no visible reason and
  break the RT-5 baseline. They land together with the weld.
- Rationale: Dropping the gate would discard the property it protects
  -- branch 5 and branch 7 hang 119 mm and 135 mm below the tool point
  over the block and would silently become admissible. Capping the
  standoff would leave the part bottom about 40 mm above the block,
  which is not a home pose. Moving the reference point keeps the intent
  word for word and keeps the same shape of result: some branches
  excluded, D-025 choosing among the rest, and the same branch 1
  selected as before. Regression checked: the ur10e path selects
  branch 3 at 0.150 m clearance both before and after the change.
- Sources: `scripts/tools/screen_home_pose_branches.py` (gate rationale
  in situ); standoff sweep 0.150...0.302 run on the laptop 2026-08-21
  (pure kinematics, no Isaac); D-025 selection criteria; D-069 tool
  chain; `ur5e_cfg.py:83` which already predicted this recomputation.

## D-073: The Creo assembly is the source of the part pose in the flange frame
- Date: 2026-08-22
- Status: accepted (numbers read from the assembly; nothing converted or
  simulated yet, so the USD side stays unverified)
- Context: D-068 fixed the asset origin on the flange mating face with
  +Z away from the robot, and D-069 split the TCP into a tool frame and
  a task frame at the part. What was still missing was the pose of the
  part itself in that frame. The user built a Creo assembly of gripper
  plus part and constrained it to the cell measurement, flange face to
  part bottom = 152.000 mm (D-069). A top view of that assembly on the
  same day appeared to show a large lateral offset between the flange
  axis and the part centre, which would have broken the "straight line
  in Z" argument the tool chain rests on.
- Options considered: treat the apparent lateral offset as real and add
  a lateral term to the tool chain / measure it and decide afterwards /
  keep the part origin where the raw STEP had it and carry the offset
  in the weld transform instead.
- Decision: Measure first. The assembly gives the part's coordinate
  system AXIS-PARALLEL to the flange frame, so the pose is a pure
  translation with identity rotation:

      dx = 0.05 mm    dy = 0.07 mm    dz = 126.747 mm

  `FLANGE_TO_PART_CENTRE = 0.126747` m is recorded in
  `insertion_tasks_cfg.py`. The lateral pair is a CHECKED zero and is
  carried as exactly zero; it gets no constant of its own, because a
  number nobody would recompute does not earn a name and would invite
  someone to wire it into a pose. The apparent offset in the top view
  came from non-central planes -- an artefact of the view.

  Two further facts are fixed here. `PART_MARGIN_END_TOOL_FRAME = "+Y"`:
  the part's 7.9 mm-margin end points to +Y of the ASSEMBLY frame. This
  is needed ALONGSIDE the cell-frame statement of D-060 (the pocket's
  7.9 mm end points to cell +Y), because the tool axis points DOWN at
  insertion while cell +Z points UP -- a right-handed frame cannot flip
  Z and keep both lateral axes, so exactly one of them is antiparallel
  between the two frames and "+Y" does not denote the same direction in
  both. Given both statements the insertion orientation of the
  asymmetric part is fully determined; it is not a convention anyone
  gets to choose.

  And the part height: `PART_HEIGHT_Z = 2 * (0.1520 - 0.126747) =
  50.506 mm`, derived from CAD alone since the measured 152 mm cancels.
  This supersedes the 50.7 mm caliper value of D-057. `CAD/fuegeteil.stp`
  is DEHN's own CAD of the part, not a redraw, and it agrees with the
  product datasheet (43.5 + 7 = 50.5 mm) to 6 um; D-057 already ranked
  CAD above measurement for the part.
- Rationale: The 0.194 mm height gap is not cosmetic -- it is the same
  order as the 0.2 mm short-axis clearance the success criterion rests
  on [CORRECTION 2026-08-25: the short-axis figure is 0.5876 mm in the
  simulation, D-121 (0.2876 until 2026-09-11); same order (3x the 0.194 mm gap), argument unchanged], so carrying the wrong one would have biased the depth threshold.
  The lateral result cuts the other way: at 0.05 / 0.07 mm it is three
  orders of magnitude below that clearance and below the grasp scatter
  GRASP_POS_NOISE will deliberately inject (D-070), so modelling it
  would add a number without adding a physical effect. Measuring both
  and then deciding which to keep is what separates a checked zero from
  an assumed one; only the first is admissible here.
- Follow-up the same day, two points that came out of checking the
  above rather than of deciding it. First, the identity rotation is
  VERIFIED, not inferred: flange face against a part datum plane known
  to be parallel to the seating plane reads 180.000 deg exactly and
  126.747 mm, so the translation is confirmed from a second reference.
  An earlier 179.978 deg was taken against an underside FACE, not that
  datum plane -- different faces, no contradiction, and it never enters
  the weld either way. Second, the part's underside is not one plane:
  a sloped strip, a recess, and a strip parallel to the seating plane,
  all three running along the LONG axis, so the underside is asymmetric
  across the SHORT axis. The 152 mm was measured to the parallel strip
  at its lowest edge, which makes that strip the leading face and
  D-069's task frame well defined. Consequence for D-071: the short
  axis is both the tighter one (0.2 mm vs 0.5 mm, D-057 [CORRECTION
  2026-08-25: 0.5876 mm vs 1.60 mm in the simulation, D-121/D-088 (0.2876 until 2026-09-11) --
  the short axis is tighter by a wider margin, not a narrower one (2.7x against 2.5x)]) and the axis
  the underside is asymmetric across, so the two tilt directions about
  the long axis are not equivalent.
- Rotation statement corrected in D-074: the identity orientation
  belongs to the datum coordinate system FUEGETEIL_MITTE, not to the
  placement of the part's geometry.
- Sources: `CAD/fuegeteil.stp` (DEHN original, coordinate system moved
  to the part centre by the user 2026-08-22, geometry untouched); Creo
  assembly gripper + part, 2026-08-22, distances read off in the
  assembly; `docs/reference/Datenblaetter/dehnvap-900360.pdf`; D-057
  (source ranking, superseded height), D-060 (cell axes), D-068 (origin
  convention), D-069 (tool chain), D-070 (grasp noise).

## D-074: The weld transform is read out of the exported STEP, and it is not the identity
- Date: 2026-08-22
- Status: accepted (verified against the STEP file). Partly confirmed in
  USD since: the conversion has run (D-067, accepted 2026-08-23) and the
  part prim lands at z = 101.490 .. 152.004 mm with its Y centre on the
  -0.07 of this entry (RT-8, RT-12), which confirms the TRANSLATION. The
  180 deg rotation about Y is NOT confirmed by that -- a bounding box
  cannot see an orientation flip of this body. It stays verified against
  the STEP only.
- Context: D-073 recorded the part pose from numbers read out in a Creo
  session and concluded the weld rotation was the identity. That
  conclusion came from a 180.000 deg reading between the flange face
  and a part datum plane. When the assembly was actually exported and
  the file inspected, the placement of the part's GEOMETRY turned out
  to carry a rotation. Both statements are true about different
  entities, which is exactly why the wrong one was plausible.
- Options considered: trust the dialog readings and weld with identity /
  read the placement out of the exported file and verify it against an
  independent quantity in the same file.
- Decision: The exported assembly is the source. Placements in the
  flange frame, with the rotation given as the images of the axes:

      FUEGETEIL    t = (0.050, -0.070, 126.746951) mm
                   X -> (-1, 0, 0)  Y -> (0, +1, 0)  Z -> (0, 0, -1)
                   = 180 deg about Y
      GREIFER_V1   t = (0, 0, 0)
                   X -> (0, 0, 1)   Y -> (0, -1, 0)  Z -> (1, 0, 0)
                   = 180 deg about (1, 0, 1)

  Recorded as `PART_WELD_ROT_COLUMNS` and `GRIPPER_WELD_ROT_COLUMNS`.
  Welding a part file with identity rotation puts the part in upside
  down. `PART_MARGIN_END_TOOL_FRAME = "+Y"` is unaffected: a rotation
  about Y leaves Y alone.

  The authoritative file is the single-file assembly export of 19:45,
  `CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp`: two solids,
  structure preserved, `GEOMETRIC_SET` 0 (the datum wireframe that the
  earlier exports carried is gone), units mm. Its geometry is ALREADY
  placed, so the transforms above must not be applied a second time on
  import -- they document what is baked in. Every other STEP in `CAD/`
  is archive; three of them sit in three different coordinate systems
  and `CAD/README.md` marks which is which.
- Rationale: A dialog reading answers the question the user pointed at,
  not necessarily the question being asked -- a datum plane and a
  component placement are different entities and here they disagreed.
  The file states the placement explicitly, and it also carries each
  component's centroid as an independent validation property, so the
  transform can be checked rather than believed: pushing each centroid
  from the part frame through the placement reproduces the centroid the
  file states in the flange frame, to 12 significant digits, for both
  components. That cross-check is the reason this entry is "accepted"
  and not "proposed".
- Sources: `CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp`
  (`ITEM_DEFINED_TRANSFORMATION` #41929 and #47550, centroids as
  `geometric validation property`); earlier export run 18:27, same
  transforms; `CAD/step_import_settings.md`;
  D-067 (conversion route), D-068 (origin convention), D-069 (tool
  chain), D-073 (superseded rotation statement).

## D-075: First STEP to USD conversion, measured -- lugs explain the width, CAD supersedes the caliper length
- Date: 2026-08-22
- Status: accepted for what it measures. The `meters_per_unit = 0.001`
  noted here was repaired afterwards (RT-10 candidate, put in place and
  re-measured in RT-12, 2026-08-23); D-067 is accepted since. The
  millimetre numbers below are unaffected by that repair -- RT-12
  reproduces them to 7e-6 mm, see the addendum at the end of this entry.
- Context: `greifer_bauteil_asm.stp` was imported through the Isaac Sim
  CAD converter on the training machine -- the first conversion this
  repo has ever run -- and checked with
  `verify_fixture_usd.py --per-prim --nominal-mm 143.34 90.3 50.506`
  (log RT-8). Three questions were open going in: does the assembly
  survive as two bodies, how much geometry does the tessellation lose,
  and does the weld chain hold in USD as it does in the STEP.
- Decision: All three are answered by the run, and two recorded part
  dimensions change.

  **Tessellation is a non-issue at the converter default.** The part
  height measures 50.514 mm against a 50.506 mm nominal: **+0.008 mm**.
  `ChordHeightRatio` never has to be touched, and the earlier worry
  that a coarse tessellation would eat into the 0.2 mm short-axis
  clearance [CORRECTION 2026-08-25: 0.5876 mm, D-121 (0.2876 until 2026-09-11); this and the
  0.2 mm figure further down in this entry are the same yardstick and
  neither conclusion changes] is retired -- it was reasoned from the chord-height formula
  without noticing that the clearance-critical faces are FLAT and
  tessellate exactly.

  **The weld chain holds in USD.** The part prim occupies
  z = 101.490 .. 152.004 mm in the flange frame. Predicted before the
  run from FLANGE_TO_PART_CENTRE 126.747 and half the part height
  25.253: 101.494 .. 152.000. The Y centre lands at -0.07, which is
  `dy` from D-074. The chain is therefore verified end to end, CAD to
  USD, and not only inside the STEP.

  **Part length is 143.50 mm, not 143.34.** The caliper reading of
  2026-08-17 is superseded, the same way and in the same direction as
  the height was in D-073 (caliper low by 0.194 mm there, by 0.16 mm
  here). Note for anyone reaching for the datasheet instead: it says
  144 x 90, quoted to whole millimetres, and cannot resolve 0.5 mm --
  it is consistent with 143.50 but is not the better source. D-057
  ranks CAD above measurement for the part; 143.50 is the CAD value.

  **The part is 96.41 mm across the short axis, not 90.3.** The extra
  6.1 mm are the two LUGS, and they sit on ONE side: the mesh spans
  x = -51.359 .. +45.050 about a part centre at +0.05, so the plus side
  matches the 90.3 body and the minus side carries 6.26 mm of lug. This
  is not a defect and not a tessellation artefact -- D-059 already
  described the pocket as "part-negative + two lugs + first stage with
  notches", and `CAD/README.md` records six wall cutouts. The lugs
  enter those cutouts. 90.3 remains correct as the BODY width between
  the lugs; 96.41 is the bounding box. [CORRECTION 2026-09-11, D-121: the 90.3 is wrong. The span above puts the body half-width at 45.050 - 0.05 = 45.00 mm, so the body is 90.00 mm (`insertion_tasks_cfg.py` `PART_BODY_X`); 90.3 was the 2026-08-17 caliper reading.]
- Rationale: The lug finding is the reason D-062 chose SDF collision
  and could not have chosen a convex approximation: a convex hull
  bridges the gap between the lugs and turns the part into a solid
  96.41 mm block, which cannot enter a 90.588 mm pocket at all. What
  looked like an anomaly in the measurement is the feature the
  collision model exists to represent.

  Two of the three surprises in this run were already written down in
  this repo before it started -- the lugs in D-059 and the wall cutouts
  in `CAD/README.md`. The measurement was read as an anomaly first and
  matched against the record second, which is the wrong order and cost
  a round trip.
- Open, sharpened rather than resolved: pocket stage-2 opening 145.1 mm
  against a part now measured at 143.50 mm leaves 1.60 mm of long-axis
  play, while the play measured on site is ~0.5 mm. The conflict noted
  in `CAD/README.md` therefore survives this run and gets larger, not
  smaller. It needs the on-site re-measurement, not more arithmetic.
  [CORRECTION 2026-08-25: closed. D-088 settles the part at 143.50 mm
  and the simulated long-axis play at 1.60 mm; D-087 records the real
  per-side play (0.9 mm total long). The 0.7 mm sim-vs-real gap is a
  property of the simplified fixture, recorded, not modelled away.]
- Also open: `verify_fixture_usd.py` does not check for instanceable
  prims, so whether the import's `Enable Instancing` checkbox took
  effect is still unverified. `fix_fixture_asset.py` reports it.
- Sources: log RT-8, 2026-08-22, `rt_logs/VERDICTS.md`; sidecar
  `greifer_bauteil_asm.verify.json` on the training machine; D-057
  (source ranking, superseded caliper values), D-059 (lugs and
  notches), D-062 (SDF collision), D-067 (conversion route), D-073,
  D-074 (weld transform).
- Addendum 2026-08-23, RT-12, after the unit repair: the millimetre
  numbers above survive it. FUEGETEIL now measures 96.408746 x
  143.500007 x 50.514071 mm where RT-11 read 96.408741 / 143.500000 /
  50.514069 -- a shift of +4.6e-6 / +6.8e-6 / +2.4e-6 mm, which is
  +4.75e-8 RELATIVE on all three edges alike. That is not noise and not
  a second tessellation: it is the float32 representation of the 0.001
  scale op `fix_stage_units.py` authors (0.001 as float32 is
  0.001000000047497451, i.e. +4.7497e-8 relative). Predicted from the
  code, then matched by the measurement to five digits on every edge.
  In physical terms it is 7e-6 mm against the 0.2 mm short-axis
  clearance, four orders of magnitude below anything the task can feel.
  The repair therefore changed the unit DECLARATION and left the
  geometry alone -- which is exactly what the check registered in
  `HANDOFF-SZENE.md` before the run demanded of it.

## D-076: The tool is one welded link, not two bodies joined inside the robot
- Date: 2026-08-23
- Status: proposed (pending verification -- authored on the dev laptop, nothing
  has run on the training machine)
- Context: The imported assembly `greifer_bauteil_asm.usd` carries two solids,
  the gripper and the part, already placed in the flange frame (D-074). To
  reach the simulation they have to become part of the UR5e articulation
  (D-019: a welded link inside the robot USD, because a runtime fixed joint
  fails and the Robot Assembler strips the articulation root). How many rigid
  bodies that link structure should contain was open.
- Options considered: (a) one rigid link carrying both meshes / (b) two rigid
  links, gripper and part, joined by a second `UsdPhysics.FixedJoint`.
- Decision: (a). One link, named `tool_link`, holding a single reference to the
  whole assembly; one `FixedJoint` named `tool_weld` from `wrist_3_link` to it.
- Rationale: The gripper is rigid by decision (CLAUDE.md, asset decisions), so
  the part cannot move relative to it. Two bodies joined by a fixed joint model
  a degree of freedom that is then immediately removed again -- the same
  kinematics at the cost of an extra body, an extra joint and an extra
  constraint for the solver. Option (b) also creates a problem that (a) does
  not have: the CAD shows the gripper adapter overlapping the part by 24.5 mm,
  because the suction cups are missing from the model and the adapter runs down
  to where they would be. Two overlapping colliders in the same articulation
  need a `FilteredPairsAPI`, a mechanism this repository uses nowhere and the
  proxy repository never needed. Under (a) the overlap is not a collision case
  at all. The cost of (a) is that a contact sensor cannot separate part
  contacts from gripper contacts; that is acceptable because the gripper is
  shielded -- 68 x 68 mm inside the part's 96.4 x 143.5 mm footprint -- and,
  under D-077, carries no collider to report contacts with in the first place.
  Reversing this decision means re-authoring the USD, which is a script run,
  not a redesign.
- Sources: `CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp` (two solids, one
  placement each); RT-8 and RT-12 measurements of the two mesh prims
  (`rt_logs/VERDICTS.md`); D-018 and D-019 (why a welded link at all);
  `scripts/author_peg_ur10e.py` (the single-link precedent that is ported);
  user decision 2026-08-23.

## D-077: Collision on the part only, as an SDF mesh; the gripper carries none
- Date: 2026-08-23
- Status: proposed (pending verification -- no PhysX run has used this asset)
  [2026-09-15, inline CORRECTION: the rationale's premise "it never touches
  the fixture" is FALSIFIED. With the block off (D-156) the part got around
  and UNDER the free-standing fixture (RT-201s3r pose report, `rt_logs/
  VERDICTS.md`), and the user watched the replay of RT-201s3 model_200: the
  part hangs below the fixture while the collider-less gripper passes the
  pocket, then the part is pulled up against the underside. The decision is
  not changed; the path is no longer paid (inbox 2026-09-15 "The pocket gate
  gets a floor"). Whether the UR5e arm links collide with the fixture is
  UNKNOWN (not checked).]
- Context: `tool_link` (D-076) is a dynamic body of the articulation. PhysX
  restricts dynamic bodies to convex shapes or signed-distance-field meshes;
  an exact triangle mesh is allowed only for static colliders, which is why
  `add_fixture_collision.py` may use `none` for the fixture and
  `author_peg_ur10e.py` had to use `convexHull` for the peg.
- Options considered: (a) `convexHull` on both meshes / (b) SDF on the part,
  `convexHull` on the gripper / (c) SDF on the part, no collider on the gripper.
- Decision: (c). `UsdPhysics.CollisionAPI` plus `MeshCollisionAPI` with
  approximation `sdf` and `PhysxSDFMeshCollisionAPI` on the part mesh; the
  gripper mesh is left without any collision API.
- Rationale: A convex hull of the part is not a coarser part -- it is a
  different part. The part's bounding box is 96.41 mm across the short axis
  while its body is 90.3 mm [CORRECTION 2026-09-11: 90.00 mm, D-121; conclusion unchanged]; the difference is the two lugs, which sit on one
  side with a gap between them (D-075). A hull bridges that gap and produces a
  solid 96.41 mm block, which cannot enter the 90.588 mm pocket at all. The
  task would be unsolvable for a reason invisible in every geometric check,
  since all edge lengths would still measure correctly. D-062 already chose SDF
  for the fixture side of the same mating pair, so the two halves are modelled
  consistently. The gripper is left uncollided for two reasons: it never
  touches the fixture -- the cups sit inside the part and the part is larger
  than the gripper in both lateral directions -- and its 24.5 mm overlap with
  the part would otherwise have to be filtered (D-076). An absent collider is a
  stronger statement than a filtered one, and it is checked as such:
  `author_tool_ur5e.py` fails if any prim below the gripper carries a
  `CollisionAPI`.
- Sources: PhysX collision-shape restriction for dynamic bodies, as recorded in
  `scripts/author_peg_ur10e.py:231-235` and `scripts/add_fixture_collision.py`;
  D-062 (SDF for the fixture); D-075 (the lugs and the 90.588 mm pocket);
  RT-11 measurement of the gripper mesh ending at z = 126.0 mm against the part
  starting at z = 101.49 mm.

## D-078: The tool's yaw about the tool axis is zero by definition
- Date: 2026-08-23
- Status: accepted (a definition, not a measurement; the quantity it defines
  away is still to be measured)
- Context: The weld needs the gripper's rotation about the tool axis on the
  real flange. That angle is a statement about how the tool was bolted on in
  the cell. It is in no CAD file -- the assembly export fixes the part relative
  to the gripper, not the gripper relative to the robot -- and it has not been
  measured.
- Options considered: (a) refuse to author until the angle is measured /
  (b) bake a measured angle into the weld once it exists / (c) define the weld
  yaw as zero and carry the real mounting angle elsewhere.
- Decision: (c). `TOOL_WELD_YAW_RAD = 0.0`: the tool frame and the flange frame
  share their yaw. `author_tool_ur5e.py` keeps a `--tool-yaw-deg` argument so a
  measured angle can be baked in later without re-deriving anything.
- Rationale: A constant rotation about the tool axis between flange and tool is
  indistinguishable from a constant offset in `wrist_3`, because `wrist_3`
  rotates about that same axis. The angle therefore does not have to be known
  to build a correct asset; it has to be known to place the part correctly in
  the cell, and it can be corrected in exactly one number -- the `wrist_3`
  home angle -- where it is also observable. Choosing the zero point of an
  angle is a convention; inventing its value would not be, and CLAUDE.md
  forbids the latter. Option (a) would block the whole scene on a measurement
  that changes one joint angle. The cost is that the home pose's `wrist_3`
  value is provisional until the cell is measured; it is recorded as such.
- Sources: `HANDOFF-SZENE.md` (open item: "Gripper rotation about the tool axis
  on the real flange is a statement about the CELL and is in no CAD file");
  D-038 (`wrist_3` already carries reset yaw noise about the same axis); user
  decision 2026-08-23.

## D-079: Mass and centre of mass are authored; the inertia tensor is left to PhysX
- Date: 2026-08-23
- Status: accepted. The two masses ARRIVED on 2026-08-23: part 0.539 kg,
  gripper 0.285 kg, total 0.824 kg. VERIFIED on the training machine in RT-16
  (the asset was actually written, 20/20 PASS) and re-measured in RT-22:
  `physics:mass` reads 0.82400000 kg and `physics:centerOfMass`
  (-0.000893, -0.000405, +0.103200) m, matching the hand-written expectation to
  seven decimals; `physics:diagonalInertia` remains unauthored.
- Context: `UsdPhysics.MassAPI` on the welded link may carry mass, centre of
  mass, and a diagonal inertia tensor. The proxy wrote all three, because a box
  peg of known density has all three in closed form. The real tool has neither
  a known density nor a complete model: the suction cups are absent from the
  CAD, so its volume is not its volume.
- Options considered: (a) write all three, deriving mass and inertia from CAD
  volume and an assumed density / (b) write nothing and let PhysX derive
  everything from the collision geometry and a default density / (c) write mass
  and centre of mass from measured quantities, leave the tensor to PhysX.
- Decision: (c). Mass is the sum of two weighings, supplied as required
  arguments with no default. The centre of mass is the mass-weighted
  combination of the two body centroids the assembly STEP states as geometric
  validation properties, read out on 2026-08-23 and recorded as
  `PART_CENTROID_FLANGE` and `GRIPPER_CENTROID_FLANGE`.
  `physics:diagonalInertia` is deliberately not authored.
- Rationale: (a) fails at its first step -- the density of the terminal-block
  part is unknown and the gripper's model is incomplete, so a volume-times-
  density mass would be an invention presented as a derivation. (b) discards
  two quantities that are known: the total mass, once weighed, and the centre
  of mass, which follows from the two centroids without any assumption beyond
  each body being of one material. What remains unknown is the tensor, and
  leaving an attribute unauthored is how USD expresses that: PhysX then derives
  it from the collision geometry and scales it to the authored mass. The result
  is an approximation, because the only collider is the part (D-077), so the
  gripper's mass is distributed over the part's shape. That approximation is
  named in `TOOL_CHAIN_PENDING` as `TOOL_DIAGONAL_INERTIA` and asserted by a
  check (`diagonal_inertia_left_to_physx`), so it cannot quietly stop being
  true. HOW THAT CHECK READS IT, corrected 2026-08-23 after run RT-13 failed
  it: it asserts `HasAuthoredValue() == False`, not `Get() is None`. USD hands
  back the SCHEMA FALLBACK for an unauthored attribute, and UsdPhysics gives
  `physics:diagonalInertia` a fallback of (0,0,0) -- which is itself the
  sentinel that asks PhysX to compute the tensor. The first form could
  therefore never pass, however correct the asset. Measured in run RT-15:
  HasAuthoredValue=False with resolve source Usd.ResolveInfoSourceFallback,
  and `physics:mass` / `physics:centerOfMass` reporting True as a control.
  A separate cross-check confirmed the centroids rather than trusting
  them: pushing each body's own-frame centroid through its D-074 weld transform
  reproduces the flange-frame centroid the file states, on all three axes --
  126.746951 + 1.774164 = 128.521115 mm for the part.
- Sources: `CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp`, entities #41921,
  #47542 (per-body centroids in the flange frame) and #47583 (the assembly
  centroid, which is volume-weighted and therefore NOT usable for two
  materials); D-074 (the weld transforms the cross-check runs through); D-077
  (why the part is the only collider).
  The two masses, 2026-08-23: the user put both bodies on a SCALE and read
  539 g for the part and 285 g for the gripper. Provenance was asked for
  explicitly and is a weighing in both cases -- not a datasheet value and not
  a Creo mass property, so D-057's CAD-over-caliper ranking does not apply
  here: it ranks GEOMETRY sources, and no CAD mass exists for either body.
  The gripper was weighed WITH the suction cups mounted (user confirmed), which
  is the whole reason the number has to be weighed at all: the cups are absent
  from the CAD. Total 0.824 kg is carried in `insertion_env_cfg.tool_mass_kg`
  and must match the sum of the two `author_tool_ur5e.py` arguments.
  Resulting centre of mass in the flange frame, from `tool_com_flange`:
  (-0.0008928, -0.0004047, +0.1032005) m -- computed on the laptop against a
  hand-written expectation and matching it to seven decimals.

## D-080: A named check states its claim type, its purchase, and the one damage it must break
- Date: 2026-08-23
- Status: accepted. Verified on the training machine in RT-21 (21 checks, 18 of
  19 mutations exact) and RT-22 (19 of 19 exact); 24 self-test cases run
  offline on the laptop via `scripts/selftest_checks.py`.
- Context: `author_tool_ur5e.py` carried 20 checks and reported 20/20 PASS. Two
  of them were themselves wrong, found by measurement rather than by reading:
  `diagonal_inertia_left_to_physx` asserted `Get() is None` where USD returns
  the schema fallback, so it could not pass for ANY asset; `self_contained`
  counted the shipped UR5e's own empty composition arc. A third defect was
  worse in kind: `tool_spans_flange_to_part_bottom` could not see a `tool_link`
  rotated 180 degrees, because `ComputeRelativeBound(tool, tool)` does not
  include the link's own xform ops (measured, RT-17). The tool would have hung
  the wrong way in the world and all twenty checks would have said PASS.
- Options considered: (a) leave the checks as free-standing assertions and fix
  the individual defects as they surface / (b) require every check to declare
  what kind of claim it makes and which single damage must break it, enforced
  by a counter-proof run that mutates the asset once per check.
- Decision: (b). A check is registered with its claim type, the measurement run
  that paid for any interface claim it makes, and the mutation set it must
  flip. `scripts/tools/checks.py` is the workshop; the per-predicate USD logic
  moved to `scripts/tools/usd_predicates.py`, one home per predicate.
- Rationale: a check is only a check if it can DISCRIMINATE. It needs a world
  in which it passes and a world in which it fails. If the first is missing it
  blocks the run and someone notices. If the second is missing it tests
  nothing -- and THAT nobody notices. The counter-proof makes the second world
  mandatory and cheap: every mutation set was derived by reading `describe()`
  and `verify()` BEFORE any mutation ran, so the run tests a prediction rather
  than recording a result.
- Four claim types: `arithmetic` (recomputable by hand), `api-shape` (what a
  foreign interface returns in the absence of a value -- not readable, only
  measurable), `invariance` (something must change), `preservation` (something
  must not change). The fourth type emerged while APPLYING the rule, not at the
  drawing board: it fired on `articulation_roots_unchanged`, which must hold
  before AND after. The vocabulary was sharpened instead of the rule weakened.
- Known limit, named not repaired: `PART_FLIPPED_IN_PLACE` in
  `TOOL_CHAIN_PENDING`. A part mounted 180 degrees about its own centre
  measures correct on all three axes while carrying its lugs and its insertion
  chamfer on the wrong side, and not one of the 21 checks sees it, because
  every geometric one is a bounding box. Catching it needs a FEATURE -- the lug
  side, or the mesh centroid against the +1.774 mm the STEP states.
- Coverage today: ONE script. Seven further asset scripts still carry unproved
  checks; `convert_step_asset.py` contains the `Get() is None` defect a third
  time and has never run.
- Sources: runs RT-17, RT-21, RT-22 (`rt_logs/VERDICTS.md`);
  `scripts/tools/checks.py`, `scripts/tools/usd_predicates.py`,
  `scripts/selftest_checks.py`, `scripts/diagnose_tool_candidate.py`;
  D-079 (the `HasAuthoredValue` correction that started this).

## D-081: A script that runs checks must exit with a non-zero code when one fails
- Date: 2026-08-23
- Status: accepted. The cause was measured in RT-22; the fix is
  `scripts/tools/isaac_exit.py`, commit 10fa18f.
- Context: RT-17's counter-proof printed `counter-proof over 18 mutations:
  FAIL` and the process exited 0, although `main()` returns 1 and `__main__`
  raises `SystemExit`. `scripts/rt_log.ps1` writes the return code into the
  log and `/rt-check` reads it, so a failed run reporting 0 makes the
  machine-readable half of the two-machine loop lie.
- Options considered: (a) accept it and judge every run by its text / (b) find
  the cause and make the exit code true.
- Decision: (b), and the cause was isolated rather than guessed. Two suspects
  stood: the wrapper misreading the code after a piped native command, or
  Isaac's shutdown ending the process first. RT-20 ruled the wrapper out in one
  second (`python -c "import sys; sys.exit(3)"` -> `exit code: 3`). RT-22
  answered the rest: `simulation_app.close()` DOES NOT RETURN. Everything after
  it is dead code, `SystemExit` included.
- Consequence: the exit must happen BEFORE the shutdown. On failure the process
  leaves without an orderly shutdown. That is a deliberate trade: the process
  is ending anyway, and the alternative is a failed run that reports success.
- Ruling one suspect out does not prove the other -- the wrapper was ruled out
  by RT-20, and the shutdown was then MEASURED in RT-22, not inferred.
- Still open, same shape and NOT to be quoted as the same cause:
  `fix_stage_units.py` reported `"pass": true`, exited 0, and printed neither
  of the two lines it must print after a passing verdict, while the original
  was not replaced. The shape matches; the LOCATION does not (there the lines
  are missing INSIDE the main function). The link is NOT established and is
  explicitly not claimed here. It has an untested hypothesis and a
  discriminating run, both recorded in `HANDOFF-SZENE.md`.
- Sources: runs RT-17, RT-20, RT-22 (`rt_logs/VERDICTS.md`);
  `scripts/tools/isaac_exit.py`; `scripts/rt_log.ps1`; PROBLEMS.md 2026-08-21
  (why the wrapper had to be suspected at all).

## D-082: The unit repair is delivered as the candidate file; the in-place replace is bypassed

- Date: 2026-08-24
- Status: accepted (user, 2026-08-24). The DEFECT stays open. What changed is
  its rank: it is no longer on the critical path.
- Context: `fix_stage_units.py` does not repair in place. It writes the metre
  file as `<name>.fixed.usd` (`scripts/fix_stage_units.py:264`), re-opens THAT
  file on a fresh stage (`:268`), measures it and runs the eight contract
  checks against it (`:273`). Only the last step -- `shutil.copy2(candidate,
  usd_path)` (`:338`) -- fails, with `OSError: [Errno 22] Invalid argument` on
  opening the destination. RT-23 measured the candidate: `metersPerUnit 1.0`,
  extents 0.09640874 / 0.14350000 / 0.15200399 m, inside 1e-4 m of the CAD
  nominals. The repaired file is therefore COMPLETE AND VERIFIED on every run;
  only its name is wrong.
- Options considered: (a) run RT-26 and keep chasing the cause before the
  fixture goes into the scene / (b) take the candidate file, rename it by hand,
  and re-certify it with an independent measurement.
- Decision: (b). RT-26 is NOT run.
- Consequence: one manual rename per repaired asset, and `verify_fixture_usd.py`
  on the RENAMED file is what certifies it -- not the exit code of
  `fix_stage_units.py`. Nothing downstream may read that exit code as a verdict
  on the asset.
- What is explicitly NOT claimed: the cause is still unknown. The memory-map
  hypothesis is untested, the discriminating step sits unused in the code, and
  RT-26 is still the run that would answer it. Skipping it is a priority call,
  not a verdict.
- NOT RECORDED, and it should have been: how `greifer_bauteil_asm.usd` came to
  equal its own `.fixed.usd`. HANDOFF-SZENE.md states the equality as fact and
  RT-12 verified the RESULT (`metersPerUnit 1.0`, asset contract PASS), but the
  step that replaced the file appears in no log. This route therefore has one
  undocumented precedent, not a proven procedure.
- Sources: `scripts/fix_stage_units.py:264,268,273,338`; RT-23 and RT-25
  (`rt_logs/VERDICTS.md`); RT-12 for the tool asset's verified end state;
  `scripts/verify_fixture_usd.py:280` for the metre contract.

## D-083: The fixture STEP is repaired by one line, not re-exported

- Date: 2026-08-24
- Status: accepted
- Context: Four imports of `CAD/Aufnahme_real_v1.stp` came out 25.4x too
  large, in two different shapes. RT-27 produced `metersPerUnit 0.0254`
  with the raw CAD numbers 150 / 189.1 / 61; RT-30 produced
  `metersPerUnit 0.001` with the numbers scaled to 3810 / 4803.14 /
  1549.4. Both are the same defect seen from two ends.
- MEASURED, on the laptop, not inferred: the file declares INCH as its
  global length unit. `CAD/Aufnahme_real_v1.stp:2682` defines
  `#3129=(CONVERSION_BASED_UNIT('INCH',#3128)...)` with `#3128 =
  LENGTH_MEASURE(2.54E1)`, and line 3366 assigns exactly `#3129` in the
  single `GLOBAL_UNIT_ASSIGNED_CONTEXT`. But all 1529 coordinate values
  are millimetre numbers: median 45.29382, which is the documented
  +-45.294 mm pocket wall, maximum 100.45. In inches that solid would be
  2.55 m across.
- The counter-example is in the repo: `Step/Greifer_Bauteil/
  greifer_bauteil_asm.stp` contains no `INCH` entity at all, and its
  import (RT-8 ... RT-12) was correct. So the importer is right and the
  export is mislabelled.
- Options considered: (a) re-export from Creo with the unit set to
  millimetres / (b) repoint the file's global unit assignment to the
  millimetre unit Creo already wrote into it / (c) scale by 1/25.4 in the
  USD after import.
- Decision: (b), as a stopgap. `CAD/Aufnahme_real_v1_mm.stp` is a copy in
  which line 3366 references `#3127` -- `(LENGTH_UNIT()NAMED_UNIT(*)
  SI_UNIT(.MILLI.,.METRE.))`, already present in the file at line 3362.
  Exactly one line differs; all 1529 coordinates are byte-identical. The
  original is untouched.
- Rationale: (a) is the correct repair and stays the goal, but the CAD
  machine was not reachable. (c) hides the defect inside the asset, where
  the next person to import the STEP would meet it again with no warning.
  (b) fixes the mislabel where it lives, is one token wide, and is
  reversible.
- VERIFIED, RT-31 and RT-36: the import from the patched file measures
  150.000 x 189.100 x 61.000 mm with a largest deviation of 7.5e-06 mm,
  z -0.046000 ... +0.015000 m, `D-016 invariants: PASS`, `asset contract:
  PASS`. The tessellation error is five orders of magnitude below the
  0.2 mm short-axis clearance of D-057 [CORRECTION 2026-08-25:
  0.5876 mm, D-121 (0.2876 until 2026-09-11); conclusion unchanged].
- NOT repaired, and named rather than assumed away: the area and volume
  units (`#3162`, `#3171`, `#3184`, `#3193`) still carry the inch label.
  They feed the `geometric validation property` entries, not the
  geometry. Anyone quoting those numbers must re-check them.
- CONFIRMED as a side effect: the derived pocket constants (walls
  +-72.55 / +-45.294, `STAGE1_DEPTH`, `INSERTION_PLANE_Z`) were read as
  millimetres and millimetres is what they are. They need no change.
- Sources: `CAD/Aufnahme_real_v1.stp:2682,3362,3366`; the coordinate
  measurement over all 1529 `CARTESIAN_POINT` values; runs RT-27, RT-30,
  RT-31 and RT-36 in `rt_logs/VERDICTS.md`; `CAD/README.md` section
  "Einheiten-Defekt".

## D-084: The fixture's place in the block is read from four measured edge distances, not from the block centre

- Date: 2026-08-24
- Status: accepted
- Context: since the scene was built, the fixture spawned at
  `(BLOCK_CENTRE_X, BLOCK_CENTRE_Y)`. That was never a measurement. The
  code said so itself: the comment on `BLOCK_CENTRE_Y` marks the lateral
  placement as an assumption and notes that the pocket is not modelled.
  Harmless while nothing touched the pocket; the moment the CAD cut-out
  enters the scene it IS the insertion target.
- MEASURED, not inferred. Three sources, ranked: (1) the STEP governs
  every dimension of the fixture itself (user, 2026-08-24: "das CAD
  gilt"); (2) `docs/Geometrie/CAD_Massblatt (1).pdf` row B12 supplies
  what the STEP cannot contain -- the edge distances inside the block,
  "links 80 / rechts 10 / oben 50 / unten 58"; (3) the user re-measured
  the 80 mm and the 10 mm wall on the real block the same day, with
  photographs.
- The STEP was re-read on the laptop over all 532 `CARTESIAN_POINT`
  entities, all of them three-dimensional. It carries TWO openings, and
  only the lower one was in the constants before: stage 1 spans
  -74.7938 ... +55.2938 (130.0876 mm) by -72.55 ... +80.45 (153.0 mm);
  stage 2 spans +-45.2938 by +-72.55. The 47 x 12 notch reaches 6.1 mm
  past the stage-1 outline at both ends of the long axis.
- Every B12 value was re-derived from the STEP before being assigned to
  a side, so no side was matched by plausibility: left = the 29.5 mm
  step (-45.2938 against -74.7938), right = the 10.0 mm rib zone
  (55.2938 against 45.2938), top = the 7.9 mm step (80.45 against
  72.55), bottom = no step at all (-72.55 against -72.55).
- DIRECTION ANCHOR, and it needs no new assumption: the 7.9 mm step
  exists at exactly one end, +y. The Massblatt calls it "am oberen
  Ende", and the user fixed that end towards +Y on 2026-08-19. So
  B12 "oben 50" is the +Y side.
- Decision: the fixture spawns at `POCKET_ORIGIN_X` = 0.500994 m and
  `POCKET_ORIGIN_Y` = -0.13345 m, both derived in
  `insertion_tasks_cfg.py` from the block's anchored edges.
  `WORKCELL_BLOCK_POS` uses them. `BLOCK_CENTRE_X/_Y` keep their meaning
  -- the centre of the block model -- and stop being the fixture pose.
- Decision: the +Y reading (50 mm) LEADS the lateral chain and the -Y
  reading (58 mm) is control only, mirroring how `T2_OVERHANG_RIGHT`
  leads the table chain. The 1.0 mm that does not close is kept as
  `WORKCELL_POCKET_Y_RESIDUAL`, not distributed.
- ~~Decision: the block is modelled 220 mm deep instead of 210, and gets
  no +X wall box. Chain: 80 + 130.0876 + 9.9124 = 220.000 mm. This is
  the Massblatt's own W2 ("Differenz 10"), CLOSED by the user on
  2026-08-24 with a photograph: a 10 mm wall really stands there, and
  the fixture's own rear wall is that wall.~~
  **SUPERSEDED the same evening by D-085**: the user re-measured the
  rear wall at 15 mm on the real block. The model is 225.0876 mm deep
  (80 + 130.0876 + 15) and DOES get a +X wall box, because the fixture's
  own 9.9124 mm wall is no longer the block's wall.
- ~~Decision: the high wall behind the pocket is modelled as a fifth box
  (10 x 260 x 100 mm on top of the rear wall)~~ — **the footprint is
  15 x 260 x 100 mm since D-085**, standing on the full rear wall. It
  GETS COLLISION (user, 2026-08-24), unchanged: it is what makes the
  part enterable only from the front; without collision the policy could
  learn an approach that reality forbids. Expected collider count in the
  block goes 1 -> 5 -> **8** (D-085).
- Consequence, and it is the expensive one: the insertion target moves
  +49.794 mm away from the robot. `scripts/tools/screen_home_pose_
  branches.py` still aims at `BLOCK_CENTRE_X/_Y` and must be re-run
  before any training. Straight-line distance to the new pose is 524 mm
  of the UR5e's 850 mm published reach.
- NOT decided here, named instead: the built CAD has a 145.1 mm cavity
  where the Massblatt computed 143.8 with 0.5 mm of play. Against the
  part that is 1.60 to 1.76 mm lengthways, roughly three times the
  intended figure, and it disagrees with the ~0.8 mm measured on site.
  The CAD governs the simulation; the conflict needs a measurement on
  the part, not a decision.
- NOT measured, named as a modelling choice: the rear wall's height
  (user's "so 10cm hoch", rounded) and its extent in y (set to the block
  width). Also the rest of the block -- handle slots, bores and steps
  visible in the photographs are not modelled. The part touches only the
  cut-out.
- Verified on the laptop, offline: box sums 70.0 + 150.0 = 220.0 mm and
  40.9 + 189.1 + 30.0 = 260.0 mm, no box with a negative edge, and
  `python scripts/check_workcell_geometry.py` reports `VERDICT: PASS`
  with exit code 0. UNVERIFIED in simulation -- no run has been made.
  (The 220.0 became 225.0876 with D-085; the lateral sum is unchanged.)
- CORRECTED 2026-08-24 (evening), by re-measurement on the real block:
  the stage-1 length is 153.0 mm. The Massblatt's computed 151.7 is
  wrong, so W1 is CLOSED in favour of the CAD, and the -1.0 mm of
  `WORKCELL_POCKET_Y_RESIDUAL` is NOT W1. It sits in one of the three
  remaining numbers -- 50, 58 or the block's 260 -- and which one is
  unknown. Because the +Y reading leads, the millimetre lands entirely
  on the -Y wall (57.0 mm modelled where B12 says 58). Nothing in the
  task touches that wall, and the residual is not split.
- FOLLOWS from that correction, and it makes the play conflict worse
  rather than better: 153.0 - 7.9 = 145.1, which is exactly the cavity
  length the built CAD has. The CAD is internally consistent, so the
  1.60 to 1.76 mm of lengthways play is real and the Massblatt's 0.5 mm
  is the number that cannot stand. Still open, still needs a
  measurement on the part.
- Sources: `CAD/Aufnahme_real_v1.stp` (532 points, read 2026-08-24);
  `docs/Geometrie/CAD_Massblatt (1).pdf` rows B12, W1, W2 and the
  measured values 2, 5, 8; `docs/Blockberichte/Block-S1_Bericht.pdf`
  holds the full derivation with photographs.

## D-085: The first render is a measurement — two open slots and a 15 mm rear wall

- Date: 2026-08-24 (evening)
- Status: accepted
- Context: the scene ran with the CAD cut-out for the first time and the
  user LOOKED at it. Two defects came out of looking that no offline
  check had caught, and both were invisible to every number in D-084.
  This entry exists to record that the render is evidence, not a demo.

### Finding 1 — the recess is a bounding box, the part is not

- The user's screenshot circles two open slots along one edge.
- MEASURED, not inferred: all 530 `CARTESIAN_POINT` entities of
  `CAD/Aufnahme_real_v1_mm.stp` were re-read for the outer contour, at
  `z = -46` and at `z = +15` separately, so the contour is known to be
  constant over the full height. The body runs to `y = +90.45`
  everywhere EXCEPT under a TAB spanning `x -32.7938 ... +34.2062`,
  which reaches the `+100.45` that sets the bounding box.
- So `POCKET_LOCAL_Y_RANGE[1]` describes the TAB, not the body. A recess
  cut to the bounding box leaves 10 mm of air beside it: 52.0 mm long at
  -x, 31.0 mm long at +x. Those are exactly the two circled slots.
- Decision: two filler boxes, `block_spalt_y_plus_minus_x` and
  `block_spalt_y_plus_plus_x`, spanning body edge to bbox edge over the
  cut-out's own height. NOT a general contour follower: the rest of the
  outline IS the full rectangle, checked, so two boxes are the whole fix
  and a sixth-order solution would be invention.
- Closure that guards it: slot + tab + slot must equal the recess width
  (52.0 + 67.0 + 31.0 = 150.0 mm), checked offline.

### Finding 2 — the rear wall is 15 mm, not 10

- The user re-measured the wall on the real block and reported that the
  back face rendered in two tones, which is what a flush block-and-part
  boundary looks like.
- Decision: `BLOCK_REAR_WALL_T = 0.015`, measured from the stage-1
  outline to the outside — what a calliper on the real block reads.
  This SUPERSEDES the Massblatt's W2 ("Differenz 10") and the earlier
  10 mm reading from the same day. A direct re-measurement outranks a
  drawing, the same precedent as the 80 mm in D-084.
- Consequence: the fixture's OWN rear wall is 9.9124 mm (STEP), so it is
  no longer the block's rear wall. `BLOCK_BEHIND_POCKET_X` = 5.0876 mm
  of block stands behind it, as the new `block_wand_x_plus` box, and the
  high wall's footprint grows from 9.9124 to the full 15 mm.
- Consequence: `BLOCK_MODEL_SIZE_X` is now read as the CHAIN
  (80 + 130.0876 + 15 = 225.0876) rather than as `BLOCK_SIZE_X +
  BLOCK_REAR_WALL_T`. The two agreed only while the wall was the derived
  9.9124. The depth chain therefore becomes over-determined and its
  residual (80 + 130.0876 against the 210 mm tape reading, +0.0876 mm)
  is reported as a number, not absorbed.
- NOT changed: the spawn pose. `POCKET_ORIGIN_X/_Y` are anchored at the
  block's NEAR face, so a deeper model does not move the target and the
  home pose does not need a sixth recompute (screener re-run, branch 1
  again at +0.1865 m).

### Consequences for the checks

- Collider count in the block: 5 -> 8. Asset name bumped to
  `aufnahme_block_v4.usd`, because a `_v3` file still has the holes in
  it and a hole loads without complaint.
- Five new named checks, and a counter-proof was run offline BEFORE they
  were trusted: deleting each of the three new boxes flips exactly the
  checks that name it.
- ONE HOLE THAT COUNTER-PROOF FOUND, repaired in the same session:
  `block_stands_proud_of_the_cutout` first read the union bounding box.
  Deleting `block_wand_x_plus` did NOT flip it, because the high rear
  wall reaches the same far face. It now reads the wall box directly.
  Same shape as the RT-17 finding: a bound cannot see a box missing
  inside it.
- NUMBERING, RESOLVED 2026-08-27 (writer session on main): `p1-gains` is
  NOT merged into main (user decision 2026-08-27). Its D-082 (gravity)
  and D-083 (gains) carry `SUPERSEDED 2026-08-25` on the branch and stay
  there. On main, D-082 and D-083 belong unambiguously to the scene line
  (unit repair, fixture STEP repair). No number is used twice.

## D-086: The tool is turned 180 degrees at the home pose, and that exposes a C4 reward built for the proxy

- Date: 2026-08-24 (evening)
- Status: accepted for the angle; the reward mismatch is NAMED, not decided
- Context: with the fixture placed, the user looked at the render again and
  circled two things: the lugs on the part, and the cut-outs in the pocket
  they have to enter. They pointed opposite ways. The part fits one way
  round, so this is a defect, not a viewing angle.
- Decision: `TOOL_HOME_YAW_RAD = math.pi` in `insertion_tasks_cfg.py`, next
  to `TOOL_WELD_YAW_RAD`. This is exactly the "ONE number" D-078 said the
  real mounting angle would live in.
- Decision: it is applied in the SCREENER's target orientation, not typed
  into `UR5E_HOME_JOINT_POS`. A hand-edited joint angle would be undone by
  the next screener run, and the person doing it would be right to.
- WHY POST-MULTIPLIED: `R = R_down @ Rz(yaw)` rotates in the TOOL frame,
  which is the frame the angle is quoted in. Pre-multiplying spins about
  world z. For a tool pointing exactly down the two look identical, and they
  stop being identical the moment the approach is tilted (D-036) -- so the
  form that stays correct was chosen over the form that happens to work.
- SELF-CHECK, and it is what makes the number believable: a pure spin about
  the tool axis must move `wrist_3` and nothing else, because `wrist_3`
  rotates about that same axis. The re-run moved `wrist_3` from -1.571096 to
  +1.570497 -- a difference of 3.141593 rad -- and left the other five
  bit-identical. Clearance unchanged, branch 1 still selected.
- The tool USD is NOT re-authored: the weld yaw stays zero (D-078). One joint
  angle changed, no asset did.
- STILL A CONVENTION: the gripper's real mounting angle on the flange is
  still unmeasured. What is known now is that the previous value was 180
  degrees wrong, read off the geometry the part has to enter. That is better
  evidence than the zero it replaces and worse than a measurement.

### The mismatch this uncovered, named and NOT fixed here

- The reward's yaw term charges `(1 - cos 4 phi) / 2` and the observation
  carries `(cos 4 phi, sin 4 phi)` (D-029). Both are C4-symmetric: ZERO
  penalty at 0, 90, 180 and 270 degrees. That was correct for the square
  proxy peg, which really does insert at any of four orientations.
- The real part is rectangular AND carries lugs. Exactly ONE of those four
  is insertable. `reward_w_yaw` is 2.0, i.e. active. So the term currently
  rewards three orientations that cannot be inserted, and it cannot teach
  the orientation this entry just corrected.
- NOT decided here on purpose: the approved scene-build scope keeps
  obs/action/reward on proxy state until the scene is done, and changing the
  yaw channels changes the observation dimension. It is written into the
  startup report as a `*** ... ***` line so no training run can start
  without it being on screen.
- The related unseen case stands: `PART_FLIPPED_IN_PLACE` (RT-21) -- a part
  mounted 180 degrees about its own centre measures correct on all three
  axes while carrying its lugs on the wrong side, and no bound-based check
  in `author_tool_ur5e.py` can see it. Today's finding is the same defect
  seen from the cell instead of from the asset.

## D-087: Real per-side clearance recorded; the simulation keeps the CAD clearance

- Date: 2026-08-25
- Status: accepted (record of a hand measurement; no simulation number changes)
- Context: While reviewing the supervisor status report, the user stated the
  real clearance between part and fixture per side, superseding the totals-only
  figures in D-057 (~0.5 mm long axis / ~0.2 mm short axis). The simulated
  fixture is a simplification of the real rig, so the simulation's clearance is
  whatever the CAD model produces, not the real rig's.
- Options considered: adopt the real per-side values into the simulation
  geometry / keep the CAD-derived geometry and record the real values as
  reference.
- Decision: The simulation keeps the CAD-derived clearance (short axis
  0.2876 mm from `POCKET_OPENING_X` [CORRECTION 2026-09-11: 0.5876 mm, D-121 -- `PLAY_X` = `POCKET_OPENING_X` - `PART_BODY_X`]; long axis per the named conflict at
  `POCKET_OPENING_Y` in `insertion_tasks_cfg.py`). The real rig's clearance is
  recorded here as the single home of that fact:
  - Long axis: **0.8 mm** on the side with the 7.9 mm ledge, **0.1 mm** on the
    opposite side (total 0.9 mm).
  - Short axis: **0.3 mm** on the side nearer the robot, **0.1 mm** on the
    opposite side (total 0.4 mm).
  - Measurement: by the user on the real hardware, reported 2026-08-25;
    instrument not recorded.
- Rationale: The CAD governs the simulation (user, 2026-08-24, restated
  2026-08-25). Mixing a hand-measured rig clearance into CAD-derived pocket
  constants would create a geometry no single source describes. The real values
  are still thesis-relevant (sim-vs-real gap discussion) and must stay
  findable. Note they do not settle the D-075/Massblatt part-length conflict:
  0.9 mm total real play still disagrees with the model's 1.60 mm, and the
  asymmetry (0.8 on the ledge side) is a property of the real rig the
  simplified fixture does not reproduce.
- Sources: user hand measurement 2026-08-25 (this conversation); D-057
  (superseded clearance totals); D-075 (part CAD length 143.50 mm);
  `insertion_tasks_cfg.py` comment at `POCKET_OPENING_Y` (the length conflict).

## D-088: Part length settled at 143.50 mm; the long-axis play conflict closes at 1.60 mm

- Date: 2026-08-25
- Status: accepted
- Context: Three part lengths were on record (143.34 mm CAD_Massblatt 18-08 /
  143.50 mm part CAD, D-075 / 144 mm `docs/Geometrie/
  Massblatt_Fuegeteil_Aufnahme.tex`, marked "final 21.08.2026"), giving
  1.76 / 1.60 / 1.10 mm play against the 145.1 mm stage-2 opening. The
  conflict was carried as a named comment beside `POCKET_OPENING_Y` and as an
  open item in `HANDOFF-SZENE.md`, with the on-site re-measurement of the
  part named as the only way to close it.
- Options considered: 143.50 mm (part CAD) / 144 mm (Massblatt "final").
- Decision: 143.50 mm. The user confirmed the value on 2026-08-25
  ("143.5mm stimmt"); the measurement basis behind the confirmation was not
  stated. The Massblatt's 144 mm entry (its correction of 21.08.2026) is
  superseded; the CAD figure D-075 measured on the mesh stands. Consequence:
  the simulated long-axis play is **1.60 mm** (145.1 − 143.50) and is no
  longer in dispute. The real rig's play stays 0.9 mm total (D-087); the
  0.7 mm sim-vs-real difference is a property of the simplified fixture,
  recorded, not modelled away.
- Rationale: The CAD governs the simulation (user, 2026-08-24), and the one
  independent check available agrees with it: D-075 measured the part mesh at
  143.50 mm with tessellation error shown to be 8 µm on the same part. The
  Massblatt's 144 was a datasheet nominal adopted as "real"; the user's
  confirmation removes it as a competing source.
- Sources: user confirmation 2026-08-25 (this conversation); D-075 (mesh
  measurement RT-8); D-087 (real per-side play);
  `docs/Geometrie/Massblatt_Fuegeteil_Aufnahme.tex` (superseded 144 mm).

## D-089: No force/torque channel in the observation; the policy trains on kinematics alone

- **Status: SUPERSEDED 2026-08-27 (Block-7 grill, concept stream).**
  The force channel IS added: observation 25 -> 28, via this entry's
  own fixed reopening route (FORGE wrench pattern,
  `body_incoming_joint_wrench_b`, EMA 0.25, 3 force components only).
  The force-abort stays and now reads an observed quantity; a FORGE
  penalty `-beta*max(0, ||F||-F_th)` enters the reward. Trigger was
  NOT this entry's reopening condition (no failed run, no jamming
  evidence) but an explicit user override, after the termination
  literature check found no work anywhere that aborts on an
  unobserved quantity. The penalty follows Beltran-Hernandez 2020:
  ONE threshold `F_max`, abort plus a fixed negative reward; the
  FORGE ramp was considered and dropped (it would need a second,
  lower threshold, for which no source exists). New home of the fact: inbox entry "D-089 is
  OVERTURNED" in `docs/decisions_inbox.md` (branch
  p1-konzept-messung) and `HANDOFF-SZENE.md` (commit 8ff7deb). The
  Factory Table IV ablation below (force costs ~35 % relative
  success) is UNRESOLVED counter-evidence, not refuted — it must be
  reported in ch. 4 and discussed in ch. 6.2. Everything below is
  kept for the record.
- Date: 2026-08-25
- Status: accepted (closes concept node N3.2) — see SUPERSEDED above
- Context: The user raised the concern that the real task's small clearance
  (short axis 0.2876 mm total in the simulation, D-087 [CORRECTION 2026-09-11: 0.5876 mm, D-121]) needs a wrist
  force/torque signal in the observation space to be learnable. N3.2 in
  `KONZEPTPHASE.md` had carried this question open since Block 3 was written,
  and it inherits the old repo's D-005 deviation: an F/T channel was specified
  there and never implemented.
- Options considered: (a) add an F/T channel before the first training run /
  (b) train kinematically first and add the channel only on evidence.
- Decision: (b). The observation stays the 25-dim kinematic vector documented
  at `insertion_env_cfg.py` L120-144. No wrench, contact-force or torque
  quantity enters the observation. `activate_contact_sensors=True`
  (`ur5e_cfg.py` L178) stays as the spawn-time option, unused.
- Rationale: In the literature a force channel substitutes for an uncertain
  object pose. This project is simulation-only and reads the fixture pose from
  the simulator, so there is nothing for force to substitute for.
  - Factory (Narang et al. 2022, RSS, Table IV) is the direct counter-evidence
    to the concern: at 0.104 mm ISO clearance — tighter than this task's
    0.2876 mm short axis [0.5876 mm since D-121; still tighter] — the observation-space ablation gives pose 0.7708,
    pose+velocity 0.7760, pose+velocity+**force** 0.5026. Adding force cost
    about 35 % relative success. Sim-only, PPO, Isaac.
  - IndustReal (Tang et al. 2023, 0.5-0.6 mm) and AutoMate (Tang et al. 2024,
    0.5-1.0 mm) both exclude F/T deliberately and still transfer to hardware.
    AutoMate states it outright: "no force-torque sensor is used".
  - Where force does pay, the pose is deliberately noisy: FORGE improves nut
    threading 0.40 -> 0.69 but peg insertion only 0.82 -> 0.84; TacSL's
    no-tactile arm collapses only under a noisy socket pose, while its
    privileged-state (ground-truth pose, no tactile) arm reaches 0.973; Lee
    et al. 2019 gains 0.49 -> 0.77 while localising a randomised box.
  - This project's own proxy task solves a 1.0 mm-per-side square peg at
    99.15-99.65 % with a purely kinematic observation and reward.
  - D-041's rigid suction chain distorts contact forces. Feeding a distorted
    signal to the policy is worse than feeding none.
- Scope: this closes N3.2 only. D-051 stands unchanged — force is not a success
  criterion. N7.3 (force-abort limit) and N5.2 (force penalty in the reward)
  stay open and are decided in their own nodes.
- Reopening condition: N3.2 is reopened when a training run on the real
  geometry with the kinematic observation stays below the success threshold
  **and** the failure analysis attributes the failures to jamming at the pocket
  entrance — not to reward shaping, reach, or collision resolution. Absent that
  evidence, an F/T channel is not added.
- If reopened, the route is fixed: not `ContactSensor` (it reports normal
  contact forces only, `ContactSensorData` has no torque field, and creating
  one switches PhysX contact processing on globally at an overhead the docs
  assert but never quantify). Use the Forge pattern instead:
  `ArticulationData.body_incoming_joint_wrench_b` (full 6-D wrench, present in
  Isaac Lab 2.3.2), as `isaaclab_tasks/direct/forge/forge_env.py:95` does —
  joint wrench at a dedicated force-sensor link, EMA-smoothed (0.25),
  noise-injected, and only the 3 force components handed to the policy. The
  ready-made observation term is `mdp.observations.body_incoming_wrench`;
  there is no built-in term for `ContactSensor`.
- Sources: Narang et al. 2022 `arXiv:2205.03532` (Table IV);
  Tang et al. 2023 `arXiv:2305.17110`; Tang et al. 2024 `arXiv:2407.08028`;
  Noseworthy et al. 2024 `arXiv:2408.04587` (No-Force ablation);
  Akinola et al. 2024 `arXiv:2408.06506` (Table IV);
  Lee et al. 2019 `arXiv:1907.13098`;
  `docs/reference/literature_check_force_criterion_2026-08-16.md`;
  old repo D-005 (the deviation this closes); D-041; D-051; D-087.

## D-090: Thesis branch never merges into main (amends D-065)
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- **SCOPED 2026-08-28 by D-129:** the ban stands, but it had a hole —
  candidates written in the thesis stream had no route to a number, and
  twelve of them sat unnumbered for two days. D-129 opens exactly one
  exception: the writer session may fetch
  `docs/decisions_inbox.md` alone from `p1-thesis`
  (`git checkout p1-thesis -- docs/decisions_inbox.md`). No merge, no
  other file. Thesis text still never reaches `main`.
- Stream: szene (user decision during the p1-szene-umbau session)
- Context: D-065 has every stream merge into main at a phase cut.
  The code and konzept streams never need thesis/ content; merging
  it into main adds ceremony without a consumer.
- Options considered: (1) keep D-065 as is — all streams merge at
  the cut; (2) thesis stays on its own branch permanently.
- Decision: Option 2. The thesis branch keeps pulling main
  (geometry docs flow in) but never merges back. Save points for
  the thesis are tagged on the thesis branch itself. At a phase
  transition the new thesis branch starts from the old thesis
  branch, not from main.
- Rationale: one-directional flow matches the real dependencies;
  the other streams never pull thesis content. Cost accepted:
  main no longer holds the complete project state — thesis state
  lives only on its branch and its tags.
- Sources: user choice 2026-08-18 (session p1-szene-umbau).

## D-091: Core block updates: corrections and additions
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (user choices during the p1-szene-umbau session)
- Context: review of the core block against the repo state and the
  old root CLAUDE.md after the per-worktree rebuild.
- Options considered: leave the core as is vs. four targeted edits.
- Decision: four core edits for the writer session on main:
  1. Delete the stale sentences "`.gitattributes` sets `CLAUDE.md
     merge=ours` ..." and "Requires once per machine ...". Verified
     2026-08-18: no .gitattributes exists on main or in any
     worktree; no merge.ours.driver is configured; the
     untracked+gitignored scheme makes a merge driver unnecessary.
  2. Re-add the primary-reference rule lost in the rebuild: Isaac
     Lab docs https://isaac-sim.github.io/IsaacLab/ with the version
     picker set to 2.3, preferred over blog posts or model memory.
  3. Move the Numbers rule (geometry and physical values from CAD or
     measurement; `[CAD pending]` / `[measure on site]`
     placeholders; no structural decision on an assumed number) from
     the stream parts into the core — it binds all three streams and
     is currently duplicated per stream.
  4. Amend "Merge into main only at a phase cut" with the thesis
     exception (see entry "Thesis branch never merges into main").
  5. Add one bullet on folder-level CLAUDE.md files: they are
     tracked pure document indexes (thesis/, scripts/, module
     folders) — no rules in them; each needs its `!path/CLAUDE.md`
     exception in .gitignore; only the worktree-root CLAUDE.md is
     private. Without this, a forgotten exception silently keeps an
     index off the other machine, and other sessions may mistake a
     tracked index for a private file.
- Rationale: 1 is a factual correction; 2 restores a binding rule;
  3 removes duplication; 4 implements today's thesis decision in
  the core text; 5 documents the index scheme (see entry "Tracked
  folder-level thesis/CLAUDE.md" in the thesis inbox) for all
  streams. After the core changes land on main, each stream
  hand-copies the new core and deletes its own Numbers-rule copy.
- Sources: session p1-szene-umbau 2026-08-18; verified there via
  `git show main:.gitattributes` and `git config merge.ours.driver`.

## D-092: UR5e articulation config written out in full, not derived from UR10e
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene
- Context: Scene-build phase C needs a UR5e `ArticulationCfg`. The
  assumption going in was that Isaac Lab ships one, as it does for
  the UR10e the proxy task uses. Checked against the v2.3.0 tag of
  `isaaclab_assets/robots/universal_robots.py`: it defines UR10,
  UR10e and three UR10 gripper/suction variants — no UR5e. The USD
  asset itself does exist (Isaac Sim 5.1 robot-asset list,
  `UniversalRobots/ur5e/ur5e.usd`); only the config is missing.
- Options considered: (1) `UR10e_CFG.copy()` with the USD path and
  gains overridden, mirroring how `ur10e_cfg.py` derives from the
  shipped config; (2) write the `ArticulationCfg` out in full, using
  the upstream UR10e definition only as a structural template.
- Decision: Option 2, in a new `ur5e_cfg.py`. The base pose stays
  imported from `insertion_tasks_cfg.py` rather than restated, so
  the scene geometry keeps one owner. Home pose and actuator gains
  are committed as explicitly labelled placeholders, not as values.
- Rationale: `ur10e_cfg.py` uses `.copy()` so that no nested field
  of the *same* robot is silently reset to a class default. That
  argument does not carry to a different robot: the UR5e's upper arm
  and forearm are each about 200 mm shorter, so every inertia- and
  length-dependent value copied across would be wrong while looking
  deliberate. Writing the fields out makes each one reviewable.
  Consequence accepted: the file drifts if upstream changes the
  UR10e boilerplate, which is cheap to re-check.
- Sources:
  https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py;
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html;
  UR published DH parameters (universal-robots.com). All three read
  2026-08-18. UNVERIFIED in simulation — no Isaac on the dev laptop.

## D-093: Workcell frame rotated to +X-towards-fixture
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (user decision, grill session)
- Context: The user built the fixture CAD with Z up, X towards the
  back wall and Y towards the 7.9 mm-margin end. The repo frame had
  +Y = robot -> fixture, and its code comment justified that with
  "Isaac Lab Factory does the same". Checked against the local
  Isaac Lab source: FALSE. Factory spawns its fixed asset at
  (0.6, 0, 0.05), i.e. in +X (factory_tasks_cfg.py:313, PegInsert);
  the old proxy repo used -Y. Only "robot base at the origin" was
  ever shared.
- Options considered: (1) keep +Y-forward, fix the comment, rotate
  the user's CAD coordinate system; (2) rotate the cell frame to
  +X-forward.
- Decision: Option 2. New frame: +X = robot -> fixture/back wall,
  +Y = robot's left (right-handed, Z up), robot base_link stays the
  origin with identity rotation. Mapping x_new = y_old,
  y_new = -x_old; block centre moves (0.133, 0.4512, 0.0775) ->
  (0.4512, -0.133, 0.0775). WORKCELL_X_SIGN became
  WORKCELL_Y_SIGN = +1 (the tape measurements' mirrored viewpoint is
  now absorbed by the axis direction). The 7.9 mm-margin end of the
  pocket points to +Y — robot's left — per user statement 2026-08-19
  (M; consistent with the right-handedness of the user's CAD frame).
  USD asset names bumped to _v2 against stale assets on the training
  machine. Home pose recomputed: same physical pose rotated with the
  cell (shoulder_pan -1.5701 -> -3.1409, wrist_3 +0.0007 -> -1.5701),
  branch 1, full 150 mm standoff, FK validation 0.07 mm unchanged.
- Rationale: The user's CAD now spawns with identity rotation (its
  axes ARE the cell axes), the Factory reference becomes true instead
  of false, and the thesis can cite it. Cost: one-time re-derivation
  of the lateral sign and the home pose, both re-verified offline.
- Sources: IsaacLab-main factory_tasks_cfg.py:313 /
  factory_env_cfg.py:156 (read 2026-08-19);
  scripts/check_workcell_geometry.py section 3b (sign + regression
  tests, PASS on the dev laptop); screen_home_pose_branches.py rerun
  2026-08-19. UNVERIFIED in simulation until the training machine
  regenerates the _v2 USDs.

## D-094: Fixture CAD authority: user's own measured model replaces the D-059 recipe
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (user decision)
- Context: D-059/D-060 planned the fixture CAD as a part-negative
  built from a recipe (docs/Geometrie/CAD_Anleitung_Taschenblock.tex,
  DELETED 2026-08-24 because this entry is what replaced it; the
  measured model is at CAD/Aufnahme_real_v1.stp).
  The user instead measured the real fixture completely on site and
  built the CAD from those measurements, outside-in, with his own
  coordinate system (origin = centre of the stage-2 opening on the
  block top face; X -> back wall, Y -> 7.9 mm end, Z up — matches the
  rotated cell frame at identity spawn).
- Options considered: (1) enforce the recipe dimensions
  (pocket = part + measured clearance); (2) adopt the user's measured
  CAD as the authority.
- Decision: Option 2, provisionally. The CAD guide is obsolete; the
  user wants it deleted (main-writer action — this stream cannot
  delete docs). Stage-2 opening in the CAD: 145.1 x 90.59 mm.
  CONFLICT RESOLVED 2026-08-21 (user, final): the CAD opening stays
  145.1 x 90.5876 deliberately. Final measured values: part length
  144.0 mm [CORRECTION 2026-08-25: superseded -- part is 143.50 mm,
  D-088; per-side real play in D-087], play long 0.8 mm, play across
  0.3 mm (total per axis, not per side). Across-axis matches the CAD (90.5876 - 90.3 = 0.2876). [CORRECTION 2026-09-11: no longer. The CAD body is 90.00 mm (D-121, `insertion_tasks_cfg.py` `PART_BODY_X`), so the CAD play across is 0.5876 mm against the measured 0.3 mm.]
  Long axis does NOT: the CAD implies 1.1 mm play, 0.3 mm more than
  measured. The user accepts this knowingly. The CAD stays the single
  source for simulation geometry; the measured play numbers stay the
  source for reality-facing statements (tolerances, success bands).
  The 0.3 mm difference is recorded, not averaged away.
- Rationale: The CAD is built from the real fixture, not from the
  recipe, so the recipe no longer describes anything that exists.
  The conflict is recorded instead of resolved by plausibility
  (CLAUDE.md hard rule).
- Sources: user statements 2026-08-19 (grill session); D-057 measured
  play 0.5/0.2; Geometrie_Fuegeteil_Aufnahme.tex (part 143.34 x 90.3; width superseded by D-121).

## D-095: D-036 tilt probe keeps its code axis through the frame rotation
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (user decision after explanation)
- Context: The deterministic tilt probe rotates the fixture about
  env y (R_z(yaw) @ R_y(tilt)). After the frame rotation env y is the
  lateral axis, so the probe's physical meaning changed from rolling
  the cell laterally to pitching the back-wall side down. The
  randomised training tilt is unaffected — it samples its direction
  uniformly over 360 deg.
- Options considered: (1) rebuild the probe about env x to preserve
  the old physical meaning; (2) keep the code, re-document the
  meaning.
- Decision: Option 2. The D-036 100%/50% baseline was measured on the
  square proxy pocket and does not carry over to the real task
  anyway. Which physical axis (pocket long vs short axis) the real
  task's probe needs is a concept-block-2 decision and is deferred
  there.
- Rationale: smallest change; no instrument the current task uses is
  altered; the real decision is made where the real probes are
  designed.
- Sources: insertion_env_cfg.py fixture_tilt_rad docstring;
  insertion_env.py tilt sampling (360 deg azimuth); user answer
  2026-08-19.

## D-096: Fixture asset origin = stage-2 opening plane (amends D-060), reduced asset scope
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (user decision after explanation)
- Context: The user's exported CAD (docs/Geometrie/Aufnahme_real_v1.stp)
  has its origin on the STAGE-2 OPENING PLANE, not on the top face as
  D-060's wording says. In the proxy both planes were the same surface;
  the real fixture separates them by stage 1. STEP checks passed:
  mm units, walls at +-72.55 / +-45.294 from the origin, origin
  centred (user re-measured both sides), the +Y side is the longer
  (7.9 mm) end. First origin check failed because the user measured
  at the insertion CHAMFER -- the CAD therefore carries the real
  chamfer, which D-059 had listed as [offen, DEHN-CAD].
- Options considered: (1) move the CAD origin up to the top face
  (D-060 wording, zero code change); (2) keep the CAD, spawn the
  asset origin on the insertion plane.
- Decision: Option 2. Insertion depth then reads directly as -z of
  the part tip relative to the asset origin/entrance (user argument:
  depth visible in graphs without offset), and D-044 tilt/yaw rotate
  the pocket about its insertion plane. This continues the proxy
  semantics "asset origin = opening plane". Code: new measured
  STAGE1_DEPTH = 15 mm (from the CAD), INSERTION_PLANE_Z =
  BLOCK_TOP_Z - STAGE1_DEPTH = +62.5 mm env; WORKCELL_BLOCK_POS and
  the entrance moved there; the coarse box in author_workcell.py now
  pokes 15 mm above its origin; checker extended.
  GUIDE_EDGE_ABOVE_T2 = 197.3 clarified by the user: 11.28 mm wooden
  base plate + 186 mm up to the STAGE-1 TOP RIM (= fixture top face).
- Asset scope, decided by the user: the CAD models the pocket insert
  with ~1 cm of surrounding block only (outer 150 x 189.1 x 61 mm),
  because that part exists physically and suffices for the task.
  Consequence for phase D: the CAD alone does not cover the full
  260 x 210 block footprint; whether the coarse box stays around it
  is a phase-D detail decision. Pocket wall contour (user answer
  2026-08-19): straight walls with SIX cutouts (Auskerbungen) -- one
  at each of the four block corners plus two extra on the back side;
  no continuous wave contour. Modelled in the CAD as built.
- Rationale: the origin follows the quantity the task measures
  (insertion depth), not a surface; the proxy precedent already
  worked this way.
- Sources: STEP point-cloud check 2026-08-19 (532 cartesian points,
  bbox 150 x 189.1 x 61, walls at the stated offsets); user
  statements 2026-08-19; check_workcell_geometry.py 3b extended
  (PASS on the dev laptop).

## D-097: STEP export convention for the user-authored fixture CAD
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: The user modelled the fixture (Aufnahme) in Creo from his
  own complete measurement of the real part and its fixture. The
  export feeds the training PC, which converts STEP to USD with SDF
  mesh collision (`docs/Geometrie/CAD_Anleitung_Taschenblock.tex`,
  step 6).
- Options considered: AP203 / AP214 / AP242; datums exported or
  suppressed; export in mm or in m; file name
  `Taschenblock_real_v1.step` or `Aufnahme_real_v1.step`.
- Decision: (a) Format AP214. (b) Export in millimetres; the mm-to-m
  conversion stays on the training PC. (c) Datum entities ARE
  exported, together with the solid. (d) The export coordinate
  system is the user's named CSYS, never `PRT_CSYS_DEF`. (e) File
  goes to `docs/Geometrie/Aufnahme_real_v1.stp` and is committed;
  the native Creo model stays local and out of the repo. The
  extension is `.stp`, matching the existing `Bild/900360.stp`.
- Rationale: (a) The existing part CAD `Bild/900360.stp` is already
  AP214 — verified in its header:
  `FILE_SCHEMA (('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'))`.
  One format for both parts means one import path and one failure
  mode. The BREP geometry is identical across AP203/214/242, so the
  choice decides nothing physical. (b) Keeping mm keeps the CAD
  numbers readable against the measurement documents and the
  drawings. The mm-to-m trap is already known and documented in
  `scripts/fix_stage_units.py`: the Isaac Sim CAD import leaves the
  stage at `metersPerUnit = 0.001`, which Isaac Lab ignores, so the
  asset spawns 1000x too large. (c) User decision against the
  recommendation. `[PROPOSAL, unverified]` risk on record: Creo
  writes datum planes as bounded surfaces, and the pipeline builds
  SDF mesh collision from surfaces, so a datum plane could become a
  collision face that does not physically exist. Nobody has verified
  how the Isaac Sim CAD importer treats datum entities — no Isaac
  installation on the laptop. Mitigation agreed: after the import on
  the training PC, check the prim list with
  `scripts/list_usd_prims.py`. If anything beyond the one block mesh
  appears, re-export with datums suppressed. (e) The name avoids the
  word "Taschenblock", because the instruction document that coined
  it is superseded (see the entry below).
- Sources: header of `Bild/900360.stp` (read 2026-08-19);
  `scripts/fix_stage_units.py` lines 4-15;
  `docs/Geometrie/CAD_Anleitung_Taschenblock.tex` line 222; user
  decisions in the grill session 2026-08-19.

## D-098: Fixture CAD origin sits at the pocket opening centre, not at the block centre
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: The cell frame was rotated so that +X points from the
  robot to the fixture (Option B). The intent is an identity spawn:
  the user's CAD drops into the scene without any rotation. That
  only works if the CAD origin is defined and its offset to the
  spawn reference is known.
- Options considered: origin at the centre of the step-2 pocket
  opening on the block top face / origin at the geometric centre of
  the outer block 260 x 210 x 116.
- Decision: The origin sits at the centre of the step-2 opening, on
  the block top face. CAD axes: +X to the rear wall (Nasen side),
  +Y to the 7.9 mm end, +Z out of the block. This matches the
  rotated cell frame one to one, so the spawn needs no rotation.
- Rationale: The origin is then the insertion target point itself,
  so the goal state needs no offset arithmetic. Consequence that
  must not be lost: the block extends from Z = 0 down to Z = -116,
  so the origin is NOT the block centre. The pocket is also offset
  laterally. Derived from the measured margins in
  `CAD_Anleitung_Taschenblock.tex` lines 197-206: in depth, step 1
  is 130 wide, the opening 90.5, the margins 10 (robot side) and
  29.5 (rear-wall side); 90.5 + 10 + 29.5 = 130 checks out, so the
  pocket sits 9.75 mm off the step-1 centre. Along the long axis,
  step 1 is 151.7, the opening 143.8, margins 7.9 and flush;
  143.8 + 7.9 = 151.7 checks out, so the pocket sits 3.95 mm off
  centre.
- CORRECTION, and it corrects the scene stream too: Z = 0 is NOT
  the block top face. It is the rim plane of the step-2 opening,
  i.e. the floor of step 1, which sits 15.0 mm BELOW the top face.
  The scene-stream entry "Fixture CAD authority: user's own
  measured model replaces the D-059 recipe" (branch
  `p1-szene-umbau`, commit 8ddbd8b) states "origin = centre of the
  stage-2 opening on the block top face". That wording is wrong by
  15.0 mm. Measured from the exported file: top face +15.0, step-2
  rim 0.0. Whoever merges these entries must keep THIS wording.
- Required scene-stream fix: `insertion_tasks_cfg.py:587` defines
  `WORKCELL_BLOCK_POS = (BLOCK_CENTRE_X, BLOCK_CENTRE_Y,
  BLOCK_TOP_Z)`, i.e. its Z is the block TOP surface. Spawning the
  CAD at that Z puts the pocket rim where the top face belongs and
  lifts the whole fixture by 15 mm. The CAD spawn translation must
  be `BLOCK_TOP_Z - 0.015`. `[verify]` — derived on the laptop from
  the STEP file, never run in Isaac.
- Measured from the exported file (see the verification entry
  below), all in mm relative to the CAD origin:
  X (depth, +X to the rear wall): -84.794 to +65.206
  Y (long axis, +Y to the 7.9 mm end): -88.650 to +100.450
  Z (up): -46.000 to +15.000
  Warning against a wrong reading of those numbers: the exported
  body is NOT the whole fixture. The user modelled the pocket plus
  about 10 mm of material below the pocket floor (user statement
  2026-08-19). The face at Z = -46.000 is therefore a cut face, not
  the real underside, and the centre (-9.794, +5.900, -15.500) is
  the centre of the modelled stub, not of the real block. The scene
  must reference the fixture by the opening plane Z = 0 only, and
  must never seat it on the table by its underside. How the stub is
  supported in the scene is an open scene-stream question — options
  are a static fixed asset that needs no support, or a filler box
  below the stub.
- Sources: `docs/Geometrie/CAD_Anleitung_Taschenblock.tex` lines
  197-206 (measured margins); `docs/Geometrie/Aufnahme_real_v1.stp`
  parsed offline 2026-08-19; user decisions in the grill session
  2026-08-19.

## D-099: CAD instruction document is superseded by the user's measured CAD
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- MERGE NOTE: overlaps the scene-stream entry "Fixture CAD
  authority: user's own measured model replaces the D-059 recipe"
  (branch `p1-szene-umbau`, commit 8ddbd8b). Both were written on
  2026-08-19 without knowledge of each other. They do not
  contradict: the scene entry settles the AUTHORITY shift, this one
  settles the DOCUMENT action (delete, and what to rescue first).
  Give them one D-number, not two.
- Context: `docs/Geometrie/CAD_Anleitung_Taschenblock.md` / `.tex` /
  `.pdf` is a step-by-step recipe to model a coarse fixture block.
  The user has since measured the real fixture completely and built
  his own CAD from those measurements.
- Options considered: correct the axis labels in the document to the
  rotated frame / mark the whole document for deletion.
- Decision: Mark the document for deletion. Do not patch it.
- Rationale: The user's measured CAD replaces the recipe entirely,
  so a corrected recipe would only compete with the real authority.
  The document also carries stale frame labels that actively
  mislead: line 243 states "unten = buendiges Ende = -X" and
  "rechts = Rueckwand = +Y", and line 201 places the 7.9 mm margin
  at the "+X end". In the rotated frame the long axis is Y and the
  depth is X, so the 7.9 mm margin is at the +Y end and the rear
  wall is at +X — the opposite of what the file says. Deletion is
  the `main` writer's job, not this stream's.
  Note for whoever executes it: the measured margin table in lines
  195-206 is the only place those step-1 margins are written down.
  Carry those numbers into
  `docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex` before deleting.
  Status 2026-08-19: the user deleted the `.md` and the `.pdf`, and
  also the drawings `Tasche_Zeichnung*`, `Tasche_Zeichnung_v3*` and
  `Tasche_Messpunkte*`, plus `Messliste_Tasche_Rueckwand.md`. The
  `.tex` still exists but no longer builds, because line 240 pulls
  in the deleted `Tasche_Zeichnung.png`. So the file is already
  half-dead — finishing the deletion is now the tidy option, after
  the margin numbers are rescued.
- Sources: `docs/Geometrie/CAD_Anleitung_Taschenblock.tex` lines
  195-206 and 240-245; user decision in the grill session
  2026-08-19.

## D-100: Open conflict: pocket opening 145.1 x 90.59 from CAD vs 143.8 x 90.5 derived from clearance
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- MERGE NOTE: the scene stream carries the same conflict inside its
  "Fixture CAD authority" entry (commit 8ddbd8b) and reaches the
  same result — 145.1 stands provisionally, `[verify on site]`.
  Merge into one entry. The one point this entry adds and the scene
  entry lacks: 143.8 and 90.5 are marked M* in the geometry sheet,
  meaning DERIVED from the clearance measurement, not measured. So
  the CAD contradicts a computed value, not a measured one.
- Context: The user reports the step-2 opening from his CAD as
  145.1 x 90.59 mm. The geometry document gives 143.8 x 90.5 mm.
- Options considered: adopt the CAD value / keep the derived value /
  hold both and measure on site.
- Decision: Adopt 145.1 x 90.59 provisionally. The conflict stays
  OPEN and is flagged `[verify on site]`.
- Rationale: The 143.8 x 90.5 pair is NOT a direct measurement. The
  geometry table marks both entries with M* and the note
  "rechnerisch: Teil + Spiel"
  (`Geometrie_Fuegeteil_Aufnahme.tex` lines 117-118, 123). The
  direct measurements are the part itself (143.34 long) and the
  clearance by paper gauge (0.5 mm long, 0.2 mm across, lines
  205-206). So the CAD value contradicts a DERIVED number, not a
  measured one — the authority gap is smaller than it looks. Still
  unresolved: 145.1 minus the part's 143.34 gives an effective
  clearance of 1.76 mm along the long axis, against 0.5 mm measured
  by paper gauge. That is a factor of 3.5 and it matters, because
  the clearance drives the success threshold. Across the axis there
  is no conflict: 90.59 against 90.5. Resolution needs a caliper
  measurement of the real opening on site. Until then no threshold
  may be derived from either number.
- Sources: `docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex` lines
  117-118, 123, 205-206; user CAD value reported in the session
  2026-08-19 (not yet cross-checked against the exported STEP file).

## D-101: Anchor the start uncertainty at a touched edge instead of the pocket centre
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: User argument during the grill session 2026-08-19: on the
  real station nobody can approach the pocket centre cleanly. The
  operator drives against two edges to get a hard reference, and
  moves from there to a defined edge of the opening. He asks whether
  the start distribution should follow that pattern.
- Options considered: start distribution centred on the pocket
  centre with symmetric noise (current proxy machinery per D-044) /
  start distribution anchored at a touched reference edge with a
  systematic offset plus smaller residual noise.
- Decision: `[PROPOSAL]` — not decided. Recorded as a candidate for
  concept block 2 or 3, where the reset distribution is settled.
  Explicitly NOT a geometry question and NOT part of the CAD export.
- Rationale: The idea matches established practice. A review of
  search strategies in robotic assembly names exactly this driver —
  uncertainty from fixture, end effector and actuator — and
  catalogues spiral, raster and random-walk search; it rates them by
  time, precision and stability. A Robotica paper treats hole search
  specifically under initial positioning uncertainty. Guarded-move
  or touch-off referencing shrinks the error but does not remove it:
  probe deflection, edge burr and gripper play remain, and the error
  changes character from random to systematic. Counter-argument that
  keeps this out of the geometry work: simulation always knows the
  true pose. IndustReal and Factory randomise part poses AROUND a
  known nominal pose; randomisation needs a nominal value, so an
  unknown offset cannot be reinterpreted as realistic noise. Second
  counter-argument: D-046 puts sim-to-real transfer out of scope, so
  the transfer motivation does not apply. What remains is a task
  difficulty question, which belongs to the reset and reward design.
  Interacts with D-044 (variation axes) — this would reshape axis 2
  (fixture offset in X/Y) from symmetric to edge-anchored.
- Sources: Jiang et al., "The state of the art of search strategies
  in robotic assembly",
  https://www.sciencedirect.com/science/article/abs/pii/S2452414X21000571
  ; "Hole search strategy for robotic peg-in-hole assembly under
  initial positioning uncertainty", Robotica,
  https://www.cambridge.org/core/journals/robotica/article/abs/hole-search-strategy-for-robotic-peginhole-assembly-under-initial-positioning-uncertainty/504E73BDCEB27B7A8BFFEB7DFFECCD7C
  ; IndustReal documentation,
  https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/industreal.md
  ; NVIDIA, "Bridging the Sim-to-Real Gap for Industrial Robotic
  Assembly Applications Using NVIDIA Isaac Lab",
  https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/
  ; "A General Peg-in-Hole Assembly Policy Based on Domain
  Randomized Reinforcement Learning", https://arxiv.org/abs/2504.04148
  ; user statement 2026-08-19; DECISIONS.md D-044, D-046.

## D-102: Verification result of the fixture STEP export (laptop, offline)
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: `docs/Geometrie/Aufnahme_real_v1.stp` was exported from
  Creo on 2026-08-19 and checked offline on the laptop by parsing
  the STEP entities. No Isaac involved, so this is a file-level
  check, not a runtime validation.
- Decision: The export is accepted as the geometry source for the
  fixture.
- Verified (evidence in the file):
  (1) Schema AP214 —
  `FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'))`.
  (2) Length unit millimetre — `SI_UNIT(.MILLI.,.METRE.)`, global
  uncertainty 3.9E-4.
  (3) Export coordinate system is the user's own CSYS, not the Creo
  default: `AXIS2_PLACEMENT_3D('CS0')` sits at (0,0,0) with Z =
  (0,0,1) and X = (1,0,0), i.e. identity, while
  `AXIS2_PLACEMENT_3D('PRT_CSYS_DEF')` sits at
  (-9.75, 6.865, -36.0) and is rotated. This was the single biggest
  risk in the export and it is clean.
  (4) The origin is exactly centred on the step-2 opening: points at
  X = +/-45.295 (opening 90.59 across) and Y = +/-72.550 (opening
  145.1 along) are all present.
  (5) The Z levels reproduce the measured pocket profile: top face
  +15.0, step-2 rim 0.0, pocket floor -36.0, block underside -46.0.
  Step-1 depth = 15.0 and pocket floor below the top face =
  15.0 + 36.0 = 51.0. Both match the measured 15 / 51 in
  `CAD_Anleitung_Taschenblock.tex` line 205 exactly.
  (6) Exactly one `MANIFOLD_SOLID_BREP`. The exported datums appear
  as named `AXIS2_PLACEMENT_3D` entities only, not as bounded
  surfaces. The SDF-collision risk noted in the export-convention
  entry therefore does NOT materialise in this file. The prim-list
  check on the training PC can be dropped for this export.
- Not verified: everything downstream of the file. The STEP-to-USD
  conversion, the metersPerUnit handling and the collision mesh are
  `UNVERIFIED` until the training PC runs them.
- Sources: `docs/Geometrie/Aufnahme_real_v1.stp` parsed on the
  laptop 2026-08-19; `docs/Geometrie/CAD_Anleitung_Taschenblock.tex`
  lines 195-206.

## D-103: Fixture CAD is a deliberate cut-out, not the whole block
- Date: 2026-08-19
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: The bounding box of the exported fixture CAD is
  150 mm in depth (X), 189.1 mm along the long axis (Y) and 61 mm
  high (Z). The coarse block used so far in the scene is
  260 x 210 x 116. The gap is roughly half the volume, so it was
  raised as a conflict.
- Options considered: adopt the CAD box as the real fixture size /
  keep the coarse box / treat the CAD as a partial model.
- Decision: No conflict. The user modelled only the pocket plus
  about 10 mm of material below it, deliberately, as a first
  working solution (user statement 2026-08-19). The 260 x 210 x 116
  coarse box was never a measurement either — the height 116 and
  the lateral position are flagged as assumptions in
  `CAD_Anleitung_Taschenblock.tex`.
- Reconciliation with the scene stream (checked 2026-08-19): the
  two numbers do NOT contradict each other. `BLOCK_SIZE_X,
  BLOCK_SIZE_Y = 0.210, 0.260` in `insertion_tasks_cfg.py:506` is
  marked `M` (measured, 210 deep x 260 wide), and the CAD footprint
  150 x 189.1 fits inside it. The CAD is a subset of the real
  block, exactly as the user described. So the coarse-box constants
  stay as they are, and the CAD mesh covers only the pocket region.
- Consequence for the scene: the fixture cannot be represented by
  the CAD mesh alone. Either the coarse box carries the body and
  the CAD mesh carries the pocket, or the fixture is a static asset
  and the stub simply floats where the pocket needs it. That choice
  belongs to the scene stream.
- Still open, deliberately deferred: the real fixture HEIGHT. The
  116 mm in the coarse box is an assumption, not a measurement
  (`CAD_Anleitung_Taschenblock.tex` lines 228-229), and the CAD
  cannot supply it because it is cut off at -46 mm.
  `[verify on site]`
- Sources: `docs/Geometrie/Aufnahme_real_v1.stp` bounding box parsed
  2026-08-19; user statement 2026-08-19;
  `docs/Geometrie/CAD_Anleitung_Taschenblock.tex` lines 206 and
  228-229.

## D-104: Zwei-Rechner-Übergabe: Datei statt Git
- Date: 2026-08-22
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)

- Decision by the user: the training-PC console log travels to the
  laptop BY HAND, not through git. `scripts/rt_log.ps1` still writes
  `rt_logs/<NAME>.txt` and now puts the text on the clipboard; the
  user pastes it into `rt_logs/inbox.txt` (always the same file) and
  runs `/rt-check`.
- Rationale: the goal of the loop was never version control, it was
  keeping the log OUT of the main chat context. Git did not help with
  that and cost four round trips of its own (PROBLEMS.md, 2026-08-21).
  The manual paste has no failure modes: the file is either there or
  it is not.
- Consequence: training-PC logs are NOT versioned any more. The
  existing RT-1/2/3/5 and SELFTEST logs were deleted on the user's
  instruction. `rt_logs/VERDICTS.md` (one line per evaluated run) is
  now the only durable record, and PROBLEMS.md / HANDOFF-SZENE.md
  cite it instead of the raw logs.
- Consequence: the sparse clone stays the way CODE reaches the
  training PC. Only the log's way back changed.
- Removed from the wrapper: `Invoke-Git`, add/commit/pull/push, the
  index guard, the `[rt_log] RESULT:` lines. Kept, because those
  defects are independent of git: automatic `python -u` and the plain
  `param($Name)` block.

## D-105: Gravity stays ON, all drive parameters come from the UR5e USD
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: szene (supervisor decision, relayed by the user)
- Context: the supervisor instructed to keep gravity enabled and to
  trust the drive values shipped inside the Isaac Sim UR5e USD: a
  previous bachelor thesis tried to optimise those stiffness/damping
  values and concluded the shipped ones are already (near-)optimal
  (that thesis is NOT cited and its title was deliberately not
  chased — user decision 2026-08-25; the decision rests on the
  supervisor's instruction plus our own runs RT-45/RT-46, not on
  that document). Screenshot evidence from the supervisor: shoulder_pan drive
  stiffness 9400.50098, damping 0.378, max force 150 in the Isaac
  Sim UI. Until now `ur5e_cfg.py` OVERWROTE those USD values with
  UR10e placeholder gains (1320/600/216 …), which produced the
  measured 21.7 mm flange droop (RT-18, HANDOFF-SZENE.md).
- Options considered: (1) keep gravity off per p1-gains D-082;
  (2) gravity on, keep UR10e placeholder gains; (3) gravity on,
  take every drive/joint parameter from the USD.
- Decision: Option 3. `ur5e_cfg.py` sets stiffness, damping,
  friction, armature to `None` in all three ImplicitActuatorCfg
  blocks and drops `max_depenetration_velocity` — `None` means "value
  from the USD joint prim" (Isaac Lab 2.3 actuator docs, verified).
  Deliberately kept overrides (setup, not robot): disable_gravity
  False, fix_root_link True, enabled_self_collisions False, solver
  iterations 16/1, activate_contact_sensors True, init_state.
  Any residual droop is accepted, not repaired — but it turned out
  there is almost none left to accept (see Verified below).
  [CORRECTION 2026-09-13 (audit): under the OSC default (D-177, since
  2026-09-03) the PhysX drive stiffness and damping are set to 0 -- the
  LETTER of "all drive parameters from the USD" is broken for those two
  values. maxForce, limits, armature and friction stay USD-authored; gravity
  stays ON. Home: D-177 Decision (2).]
- Supersedes: p1-gains DECISIONS.md D-082 ("disable_gravity=True",
  verified G-5a) and D-083 ("the gains stay as they are: 1320 / 600 /
  216"). Inline corrections placed on both. NOTE for the numbering
  guardian, RESOLVED 2026-08-27: `p1-gains` is not merged into main
  (user decision 2026-08-27), so D-082 is NOT used twice on main — it
  belongs to the scene line (metre-file/unit repair). The p1-gains
  D-082/D-083 stay on their own branch, marked SUPERSEDED there.
- Verified on the training PC the same day, both halves:
  RT-45 (`probe_assets.py --only-robots`) — the WELDED robot+tool+peg
  USD (`ur5e_tool.usd`, the asset insertion_env.py:543 swaps in)
  carries the same authored drives as the bare arm: shoulder_pan
  stiffness 9400.5009765625, damping 0.37800323963165283, maxForce
  150.0. The weld preserved them.
  RT-46 (`zero_agent.py`), t = 3.000 s, envs 0..1, reset noise off —
  Z standoff (flange) 0.317026 m (nominal 0.3170, so 0.000026 m off);
  max joint deviation 0.000029 rad (0.002 deg) on shoulder_lift;
  XY offset to entrance 0.000014 m; tip Chebyshev offset 0.000019 m.
  Against the RT-18 baseline under placeholder gains (flange standoff
  0.309950 m, elbow error 0.031595 rad, lateral tip offset 0.013456 m)
  the droop is effectively gone. NOT re-measured: tool tilt and flange
  drop — the report prints neither, so no new number exists for them.
- Nothing left open on this entry.
- Sources: supervisor statement 2026-08-25 + Isaac Sim UI screenshot;
  Isaac Lab 2.3 API docs (ImplicitActuatorCfg, None -> USD prim);
  runs RT-45 and RT-46 (rt_logs/VERDICTS.md); HANDOFF-SZENE.md for
  both the RT-18 record and the new reference numbers.

## D-106: The three D-052 thresholds filled from the CAD numbers
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- **INPUT SUPERSEDED 2026-08-28 by D-121:** all alignment bounds here are
  derived from the cross play, which this entry took as 0.2876 mm. The
  scene stream re-read the part CAD — the body is 90.00 mm wide, not
  90.3, so **the cross play is 0.5876 mm**. Affected: the offset bound
  (half the play), the yaw bound (play / 143.5), the tilt bounds
  (play / depth) and the interpenetration sweep. **The numbers here are
  NOT replaced yet** — recomputing them is a concept-stream decision of
  its own, and D-121 deliberately stops at supplying the input.
- Stream: konzept
- Context: D-052 fixed the derivation METHODS for the three success
  thresholds and left all three values `[CAD pending]`. The CAD
  numbers now exist: part 143.50 x 90.3 x 50.506 mm (D-088, D-075),
  stage-2 opening 145.1 x 90.5876 mm, sim play 1.60 mm long /
  0.2876 mm across (D-088), stage-2 depth 36 mm and stage-1 depth
  15 mm (Aufnahme_real_v1.stp, verified offline 2026-08-19). This
  entry closes D-052's `[CAD pending]` state and the obligation
  attached to its point (a).
- Options considered: depth band 3 mm (IndustReal precedent) / 2 mm
  (Isaac Lab fraction rule 0.04 x part height 50.506 = 2.02 mm).
  Interpenetration headline bound at half the across-axis play /
  adopting the IndustReal ladder 0.5-2 mm unchanged (rejected: every
  rung exceeds our 0.2876 mm across-axis play).
- Decision: (1) Depth band: success requires the part at most
  **3 mm** above full seat (part bottom on the stage-2 floor).
  Stated per the D-052 obligation: 3 mm = 8.3 % of the 36 mm
  engaged stage-2 depth (and 5.9 % of the 50.506 mm part height).
  (2) Alignment tolerance: the geometric bounds implied by the sim
  play at full depth — offset across +/-0.144 mm (half of
  0.2876), offset long +/-0.80 mm (half of 1.60), yaw <= 0.11 deg
  (0.2876/143.5 rad), tilt across <= 0.46 deg (0.2876/36 rad),
  tilt long <= 2.5 deg (1.60/36 rad) [CORRECTION 2026-08-26: the
  2.5 deg is the small-angle approximation 1.60/36; the CAD-exact
  full-seat bounds are 2.821 deg (shoulder side) / 1.905 deg
  (other side) — see the Block-5 entry above]. No separate alignment check
  is evaluated: depth check + interpenetration filter imply these
  bounds, because the pocket walls enforce them on any
  non-penetrating pose at depth. The table documents the enforced
  bound; this is the "derived from clearance" route D-052 point (b)
  demanded.
  (3) Interpenetration filter: sweep over
  {0.05, 0.1, 0.144, 0.2876, 0.5} mm; the **0.144 mm** bound
  (half the across-axis play) carries the headline success rate.
  Rationale for the tie: interpenetration beyond half the play is
  physically impossible even at perfect centering, so 0.144 mm is
  the largest bound that still rejects only impossible poses.
- Rationale: 3 mm adopts the IndustReal band, the only absolute-band
  precedent in the Block-2 literature check, and lands in the same
  regime relative to depth (8.3 %). The 2 mm alternative is stricter
  without discriminating anything: below 3 mm remaining travel the
  part is already captured by the tight stage-2 walls. User chose
  3 mm ("3mm passt", 2026-08-25).
- Sources: D-052 (methods + obligation); D-088 (part length, sim
  play); D-075 (part mesh dims); D-087 (real play, reference only);
  `docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex` (stage depths
  15/36/51); `docs/reference/literature_check_success_definitions_2026-08-16.md`
  (IndustReal 3 mm band, SAPU sweep, Isaac Lab 0.04 fraction);
  user decision 2026-08-25.

## D-107: Block 3: Observation space — proxy 25-channel layout adopted, part-anchored, cos/sin yaw, no F/T
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: Concept Block 3 (nodes N3.1-N3.4, KONZEPTPHASE.md). The
  proxy observation is 25 channels (read from the old-repo
  `proxytask_env.py:_get_observations`): joint pos (6), joint vel by
  finite difference (6), tip relative to the pocket opening in the
  pocket frame (3), EE quaternion world-frame (4), yaw encoded as
  cos 4*phi / sin 4*phi (2), pocket quaternion sign-canonicalised
  (4, old-repo D-037).
- Options considered: N3.1 adopt / redesign. N3.2 F/T channel
  implement / omit with justification (old-repo D-005 had DECIDED an
  F/T channel; the proxy code never implemented it). N3.3 grasp
  uncertainty visible via (a) explicit new part-in-flange channels
  (+7, privileged) / (b) anchoring the existing pose channels at the
  part instead of the flange, zero new channels. Yaw encoding: keep
  cos 4*phi / switch to cos phi, sin phi.
- Decision (user, grill session 2026-08-25):
  (1) N3.1: ADOPT the 25-channel layout (adoption per D-042).
  (2) Yaw encoding becomes **cos phi / sin phi**: the real part fits
  the pocket in exactly ONE rotational position (user statement — the
  wave contour side is unique; the proxy peg was 4-fold symmetric,
  hence its cos 4*phi). Channel count unchanged.
  (3) [CORRECTED 2026-08-27 — SUPERSEDED by the entry "D-089 is
  OVERTURNED" at the end of this file: the F/T channel IS added,
  observation 25 -> 28. Original text follows for the record.]
  N3.2: **No F/T channel** — MERGE NOTE (added 2026-08-25,
  after the grill session): this half is ALREADY DECIDED as D-089
  (branch p2-szene-umbau, same day, accepted, closes N3.2 with the
  full literature case incl. the Factory ablation where adding force
  COST ~35 % relative success). D-089 is the home of this fact; do
  not give this half a second number. The grill session reached the
  same result independently; its "add F/T later if contact fails"
  expectation is formalised by D-089's reopening condition (jamming
  evidence required) and its fixed route (Forge wrench pattern, not
  ContactSensor).
  (4) N3.3: **(b) part-anchored channels** ("like Factory", user):
  the tip-relative-to-opening channels and the orientation quaternion
  are computed from the part's task frame (D-069), not the flange.
  The per-reset grasp offset (D-070) then shows up in the existing
  channels; no new channels, layout stays 25, warm start untouched.
  [Corrected 2026-09-11 by D-183: the grasp offset is now added to the
  OBSERVATION only, so it is HIDDEN inside `tip_rel` — the policy does not
  see that it is there; the body stays nominal.]
  (5) N3.4 observability check passed for all four D-044 axes:
  axis 1 via part-anchored channels; axis 2 via pocket-relative tip +
  joint state (empirically proven in the proxy at +-2 cm, >=99 %);
  axes 3-4 via the pocket quaternion (old-repo D-037).
- Rationale: Adoption: verified >=99 % in the pre-study on exactly
  the variation axes the real task reuses (D-044 axes 2-4); keeps
  the proxy warm start (lesson 3); quaternion input is the
  GenPiH-precedented standard (old-repo D-037). No F/T: force is not
  a success criterion (D-051, nine sources); the pre-study proves the
  task solvable without it; no sim-to-real (D-046) removes the
  real-sensor argument that motivated old-repo D-005.
- Open flag for the code stream: `[verify]` whether the welded
  gripper+part link (D-076) actually allows a per-reset
  flange-to-part offset in Isaac; D-070 decided the intent
  (Factory `held_asset_pos_noise` pattern), no runtime evidence yet.
- Sources: old repo `proxytask_env.py` (obs assembly, read
  2026-08-25); old-repo D-005, D-037; DECISIONS.md D-042, D-044,
  D-046, D-051, D-069, D-070, D-076; Isaac Lab 2.3.2
  `factory_env.py:763-775` (held-asset noise, via D-070); user
  decisions in the grill session 2026-08-25.

## D-108: Block 4: Action space — joint deltas kept, with failure hypotheses, fallback, and a gating scripted test
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- **DECISION (1) OVERTURNED 2026-09-03 — THE ACTION IS NOT SIX JOINT
  DELTAS ANY MORE.** What this entry calls the *fallback* — "task-space
  delta actions + impedance control after the Factory pattern" — IS the
  running code. The action is a 6-D pose delta under Isaac Lab's
  `OperationalSpaceController`: `control_mode = "osc"` is the cfg default
  (`insertion_env_cfg.py:210`), `impedance_mode="fixed"`,
  `motion_stiffness_task` 100 N/m translational and 30 rotational,
  `motion_damping_ratio_task` 1.0, `gravity_compensation=True`,
  `inertial_dynamics_decoupling=True`, `nullspace_control="none"`
  (`insertion_env.py:1433-1442`). The PhysX position drives go INERT
  under this mode — stiffness and damping of every actuator block are set
  to 0 (`insertion_env.py:1313-1316`), which breaks the LETTER of D-105
  for those two values; maxForce, limits, armature and friction stay
  USD-authored. `joint_pd` survives only as a per-run override
  (`env.control_mode=joint_pd`).
  **Home of the fact: D-177** (numbered 2026-09-10). Its own source is
  the inbox entry "Audit 2026-09-03 (a)" (six decision branches) in
  `docs/decisions_inbox.md` on branch `p4-konzept`; summarised in
  `HANDOFF-RL.md` § Frozen. The supervisor question D-177 (2) raises is
  UNANSWERED, so the concept has not fixed the point — the code runs OSC
  regardless. D-166 derives `osc_kp_pos` = 100 from a 20 N search force;
  D-167 records that under OSC the force abort is no longer the brake
  against ramming.
  **What this does to the rest of this entry:** H2 (stiff position
  control produces high contact forces under jamming) is the reason for
  the switch, measured at RT-144a — 250.12 N with all six joints
  saturated (see D-167). H3 (all six joints couple in joint space) no
  longer applies to the action. The 2026-08-29 correction below that
  declares `action_scale = 0.02` CONFIRMED belongs to `joint_pd` ALONE;
  under OSC the step size is `osc_pos_step_limit_m` /
  `osc_rot_step_limit_rad`, and D-168 owns the action-rate scale.
- **INPUT SUPERSEDED 2026-08-28 by D-121:** the "7x tighter" comparison
  (proxy 2 mm against the real 0.288 mm) uses a cross play that the scene
  stream has since overturned by CAD — the real value is 0.5876 mm, so
  the factor is about 3.4x, not 7x. **The factor here is NOT replaced
  yet.** The decision itself (six joint deltas, the three hypotheses, the
  scripted-insertion gate) does not depend on the exact factor.
- Stream: konzept
- Context: Concept Block 4 (nodes N4.1-N4.3, KONZEPTPHASE.md). The
  proxy (and the copied `insertion` package, D-063) uses 6 joint-delta
  actions: each step clamps the action to [-1,1], scales by
  `action_scale = 0.02` rad, integrates into joint position targets,
  clamps to joint limits, applies via PD position control
  (`insertion_env.py:572`, control rate 60 Hz per D-024). The proxy
  cfg itself labels this "interim (joint deltas, no OSC)".
  HARD FACT from the grill session: the proxy's verified >=99 % ran
  at 2 mm total play (30 mm peg in a 32 mm pocket,
  `proxytask_tasks_cfg.py` default); the real task's across-axis play
  is 0.288 mm - about 7x tighter. The pre-study therefore does NOT
  prove joint deltas reach the real precision regime (user
  requirement 2026-08-25: do not lean on the 99 %; think critically
  about why it could fail here).
- Options considered: keep joint-delta actions / switch to task-space
  delta actions with an impedance controller (Factory/IndustReal
  pattern).
- Decision (user, grill session 2026-08-25):
  (1) N4.1: KEEP the 6 joint-delta actions (adoption per D-042), with
  three recorded failure hypotheses and the alternative documented
  beside it as the next-to-test:
  - H1 precision: terminal accuracy is set by PD tracking, not by the
    action scale (actions are continuous); whether the control chain
    can command a 0.144 mm placement is unproven.
  - H2 contact: stiff position control under jamming produces high
    contact forces; Factory/IndustReal use compliant task-space
    control for exactly this reason.
  - H3 reorientation: the single valid yaw pose demands fine
    reorientation near the wall; in joint space all six joints couple.
  - Fallback (next to test if joint deltas fail): task-space delta
    actions + impedance control after the Factory pattern. A switch
    costs the warm start (policy head semantics change) - that is the
    price of the fallback, stated openly.
  (2) N4.2: no new concept decision. UR5e config + home pose are
  scene-stream work (done, status report 2026-08-25). Controller
  gains: CORRECTED 2026-08-25 (user) — they do NOT live in stream
  `p1-gains` any more. All drive parameters (stiffness, damping,
  friction, armature) come from the UR5e USD itself: `ur5e_cfg.py`
  sets them to None, which loads the USD-authored values
  (shoulder_pan 9400.5 / 0.378 / 150). Supervisor decision
  2026-08-25, supersedes p1-gains D-082/D-083; verified RT-45/RT-46
  (flange droop 21.7 mm -> 0.000026 m). See the szene-stream inbox
  entry "Gravity stays ON, all drive parameters come from the UR5e
  USD" (branch p2-szene-umbau). Joint limits come from the robot cfg
  and the target clamp already in the code.
  (3) N4.3: `action_scale = 0.02` rad/step ADOPTED with a derivation
  instead of an assumption: 60 Hz x 0.02 rad = 1.2 rad/s commanded
  joint speed = 38 % of the UR5e datasheet limit (180 deg/s all
  joints, `ur5e-datasheet.pdf`) - safe upper bound; continuous
  actions mean the scale imposes no resolution floor; the 4 s episode
  budget is sufficient at these speeds.
- GATE attached to (1) and (3) (user requirement: the value must be
  CHECKED, not assumed): before the first training run, a SCRIPTED
  insertion at the real 0.288 mm play must run on the training PC
  (debug protocol "scripted solvability"). It discriminates H1/H2
  from policy failure: if the control chain cannot place the part
  scripted, no reward tuning can fix it - go to the fallback. The
  0.02 value counts as confirmed only after this measurement.
  **[CORRECTION 2026-08-29: THIS GATE IS SUPERSEDED. The scripted descent is
  no longer required. RT-81 measured the seated state directly -- 32.965 mm at
  8.08 N against `SEATED_SUCCESS_DEPTH` 33.000 mm -- so "is the seat reachable"
  is answered by measurement rather than by a controller, and hand-building the
  search motion a 0.2-0.8 mm clearance needs is classical-control work of its
  own size (user, 2026-08-29). What replaces it: a TELEPORT identity test on
  the real success path, plus separate instruments for H1 (command a lateral
  offset of the SAPU order in free space, measure the tracking error) and H2
  (already measured -- RT-74 climbs to 122.2 N at 19 mm and stalls at 227.2 N).
  `action_scale = 0.02` IS NOW CONFIRMED: RT-84 (2026-08-29 12:19:44, git
  f43681a) held sixteen envs at lateral targets spread over the whole 0.5876 mm
  cross play, in free space, and measured a steady residual of p95 0.0461 mm
  against the 0.2938 mm half-play -- six times inside it, settled after 0 to 1
  control step. Verdict `H1_TRACKS`, exit 0. H1 is answered: the control chain
  reaches the precision the task needs. Full
  entry: `docs/decisions_inbox.md`, "The D-108 gate drops the scripted
  descent".]**
- Sources: old repo `proxytask_tasks_cfg.py` (2 mm play),
  `insertion_env.py:572`, `insertion_env_cfg.py:114-148` (D-024
  60 Hz, action_scale); `docs/reference/Datenblaetter/ur5e-datasheet.pdf`
  (180 deg/s); D-042, D-045, D-063, D-071; D-088 (0.288 mm play);
  user decisions in the grill session 2026-08-25.

## D-109: Block 5: N5.0–N5.4 — SDF-Reward + Quetschfunktion, Kraft-Abbruch statt Kraftstrafe, SAPU 0.144 mm, Kernelbreiten aus der Toleranz
- Date: 2026-08-26
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- **INPUT SUPERSEDED 2026-08-28 by D-121:** every kernel width here
  divides by the cross play, which this entry took as 0.2876 mm. The
  scene stream re-read the part CAD with a control on the long axis and
  found the body is 90.00 mm, not 90.3 — **the cross play is 0.5876 mm**.
  The kernel constant a = 10408 and the SAPU 0.144 mm therefore rest on a
  superseded input. **The numbers here are NOT replaced yet**; the
  recomputation is a concept-stream decision of its own. Do not cite
  a = 10408 or 0.144 mm until it is done.
- Stream: konzept
- Context: Block-5 grill in a fresh chat (per 2026-08-25 user
  decision), questions Q1–Q18 resolved. Full record with sources in
  the plan file `zu-block-5-transient-badger.md` and in the new
  reward literature check (see Sources). Replaces the proxy's
  7-term reward structure for the real insertion task. The six
  exploit questions (project rule, old-repo
  `insertion_env.py:130-198`) were re-run against the new term set.
- Options considered: adopt the proxy 7-term structure vs derive
  top-down (criterion -> measured quantity -> term); force penalty
  in the reward (FORGE) vs force-abort only (Schoettler) vs
  nothing; separate approach/depth/yaw/align terms vs one
  SDF-distance term (IndustReal) vs keypoints (Factory/FORGE);
  -log mapping (IndustReal) vs squashing function
  (Factory/FORGE code) vs 1-tanh(d/std) vs -log(phi+eps) vs
  1/(x+0.1); terminate on success vs run to timeout; kernel
  widths from clearance vs copied ratios vs dm_control
  tolerance() solve; SAPU threshold 1 mm (IndustReal) vs 0.144 mm
  (D-052 headline) vs 0.1847 mm (voxel fallback).
- Decision (user, grill session 2026-08-26):
  (1) Q2/Q6 method: derivation TOP-DOWN (criterion -> measured
  quantity -> term). The proxy is the cross-check, not the
  template. Q3: the reward reads ONLY the 25 observation channels
  (Block-3 layout); a term needing a new quantity must first put
  it into the observation.
  (2) [CORRECTED 2026-08-27 — this point is SUPERSEDED by the
  entry "D-089 is OVERTURNED" at the end of this file: force IS in
  the observation and the FORGE penalty IS in the reward. The
  original text follows for the record.]
  Q4/Q5/Q5b: NO force penalty in the reward — D-089 holds.
  Ramming is punished by the FORCE-ABORT (Block 7, N7.3): episode
  ends, remaining reward and success bonus are lost. First stage
  follows the Schoettler 2020 pattern (10 N stop / 6 N retract as
  the FORM; our Newton limit comes from a measurement, not an
  estimate). Planned escalation stage, documented as a stage and
  not as an emergency: force channels into the observation + FORGE
  penalty `-beta*max(0, ||F||-F_th)`. Reopening only under the
  D-089 condition.
  (3) Q12: ONE SDF-distance term replaces approach, depth, yaw and
  align (Q7/Q8/Q9 are thereby VOID). Source: IndustReal Tab. VI /
  AutoMate Eq. 9; code exists in Isaac Lab 2.3.2
  (`isaaclab_tasks/direct/automate/industreal_algo_utils.py::
  get_sdf_reward`, called in `assembly_env.py:583`); N = 1000
  surface points sampled once (`trimesh.sample.sample_surface_even`,
  `num_mesh_sample_points = 1000`). Three adaptations are needed
  and go to the code stream: (a) the target pose must come from
  OUR CAD (the Isaac Lab code equates target pose with pocket pose
  because its meshes are drawn coincident — ours are not); (b)
  compute load: Python loop over envs, mesh rebuild + BVH refit
  per env per step, reference configs run only 128 envs — check
  against the proxy env count; (c) `-log(0) = inf` at full seat is
  NOT caught by Isaac Lab (see (6)).
  (4) Q10: the reward prescribes NO sequence — neither corner nor
  side is named. If the other side is easier, the policy may take
  it. Q11: void — no progress term, hence no gate.
  [CORRECTED 2026-09-02 by D-165: a depth-progress term is paid BESIDE
  the SDF term — (3)'s "replaces … depth" and (4)'s "no progress term"
  no longer hold. The proxy's gated max-depth increment returns as
  `w_depth_progress`, gated by D-157's box and not by a corner/side
  gate, so (4)'s "no sequence" stands. Reason: RT-135 measured the
  kernels flat beside the stage-2 opening (5.82e-5 per step across
  4.88 mm). Widths, (9), are untouched.]
  (5) Q13: engaged + success display bonuses, weight 1.0 each,
  paid PER TIMESTEP without a latch (Isaac Lab factory pattern,
  literals in `factory_env.py:477-485`). Noted as concept, to be
  checked against common practice later. OPEN: the depth threshold
  for "engaged" `[CAD pending]`. CAVEAT (named, not resolved):
  over an episode these two bonuses are about TWO THIRDS of the
  total return — the "sparse bonus" is the dominant term.
  (6) Q15: the -log mapping is REPLACED, not guarded: SDF distance
  (IndustReal) mapped through the Factory/FORGE squashing function
  `1/(exp(a*x) + b + exp(-a*x))` — finite at x = 0 (value
  `1/(2+b)`), always positive. Only the mapping is swapped, the
  measured quantity stays the SDF distance. Rejected:
  `-log(phi+eps)` (SB3 pattern protects a probability, not a
  metre distance — would be self-invented) and IndustReal's
  `1/(x+0.1)` (the 0.1 is metre-sized, our phi is ~1e-4 m).
  Motivation: rsl_rl v3.0.1 has no NaN/inf check and no reward
  clamp at all (corrected 2026-08-27 in the Block-8 session; the
  earlier `check_nan` reference was wrong — that function does not
  exist) — an inf would surface one iteration later in the advantage
  normalisation. NOTE: the squashing function is an Isaac-Lab
  CODE artefact, it is NOT in the Factory paper; cite the code or
  FORGE. The FORGE curve exists in two versions (with/without
  prefactor) — the Isaac Lab code has no prefactor, WE CITE THE
  CODE.
  (7) Q16: do NOT terminate on success (Isaac Lab factory/forge/
  AutoMate pattern); the literature is split about half-half
  (terminate: Inoue 2017, GenPiH, Soft-Wrist, Beltran-Hernandez
  2020b, Petrovic 2022). Named consequence: the per-step success
  bonus is a HIDDEN TIME REWARD (success at step 50 pays 190x, at
  step 230 only 10x). Switch trigger stays `[measure]`.
  (8) Q17: NO action penalty copied from Factory — Factory has
  none (`action_penalty_ee_scale = 0.0`,
  `action_grad_penalty_scale = 0.0`, `factory_tasks_cfg.py:70-71`).
  If an action penalty is wanted, it comes from FORGE:
  `-0.1 * ||a_t - a_(t-1)||` (`forge_tasks_cfg.py:15`), with the
  origin stated.
  (9) Q18: kernel widths are SOLVED, not tuned, via dm_control
  `tolerance()` (Tassa et al. 2018): `a = arccosh(1/value_at_
  margin)/margin = 2.9932/margin` at the DMC convention
  `value_at_margin = 0.1`. Table: coarse margin 10 mm (start
  scatter, Block 6) -> a = 299, b = 2; middle margin 3 mm
  (success band, D-052 family) -> a = 998, b = 2; fine margin
  0.2876 mm (cross play) -> a = 10408, b = 0. Peak sum = 1.0.
  Package: (i) widths from measured tolerances, zero runs;
  (ii) avoid constants — the SDF distance measures position and
  rotation in ONE quantity, no weight between them to defend
  (Factory: "computes distance on a single manifold, obviating
  tuning"); (iii) sensitivity table: three values for a_fine over
  a decade, 5 seeds each, metric = SUCCESS RATE, never the reward
  (IndustReal Tab. I template; optional IQM + bootstrap CI per
  Agarwal et al. 2021). Cost: 15 runs.
  (10) Q14: SAPU interpenetration scaling with threshold
  0.144 mm — the SAME number as the D-052 headline. Return is
  multiplied by `1 - tanh(d/0.144mm)`; above, discarded. SAPU
  reads pure Warp mesh queries (`get_interpen_dist`,
  `industreal_algo_utils.py:343-379`), NEVER PhysX — contact/rest
  offsets and solver settings are irrelevant. Three conditions:
  (a) Isaac Lab 2.3.2 has the kernel but NO caller
  (`interpen_thresh` does not exist) — port from IsaacGymEnvs
  (`get_max_interpen_dists`, `get_sapu_reward_scale`) -> code
  stream; (b) the fixture SDF resolution must be >= 1314,
  practically 1536 or 2048 (at D-062's 1024 the voxel is
  0.1847 mm, COARSER than 0.144 mm — the filter would punish the
  collider, not the policy) -> scene stream; (c) the pocket mesh
  must be watertight (sign flips silently otherwise), measurement
  mesh and collider CAD must be identical, and 1000 sample points
  are thin for a 143-mm part (too few points systematically
  UNDER-report penetration). Fallback if the resolution does not
  rise: 0.1847 mm (the voxel pitch itself, still inside the
  0.2876 mm cross play). IndustReal's 1 mm is excluded (3.5x our
  entire cross play).
  (11) Exploit questions re-run: five of six dissolve (gate
  farming — no gate; submarining, side-parking, yaw-parking — SDF
  distance grows there; suicide/early-abort — the squashed reward
  is ALWAYS positive, every step pays). Tunneling REMAINS and is
  what SAPU is for. NEW exploit class, named: since everything is
  positive, existing is rewarded; the gradient still points to
  the goal (Factory has the same), goes into the report.
- Rationale: IndustReal Tab. I (5 seeds x 1000 trials): collinear
  keypoints 15.4 %, 6-DOF keypoints 54.2 %, SDF 88.6 % — and the
  stated reason "collinear keypoints underspecify the assembly of
  non-axisymmetric parts" matches our non-axisymmetric part.
  Factory rejected `||dp|| + lambda*||dq||` as "sensitive to
  lambda" — the argument against separate orientation terms.
  Factory Tab. XII shows the reward FORM is decisive (linear
  83.6 % vs exp 0.5 %). Booth et al. 2023 (92 % trial-and-error,
  89 % self-judged suboptimal) is the evidence that reward
  constants need a derivation method — hence Tassa instead of
  tuning. Schoettler 2020 proves force-abort works with NO force
  in observation or reward. HONESTY ITEMS for the report: the
  combination SDF distance + squashing function has NO precedent
  (IndustReal uses -log, Factory/FORGE squash keypoints — nobody
  combined both; this is our contribution and must be stated as
  such); Ng, Harada & Russell 1999 does NOT protect this shaping
  (the theorem covers only potential-based shaping
  `gamma*Phi(s')-Phi(s)`, our kernel sum is a state-dependent
  step reward); rsl_rl v3.0.1 does NOT normalise the reward, so
  relative weights are absorbed by nothing.
- Note for the writer session: D-052's tilt-long bound 2.5 deg
  was the small-angle approximation 1.60/36; CAD-exact full-seat
  values are 2.821 deg (shoulder side) / 1.905 deg (other side)
  — see the inline correction in "The three D-052 thresholds"
  below. Also: D-059 and the D-075 surroundings need an inline
  correction (the lugs sit on the PART, the notches in the
  pocket) — this stream may not touch DECISIONS.md.
- Sources: `docs/reference/literature_check_reward_2026-08-26.md`
  (full source list with pinned locations); IndustReal (Tang et
  al. 2023) Tab. I/VI + released code; AutoMate Eq. 9 + code;
  Factory (Narang et al. 2022) V-B, Tab. IV/XII; FORGE
  (Noseworthy et al.) Tab. I + `forge_tasks_cfg.py`; Schoettler
  et al. 2020 (arXiv:2004.14404); Tassa et al. 2018
  (arXiv:1801.00690, `dm_control/utils/rewards.py`); Booth et al.
  AAAI 2023 (DOI 10.1609/aaai.v37i5.25733); Petrovic et al. 2022
  (MMAR, PDF at the user); Agarwal et al. 2021
  (arXiv:2108.13264); Isaac Lab 2.3.2 source
  (`industreal_algo_utils.py`, `factory_env.py:477-485`,
  `factory_tasks_cfg.py`, `factory_utils.py:105-107`); rsl_rl
  v3.0.1 source; user decisions in the grill session 2026-08-26.

## D-110: Block 6 (Teil 1): N6.1 + N6.2 — frisch trainieren, mehrachsige manuelle Leiter, 80/10-Regel, Ranges aus der Zelle, 0,6-mm-Vorstufe optional
- Date: 2026-08-26
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: Grill session Block 6 (this session), nodes N6.1 and
  N6.2 decided; N6.3 (DR + obs noise K3) and N6.4 (seeds) still
  OPEN — grill continues in a fresh chat (resume point in
  HANDOFF-KONZEPT § Open). Basis: Block-6 draft report + curriculum
  check (Rounds 1–3) + Parnada check.
- Options considered: N6.1a warm start from proxy checkpoint vs
  fresh; N6.1b no curriculum (Factory/FORGE/GenPiH) / depth-only
  SBC (IndustReal) / manual multi-axis rungs (proxy form) /
  ADR (automatic multi-parameter); N6.1c IndustReal paper
  thresholds 80/10 vs released-code 75/50; N6.2 literature-copied
  ranges vs cell-derived; 0.6-mm fixture variant as eval-only vs
  mandatory first stage vs optional pre-stage.
- Decision (user, grill session 2026-08-26):
  (1) N6.1a: Train FRESH — no warm start from the proxy
  checkpoint. Checkpoint widening is not needed; the Block-3
  "keeps warm start" rationale loses that one leg (layout adoption
  stands on its other grounds); the yaw-channel reinterpretation
  (cos 4phi -> cos phi) is defused.
  (2) N6.1b: Curriculum YES, as MANUAL MULTI-AXIS rungs: start
  height above the opening + fixture offset + tilt widened
  together per rung (adoption of the proxy ladder machinery per
  D-042; simultaneity precedent = the proxy itself, published
  relatives Inoue 2017 / Florensa 2017; ADR rejected as
  implementation cost beyond bachelor budget, D-042 custom-code
  rule). Hard constraint: widen, never shift (IndustReal
  ablation).
  (3) N6.1c: Advance rule in the IndustReal PAPER form: advance
  at >80 % success, retreat below 10 %; retreat for a manual
  ladder = one rung back, re-widen with HALF the step. The
  paper-vs-code discrepancy (0.75/0.5) is stated as a footnote.
  Rung step sizes come from the project's own learning curves,
  not from IndustReal's 5 mm (different ladder variable).
  [CORRECTION 2026-09-13 (audit): points (2) and (3) -- the manual
  multi-axis ladder and the 80/10 advance rule -- are SUPERSEDED by AutoDR
  since 2026-09-11: D-178 (lateral disk, tilt, yaw), D-179 (start height),
  D-181 (friction). `curriculum_enabled` is False. The ADR rejection in (2)
  ("implementation cost beyond bachelor budget") no longer holds;
  `autodr.py` exists.]
  (4) N6.2 end ranges (final rung): derived from the real cell
  (fixture mounting, suction-grasp scatter) via CAD/measurement,
  `[CAD pending]` / `[verify on site]`; literature values remain
  order-of-magnitude anchors only.
  (5) N6.2 first rung: SMALL, symmetric around the target pose;
  concrete numbers only after the scripted-insertion gate. The
  copied insertion-package reset values (0.01 rad on five joints,
  0.7854 rad = +-45 deg on the yaw joint,
  `insertion_env_cfg.py:131-132`) are NOT adopted: the code's own
  comment derives +-45 deg from the proxy's C4 symmetry ("largest
  distinct range"), which the one-orientation real part does not
  have.
  (6) N6.2 geometry pre-stage: the user builds a SECOND fixture
  variant with ~0.6 mm play (user CAD task). It is a PLANNED,
  OPTIONAL stage 0: skipped if training at the 0.2876 mm sim play [CORRECTION 2026-09-11: the sim play is 0.5876 mm since D-121 -- already about the ~0.6 mm planned for this stage 0; what that means for stage 0 is NOT decided here]
  gets off the ground; if used, the stage switch follows the 80 %
  rule with weight inheritance + learning-rate decay to 10 %
  (Cheng 2026 C-KPM as the documented mitigation for the switch
  collapse). Headline success (Block 2) counts ONLY on 0.2876 mm [0.5876 mm since D-121];
  a used stage 0 derives its interpenetration filter from its own
  play.
- Rationale: Fresh training because the real task is a different
  task, not a proxy rung (new contour, ~7x tighter play, ONE valid
  yaw pose — two input channels change semantics; negative-transfer
  risk per Narvekar; IndustReal shift ablation). Manual multi-axis
  rungs satisfy the user requirement "existing methods only":
  proxy-verified machinery + published threshold FORM, near-zero
  code (rsl_rl `OnPolicyRunner.load()` verified for resume).
  Geometry pre-stage compromise keeps Cheng as the only published
  geometry-ladder precedent available WITHOUT committing to a
  shift-type curriculum unless the start proves infeasible.
- Sources: curriculum check
  `docs/reference/literature_check_curriculum_randomisierung_2026-08-26.md`
  (§§1, 8.1, 9; IndustReal ablation, Inoue, Florensa via Parnada
  §2, Cheng C-KPM); `source/insertion/.../insertion_env_cfg.py`
  lines 131-132, 229-231 (read 2026-08-26); D-042, D-044, D-053
  family (5 eval seeds), D-055, D-088; HANDOFF-KONZEPT geometry
  conflict entry 2026-08-26; user decisions in the grill session
  2026-08-26.

## D-111: Block 6 (Teil 2): N6.3 + N6.4 — keine Dynamik-DR, Obs-Rauschen-Form fixiert (Aktivierung an F1), 5 Trainings-Seeds
- Date: 2026-08-26
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: Continuation of the Block 6 grill (resume point N6.3 per
  HANDOFF-KONZEPT § Open). Closes Block 6: all four nodes decided.
  Basis: Block-6 draft report, curriculum check §8.9 (obs-noise
  candidate K3), Kirk criterion, IndustReal precedent.
- Options considered: N6.3-A dynamics DR (friction/mass/stiffness
  randomised) vs a single fixed value; N6.3-B observation noise
  per-step white noise vs per-episode constant offset vs none;
  N6.4 training seeds 3 (FORGE minimum) vs 5 (frozen eval
  protocol).
- Decision (user, grill session 2026-08-26):
  (1) N6.3-A: NO dynamics domain randomisation. Explicit written
  reason: the thesis has NO sim-to-real, so DR has no target.
  Documented as re-openable: "bei genug Zeit kann man das wieder
  aufnehmen" (time-permitting extension, ch. 6.2 outlook).
  (2) N6.3-A follow-up (user, same session): the SINGLE sim
  friction value is set to µ = 0.4 — the midpoint of the dry
  polymer-polymer literature span — INSTEAD of the sim default.
  Reason: both part and fixture are the DEHNvap housing material
  "Thermoplast, Farbe rot, UL 94 V-0" (datasheet 900 360); the
  datasheet does NOT name the polymer, public DEHN sources
  neither, and there is NO single plastic-on-plastic mean: dry
  polymer-polymer µ spans ~0.2–0.8 (PA66–PA66 ≈ 0.6, PPS–PPS ≈
  0.8; Wear 2008 polymer-polymer study, Unal/Mimaroglu polyamide
  study). Status: PROVISIONAL — supervisor question F3
  (Fragen_an_Betreuer) asks whether the midpoint is acceptable;
  upgrade path: DEHN/supplier names the polymer → matching
  literature value replaces 0.4.
  [Corrected 2026-09-11 by D-181: two DIFFERENT thermoplastics are now
  assumed (PA6 part, POM fixture); band 0.08-0.20, centre 0.14, from the
  DuPont PA66/POM tables. Point (1) had already fallen on 2026-09-10,
  inbox entry "Contact friction is randomised PER RESET".]
  (3) N6.3-B (K3): observation-noise FORM fixed now, ACTIVATION
  tied to the supervisor's F1 answer. Form: per-EPISODE constant
  offset on the pose channels (models a calibration/localisation
  error, not sensor jitter); amplitude is a config parameter;
  literature anchor IndustReal ±1 mm — larger than the 0.2876 mm [0.5876 mm since D-121; still larger]
  sim play, hence a real perturbation.
  [Corrected 2026-09-11 by D-182: switched ON before F1 is answered
  (user); pocket-position bias sigma 2.5 mm per episode plus force sigma
  3.5 N per step, through Isaac Lab's `observation_noise_model`.]
  (4) N6.4: 5 training seeds, fixed in the training config;
  per-seed reporting plus mean and spread (Henderson). Fallback 3
  (FORGE minimum) ONLY if the training PC cannot afford 5 runs
  per experiment — fallback not triggered, budget unverified.
- Rationale: Kirk et al.: randomise only what the evaluation
  varies — our robustness eval varies geometry, not friction;
  IndustReal uses dynamics DR only for real transfer; the proxy
  reached ≥99 % without it. Obs-noise split (form now / switch
  later) keeps the concept complete while F1 is pending. 5 seeds
  keeps ONE seed count across training and the frozen 5-seed eval
  protocol (D-053 family).
- Sources: curriculum check
  `docs/reference/literature_check_curriculum_randomisierung_2026-08-26.md`
  (§8.9, Kirk, IndustReal); DEHNvap datasheet 900 360
  (`docs/reference/Datenblaetter/`, "Gehäusewerkstoff" row); Wear
  2008 polymer–polymer sliding study
  (sciencedirect S0043164807001949) + Unal/Mimaroglu polyamide
  counterpart study (searched 2026-08-26, abstract-level values);
  Henderson et al. (per-seed reporting); D-053 family; user
  decisions in the grill session 2026-08-26.

## D-112: Eskalationspfad F/T-Kanal: erst ohne, F/T nur als Rückfallebene
- SUPERSEDED 2026-08-27 by the entry "D-089 is OVERTURNED" at the
  end of this file. The staged path is not waited out; the F/T
  channel is added now by user override. Kept for the record.
- Date: 2026-08-26
- Status: superseded by D-114
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: While filing the Parnada review (2026-08-26) the user
  stated: "ich glaube wir werden den F/T Sensor tatsächlich
  einsetzen müssen. Aber wir versuchen es erstmal ohne." The review
  calls F/T + proprioception the field's minimal sensor composition
  and shows the classic hole search presupposes force measurement
  (`docs/reference/literature_check_parnada_review_2026-08-26.md`
  §§1, 6). D-089 (frozen) decided NO F/T channel, with a reopening
  condition (jamming evidence) and a fixed route (FORGE wrench
  pattern, not ContactSensor).
- Options considered: add the F/T channel now (pre-empts D-089
  without new evidence) / never add it (ignores the user's stated
  expectation) / staged escalation path.
- Decision: Staged escalation path. (1) Training starts WITHOUT an
  F/T channel — D-089 stands untouched. (2) F/T is the named
  fallback level, to be considered ONLY if the Block-2 success
  criterion is not reachable without it AND D-089's reopening
  condition (jamming evidence from training runs) is met. (3)
  Activating the fallback is a separate explicit decision (grill
  session against D-089's evidence, incl. the Factory ablation
  −35 %), not an automatic step.
- Rationale: Keeps the frozen decision intact while recording the
  user's expectation as a plan, not a hunch. The escalation trigger
  reuses D-089's own reopening condition, so there is exactly one
  gate and one home for it. The literature counter-voice (review
  §6) is filed where it belongs — as ch.-6.2 discussion material,
  not as a reopening reason (it describes real-robot practice,
  where force substitutes for missing ground truth; our sim has
  ground truth).
- Sources: user statement 2026-08-26; DECISIONS.md D-089 (branch
  p2-szene-umbau);
  `docs/reference/literature_check_parnada_review_2026-08-26.md`
  §§1, 6; Factory ablation via D-089.

## D-113: Block 7: N7.1–N7.4 — no success termination, success-and-hold, 256 steps, two exits only, success rate AND abort rate reported separately
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: Block-7 grill. Basis: the termination check
  (`docs/reference/literature_check_termination_2026-08-26.md`,
  Rounds 1+2), the new abort-reporting check
  (`docs/reference/literature_check_abort_reporting_2026-08-27.md`),
  and the Block-7 draft report. The tree was extended at session
  start by N7.1b (success latch), N7.2b (timeout channel), N7.3b
  (force source), N7.4b (abort in evaluation), N7.4c (position abort,
  noted only). The force half of N7.3 has its own entry above
  ("D-089 is OVERTURNED") and is not repeated here.
- Options considered: N7.1 terminate on success (Inoue, Vecerik,
  Factory 2022 Screw subpolicy) vs run to timeout (Isaac Lab
  factory/forge/AutoMate); success counted at any step vs at the last
  step; N7.2 proxy 240 @ 60 Hz vs IndustReal 256 @ 60 Hz vs Isaac Lab
  Factory 150 @ 15 Hz vs derive from measured insertion time; N7.2b
  time-limited (Pardo time-awareness, remaining time in the
  observation) vs time-unlimited (bootstrap at timeout); N7.4 add
  workspace / joint-limit / stuck aborts vs none; reporting one
  success number vs two success numbers vs success rate plus abort
  rate.
- Decision (user, grill session 2026-08-27):
  (1) **N7.1: no termination on success.** The episode runs to the
  timeout. Follows the Isaac Lab factory/forge/AutoMate pattern
  (lockstep reset); the terminate-on-success precedents (Inoue 2017,
  Vecerik 2019) come from the real-robot line, whose motive is
  expensive robot time and does not apply to GPU-parallel sim.
  Already fixed in Block 5 point (7); restated here as the Block-7
  home of the node.
  (2) **N7.1b: success-and-hold.** An episode counts as a success
  only if the part is in the success region (D-052 depth band +
  interpenetration filter) at the LAST step. IndustReal form.
  Matches the per-timestep bonus without a latch (Block 5 point (5)).
  Named consequence, unchanged from Block 5: the per-step success
  bonus is a hidden time reward.
  (3) **N7.2: 256 steps @ 60 Hz = 4.27 s.** IndustReal's value; the
  proxy's verified 240 steps @ 60 Hz sits next to it. ADOPTED, not
  derived — the start height over the opening is still
  `[CAD pending]` (Block 6). CHECK OBLIGATION: re-measure against the
  scripted-insertion gate and correct there if the travel needs more
  steps.
  [CORRECTION 2026-09-13 (audit): 256 steps were adopted at 60 Hz =
  4.27 s. Under the OSC default (D-177, policy rate 15 Hz) the same 256 steps
  last 17.07 s; the re-derivation is OWED, as D-177 records. Point (1) fell
  on 2026-09-01 (success terminates; inbox "Reward-Ueberarbeitung" (7)) and
  the force abort is `truncated` since D-164.]
  (4) **N7.2b: the task is time-unlimited in Pardo's sense.** D-051
  makes the success rate the only acceptance criterion — there is no
  time criterion, so the time limit is a training device, not part of
  the task. Consequences: timeout goes into `truncated`, and the
  stack bootstraps it by itself (Isaac Lab wrapper sets
  `extras["time_outs"] = truncated`; rsl_rl `ppo.py::
  process_env_step` adds `gamma*V(s)` back). The force-abort goes
  into `terminated` (a true terminal, CaT form). NO remaining-time
  channel in the observation — Pardo prescribes time-awareness only
  for the time-limited regime, which we are not in.
  (5) **N7.3b: the abort reads the wrench**, and the same 3 force
  components are now also in the observation (see the D-089 entry).
  (6) **N7.4: NO other failure terminations.** Exactly two exits:
  timeout (`truncated`) and force-abort (`terminated`). No workspace,
  joint-limit, stuck or divergence abort. Precedent: the NVIDIA
  assembly tasks define none at all; Isaac Lab's library terms
  (joint limits, orientation, root height, illegal contact) are used
  by locomotion tasks, not by assembly. "Part dropped" is impossible
  under D-041 (rigid chain) — stated explicitly in ch. 4.5, no source
  needed (model geometry).
  (7) **N7.4b: the force-abort runs in the D-053 evaluation too.** An
  aborted episode counts as a failure.
  (8) **N7.4c: position abort NOTED, NOT adopted.** Precedents exist
  (Inoue 2017 >10 mm from goal with reward -1; Vecerik 2019 workspace
  and end-effector rotation bounds). Kept as a named option if the
  force-abort proves insufficient.
  (9) **Reporting (Block-7 exit check, point 3): TWO separate
  numbers** — the success rate (aborts count as failures; D-051's
  headline) AND the abort rate (share of episodes that ended at
  `F_max`). NOT two different success rates, and force does NOT enter
  the success definition. Form: Neunert et al. 2020 Table 2, which
  reports "early episode terminations" and "successful episodes" as
  separate rows over 100 episodes; the pair also exposes the failures
  that did not hit the limit. Backed by the safe-RL convention of
  keeping task and safety metrics decoupled (Safety Gym;
  SafeVLA-Bench).
- Block exit — consistency check against Block 2, all four points:
  (a) D-051/D-052 success vs the per-step bonus: no contradiction;
  the headline measures "seated at the end", the reward pays "seated
  early and long". Intended, and named.
  (b) No time criterion (D-051) vs the 256-step cap: no
  contradiction — the cap is a training device, which is exactly why
  it is `truncated`.
  (c) Force is not a success criterion (D-051) vs the abort running
  in the evaluation: resolved by (9). Force bounds the admissible
  ROUTE, it does not define success; the two numbers keep D-051
  literally intact.
  (d) N5.4 reward-vs-criteria: the force penalty can make the policy
  avoid the contact insertion needs. FORGE mitigates by conditioning
  on F_th; we do not. OPEN watch item for the first runs, carried
  over from Block 5.
- Open after this block: `F_max` in Newton `[measure]`; the magnitude
  of the fixed negative reward on abort `[open]`, decided with
  `F_max`; the 256-step check obligation at the scripted-insertion
  gate; the N5.4 watch item (d).
- Sources: `docs/reference/literature_check_termination_2026-08-26.md`
  (§1-§10 lineage and stack facts, §7 Pardo, §8 rsl_rl bootstrap,
  §9 CaT, §10 Isaac Lab termination library, §11-§16 Round 2);
  `docs/reference/literature_check_abort_reporting_2026-08-27.md`
  (Neunert Table 2, Safety Gym, SafeVLA-Bench, DSL-Assembly negative
  finding); Neunert et al. 2020 `arXiv:2001.00449` (CoRL 2019);
  Pardo et al. 2018 `arXiv:1712.00378`; Chane-Sane et al. 2024
  `arXiv:2403.18765`; Isaac Lab 2.3 `factory_env.py`,
  `isaaclab_rl/rsl_rl/vecenv_wrapper.py`, rsl_rl `algorithms/ppo.py`;
  old repo `proxytask_env.py:697-724`, `proxytask_env_cfg.py:84-89`;
  D-041, D-051, D-052, D-053; user decisions in the grill session
  2026-08-27.

## D-114: D-089 is OVERTURNED: the force channel enters the observation now, plus a FORGE force penalty in the reward
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: Block-7 grill (N7.3, force-abort). The termination
  literature check
  (`docs/reference/literature_check_termination_2026-08-26.md`,
  Round 2) established a gap: NO surveyed work terminates an episode
  on a quantity the policy does not observe. Block 5 point (2) had
  chosen exactly that — force-abort with force unobserved (D-089).
  Confronted with the gap, the user decided to close it from the
  observation side rather than drop the abort. Supersedes D-089 (no
  F/T channel) and Block 5 point (2) (no force penalty). Also
  supersedes the inbox entry "Eskalationspfad F/T-Kanal: erst ohne,
  F/T nur als Rückfallebene" (2026-08-26) — its staged plan is
  executed now instead of on evidence.
- Options considered: (a) keep Block 5 as is — force-abort, force
  unobserved, no precedent, own reasoning flagged; (b) put force into
  the observation, making the abort literature-conform; (c) drop the
  force-abort, force stays diagnostic only (fully conform, but
  removes the only anti-ramming mechanism Block 5 has); (d) replace
  the force-abort by a POSITION abort on an observed quantity
  (Inoue 2017 >10 mm from goal, reward -1; Vecerik 2019 workspace +
  orientation aborts). On the reward half: force-abort alone
  (CaT 2024, termination instead of penalty) vs abort + penalty
  (FORGE observes and penalises; Beltran-Hernandez 2020 aborts at
  F_max AND pays -50).
- Decision (user, grill session 2026-08-27):
  (1) **(b)**: the force channel enters the observation. D-089's own
  reopening ROUTE is used unchanged, so nothing is invented:
  `ArticulationData.body_incoming_joint_wrench_b` (6-D wrench,
  present in Isaac Lab 2.3.2) at a dedicated force-sensor link,
  EMA-smoothed (0.25), noise-injected, and **only the 3 force
  components** handed to the policy — the FORGE pattern,
  `isaaclab_tasks/direct/forge/forge_env.py:95`. NOT `ContactSensor`
  (normal forces only, no torque field, global PhysX contact
  processing). Ready-made term: `mdp.observations.body_incoming_wrench`.
  Observation grows **25 -> 28 channels**. The Block-3 layout
  (part-anchored, cos phi / sin phi yaw) is otherwise untouched.
  [CORRECTED 2026-08-28 — the two names above are the SAME tensor, and
  its frame is the PARENT BODY frame, not the world frame. The decision
  content is untouched; only the frame label was never stated and the
  cited variable name is a misnomer. Home of the correction:
  `docs/decisions_inbox.md`, entry "The force wrench is in the parent
  body frame, not the world frame".]
  (2) The **force-abort stays** (Block 5 point (2), Block 7 N7.3) and
  is now backed by observed force, which removes the aliasing
  objection.
  (3) A force penalty enters the reward, in the
  **Beltran-Hernandez 2020 form, NOT the FORGE form** (revised later
  the same session, see the correction below): ONE threshold
  `F_max`; exceeding it terminates the episode AND pays a fixed
  negative reward. `F_max` is `[measure]`; the magnitude of the fixed
  negative reward is `[open]` (Beltran-Hernandez uses -50 on a
  different reward scale, so the number does not transfer).
  [CORRECTED 2026-09-02 — the number is **-10**, not -50.
  Beltran-Hernandez et al. 2020, arXiv:2003.00628 v3, Eq. (6):
  `kappa = {200 task completed; -10 safety violation; 0 otherwise}`.
  There is no -50 anywhere in the paper; the RA-L version of record is
  paywalled and was NOT checked. The decision content is untouched --
  only the cited magnitude was wrong, and it never transferred anyway.
  The same wrong attribution is inherited by
  `insertion_math.compute_rewards_insertion`'s docstring and must be
  fixed there too. Home of the correction:
  `docs/reference/literature_check_abort_return_forfeiture_2026-09-02.md`.]
  [REOPENED 2026-09-03 — the "NOT the FORGE form" half of this point is
  withdrawn: a graded per-step contact penalty in the FORGE form now sits
  BESIDE the abort. RT-138 measured the policy pressing up to the abort
  limit (force p50 305 N at F_max 300 N, 64 % aborts) because nothing
  below the limit priced force. Home:
  `docs/decisions_inbox.md`, entry "The contact force is priced per step
  ..." (2026-09-03). The abort and its payment are unchanged.]
  [CLOSED 2026-09-11 by D-184: the per-step contact penalty is removed; the
  abort and its payment stand alone again.]
  Reason the Block-5 rejection no longer holds: it rested on "D-089
  holds" — that leg is gone.
  **CORRECTION within the same session (2026-08-27):** the FORGE
  ramp `-beta*max(0, ||F||-F_th)` was first chosen here and then
  DROPPED. Reason: a ramp only works if its threshold sits BELOW the
  abort limit, i.e. two numbers — and no source anywhere runs a
  continuous penalty threshold plus a separate higher abort limit.
  FORGE penalises and never aborts; Schoettler's 10 N / 6 N are stop
  and retract, both parts of the abort, with no penalty at all;
  Beltran-Hernandez 2020 is the only work that observes force, aborts
  on it AND pays for it — and it uses ONE threshold. With one
  threshold a ramp degenerates into a one-off payment at the moment
  of abort, which IS the Beltran-Hernandez form. Chosen under the
  global rule "nothing new is invented"; the two-number variant would
  have been an own construction.
  (4) D-089's reopening condition (failed training run + jamming
  evidence) is **NOT** met. This is a deliberate user override, not a
  triggered escalation.
- Rationale: Under the global rule "nothing new is invented", option
  (a) was the only branch with zero precedent in either lineage. Two
  of the three force-handling precedents take observation AND penalty
  together (FORGE; Beltran-Hernandez 2020); CaT 2024 is the lone
  counter-voice for termination-instead-of-penalty.
- Counter-evidence, stated openly and NOT resolved: Factory
  (Narang et al. 2022, RSS, Table IV) is the direct measurement
  against this decision. At 0.104 mm ISO clearance — TIGHTER than
  this task's 0.2876 mm short axis [0.5876 mm since D-121; still tighter] — the observation-space ablation
  gives pose 0.7708, pose+velocity 0.7760, pose+velocity+**force**
  0.5026: adding force cost about 35 % relative success. IndustReal
  and AutoMate exclude F/T deliberately and still transfer to
  hardware ("no force-torque sensor is used"). D-041's rigid suction
  chain distorts contact forces, so the channel carries a distorted
  signal. These are the reasons D-089 existed; the decision is taken
  against them by explicit user choice and must be reported as such
  in ch. 4 and discussed in ch. 6.2.
- Named risk for the N5.4 consistency check: a force penalty can make
  the policy avoid the contact that insertion requires. FORGE
  mitigates this by CONDITIONING the policy on F_th; we do not. Watch
  this in the first runs.
- Abort payment, decided 2026-08-27: the FORM is a fixed negative
  reward on abort, and the MAGNITUDE stays `[open]` — it is decided
  together with `F_max` at the measurement. Rejected in the same
  step: paying NOTHING on abort (the abort's own loss of future
  reward as the whole signal, CaT 2024 form). Reason for the
  rejection: every insertion work that aborts AND documents its
  termination pays something on top — Beltran-Hernandez 2020 -50,
  Inoue 2017 -1; Schoettler 2020 (arXiv:1906.05841) documents no
  termination design at all (negative finding); CaT is locomotion,
  not assembly. Also rejected: pricing the abort at the episode's
  remaining reward (the proxy's `_last_remaining` trick,
  `proxytask_env.py:713-716`). It does not transfer — the proxy's
  terms were NEGATIVE, so paying the remainder made leaving NEUTRAL;
  our Block-5 terms are positive, so an abort already forfeits the
  remainder and the success bonus, and neutralising would be the
  opposite of the intent.
- Open: `F_max` in Newton is `[measure]`. The magnitude of the fixed
  negative reward is `[open]` — it must fit OUR reward scale (Block-5
  terms are order 1.0 per step), so Beltran-Hernandez's -50 cannot be
  copied as a number, only as a form; Inoue's -1 was considered and
  left unchosen because our check does not record Inoue's own reward
  scale.
- Code-stream consequences (do NOT change here — territory of
  `p2-szene-umbau`): observation dimension 25 -> 28 in
  `insertion_env_cfg.py`; a dedicated force-sensor link on the UR5e;
  `activate_contact_sensors=True` (`ur5e_cfg.py` L178) is still NOT
  the chosen route.
- Sources: DECISIONS.md D-089 (branch p2-szene-umbau, incl. its
  fixed reopening route and the Factory ablation);
  `docs/reference/literature_check_termination_2026-08-26.md`
  (§3, §5, §6, §8, §9, §11-§16, Round-2 synthesis);
  `docs/reference/literature_check_force_criterion_2026-08-16.md`
  (Beltran-Hernandez 2020); `docs/reference/literature_check_reward_2026-08-26.md`;
  Narang et al. 2022 `arXiv:2205.03532` (Table IV);
  Noseworthy et al. 2024 `arXiv:2408.04587` (FORGE);
  Chane-Sane et al. 2024 `arXiv:2403.18765` (CaT);
  Isaac Lab 2.3 `forge_env.py:95`, `forge_tasks_cfg.py`;
  user decisions in the grill session 2026-08-27.

## D-115: Block 8 / N8.1: Simulation stack is Isaac Lab 2.3.2 on Isaac Sim 5.1.0-rc.19, PhysX with the TGS default, env count as a rule instead of a number
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- **CORRECTED 2026-08-28 by D-127:** this entry states that the scene
  authors NO solver iteration counts and leaves the effective value
  `[verify on site]`. The scene stream measured it: the fixture DOES
  author them. Read D-127 for the measured state; the sentence here is
  superseded, not the rest of the entry.
- Stream: konzept
- Context: Chapter 4.4 needs a justified stack description. The Block-8
  draft report had pre-formulated the entry (anchor risk); this grill
  session re-opened it. The draft also carried a hard "4096 envs" that
  no repo source backs for the NEW geometry.
- Options considered: (a) name the stack only, leave every physics
  number to the code stream; (b) N8.1 owns the physics numbers as
  read-off values with a source per line; (c) write "4096 envs" as a
  fixed number vs. write the rule that produces it.
- Decision: (b) + the rule form. Chapter 4.4 states: Isaac Lab
  **2.3.2** on Isaac Sim **5.1.0-rc.19+release.26219.9c81211b.gl**
  (a release candidate — say so), rsl_rl **3.0.1**; PhysX with
  `solver_type = 1` (TGS); GPU simulation via `SimulationCfg.device`.
  Physics values are given as READ-OFF values with their file and line,
  never as new decisions. Env count is written as a rule: "as many
  parallel environments as GPU memory allows; target 4096 (ran in the
  pre-study on the same RTX 3080, comment `insertion_env_cfg.py:170`);
  the final value is measured before the main run with the new
  geometry and reported in chapter 5."
- Citation rule: the Isaac Lab v2.3.0 docs prescribe the **Orbit paper**
  (Mittal et al., IEEE RA-L 8(6):3740-3747, 2023) — Isaac Lab has no
  peer-reviewed paper of its own. Cite Orbit + the version-pinned
  documentation. NEVER attribute "TGS" to the Factory paper; that text
  says "Gauss-Seidel". The TGS naming comes from the Isaac Lab v2.3.0
  API docs.
- Rationale: three pillars carry the stack — GPU-pipeline speedup
  (Makoviychuk et al. 2021, NeurIPS D&B: "2-3 orders of magnitude";
  Rudin et al. 2021, CoRL: locomotion in minutes on one workstation
  GPU), PhysX/SDF as the established method of the insertion lineage
  (collision check + Factory), and continuity with the pre-study.
  Against the hard 4096: the number comes from the proxy scene WITHOUT
  SDF collision and WITHOUT the Factory GPU buffers; the code default
  today is `num_envs=128` (`insertion_env_cfg.py:293`). Counter-data
  found in this session: Isaac Lab's own Factory config trains with
  `num_actors: 128`, so "many envs" is not established practice for
  insertion. Chapter 4 is concept, chapter 5 is measurement — the rule
  belongs in 4, the number in 5.
- Read-off physics values (2026-08-27): `dt = 1/120`
  (`insertion_env_cfg.py:182`), `decimation = 2` -> 60 Hz
  (`insertion_env_cfg.py:89`, adopted from proxy D-024),
  `solver_type = 1` (TGS, Isaac Lab `sim/simulation_cfg.py:37`),
  `gpu_max_rigid_contact_count = 2**23` (`:153`),
  `gpu_max_rigid_patch_count = 5*2**15` (`:156`),
  `gpu_heap_capacity = 2**26` (`:178`),
  `gpu_max_num_partitions = 8` (`:185`).
  `solver_position_iteration_count` and
  `solver_velocity_iteration_count` are **`None`**
  (Isaac Lab `sim/schemas/schemas_cfg.py:31,34,107,110,357`) and our
  scene sets neither. `None` means Isaac Lab writes nothing; the value
  comes from the USD asset, otherwise from the PhysX SDK default. The
  effective number for our scene is therefore UNKNOWN -> `[verify on
  site]`. Not estimated.
- Open / honest points for the report: (1) our scene sets NONE of the
  seven physics values that Isaac Lab's Factory sets (see the Block-8
  report table); (2) Factory paper (16 position iterations at
  dt = 1/60) and Isaac Lab Factory code (192 at dt = 1/120) disagree
  strongly — state both, do not average; (3) the supervisor raised the
  interpenetration risk orally and it was nowhere in writing before
  this session; (4) two v2.3.0 doc fetches may have rendered
  develop-branch banners — re-check the benchmark table and the
  four-library list with the version picker set to 2.3 before citing;
  (5) Rudin 2021 env count and GPU model are not in the abstract.
- Sources: `docs/reference/literature_check_stack_rl_library_2026-08-26.md`
  (S1-S4); Isaac Lab v2.3.0 documentation (version-pinned URLs);
  Makoviychuk et al. 2021 (NeurIPS 2021 Datasets and Benchmarks,
  arXiv:2108.10470); Rudin et al. 2021 (CoRL, PMLR 164:91-100);
  Mittal et al. 2023 (IEEE RA-L 8(6):3740-3747, arXiv:2301.04195);
  Narang et al. 2022 `arXiv:2205.03532` (Sec. II-A3, III-B, Tab. II);
  local Isaac Lab 2.3.2 install (`simulation_cfg.py`,
  `schemas_cfg.py`, read off 2026-08-27); GitHub tag `v2.3.2`
  (`isaaclab_tasks/direct/factory/factory_env_cfg.py`,
  `factory_tasks_cfg.py`); repo file
  `source/insertion/insertion/tasks/direct/insertion/insertion_env_cfg.py`;
  user decisions in the grill session 2026-08-27.

## D-116: Block 8, N8.2: rsl_rl stays — and the rl-games precedent is weaker than its 3-of-3 count suggests
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-27, writer session on main (see "The 2026-08-27 numbering batch" above)
- Stream: konzept
- Context: The user noticed that Factory uses rl-games although the
  proxy task was built "after Factory's model", and asked whether
  rl-games is the Isaac Lab standard — because if it were, the
  rsl_rl choice would have no basis. Round 2 of the stack check had
  only established the narrower fact that the three insertion PAPERS
  use rl-games. A local census settled the wider question. NOTE: the
  RL library had never been decided in this project — the current
  `DECISIONS.md` (2225 lines, through D-089) contains ZERO
  occurrences of `rsl_rl` / `rl_games`; rsl_rl arrived via the Isaac
  Lab template and travelled with the proxy. This entry therefore
  overturns nothing; it creates the missing record. N8.1 (stack
  entry, ch. 4.4) is NOT covered here and is still open.
- Options considered: (a) keep rsl_rl; (b) switch to rl-games to
  match Factory/IndustReal/AutoMate; (c) keep rsl_rl but adopt
  Factory's ARCHITECTURE instead (LSTM + asymmetric critic), which
  rsl_rl also supports. skrl and SB3 were on the original node list
  and stay rejected (skrl: breadth the task does not need; SB3: no
  vectorised training in the Isaac Lab integration, slowest
  benchmark).
- Decision (user, 2026-08-27, after reading the v2.3.2 docs page):
  **rsl_rl stays.** Grounds, in this order and WITHOUT a superiority
  claim (D-040 requires none):
  (1) rl-games is NOT the Isaac Lab standard. Census of the local
  checkout `C:\IsaacLab` at v2.3.2: rsl_rl ships in **37 of 48**
  task dirs, rl-games in 28. By `gym.register` kwargs over all 175
  registrations: skrl 97, rsl_rl 96, rl-games 62, SB3 10 — rl-games
  is THIRD and appears on 35 % of registrations. No document in
  `docs/` names any library as default or recommended; the official
  tutorial uses SB3, and where a default IS set it is rsl_rl
  (`tools/run_train_envs.py:33`, and the post-install verification
  commands).
  (2) The precedent is one inherited decision, not three. All three
  insertion works run on **Isaac Gym**, not Isaac Lab; IndustReal is
  built "within the Factory simulation framework", AutoMate on
  IndustReal. And **none of the three gives a reason** for the
  library — it is named, never argued.
  (3) Benchmark parity: 198 s (rsl_rl) vs 201 s (rl-games) on
  Isaac-Humanoid-v0, 4096 envs, RTX 4090 — a 1.5 % gap, not a
  criterion.
  (4) Pre-study continuity (D-042): the proxy reached >=99 % on
  rsl_rl; pipeline, hyperparameters and scripts carry over.
  (5) D-054 baseline (2) is the proxy policy zero-shot, and that
  checkpoint is rsl_rl. A switch would need BOTH libraries at
  evaluation time — there is no checkpoint converter anywhere.
  (6) The switch cost is real and one-sided: the environment side is
  free (our `_get_observations` already returns `{"policy": obs}`,
  which is what `RlGamesVecEnvWrapper` requires), but
  `--stop-at-success-rate` would have to be rebuilt as an
  `AlgoObserver` because rl-games has no chunked-learn API, legacy
  `gym` comes in beside `gymnasium`, `max_epochs` is not the same
  unit as `max_iterations`, and all checkpoints are lost.
  Option (c) is NOT decided here and is explicitly left open for the
  code stream: Factory's real edge over our config is architectural
  (LSTM 2x1024 + `central_value_config`) and rsl_rl exposes both
  (`RslRlPpoActorCriticRecurrentCfg`, `obs_groups`). What is not
  free in either library is the privileged state tensor itself
  (Factory's is 72-dim).
- Rationale: The question was decided on a census rather than on the
  paper count, because the paper count measures one lineage and not
  the framework we actually use. D-040 removes any duty to show
  rsl_rl is better; the duty is to show the choice is defensible and
  its costs are named.
- Costs, named and NOT resolved: (i) **PBT is unavailable** — Isaac
  Lab verbatim "PBT is currently supported only with the rl_games
  library", and the Ray `JobCfg` is rl-games-only too.
  **CORRECTED 2026-08-28 by D-119:** the clause that once stood here —
  "Optuna does work with rsl_rl (`scripts/reinforcement_learning/ray/
  tuner.py`), which is the mitigation for the Block-5 sensitivity
  table" — is WITHDRAWN. Isaac Lab documents its Ray path as "tested
  only on Linux" (`ray.rst:17`) and `ray/util.py:239` calls
  `select.select()` on a pipe, which fails on Windows. Status
  `[verify on site]`; no plan may rest on it until one local run has
  started on the training PC. PBT stays unavailable either way.
  (ii) **No value normalization.** `RslRlPpoAlgorithmCfg` has 16
  fields and no such switch. Two counter-examples say we do not need
  it: our own proxy trained under rsl_rl at returns of roughly
  [-40, +15] — sign-flipping, with a ±34-point terminal
  discontinuity — and every documented failure there was a
  reward-shape exploit, never a value-scale problem; and Isaac Lab's
  own `manager_based/manipulation/lift` runs rsl_rl at returns in
  the thousands. Both are evidence, not proof; if a training run
  ever shows critic divergence, this is the entry to reopen.
  (iii) **CORRECTED 2026-08-27 (Block-8 grill session, verified
  against the installed rsl_rl **3.0.1** and the GitHub tag
  `v3.0.1`):** the claim "rsl_rl checks only `isnan`, not `isinf`"
  is WRONG for our version. `check_nan` does not exist;
  `rsl_rl/utils/utils.py` defines seven functions and uses none of
  `isnan`, `isinf`, `nan_to_num`, `clamp`; `algorithms/ppo.py`,
  `runners/on_policy_runner.py` and `storage/rollout_storage.py`
  contain no NaN/inf check and no reward clamp either. The correct
  statement is STRONGER: rsl_rl 3.0.1 has NO NaN/inf check at all.
  Its only numerical guards are gradient-norm clipping, value
  clipping and advantage standardisation. Block 5's squashing
  function therefore does not merely remove one `inf` source — it is
  the ONLY thing keeping the reward finite.
  (iv) Two further Factory settings have no rsl_rl equivalent
  (found 2026-08-27, additional to the value normalisation in (ii)):
  `bounds_loss_coef: 0.0001` and `mixed_precision: True` from
  `direct/factory/agents/rl_games_ppo_cfg.yaml`. `mixed_precision`
  is speed only; the bounds loss is largely redundant because the
  environment clips actions anyway. Neither changes the decision;
  both are named so the chapter does not claim parity it does not
  have.
- Open / not decided here: N8.1 (stack entry, ch. 4.4) is
  untouched. Whether PBT would even fit the training-PC budget was
  never measured. Whether to adopt Factory's architecture (option c)
  belongs to the code stream.
- Sources: `docs/reference/literature_check_stack_rl_library_2026-08-26.md`
  Round 3, S10–S17 (census, docs negative finding, switch-cost
  inventory, value-normalization evidence, "why rl-games", S1
  re-check); local checkout `C:\IsaacLab` at git
  `37ddf62687 Bumps version to v2.3.2`
  (`docs/source/overview/reinforcement-learning/rl_frameworks.rst`,
  `docs/source/features/population_based_training.rst:90`,
  `tools/run_train_envs.py:33`,
  `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:75-129`,
  `source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py:124-132`);
  D-040 (no superiority claim), D-042 (pre-study adoption), D-054
  (baselines); Factory §V-A, IndustReal §IV-B, AutoMate App. F;
  user decision 2026-08-27.


## D-117: Block 9: hyperparameter study — gate, scope, tool and budget (N9.0-N9.2)
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: konzept
- Context: Block 9, nodes N9.0, N9.1, N9.1b, N9.1c, N9.2, N9.2b, N9.2c.
  The work plan for the training month names a hyperparameter study as
  its own strand and no concept block covered it. Block 8 left three
  side notes only: no PBT on one GPU, `ray/tuner.py` named as viable,
  and the Factory hyperparameter file targets a different geometry.
  The tree draft of 2026-08-28 was confirmed at session start and
  extended by N9.0, N9.1b, N9.1c, N9.2b and N9.2c.
- Options considered:
  (N9.0) study before the main run / study as a fixed block regardless
  of the outcome / gated study that starts only after a first run has
  been evaluated.
  (N9.1) no study at all (pure D-042 adoption) / a small targeted set /
  a defined Optuna search space.
  (N9.2b) `ray/tuner.py` locally / Optuna as a plain library around our
  own `train.py` / hand-guided sweep, one knob at a time / grid.
  (N9.2) budget as a fixed number of GPU hours / budget as a rule with
  a measurement gate and a pre-declared shortening order.
- Decision:
  (a) **N9.0 — gated.** The study starts only AFTER the first full
  training run with the current configuration has been run and
  evaluated. No invented threshold is used as the gate: the first run
  IS the gate. An earlier proposal — a self-built "one standard
  deviation above the random-policy baseline" rule — was withdrawn
  during the grilling because it had no source.
  (b) **N9.1 — four knobs, plus two disputed inherited values.**
  Varied, in this order: `learning_rate`, `num_steps_per_env`,
  `init_noise_std`, `gamma`. Checked separately as one on/off
  comparison each: `use_clipped_value_loss` and `entropy_coef`. Value
  normalisation is NOT a study variable — rsl_rl 3.0.1 does not have
  it; it stays the Block-8 watch item.
  (c) **N9.1b — forbidden list.** The study never varies what another
  block owns: `action_scale` (Block 4), reward weights and kernel
  widths (Block 5), curriculum rungs and friction (Block 6), the
  256-step episode (Block 7), the environment count (Block 8).
  `max_iterations` is held FIXED across the study; its value comes
  from the first runs and is `[measure]` until then.
  (d) **N9.1c — reference configuration** ("configuration A") is the
  current `rsl_rl_ppo_cfg.py` adopted from the pre-study, i.e. the
  configuration the first main run uses. No second candidate.
  (e) **N9.2b — hand-guided sweep, one knob at a time.** Optuna as a
  plain library around our own `train.py` is the NAMED upgrade path,
  not the plan; it depends on two unknowns (installability in the
  Isaac Sim Python environment, wall-clock time per run). Isaac Lab's
  `ray/tuner.py` enters no plan until a short `--run_mode local` test
  on the Windows training PC has actually started (see **D-119**).
  (f) **N9.2 — budget as a rule, not as hours.** At most **14 search
  runs** (4 knobs x 3 levels = 12, plus 2 on/off comparisons), **one
  seed each**. Wall-clock time per run is measured at the first main
  run. If the measurement shows 14 runs do not fit the window, the
  knob list is shortened FROM THE BACK — `gamma` first, then
  `init_noise_std` — following the importance ranking of
  Andrychowicz et al. 2021. The shortening order is fixed in advance,
  not decided mid-study.
  (g) **N9.2c — SAC placeholder. RESOLVED the same day**, see the
  entry **D-120** ("F4 answered: a methodological
  PPO-vs-SAC comparison, no SAC training run"). The supervisor's answer means no SAC run at
  all, so no SAC effort is added and the 14 search runs are not at
  risk from F4. The placeholder is withdrawn, not carried.
- Rationale:
  (a) The scene is not verified in simulation yet (`HANDOFF-SZENE.md`,
  NS.10 open, D-071). Tuning against a scene that solves nothing
  measures noise. The user's own framing removed the need for an
  invented gate number: run first with the pre-study values, look at
  the result, then tune. That is hand-tuning-first, which Eimer et al.
  2023 criticises — but our starting point is a configuration verified
  at >=99 % on the proxy, not a guess. This legitimacy question goes to
  the supervisor as F5.
  (b) There is NO in-family precedent for a search space: Factory used
  "a shared set of hyperparameters (Table IX)" for three subpolicies
  and IndustReal, AutoMate and FORGE inherited it. The ranking
  therefore comes from the only large-scale study of PPO knobs found,
  Andrychowicz et al. 2021 (ICLR, >250 000 agents): high importance =
  initial action noise, discount factor, learning rate, transitions per
  iteration; low = entropy coefficient, gradient clipping. The four
  chosen knobs are exactly its high-importance list, and three of them
  are the places where we deviate most from Factory (lr 1e-3 vs 1e-4,
  16 vs 128 steps per iteration, gamma 0.99 vs 0.995). The two extra
  comparisons exist because the same paper directly contradicts two
  values we inherited without ever deciding them: PPO-style value-loss
  clipping "hurts the performance regardless of the clipping
  threshold", and no regularizer (entropy included) helped.
  (c) Without the forbidden list the study becomes a back door for
  six decided values. Factory supports the separation: it treats
  observation space and controller as SEPARATE studies from the PPO
  parameters.
  (e) The hand-guided sweep needs zero new code and reuses
  `scripts/compare_runs.py`, which already ranks runs by
  `success_rate_recent`. Its named weakness is that it cannot find
  interactions between knobs, and Eimer et al. 2023 criticises exactly
  this practice — this is a budget-and-tooling choice, not a
  methodological preference, and the report says so.
  (f) One seed per search run is not thrift: Hertel/Baldi/Gillen 2020
  find that random search with ONE sample outperforms three or five
  repetitions at matched compute. The hours cannot be stated because
  the wall-clock time of one run at our environment count does not
  exist yet; the same rule-with-measurement form was already used in
  Block 8 for the environment count. The only comparable anchor from
  the literature is Factory's 1-1.5 h per 1024 policy updates for 3-4
  policies at 128 envs on ONE RTX 3090 — not transferable to 4096 envs.
- Sources: `docs/reference/literature_check_hyperparameter_study_2026-08-28.md`
  §1 (Factory, no search documented, hardware anchor), §4 (Hertel 2020,
  one seed beats repeats), §5 (Eimer 2023), §7 (`ray/tuner.py` on
  Windows), §8 (existing instruments), §10 (Andrychowicz 2021 ranking).
  Own code read: `scripts/rsl_rl/train.py`, `scripts/compare_runs.py`,
  proxy `rsl_rl_ppo_cfg.py`.



## D-118: Block 9: hyperparameter study — objective, seed separation, decision rule, reporting (N9.3-N9.4)
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: konzept
- Context: Block 9, nodes N9.3, N9.3b, N9.3c, N9.4, N9.4b. With the
  scope and budget settled, the study still needed an objective, a rule
  for when configuration A is beaten, and a reporting form. The draft
  premise of N9.3 — "the full D-053 evaluation per candidate is
  unaffordable, a reduced protocol is needed" — was REFUTED during the
  literature check and is not carried forward.
- Options considered:
  (N9.3b) success rate / training reward / a mixed objective.
  (N9.3c) tune on the D-053 evaluation grid / tune on the training
  distribution with separated seeds.
  (N9.3) mean comparison without a test / Welch's t-test /
  bootstrap confidence intervals.
  (N9.4) table / curves / both.
- Decision:
  (a) **N9.3b — objective is the success rate.** Specifically the
  trailing-window success rate the environment already writes to
  `demo_metrics.json`, read at the end of a search run. Tie-breaker:
  iterations to reach the target, which `train.py --stop-at-success-rate`
  already produces. Reward is logged but is NEVER an objective.
  (b) **N9.3c — separation.** The search runs on ONE fixed tuning seed,
  named in the report, which is NOT one of the five D-053 evaluation
  seeds. The study never tunes on the D-055 robustness ladder.
  (c) **Protective rule: the search may only propose, never decide.**
  The winner of the search is measured against configuration A under
  the full D-053 protocol (5 seeds x 1000 episodes) on fresh seeds. If
  it does not beat A there, **A stays**.
  (d) **N9.3 — decision rule** for that comparison: Welch's t-test over
  the five seed means at alpha = 0.05. No bootstrap. Not significant →
  A stays.
  (e) **N9.4b — the null result is pre-declared** and is reported as
  "not distinguishable at this seed count", never as "equally good".
  (f) **N9.4 — reporting in ch. 5** is two tables and one figure.
  Table 1, the search: up to 14 rows grouped by knob; columns knob,
  value, success rate, iterations to target; the caption states
  explicitly that each row is ONE seed. Table 2, the decision:
  configuration A vs. the winner, 5 seeds, mean and standard deviation
  per D-053. One figure: the learning curves of A and the winner
  overlaid. p-values do NOT appear in ch. 5.
  (g) **Boundary sentence (D-040):** the study is engineering tuning
  for this one task; it does not show that these values are better in
  general.
- Rationale:
  (a) D-051 makes the success rate the only acceptance criterion, and
  reward no criterion at all. The objective was initially assumed to be
  expensive; it is not — the environment already tracks the trailing
  success rate and `scripts/compare_runs.py` already ranks on it, so
  tuning on it costs zero extra runs. Tuning on the same quantity that
  is reported removes a translation step. Named weakness: the trailing
  training success rate is a PROXY for the D-053 figure — it is
  measured on the training distribution with a policy still moving.
  `compare_runs.py` warns about this in its own docstring; the warning
  is weakened here because all study runs share one environment
  configuration and differ only in PPO knobs.
  (b)+(c) Eimer et al. 2023 (ICML) require the separation of tuning and
  testing seeds because the hyperparameter landscape can depend on the
  tuning seed. That collides with Hertel 2020, which says one seed per
  configuration is the best use of the budget. Both are right about
  different things — budget efficiency vs. generalisation of the chosen
  configuration. The protective rule resolves it: a one-seed lucky hit
  can cost 14 runs but cannot change the reported configuration. The
  worst case is a documented search that ends in "A stays", which is a
  valid outcome, not a failure.
  (d) Any threshold we invent ourselves is unsourced — an earlier
  "one standard deviation" proposal was withdrawn for that reason.
  Colas et al. 2018 provide the sourced procedure for comparing two
  configurations at few seeds: Welch's t-test, because it does not
  assume equal variances, and NOT the bootstrap test, whose type-I
  error runs at about 10 % against a nominal 5 % at N = 5.
  (e) Colas et al. 2018 also state that a non-significant result is not
  evidence of equality; in their worked example N = 5 leaves a 51 %
  chance of missing a real effect. The wording is fixed in advance so
  the outcome cannot be re-framed afterwards.
  (f) Factory's ablation tables are the in-family reporting form: one
  row per configuration, several columns, seed count named in the
  caption. Naming the seed count is what stops the 14 search rows being
  read as final results.
  INTERPRETATION, stated as such: D-053 governs how the RESULT is
  reported (mean over 5 seeds, standard deviation, no confidence
  intervals) and stays untouched. The Welch t-test governs an INTERNAL
  go/no-go between two configurations and does not appear in ch. 5.
  These are two different jobs.
  NAMED COUNTER-VOICE: Agarwal et al. 2021 (NeurIPS) recommend
  stratified bootstrap confidence intervals for exactly this kind of
  comparison, which contradicts D-053. D-053 is NOT reopened here; the
  tension belongs in the Block-9 report's critical section and, if it
  survives, in thesis ch. 6.2.
- Sources: `docs/reference/literature_check_hyperparameter_study_2026-08-28.md`
  §2 (in-family comparison protocol 3x1024 to 5x1000 — D-053 already IS
  the field standard, the "reduced protocol" premise is unsupported),
  §3 (Colas 2018), §5 (Eimer 2023), §6 (Agarwal 2021 tension), §8
  (`compare_runs.py`, `--stop-at-success-rate`), §9 (objective
  conflict). Own decisions: D-040, D-051, D-053, D-055.



## D-119: Correction: the Block-8 note "Optuna via ray/tuner.py is viable" does not hold on our machine
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: konzept
- Context: Block 8 (N8.2) named two costs of staying with rsl_rl: no
  PBT, and no `normalize_value`. It softened the PBT cost with the side
  note that Optuna is available via
  `scripts/reinforcement_learning/ray/tuner.py`. That note appears in
  the Block-8 report, in the Block-8 inbox entry and in
  `KONZEPTPHASE.md`. The evidence behind it was that the script exists.
  Block 9 needed the note to be load-bearing for its budget, so it was
  checked against the local Isaac Lab 2.3.2 install.
- Options considered: keep the note as it stands / drop it entirely /
  replace it with the checked state plus an on-site verification.
- Decision: **replace it.** The corrected statement is:
  Isaac Lab's Ray integration is documented as "experimental, and has
  been tested only on Linux" (`docs/source/features/ray.rst:17`); the
  included `JobCfg` supports only the rl_games workflow
  (`ray.rst:193,383`); and the local job runner calls
  `select.select()` on a subprocess pipe (`ray/util.py:239`), which on
  Windows accepts sockets only. The documented local quickstart is
  Docker-based. Against that: `--workflow` is overridable
  (`tuner.py:405`), the tuner reads plain TensorBoard scalars
  (`tuner.py:131`, `util.py:25`), and both Isaac Lab's rsl_rl
  `train.py` (lines 148, 153) and OUR fork (`scripts/rsl_rl/train.py`
  lines 169, 174) print the two lines the tuner greps for — our fork
  even carries a comment saying that line must not be changed for Ray
  Tune. So the incompatibility is with the operating system, not with
  the library. STATUS: `[verify on site]` — one short
  `--run_mode local` run on the Windows training PC decides it. Until
  that run exists, no budget and no plan may rest on `ray/tuner.py`.
  Also corrected: the tuner's own defaults are 100 configurations x 3
  repetitions (`tuner.py:432,438`) — 300 training runs, out of reach on
  one RTX 3080 — and its default objective is `rewards/time`
  (`tuner.py:423`), which contradicts D-051.
- Where the old wording stood: inside D-116, cost (i). FIXED
  2026-08-28 by the writer session on `main` — the clause is now
  marked WITHDRAWN there with a pointer to this entry. Also corrected in this stream's own files: the Block-8
  report (inline correction pointing here), `KONZEPTPHASE.md` (Block-8
  status line and the Block-9 tree) and `HANDOFF-KONZEPT.md`.
- Rationale: two sources disagreed (the Block-8 note vs. Isaac Lab's
  own documentation and code), and the project rule forbids resolving
  that by plausibility. The check went to the authoritative source, the
  installed code. The note is overwritten rather than appended to,
  because a decided value has exactly one home: the Block-8 report gets
  an inline correction pointing here, and the sentence must be replaced
  everywhere it occurs. The correction does NOT touch N8.2 itself —
  rsl_rl stays, argued as lower risk, and PBT stays a deliberate
  omission. Only the softening fallback loses its support.
- Sources: `docs/reference/literature_check_hyperparameter_study_2026-08-28.md`
  §7. Local install `C:/IsaacLab` at 2.3.2. Python `select` behaviour
  on Windows (standard library documentation: file objects are not
  acceptable, sockets are).



## D-120: F4 answered: a methodological PPO-vs-SAC comparison, no SAC training run
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: konzept
- Context: Supervisor question F4 asked whether the thesis must compare
  PPO against SAC, and in which of three costed forms (no comparison /
  a light untuned SAC run / a full fair comparison with its own tuning
  and seeds). The supervisor answered: **a methodological comparison is
  enough.** Reading confirmed with the user 2026-08-28: the comparison
  happens on the method level, in ch. 4.6, with arguments and
  literature — NOT as an experiment.
  This does not conflict with the professor. `Professor-Feedback.md`
  already carries both halves: the answer table says a PPO-vs-SAC
  demonstration is not required ("kein Forschungs-Nachweis nötig"),
  and consequence 4 requires that every concept decision be justified,
  naming "Warum PPO statt SAC" as an example. Justify in writing, do
  not demonstrate by experiment. Both voices agree; only the Block-0
  deletion of the node went further than either of them.
- Options considered: the three forms offered in F4 (no comparison /
  light empirical / full empirical), plus the form the supervisor
  actually named, which was not among them: a written comparison with
  no run at all.
- Decision:
  (a) **No SAC training run.** No second RL library is installed, no
  SAC seeds, no empirical comparison. F4 option 2 and option 3 are
  both rejected by the answer.
  (b) **Chapter 4.6 gains a methodological PPO-vs-SAC section.** It
  argues from properties and literature: on-policy vs. off-policy,
  what the insertion family uses, what our stack can run, and what
  the counter-evidence is. D-040 continues to hold — the section
  states why PPO was CHOSEN, never that PPO is superior.
  (c) **The Block-0 deletion of the "PPO vs. SAC" node is partly
  reversed.** Block 0 dropped the node completely on the user's
  decision of 2026-08-16. It comes back in the narrower written form
  only. The empirical comparison stays dropped. The Block-0 report
  gets an inline correction pointing here.
  (d) **N9.2c is resolved.** The SAC placeholder in the Block-9 budget
  is removed. The 14 search runs are no longer at risk from F4.
  (e) **F4 gets the BEANTWORTET pill** and a pointer to this entry;
  the open count in `Fragen_an_Betreuer.tex` goes from 5 to 4.
- Rationale: the answer is the cheapest of the three costed forms and
  costs no training time at all, which protects both the main result
  and the Block-9 study. The written form is also the only one that is
  defensible under D-040: an untuned SAC run presented as a loser was
  named in F4 itself as methodically attackable, and a fair comparison
  would have displaced either the study or the depth of the robustness
  evaluation. Point (c) is recorded explicitly because it changes a
  settled Block-0 decision; changing one silently is forbidden.
- Material already on file for the ch. 4.6 section, no new research
  needed: rsl_rl 3.0.1 offers PPO only, so SAC would change algorithm
  AND library at once (Block-8 report, N8.2); Factory, IndustReal,
  AutoMate and FORGE all use PPO
  (`literature_check_stack_rl_library_2026-08-26.md` S12); the
  documented counter-voice is Cheng 2026, where SAC exceeds 97 % and
  PPO stalls in the hard stages, with evidence strength LOW for us
  (own curriculum framework, cm scale, PyBullet, no seed count) —
  `literature_check_curriculum_randomisierung_2026-08-26.md` §9.
- Sources: supervisor answer (verbal, reported by the user
  2026-08-28); `docs/Zwischenbericht/Professor-Feedback.md` §3 answer
  table and §4 point 4; `docs/Zwischenbericht/Fragen_an_Betreuer.tex`
  F4; Block-0 report §"Knoten PPO vs. SAC gestrichen".


## D-121: The part body is 90.00 mm across, not 90.3 -- the cross play doubles to 0.5876 mm
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: Every bound in D-106 and every kernel width in D-109 divides by
  the cross play, and the cross play is `POCKET_OPENING_X` minus the part's
  body width across the short axis. The opening is CAD (90.5876 mm,
  `POCKET_WALL_X`, read out of the fixture STEP). The body width was 90.3 mm
  -- a caliper reading from the session of 2026-08-17 (D-057), never
  replaced by CAD, and never a named constant in `insertion_tasks_cfg.py`;
  it lived only in prose. The user stated on 2026-08-28 that the CAD says
  90 mm.
- Options considered: (a) accept the caliper 90.3 and keep 0.2876 mm;
  (b) derive the body width as `POCKET_OPENING_X - 0.2876 mm` -- rejected as
  circular, because D-094 produced the 0.2876 by subtracting 90.3 in the
  first place, so the inverse would launder a caliper reading into a CAD
  one; (c) read the body edge out of the part CAD the same way
  `POCKET_WALL_X` was read.
- Decision: (c), and it was carried out. **`PART_BODY_X = 0.0900` m
  (C: part CAD). `PLAY_X` = 0.5876 mm, not 0.2876 mm.** Both are now named
  constants with exactly one home each, `PLAY_X` derived rather than
  written. `PLAY_Y` is unaffected at 1.60 mm.
- Rationale: The measurement was read from `CAD/Step/fuegeteil_prt.stp`,
  all CARTESIAN_POINTs, `SI_UNIT(.MILLI.,.METRE.)` -- the identical method
  that produced `POCKET_WALL_X`. Two dense point planes at -45.0 (138
  points) and +45.0 (94 points). `Fuegeteil_Neuer.stp` and
  `CAD/fuegeteil.stp` return the same spans.

  The read carries its own CONTROL and that is what makes it evidence
  rather than a second opinion: on the LONG axis the same method returns
  -71.75 / +71.75, i.e. 143.50 mm, which is D-088's independently settled
  value. A method that reproduces the known axis exactly is trusted on the
  unknown one.

  This is the THIRD number from that one caliper session to be overturned
  by CAD -- length 143.34 -> 143.50 (D-088), height 50.7 -> 50.506 (D-073),
  width 90.3 -> 90.00 (here). D-075 already held the disproof and did not
  use it: it recorded the mesh spanning x = -51.359 .. +45.050 about a part
  centre at +0.05, which puts the body half-width at 45.00 and the body at
  90.00, and then wrote "the plus side matches the 90.3 body" anyway.

  CONSEQUENCES, computed but NOT decided here -- the derivations belong to
  D-106 and D-109, i.e. the concept stream. This entry supplies the input
  and lists what divides by it:

  | quantity | old (0.2876 mm) | new (0.5876 mm) |
  |---|---|---|
  | SAPU threshold, half the play | 0.144 mm | 0.2938 mm |
  | fine kernel width `a = arccosh(10)/margin` | 10408 | 5094 |
  | yaw bound, play / 143.5 | 0.115 deg | 0.235 deg |
  | across-tilt bound, play / 36 | 0.458 deg | 0.935 deg |
  | SDF resolution floor | 1314 | far below 1024 |

  The SDF floor matters in one direction only: at 1024 the voxel is
  0.1847 mm, which was COARSER than the old 0.144 mm filter and is FINER
  than the new one. The authored 2048 (D-062, RT-56) therefore stays valid
  -- but **its justification is gone**, and that is a decision, not a
  recomputation.

  Two code sites on branch `p2-rl-code` carry the old numbers as live check
  constants and must be corrected there, not here:
  `scripts/check_insertion_math.py:70` (`A_FINE = 10408.0`) and `:72`
  (`INTERPEN_THRESH = 0.000144`). The env code path is unaffected, because
  `insertion_math.py` owns no numbers by design.
- Sources: `CAD/Step/fuegeteil_prt.stp` (read 2026-08-28, with the 143.50
  control); user statement 2026-08-28 ("Im CAD ist das Teil 90 breit");
  D-057 (the superseded caliper session), D-073, D-088 (the two earlier
  corrections from the same session), D-075 (holds the unused disproof),
  D-094 (the circular derivation), D-106, D-109 (9)/(10).


## D-122: 1000 SDF sample points give 7 mm spacing on our part -- 12x the cross play
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: D-109 (3) names N = 1000 surface points via
  `trimesh.sample.sample_surface_even`, taken from IndustReal, and D-109
  (10c) warns without a number that 1000 points are "thin for a 143 mm
  part" and that too few points systematically UNDER-report penetration.
  Branch `p2-rl-code` lists this as one of three open questions blocking
  the Warp half of its SDF work (`HANDOFF-RL.md:324-331`). The warning was
  never quantified. It is arithmetic, so it was.
- Options considered: not a choice yet -- this entry supplies the number
  the choice needs.
- Decision: OPEN. The count is `p2-rl-code`'s to set; this is the input.
  **CLOSED 2026-08-30 by D-154: the count is 64,000.** D-154 also
  MEASURED the surface area this entry could only bound: 701.2 cm2, not the
  494.2 cm2 box, so every spacing above is 1.19x coarser than printed and
  "about 143,000 points" for spacing equal to the play is really about
  203,000.
- Rationale: Treating the part as a box from the CAD constants (90.00 x
  143.50 x 50.506 mm) gives 494 cm2 of surface. Even sampling puts the
  mean point spacing at `sqrt(area/N)`:

  | N | spacing | vs the 0.5876 mm cross play |
  |---|---|---|
  | 1000 | 7.03 mm | 12.0x |
  | 2000 | 4.97 mm | 8.5x |
  | 4000 | 3.52 mm | 6.0x |
  | 8000 | 2.49 mm | 4.2x |
  | 16000 | 1.76 mm | 3.0x |

  Spacing equal to the cross play would need about 143,000 points; half
  the play, about 572,000.

  The box is a LOWER bound on the area -- the real part carries two lugs
  and a stepped, three-strip underside -- so the true spacing is coarser
  than every row above.

  What the number does and does not say, stated plainly: the spacing is a
  LATERAL resolution. It bounds the smallest FEATURE whose penetration can
  be seen at all, not the smallest DEPTH that can be measured once a point
  lands on it. A lug corner biting 0.3 mm into a wall over a patch a few
  millimetres wide can fall entirely between samples at N = 1000 and be
  reported as zero penetration. That is exactly the failure D-109 (10c)
  names, now with a size on it.

  This does not by itself argue for a larger N: the cost is a mesh query
  per point per env per step, and that cost has never been measured for
  our env count either. Both numbers belong in the same decision.
- Sources: D-109 (3) and (10c); IndustReal `industreal_algo_utils.py`
  (`get_sdf_reward`, `sample_surface_even`); `HANDOFF-RL.md:324-331` on
  branch `p2-rl-code`; part dimensions from `insertion_tasks_cfg.py`
  (`PART_BODY_X`, `PART_BBOX_M`, `PART_HEIGHT_Z`); cross play from
  `PLAY_X` (this file, entry "The part body is 90.00 mm across").


## D-123: Open: the lug side of the part does not close, by about 1.8 mm
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: Found while reading the part CAD for `PART_BODY_X`.
  `PART_BBOX_M[0] = 96.409 mm` was measured in the imported USD (RT-12).
  Reading the same part STEP, the dense lug point cloud ends near +49.5 mm
  against a body edge at -45.0, i.e. a span near 94.5 mm. The gap of about
  1.8 mm is unexplained and sits close to the +1.774 mm offset named in the
  `PART_FLIPPED_IN_PLACE` note.
- Options considered: none yet -- not investigated, only observed.
- Decision: OPEN, recorded as `PART_LUG_SPAN_VS_BBOX` in
  `TOOL_CHAIN_PENDING`. It blocks nothing: the success criterion is depth
  plus interpenetration (D-106 (2)), and the lugs are carried by the SDF
  collider rather than by any bound.
  **[CORRECTION 2026-08-29: CLOSED. There was no disagreement. The lug is a
  cylinder of R 4.203 mm about x -47.207 mm, so its outermost point lies
  mid-arc and no point cloud returns it; read from the surface the tip is at
  -51.410 mm and the span is 96.410 mm against the USD's 96.409 mm.
  `PART_LUG_SPAN_VS_BBOX` is removed from `TOOL_CHAIN_PENDING` and no constant
  replaces it. See the inbox entry "The lug/notch pair does not bound the
  lateral play" (`docs/decisions_inbox.md`, 2026-08-29) and the note above
  `PART_BBOX_M` in `insertion_tasks_cfg.py`.]**
- Rationale: the body width is settled on two dense planes and a passed
  control, so this disagreement does not touch it. Recorded so it is not
  met later as a mystery in a mesh query.
- Sources: `CAD/Step/fuegeteil_prt.stp`; `PART_BBOX_M` (RT-12);
  `insertion_tasks_cfg.py` `TOOL_CHAIN_PENDING` note.


## D-124: Closed: the part is welded the right way round
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: `PART_FLIPPED_IN_PLACE` was named on 2026-08-23 as a MEASURED
  blind spot: RT-21 rotated the part 180 degrees about its own centre and
  not one of the twenty-one authoring checks moved, because an
  axis-aligned bounding box is the same box either way.
- Options considered: detect it by a geometric feature (the lug side, or
  the mesh centroid against the +1.774 mm offset the STEP states).
- Decision: CLOSED by the user, 2026-08-28: the part is welded correctly.
  The flange was turned by 180 degrees, and the code has carried that
  correctly for some time. Removed from `TOOL_CHAIN_PENDING`.
- Rationale: the blind spot itself was real and the lesson is kept in the
  code note -- a wrong orientation is caught by a geometric FEATURE, never
  by another bound.
- Sources: user statement 2026-08-28; RT-21; `insertion_tasks_cfg.py`
  `TOOL_CHAIN_PENDING` note.


## D-125: Open conflict: fixture pose noise moves the cut-out, the block stays
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: Since 2026-08-24 the fixture is the CAD cut-out
  (`Aufnahme_real_v1_mm.usd`, a kinematic RigidObject whose pose the
  reset can write) and the block (`aufnahme_block_v4.usd`) is static
  scenery spawned separately at the same translation. The recess in
  the block is the cut-out's bounding box with ZERO clearance. The
  consequence is documented in code (`insertion_env.py`, block spawn
  comment): with `fixture_pos_noise_xy > 0` or yaw noise, the cut-out
  moves per reset and the block walls do not -- every noisy reset
  drives the two bodies into each other. All fixture noises are 0.0
  today, so nothing is broken yet. D-110 (4)/(5) plans start-pose
  scatter rungs that include fixture offset, so this becomes real at
  the first rung with fixture noise.
- Options considered: (a) author block and cut-out into ONE USD, and
  the noise moves both (the block then wrongly moves relative to the
  tables it stands on -- by the noise amplitude, order mm); (b) give
  the recess clearance equal to the maximum noise amplitude (block
  geometry no longer flush with the cut-out; visually a gap);
  (c) make the block a second kinematic RigidObject and teleport both
  in the same reset write; (d) keep fixture noise at 0 and put ALL
  start scatter on the robot side (grasp/joint noise only). Not
  chosen here -- the choice affects what D-110's "fixture mounting
  scatter" physically means in the scene.
- Decision: OPEN -- user call needed before the first curriculum rung
  that widens fixture offset.
- Rationale: recorded so the zero-clearance coupling is decided, not
  discovered as an exploding sim at rung 2.
- Sources: `insertion_env.py` `_setup_scene` block-spawn comment
  (2026-08-24); D-110 (4)/(5); D-034 (kinematic fixture reset write).


## D-126: Open route: the per-reset flange-to-part offset has no runtime path on the welded tool
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: D-070 decided the INTENT -- grasp uncertainty enters as a
  per-reset flange-to-part offset, after Factory's
  `held_asset_pos_noise`. D-107 left an explicit `[verify]` flag on it:
  whether the welded gripper+part link (D-076) allows that offset at
  runtime in Isaac had never been measured. Factory teleports a
  SEPARATELY SPAWNED rigid body (`write_root_pose_to_sim`); our part is
  a welded LINK of the articulation, joined to `wrist_3_link` by the
  fixed joint `tool_weld`. There is no root pose to write.
  MEASURED, RT-51/RT-52 (2026-08-27, git f0ac527,
  `scripts/verify_grasp_offset.py`): a runtime USD write of
  `physics:localPos0` on `tool_weld` DOES NOT REACH PhysX. With 0.002 m
  written on env 0, the part's pose relative to the flange changed by
  4.77e-7 m -- bit-identical to the noise floor the negative control
  (`--offset 0.0`) produced twice, and three orders under the written
  value. env 1 printed `still -- no leak`, so nothing crossed envs
  either. The joint frame is consumed when the articulation is parsed.
  This route is dead. D-070's INTENT is untouched by this; only this
  one mechanism is ruled out.
- Options considered: (a) re-author `ur5e_tool.usd` per curriculum rung
  with a fixed offset, so the scatter is between rungs and not between
  resets; (b) drop the weld and spawn the part as a separate rigid body
  held by a runtime-writable joint, which is exactly Factory's shape but
  reopens D-076 and changes the articulation; (c) move the offset out of
  the geometry and into the observation, i.e. perturb the part-anchored
  channels instead of the body (changes what the policy is told, not
  what the sim does); (d) drop per-reset grasp scatter and put the whole
  start-pose scatter on the robot joints (D-110 rungs would then carry
  no grasp component). None of these is measured; (b) is the only one
  known to work in Isaac, and it is the most expensive.
- Decision: (d), user 2026-08-27. Per-reset grasp scatter is DROPPED.
  The whole start-pose scatter stays on the robot side -- the joint and
  yaw reset noise that `insertion_env.py` already applies
  (`reset_joint_noise`, `reset_yaw_noise`). D-110's rungs carry NO grasp
  component, and D-107's `[verify]` flag on D-070 closes as "measured,
  route dead, feature dropped". The welded tool (D-076) is untouched.
  NOTHING IS REMOVED FROM THE CODE: no grasp-noise parameter was ever
  written (checked 2026-08-27, no match in the env package), so this
  decision only stops one from being added.
  REOPENING CONDITION, written down so it is not re-derived: if grasp
  scatter comes back, the route is (b) -- drop the weld, spawn the part
  as a separate rigid body on a runtime-writable joint, Factory's shape.
  (a) and (c) stay unmeasured and are not the fallback.
  [Superseded 2026-09-11 by D-183, user: grasp scatter RETURNS through
  option (c), observation only, ±3 mm along the short axis. Decision (d)
  and the sentence above no longer hold; the RT-51/RT-52 measurement does.]
- Rationale: the user's call, 2026-08-27 -- the remaining scatter
  (robot joints, yaw, fixture pose) is judged enough for the first
  training runs, and (b) is too expensive to pay before any policy has
  trained. The measurement was run before the first training rung
  exactly so this is a design choice and not a silent gap. Recording the
  dead route matters as much as the live one: without it, the next
  session would try the same write again.
- Thesis consequence, do not lose: the thesis must STATE that grasp
  uncertainty is not simulated, and why (RT-51/RT-52). It is a named
  limitation, not an omission.
- Sources: RT-51 (2026-08-27 18:18:50, twice, negative control PASS),
  RT-52 (2026-08-27 18:21:34, NOT MOVED); `rt_logs/VERDICTS.md`;
  `scripts/verify_grasp_offset.py` docstring; D-070, D-076, D-107,
  D-110 (4)/(5).


## D-127: Correction to D-115: the scene DOES author solver iteration counts
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: D-115 states that the scene sets neither
  `solverPositionIterationCount` nor `solverVelocityIterationCount`.
  The szene stream flagged this as a NAMED CONFLICT before the run,
  because `ur5e_cfg.py:173-178` authors 16 and 1 in `articulation_props`.
  The startup report was extended to read the AUTHORED values on site
  (D-115 `[verify on site]`) and RT-50 ran it.
  MEASURED, RT-50 (2026-08-27, both runs, git c39f946):
  `solver iterations (D-115 verify): /World/envs/env_0/Robot/root_joint:
  solverPositionIterationCount=16, solverVelocityIterationCount=1`.
  The conflict is resolved against D-115: the cfg is right, D-115's
  sentence is wrong.
- Options considered: not applicable — this is a measurement correcting a
  written statement, not a choice between designs.
- Decision: D-115 needs an inline correction on main pointing at this
  measurement. The VALUES (16 / 1) stay where they are owned,
  `ur5e_cfg.py:173-178`; do not restate them in D-115.
- Rationale: CLAUDE.md forbids resolving a conflict by plausibility, so
  the report was made to read the stage rather than the config. The run
  is the authority.
- Sources: RT-50 (2026-08-27 18:03:03 and 18:08:33), `rt_logs/VERDICTS.md`;
  `ur5e_cfg.py:173-178`; D-115.
- STILL OPEN, do not fold into the above: the report prints NO solver line
  for `/World/envs/env_0/Fixture` in either run. Whether nothing is
  authored there or the report does not look is UNKNOWN and untested.


## D-128: Fixture SDF resolution is 2048
- Date: 2026-08-27
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: szene
- Context: D-109 (10b) raised the fixture's SDF resolution floor to >= 1314
  and named two practical values, 1536 and 2048, without choosing. The
  driver is the SAPU interpenetration filter at 0.144 mm: at D-062's 1024
  the voxel is 0.1847 mm, COARSER than the filter, so the filter would
  punish the collider instead of the policy. The fixture's longest edge is
  189.100 mm, measured and printed by the startup report's
  `asset check (fixture)` line against the STEP constants.
  Voxel pitch = 189.1 / resolution: 1024 -> 0.1847 mm (matches the number
  D-109 states, so the arithmetic checks against its own source),
  1314 -> 0.1439 mm, 1536 -> 0.1231 mm, 2048 -> 0.0923 mm.
- Options considered: 1536 (voxel 0.1231 mm, clears the filter, 2.37x fewer
  voxels than 2048) vs 2048 (voxel 0.0923 mm, the largest margin). Cost was
  NOT decidable from documentation: PhysX does not store the SDF as a dense
  grid, so the cubic voxel-count ratio is an upper bound on memory, not a
  measurement. That remains UNKNOWN and is what the run measures.
- Decision: 2048, user 2026-08-27. Applied via
  `scripts/add_fixture_collision.py --approximation sdf --sdf-resolution 2048`;
  the value has NO default in the script on purpose (D-109 (10b)) and lives
  in the command, not in a constant.
- Rationale: the user's call. The larger margin is bought before any
  training run, while re-running the authoring step is still cheap.
- DO NOT ALSO CHANGE THE PART. `author_tool_ur5e.py` authors the part's SDF
  at 1024 and that is correct: the part's longest edge is 143.5 mm, so its
  voxel is 0.1401 mm, already under the 0.144 mm filter. The D-109 (10b)
  floor is a FIXTURE number and does not transfer.
- Sources: D-109 (10b); D-062; `scripts/add_fixture_collision.py`;
  `docs/reference/literature_check_fixture_collision_sdf_2026-08-24.md`;
  fixture edges from the startup report `asset check (fixture)` line
  (RT-55, 2026-08-27 18:52:07).

## D-129: D-090 is scoped, not lifted: the inbox file alone may travel from the thesis branch to main
- Date: 2026-08-28
- Status: accepted
- Numbered: 2026-08-28, writer session on main
- Stream: konzept (writer session), on behalf of p1-thesis
- Context: D-090 forbids `p1-thesis` from merging into `main`, so the
  thesis text lives on its own branch by design. The writer session,
  however, numbers only what reaches `main`. The two rules together left
  a hole nobody had noticed: **candidates written in the thesis stream
  have no route to a D-number at all.** On 2026-08-28 twelve of them were
  found sitting in `p1-thesis:docs/decisions_inbox.md`, the oldest from
  2026-08-26, none of them present in `DECISIONS.md`. Two of the twelve
  contradict numbered entries (the pass-criterion conflict against D-051,
  and the 143.50 mm part length against the chapter-1 text), so the hole
  was not cosmetic — it hid disagreements from anyone reading `main`.
- Options considered: (a) leave D-090 as it is and accept that thesis
  candidates never get numbers; (b) lift D-090 and let the branch merge
  normally; (c) scope D-090 so that exactly one file may travel, without
  merging the branch.
- Decision: **(c).** D-090 stands: the thesis branch never merges into
  `main`, and thesis text never appears there. The single exception is
  the candidate file:

  > The writer session on `main` may fetch
  > `docs/decisions_inbox.md` from `p1-thesis` on its own
  > (`git checkout p1-thesis -- docs/decisions_inbox.md`), number the
  > candidates and empty the file. No merge, no other path, no other
  > file.

  After numbering, the thesis stream removes the same candidates from
  its own copy when it next merges `main` in. The direction of travel for
  everything else is unchanged: `main` flows INTO `p1-thesis`, never out.
- Rationale: (a) is the status quo and it demonstrably loses decisions —
  twelve of them, for two days, including two that contradict numbered
  entries. (b) throws away the reason D-090 exists: the thesis text is
  large, is rewritten constantly, and has no business in the shared
  history. (c) keeps that protection intact and costs one command. It
  also preserves the guarantee that makes the numbering safe — exactly
  one guardian assigns numbers, on `main`, from files that reach `main`.
  The file is the only cross-territory write every stream already has
  (D-065 rule 2), so extending its route does not open a second one.
- Sources: D-090 (the rule being scoped), D-065 rule 2 (the inbox is the
  one shared write) and rule 4 / D-066 (one writer per handoff file);
  the twelve stranded candidates in `p1-thesis:docs/decisions_inbox.md`
  as of 2026-08-28; user decision 2026-08-28.


## D-130: CLAUDE.md becomes a per-worktree private file (amends D-065/D-066)
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: The single tracked CLAUDE.md carried all streams' rules; every
  session loaded rules that did not apply to it (e.g. code-phase rules in
  the thesis stream). CLAUDE.local.md filtered but did not slim the load.
- Options considered: (1) status quo (one identical tracked CLAUDE.md +
  per-folder CLAUDE.local.md); (2) short tracked core + tracked per-stream
  rule files; (3) per-branch tracked CLAUDE.md protected by merge=ours
  (briefly implemented, superseded the same day); (4) untrack CLAUDE.md
  entirely: gitignored, each worktree keeps its own private copy
  (core block + own stream rules); no tracked templates.
- Decision: Option 4. `CLAUDE.md` and `CLAUDE.local.md` are gitignored;
  `git rm --cached CLAUDE.md` must be run once per branch BEFORE the next
  `git merge main` there (wrong order lets a merge delete the file from
  disk). Done on p1-thesis 2026-08-18; main, p1-szene-umbau and
  p1-konzept-messung still pending. Accepted costs (user informed): no
  git backup of current rule text (last tracked version survives in
  history at commit d1811db) and every new phase worktree needs its
  CLAUDE.md hand-copied in. The training PC needs NO CLAUDE.md at all:
  it only pulls code and assets and runs Isaac Sim/Lab; no Claude
  sessions, thesis documents or images are used there (user, 2026-08-18).
  Two additions (same session): (a) cross-stream awareness — at session
  start each stream skims its relevant neighbours' handoff files
  directly from their worktree folders (READ-ONLY; mid-phase state does
  not flow through git). Asymmetric: thesis reads HANDOFF-SZENE.md and
  HANDOFF-KONZEPT.md; szene and konzept read each other and do NOT
  read HANDOFF-THESIS.md. (b) Phase transition — new pN+1 worktrees
  start without a CLAUDE.md (gitignored); the user copies each one by
  hand from the predecessor folder; the thesis version normally stays
  unchanged.
- Rationale: user choice after grilling; each session loads only its own
  rules and no CLAUDE.md is ever merged again.
- Sources: session p1-thesis 2026-08-18; incident 2026-08-16 (overwritten
  CLAUDE.md, duplicated D-number).


## D-131: CLAUDE.md core: add adversarial-mode and rule-hygiene rules
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Review of the CLAUDE.md against the article "How to Write a
  CLAUDE.md That Actually Works" (M. Garramon, Medium, 2026-02-25). The
  core block has no rule that makes sessions challenge unverified
  claims, and no meta-rule that keeps the rule set corrective and lean.
- Options considered: (1) leave the core unchanged (streams add their
  own variants); (2) add two short core rules on main and let streams
  copy them at next session start.
- Decision: Option 2 proposed. Text for the core block (writer session
  applies on main, streams copy by hand):

  ```
  - Adversarial mode: challenge unverified claims and assumptions;
    distinguish verified (DECISIONS.md / PROBLEMS.md / ran on the
    training PC) from assumed, and say which one it is.
  - CLAUDE.md hygiene: every rule is corrective (born from a real
    mistake, cf. PROBLEMS.md); no generic best practices; prune rules
    that were irrelevant for a month.
  ```

  Also: trim explanatory sub-clauses in core rules where the rule alone
  is enough; KEEP decision IDs and dates as cross-references.
- Rationale: The article's two highest-value practices (adversarial
  mode, corrective-only rules) are the only gaps in the current core;
  everything else (short file, concrete rules, status in handoffs) is
  already in place. The thesis stream added tailored variants to its
  own stream section the same day.
- Sources: session p1-thesis 2026-08-18;
  https://medium.com/@martin_50671/how-to-write-a-claude-md-that-actually-works-76a184042444


## D-132: Tracked folder-level thesis/CLAUDE.md (scopes the untrack decision)
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Second CLAUDE.md review (Stulberg's 4-step setup: one
  CLAUDE.md per active folder, with a document index). thesis/ had no
  document index; the untrack decision of 2026-08-18 made all
  CLAUDE.md files gitignored at every level.
- Options considered: (1) index block inside the untracked root
  CLAUDE.md; (2) new tracked `thesis/CLAUDE.md` plus a .gitignore
  exception.
- Decision: Option 2 (user choice). The untrack decision is scoped: it
  applies only to the worktree-root CLAUDE.md files. Folder-level
  files — here `thesis/CLAUDE.md`, a pure document index with no
  rules — are tracked. `.gitignore` gets the exception
  `!thesis/CLAUDE.md`. No people block (user choice).
- Rationale: thesis/ is exclusive thesis-stream territory, so no merge
  conflict is possible; a tracked file survives phase transitions via
  git instead of hand-copying; the index loads only when working in
  thesis/.
- Sources: session p1-thesis 2026-08-18; user choice; Stulberg,
  "How to set up your CLAUDE.md files" (4-step excerpt provided in
  session).


## D-133: Part naming in the introduction: "Unterteil eines Kombi-Ableiters"
- Date: 2026-08-18
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: The thesis-writer agent flagged that the material uses three
  different names for the joined part. Chapter 1 needs one fixed term.
- Options considered: (A) neutral "das Unterteil eines Kombi-Ableiters",
  full type name deferred to the geometry chapter; (B) full designation
  "Unterteil des Kombi-Ableiters DEHNvap DVA CSP 3P 100 FM" already in
  chapter 1.
- Decision: Option A. The neutral term is used consistently in chapter 1;
  the full type designation appears once in the geometry chapter.
- Rationale: Readability at first mention; no manufacturer name in the
  introduction; precision is preserved where the geometry is specified.
- Sources: user choice 2026-08-18 (session p1-thesis); handoff of the
  section-1.1 rewrite session (Phase 0, 2026-08-18).


## D-134: Running term for the joined part: "Fügeteil"
- Date: 2026-08-21
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Section 1.1 called the part "das Unterteil eines
  Kombi-Ableiters" (wavy outer contour), section 1.2 called the same
  part "das vorgegriffene rechteckige Führungsstück". The two
  descriptions contradict each other. STYLE.md requires one fixed term
  per thing; the terminology table did not cover this one.
- Options considered: (A) introduce once in 1.1 as "das Unterteil
  eines Kombi-Ableiters, im Folgenden Fügeteil" and use "Fügeteil"
  everywhere after; (B) repeat the full name "Unterteil des
  Kombi-Ableiters" throughout.
- Decision: Option A. Fixed term is "Fügeteil"; "Führungsstück",
  "Peg" and "Werkstück" are banned for this part. "Zapfen" stays
  allowed for the laboratory peg geometry of the cited literature, never
  for this part. Refines the entry "Part naming in the introduction"
  (2026-08-18), which fixed the first-mention wording only.
- Rationale: "Fügeteil" is the term the geometry fact source
  (docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex) uses throughout, it
  is short enough for running text, and "rechteckig" was factually
  wrong.
- Sources: user choice 2026-08-21 (session p1-thesis); terminology table
  in thesis/STYLE.md.


## D-135: Chapter review runs as a skill, not ad hoc
- Date: 2026-08-24
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: A critical re-read of chapter 1 produced a usable ranked list
  of weak spots (redundancy at the 1.1/1.2 seam, an undecided pass
  threshold, a project-log date in the body text, the coined word
  "vorgegriffen"). The procedure was ad hoc and would not repeat the
  same way on chapters 2-7.
- Options considered: (A) capture the procedure as a repeatable skill
  `.claude/skills/kapitel-lesen/`; (B) leave it ad hoc per session.
- Decision: Option A. The skill reads the chapter, STYLE.md,
  HOCHSCHULVORGABEN.md, the Frozen section of HANDOFF-THESIS.md and
  main.toc, then reports at most 7 ranked findings grouped as
  Inhalt / Wortwahl / Kosmetik. It reports only and never edits a .tex
  file; rewriting stays with the thesis-writer agent.
- Rationale: Every chapter needs the same pass before its close
  checklist. A fixed procedure keeps the review comparable across
  chapters and stops it from silently turning into a rewrite.
- Note on territory: the skill file lives in `.claude/skills/`, outside
  this stream's declared territory (thesis/ + HANDOFF-THESIS.md +
  inbox). `.claude/skills/handoff/SKILL.md` is already branch-specific,
  so per-stream skill files follow the existing pattern. Confirm or
  reject at the next main-writer session.
- Sources: user request 2026-08-24 (session p1-thesis).


## D-136: Term for the pre-grasped state: "bereits gegriffen"
- Date: 2026-08-24
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Chapter 1 used the coined word "vorgegriffen" ("das
  vorgegriffene Fügeteil", 1.2; "da es vorgegriffen ist", 1.2
  delimitation paragraph). "Vorgegriffen" is not standard German and
  was in no terminology table.
- Options considered: (A) replace with "bereits gegriffen"; (B) keep
  "vorgegriffen" and register it as a defined term.
- Decision: Option A. Both occurrences replaced; a row was added to the
  terminology table in thesis/STYLE.md (Zustand des Fügeteils zu
  Episodenbeginn | bereits gegriffen | vorgegriffen).
- Rationale: STYLE.md requires plain German terms; a coined word needs a
  justification that this one does not have.
- STILL OPEN (raised by the thesis-writer agent, not resolved): 1.1
  describes the same state as "Zu Beginn des Fügeprozesses ist das
  Fügeteil bereits aufgenommen". Either align that sentence to "bereits
  gegriffen" or admit "bereits aufgenommen" as a permitted variant.
  STYLE.md demands one term per thing, so this needs a decision.
- Sources: user approval 2026-08-24 (session p1-thesis).


## D-137: CONFLICT (unresolved): is the pass criterion a success rate at all?
- Date: 2026-08-24
- Status: OPEN — conflict recorded, not resolved
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Chapter 1, section 1.2 states "Die Schwelle, ab der die
  Erfolgsrate als ausreichend gilt, ist noch offen." Asked for the
  number, the user answered that this is not a success rate at all:
  what is required is a robust policy that inserts the part under
  defined conditions, including noise.
- Conflict: D-051 (accepted) fixes the success rate as the SINGLE
  pass/fail acceptance criterion, with engagement, step count and
  contact-force distribution reported as diagnostics only. The user's
  statement points at a different acceptance concept (robustness over a
  range of disturbance conditions).
- NOT resolved here, and deliberately not resolved by plausibility.
  Either the user's robustness view is expressed as a success rate over
  a defined disturbance range (then D-051 holds and only the number is
  missing), or D-051 needs an amending entry.
- Effect on the text: the sentence in 1.2 stays untouched until this is
  settled. It is honest but reads as a hole in the goal chapter.
- Sources: user statement 2026-08-24 (session p1-thesis); DECISIONS.md
  D-051, D-052.


## D-138: Chapter 1 part length: 143.50 mm, not 144 mm — cross-stream correction not yet merged here
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: Chapter 1, Figure 1.1 caption used 144 mm (the Massblatt's
  "final 21.08.2026" value, itself the datasheet nominal adopted as
  real). Today D-088 (szene-umbau branch, not yet in this branch's
  DECISIONS.md or in docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex)
  settled the part length at 143.50 mm from the CAD mesh (D-075, log
  RT-8 22.08., reproduced in RT-12 23.08. to 7e-6 mm on the training
  machine), user-confirmed 2026-08-25. The datasheet itself cannot
  resolve 143.5 vs. 144 (whole-mm rounding only).
- Decision: use 143.50 mm in chapter 1, sourced to D-088/D-075, even
  though this branch's local geometry fact source has not been updated
  yet. Width stays 90.3 mm [CORRECTION 2026-09-11: D-121 overturned this on 2026-08-28 -- the CAD body width is 90.00 mm, `insertion_tasks_cfg.py` `PART_BODY_X`; D-075's "90.3 body" was wrong] (D-075 confirms this as the CAD body width
  between the lugs, independent of the length question — no change from
  the value already in the text). Height stays 50.7 mm (real
  measurement); the user explicitly declined switching to the CAD mesh
  value (50.514 mm) here, unlike for the length, so this is NOT a
  blanket "always prefer CAD" rule.
- Rationale: D-088 is dated, sourced, and user-confirmed; treating a
  stale local copy as authoritative over a newer cross-stream decision
  would contradict the "no assumptions" and "conflicts are not
  resolved by plausibility" rules in the other direction.
- Open action for the main-writer / szene-umbau session: propagate
  D-088 into this branch's DECISIONS.md and into
  docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex, which still shows
  144 mm as "final".
- Sources: D-088, D-075 (szene-umbau branch DECISIONS.md); user
  statements 2026-08-25 (session p1-thesis).


## D-139: Territory note: new literature-check file written from p1-thesis
- Date: 2026-08-25
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: user asked to source two claims in chapter 1 paragraph 2
  (relative pose governs insertion success; how the field models
  part-in-gripper pose uncertainty) via the research skill, which
  writes findings to `docs/reference/`, matching this repo's existing
  literature-check convention. `docs/` is normally out of this
  stream's territory (thesis/ + HANDOFF-THESIS.md + inbox only).
- Decision made in-session, not by main: wrote
  `docs/reference/literature_check_pose_governs_success_and_grasp_uncertainty_2026-08-25.md`
  anyway, following the same precedent as the `kapitel-lesen` skill
  file added 2026-08-24 (also outside declared territory, also flagged
  here without objection). Rationale: it is a pure addition (no
  existing docs/ file touched), it is exactly the kind of material
  this stream already reads from `docs/reference/`, and duplicating it
  under `thesis/` would split one literature-check convention into two
  incompatible locations.
- Confirm or reject at the next main-writer session; if rejected, the
  file should move under `thesis/` or a szene/konzept stream should
  own it instead.
- Sources: user request 2026-08-25 (session p1-thesis).


## D-140: Introduction states concrete asymmetric clearance values (supersedes the vague wording of 2026-08-21)
- Date: 2026-08-26
- Status: accepted
- Numbered: 2026-08-28, writer session on main (route opened by D-129)
- Stream: thesis
- Context: On 2026-08-21 the user decided to keep the clearance in
  section 1.1 deliberately vague ("am realen Aufbau in beiden Achsen
  unter 1 mm") because the clearance is not uniform and simulation
  and hand measurement use different numbers. During the chapter-1
  rework of 2026-08-26 the user asked for the concrete values instead.
- Options considered: keep the vague wording / state the measured
  values with their asymmetry.
- Decision: Section 1.1 states the measured clearance: 0.8 mm
  longitudinal, 0.3 mm transverse (total per axis), sitting
  asymmetrically with <= 0.1 mm remaining on the opposite side per
  axis. The detour via the UR5e repeatability spec is cut. The
  causal attribution to nubs/notches is NOT claimed (not in the
  geometry source).
- Rationale: The concrete asymmetric values carry the actual
  problem statement; the vague wording hid the tight side. User
  instruction 2026-08-26 supersedes the user decision of 2026-08-21.
- Sources: docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex (final
  values 2026-08-21); user instruction 2026-08-26.

## D-152: The SDF probe's on-surface tolerance is the success-depth uncertainty, and the 17 outliers are recorded, not repaired
- **RENUMBERED 2026-08-30. This entry was written as D-141 on branch `p2-rl-code` and is now D-152.** D-141 was already taken on `main` (commit `5fc10bc`, 2026-08-28, where the concept stream numbered eleven of this stream's inbox candidates D-141..D-151); this branch had not merged that commit and reused the range a day later. Every reference in this repo moved with it. **Training-PC logs, RT verdicts quoted in chat, and commit messages written before this date still say D-141** — read them through this line.
- Date: 2026-08-29
- Status: accepted
- Stream: rl-code
- Context: `scripts/check_insertion_sdf.py` P4 lays the part on itself at
  the identity transform and asserts that every sampled surface point reads
  ~0 against its own mesh. The tolerance was `1e-6` m. That number has no
  entry in this repo -- it was written into the argparse default and never
  decided. RT-88 failed P4 on it (max |d| = 5.358e-06 m) in both the
  ordinary and the mutation run, which turned an otherwise green sign probe
  into `SIGN_SUSPECT` and `MUTATION VERDICT: FAIL`.
- Measurement: RT-89 (git 9e152a6, 2026-08-29 16:43:23) printed the spread
  instead of the max alone. Median 3.725e-09 m, p99 1.799e-06 m,
  max 5.358e-06 m, and **17 of 953 points above 1e-06 m**. Worst point
  index 809 at xyz (0.018063, -0.039084, 0.149754) m in the mesh frame.
  936 points at ~4 nm rules out arithmetic: the outliers are local, and the
  part mesh carries 380 unpaired edges (RT-87).
- Options considered: (a) keep 1e-6 and repair the part mesh in the scene
  stream, effort unknown; (b) derive the tolerance from a number the task
  already carries and record the outliers.
- Decision: (b). `ON_SURFACE_TOL_M = 3e-5` m in
  `scripts/check_insertion_sdf.py`. The value is the +/- 0.03 mm to which
  `FLANGE_TO_PART_BOTTOM` is defined (`insertion_tasks_cfg.py` L1291, user
  2026-08-22, which owns the fact); the success-depth threshold rests on
  that number. A geometry error below the uncertainty of the threshold it
  feeds cannot move a task decision. The measured 5.358e-06 m passes it by
  a factor of 5.6.
- Why this is not fitting the threshold to the result: the +/- 0.03 mm was
  written on 2026-08-22, seven days before the measurement, for an unrelated
  reason (a tilted underside face). It was not chosen to make P4 green.
- What this does NOT settle, and is an open point, not a closed one: P4
  tests DISTANCE at the identity, and P2/P3/P5 test the SIGN far outside the
  mesh -- at least two full extents away by construction. Nothing tests the
  sign within microns of an open edge, which is exactly where
  `wp.mesh_query_point` may get it wrong on a non-watertight mesh. The
  exploit that opens is already named in `insertion_sdf.clamp_outside`: a
  far-outside point reading negative is clamped to zero, so a bad pose
  scores like a seated one. No instrument for this exists yet.
- Sources: RT-88 and RT-89 in `rt_logs/VERDICTS.md`; RT-87 (380 unpaired
  edges at a 1.8 nm weld); `insertion_tasks_cfg.py` L1285-1294;
  `wp.mesh_query_point` documents "NOTE: Mesh must be watertight!" without
  saying what happens when it is not.

---

## D-153: The fixture mesh keeps its 1.5 mm hole; the seated state is proven clear of it and the approach is not
- **RE-MEASURED AND STILL TRUE (2026-09-12, RT-183, git `379aab2`,
  `rt_logs/VERDICTS.md`).** Rule L-09 (`docs/reference/pruefregeln.md`) forbids
  reading a 2026-08-30 measurement as today's asset state, and the asset lives
  outside git, so the check was re-run against the current
  `assets/Werkzeug/aufnahme.obj`. Same result, same address: G1 903/757 PASS,
  G2 frame guard PASS with every error 0.0000 mm, 903 authored boundary edges
  welding to 9, `ZERO-EDGE COUNT 3` at x 35.9863..37.4512, y 79.6347..79.7500,
  z 0.0000 mm, verdict `OPEN_EDGES_NOT_CLEAR`, exit 1 — which this decision
  wants. Both code constants are confirmed against today's numbers:
  `POCKET_HOLE_MIN_Y` 79.6347 mm is the measured edge, and
  `HOLE_REACH_OFFSET_Y` = 79.6347 − 71.75 = 7.8847 mm recomputes exactly.
  Stage 1 allows 8.7000 mm, so the reachable window is 0.8153 mm wide. What
  is STILL not measured is whether reaching the hole does any harm; the
  instrument for that is `stage1_lateral_y_mm.over_reach`.
- **RENUMBERED 2026-08-30. This entry was written as D-142 on branch `p2-rl-code` and is now D-153.** D-142 was already taken on `main` (commit `5fc10bc`, 2026-08-28, where the concept stream numbered eleven of this stream's inbox candidates D-141..D-151); this branch had not merged that commit and reused the range a day later. Every reference in this repo moved with it. **Training-PC logs, RT verdicts quoted in chat, and commit messages written before this date still say D-142** — read them through this line.
- Date: 2026-08-30
- Status: accepted
- Stream: rl-code
- Context: SAPU (D-109 point (10)) queries the part's surface samples
  against the FIXTURE mesh, and `wp.mesh_query_point` carries the note
  `Mesh must be watertight!` without documenting what happens when it is
  not. The fixture surface is provably open. RT-61 had reported that all
  nine surviving boundary edges lie on the outer top rim, i.e. nowhere the
  part goes; that report was a note, never a check, and it was taken on the
  USD after a position weld, while Warp receives the OBJ as authored
  (`SdfDistanceQuery` builds `wp.Mesh` from
  `trimesh.load(..., process=False)`).
- Measurement: RT-91 and RT-94 (`scripts/check_fixture_mesh.py`,
  2026-08-30, git `6d3cc8a` and `db72bae`, judged in `rt_logs/VERDICTS.md`).
  Three results, in the order they matter:
  1. **The OBJ as authored carries 903 unpaired edges, all boundary.** A
     position weld at 2.49 um leaves **9**, so 894 are index artefacts. This
     settles a question RT-86's parity argument could not: `2271 = 2*1131 + 9`
     and `2271 = 2*684 + 903` are both true, and the answer is 903.
  2. **RT-61's address is wrong for this OBJ.** Six of the nine welded
     survivors sit at `y = 90.4500`, `z = 15.0000` mm and clear the stage-1
     hull by 10.0000 mm -- the rim RT-61 named. The other **three form one
     triangle about 1.5 mm across** on the stage-1 shoulder, at
     `x 35.9863..37.4512`, `y 79.6347..79.7500`, `z 0.0000` mm. RT-61's own
     listed break points span six edges on that line, not nine, and its
     sweep rode the unit defect `HANDOFF-SZENE.md` records (it reached
     0.1 um, not the 0.1 mm it claimed), so its address was never load
     bearing.
  3. **Reachability, computed from constants that already have homes**, not
     argued: `PART_BBOX_M[1]` = 143.5000 mm, half-span 71.7500 mm, so the
     part's `+y` face reaches the hole only at a centre offset of
     **+7.8847 mm**. Inside stage 2 the wall (`POCKET_WALL_Y` 72.55,
     `PLAY_Y` 1.6) allows **+0.8000 mm**, leaving **7.0847 mm**. Inside
     stage 1 (`POCKET_STAGE1_Y_RANGE[1]` 80.45) it allows **+8.7000 mm**,
     and the stage-1 floor is z = 0, the hole's own plane.
- Options considered: (a) repair the fixture mesh -- weld that triangle or
  re-export the asset, work owned by the scene stream and of unmeasured
  size; (b) accept the hole, record what it does and does not touch, and
  reopen the point when the condition that makes it live is switched on.
- Decision: (b). The mesh is not repaired. `check_fixture_mesh.py` keeps its
  gate sharp and keeps exiting 1 on this asset; the failure is the record,
  not a defect to silence. Logged in `InBachelorErwähnen.md`, 2026-08-30.
- Rationale, and it rests on the split the measurement produced rather than
  on convenience:
  * **The success predicate is provably unaffected.** D-106's
    interpenetration filter is read at the seated state, and there the
    pocket wall holds every part surface point 7.0847 mm from the hole. No
    sample point can reach a face adjacent to it, so no sign it returns can
    enter the success decision.
  * **The SAPU factor during the approach is NOT provably safe.** In stage 1
    the part can slide +8.7000 mm, more than the +7.8847 mm required, with
    its underside in the hole's plane. This is stated as an open exposure,
    not waved away.
  * **The exposure cannot occur under the present configuration.**
    `fixture_pos_noise_xy = 0.0` (`insertion_env_cfg.py:340`), so the part
    never leaves the pocket axis. Repairing an asset to remove an exposure
    that no configured run can produce spends another stream's effort on a
    condition that does not exist yet.
- Reopening condition, stated so it is not forgotten: **the moment
  `fixture_pos_noise_xy` is set above zero**, this decision is reopened.
  **ANSWERED 2026-09-12 by D-187**: the condition was met in plan, the hole
  was re-measured (RT-183) and its effect on the reward was measured
  (RT-184). The mesh still is not repaired, and the reason is now a
  resolution argument, not the absence of the condition. Read D-187. The
  documented target is 0.05 m (Isaac Lab 2.3.2 PegInsert
  `fixed_asset_init_pos_noise`, cited at `insertion_env_cfg.py:336-339`),
  which is 6.3x the offset that reaches the hole.
- What this does NOT settle: whether `wp.mesh_query_point` pairs edges by
  INDEX or rebuilds adjacency by POSITION. If by index, Warp sees 903 open
  edges rather than 9, and the 894 artefacts are not artefacts to it. Warp's
  documentation states neither, and no run in this repo has measured it.
  The far-outside sign is covered by RT-92/RT-93
  (`check_insertion_sdf.py --query-obj`); the sign AT the surface is not
  covered by any test, the same gap D-152 records for the part mesh.
- Sources: `rt_logs/VERDICTS.md` entries RT-91 (2026-08-30 13:48:13) and
  RT-94 (2026-08-30 14:08:52); `HANDOFF-RL.md` § `Open` NEXT item 1;
  `HANDOFF-SZENE.md` § `Open`, "RESULT of RT-61", including its recorded
  unit defect; `insertion_tasks_cfg.py` (`PART_BBOX_M`, `POCKET_WALL_Y`,
  `PLAY_Y`, `POCKET_STAGE1_Y_RANGE`, `POCKET_RIM_Z`);
  `insertion_env_cfg.py:336-340`; Warp `mesh_query_point` note quoted at
  `isaaclab_tasks/direct/automate/industreal_algo_utils.py` L356-L358.


## D-154: The SDF sample count is 64,000 — the cost curve, not the geometry, sets where the ladder stops

- **RENUMBERED 2026-08-30. This entry was written as D-143 on branch `p2-rl-code` and is now D-154.** D-143 was already taken on `main` (commit `5fc10bc`, 2026-08-28, where the concept stream numbered eleven of this stream's inbox candidates D-141..D-151); this branch had not merged that commit and reused the range a day later. Every reference in this repo moved with it. **Training-PC logs, RT verdicts quoted in chat, and commit messages written before this date still say D-143** — read them through this line.
- Date: 2026-08-30
- Status: **accepted 2026-08-30** — RT-100 and RT-101 are green on the
  training PC (`SIGN_HOLDS` exit 0, `MUTATION VERDICT: PASS` exit 0, both git
  `1c79b11`). The route there was not straight: RT-98 failed on P4 exactly as
  this entry predicted, and D-155 settled it without touching the count. See
  the RT-98 block below.
- Stream: rl-code
- Context: D-109 point (3) names N = 1000 surface points via
  `trimesh.sample.sample_surface_even`, a value taken from Isaac Lab's
  `num_mesh_sample_points`, which was chosen for another task's env count.
  D-109 point (10c) warned without a number that 1000 points are "thin for a
  143 mm part" and that too few points systematically UNDER-report
  penetration. D-122 put a size on that warning — the mean point spacing
  `sqrt(area/N)` — and then stopped, explicitly: "OPEN. The count is
  `p2-rl-code`'s to set; this is the input", because "the cost is a mesh query
  per point per env per step, and that cost has never been measured for our env
  count either. Both numbers belong in the same decision." This entry supplies
  the cost half and closes D-122.
- Measurement: RT-97 (`scripts/check_insertion_sdf.py --bench --seed 0`,
  2026-08-30 15:16:02, git `016e644`, judged in `rt_logs/VERDICTS.md`, PASS on
  all five expectations). Five rungs at the scene's own 128 envs
  (`insertion_env_cfg.py:456`), warm-up excluded, 20 timed calls per rung:

  | N | returned | ms per call | spacing | vs the 0.5876 mm play | device MiB |
  |---|---|---|---|---|---|
  | 1000 | 953 | 4.0 | 8.578 mm | 14.60x | 36 |
  | 4000 | 4000 | 11.5 | 4.187 mm | 7.13x | 2 |
  | 16000 | 16000 | 43.5 | 2.093 mm | 3.56x | 2 |
  | **64000** | **64000** | **174.2** | **1.047 mm** | **1.78x** | **66** |
  | 143000 | 143000 | 396.5 | 0.700 mm | 1.19x | 98 |

  Two of these numbers were PREDICTED in writing before the run and both held:
  the sampler returns 953 of 1000 requested (RT-95's figure, same mesh and same
  seed), and the measured surface area exceeds D-122's box.

  **The spacing column is not D-122's.** D-122 computed it from the part
  treated as a box, 494.2 cm², and said in the same entry that the box is a
  LOWER bound on the area because of the two lugs and the stepped underside.
  RT-97 measured the real area: **701.2 cm², a ratio of 1.419**. Every spacing
  is therefore `sqrt(1.419)` = 1.19x coarser than D-122 printed, and D-122's
  "about 143,000 points" for spacing equal to the play is short: the real
  figure is `area / play²` = 0.07012 / 0.0005876² ≈ **203,000 points**.
- Options considered:
  (a) **Keep N = 1000.** Rejected: 8.578 mm spacing is 14.6x the cross play,
      and D-109 (10c)'s failure mode is live at that resolution — a lug corner
      biting into a wall over a few millimetres falls entirely between samples
      and is reported as zero penetration.
  (b) **N ≈ 203,000, spacing equal to the cross play.** The only value with a
      geometric argument rather than a habit. Rejected on cost: extrapolating
      the measured curve gives about 560 ms per RL step for the SDF query
      alone, roughly 9 minutes of added wall-clock per 1000 RL steps.
  (c) **N = 64,000.** Chosen.
  The user was shown the measured table and the 203,000 figure with its price
  and chose (c) on 2026-08-30.
- Decision: **`num_sample_points` = 64,000.** The point spacing is then
  1.047 mm on the measured surface, 1.78x the 0.5876 mm cross play, at 174 ms
  per call for 128 envs and 66 MiB of device memory. D-122 is CLOSED by this
  entry; D-109 point (3)'s N = 1000 is superseded for this task.
- Rationale: The measurement turned a one-sided question into a curve, and the
  curve is close to linear in N (149x the points cost 98x the time from 953 to
  143,000). So the choice is a straight trade and neither end of it is free.

  Against (a): the resolution argument is D-109 (10c)'s own and it is
  one-directional — under-sampling can only hide penetration, never invent it,
  so a too-coarse cloud makes the interpenetration filter permissive exactly
  where D-106 point (2) needs it strict. 14.6x the play is not a margin, it is
  a blind spot the size of the tolerance the task is about.

  Against (b): the geometric target buys a factor 1.7 in spacing over (c) for a
  factor 3.2 in time. The spacing is a LATERAL resolution — D-122 states this
  plainly: it bounds the smallest FEATURE whose penetration can be seen at all,
  not the smallest DEPTH that can be measured once a point lands on it. Depth
  resolution is set by the query, not by the sample count, and the query's own
  accuracy is measured: RT-89/RT-90 put the on-surface error at a median of
  3.7 nm with a maximum of 5.4 µm. So sampling AT the play does not make the
  filter exact; it only makes the smallest visible feature equal to the play
  rather than 1.78x it, and features at that size are already bounded by the
  pocket walls the part cannot pass.

  For (c): 1.047 mm is the first rung below the 1.6 mm long-axis play
  (`PLAY_Y`), so every feature the LONG axis can hide is visible, and it is
  within a factor two of the cross play. Device memory is not a constraint at
  any rung — 98 MiB at the largest — and the user stated on 2026-08-30 that
  memory and runtime are ample; the cost was measured anyway, because "ample"
  is a weighing and D-122 asked for a number.

  What this does NOT claim: that 64,000 is optimal. It is a point on a measured
  curve, chosen by the user with the curve in front of them. If a later run
  shows the interpenetration filter under-reporting, the curve is on file and
  the next rung is one line.
- **RT-98 RAN AND THE NAMED RISK MATERIALISED (2026-08-30 15:37:05, git
  `0fd57c2`): P4 FAILED, `max |d| = 7.387e-05 m` against the 3e-5 tolerance,
  11 of 64,000 points above it.** Everything else was green, and median and
  p99 were BETTER than at 953 points. Settled by **D-155**: P4 judges the 99th
  percentile, not the maximum, because the pair it guards feeds an averaging
  term; the tolerance itself does not move. The COUNT decided here is
  unaffected -- D-155 explicitly rejected lowering it as a way out. This entry
  stays `proposed` until RT-100 and RT-101 are green.
- What is not yet verified: the sign probe's P4 tolerance at this count. P4
  asserts that a sampled point reads ~0 against its own surface at the identity
  transform, with `ON_SURFACE_TOL_M = 3e-5` derived in D-152 from the
  success-depth uncertainty. RT-89 measured 17 of 953 points above 1e-6 m, and
  those outliers are the part mesh's 380 unpaired edges, not arithmetic. A 67x
  larger cloud lands 67x more points near those edges, so the maximum may rise.
  RT-98 is the one-mesh probe at 64,000 and is the run that can fail.
- Sources: RT-97 (the measurement, `rt_logs/VERDICTS.md`); D-122 (the spacing
  half and its explicit hand-off of the count); D-109 (3) (the superseded 1000)
  and D-109 (10c) (the under-reporting failure mode); D-106 (2) (the
  interpenetration filter the count serves); D-121 (`PLAY_X` = 0.5876 mm);
  D-152 (P4's tolerance and the 17 outliers); RT-89 / RT-90 (the on-surface
  error distribution); user decision 2026-08-30, taken against the RT-97 table.


## D-155: P4 judges the 99th percentile, not the maximum — the statistic was wrong for an averaging term, and the tolerance does not move

- **RENUMBERED 2026-08-30. This entry was written as D-144 on branch `p2-rl-code` and is now D-155.** D-144 was already taken on `main` (commit `5fc10bc`, 2026-08-28, where the concept stream numbered eleven of this stream's inbox candidates D-141..D-151); this branch had not merged that commit and reused the range a day later. Every reference in this repo moved with it. **Training-PC logs, RT verdicts quoted in chat, and commit messages written before this date still say D-144** — read them through this line.
- Date: 2026-08-30
- Status: **accepted 2026-08-30** — RT-100 (2026-08-30 15:51:33) and RT-101
  (15:53:17), both git `1c79b11`, judged in `rt_logs/VERDICTS.md`. RT-100:
  `P4 ... p99 |d| = 1.348e-06 m against tol 3e-05 m -- PASS`, with
  `max |d| = 7.387e-05 m` and `11 of 64000 points above tol` REPRODUCING RT-98
  to the digit, so only the verdict rule moved; P2, P3, P5 PASS;
  `VERDICT: SIGN_HOLDS`, exit 0. RT-101: P2 FAILS on all 12 asserted rows, P5
  on 11, P4 stays green, `MUTATION VERDICT: PASS`, exit 0 — the probe still
  catches a flipped sign under the new rule.
- Stream: rl-code
- Context: P4 is the sign probe's identity check: the sampled points lie ON
  the queried surface, so at the identity transform every distance must read
  ~0. Its tolerance is `ON_SURFACE_TOL_M = 3e-5` m, DERIVED in D-152 from the
  success-depth uncertainty (`FLANGE_TO_PART_BOTTOM` is defined to about
  ±0.03 mm), on the argument that a geometry error below the uncertainty of
  the threshold it feeds cannot move a task decision. Until 2026-08-30 P4
  judged the MAXIMUM absolute distance.

  D-154 raised the sample count from 953 to 64,000 and named this as the one
  thing it had not verified: RT-89 had found 17 of 953 points above 1e-6 m,
  and those outliers are the part mesh's 380 unpaired edges, not arithmetic,
  so 67x more points would land 67x more samples near them.
- Measurement: RT-98 (`scripts/check_insertion_sdf.py --seed 0`, 2026-08-30
  15:37:05, git `0fd57c2`, judged in `rt_logs/VERDICTS.md`). It failed, exit 1,
  on P4 alone, and the distribution is the finding:

  | statistic | 953 points (RT-89/RT-90) | 64,000 points (RT-98) |
  |---|---|---|
  | median | 3.725e-09 m | **3.455e-09 m** |
  | p99 | 1.799e-06 m | **1.348e-06 m** |
  | max | 5.358e-06 m | **7.387e-05 m** |
  | points above 3e-5 m | 0 of 953 | **11 of 64,000** |

  **The bulk got BETTER and the extreme got worse.** Median and p99 both
  improved; the maximum grew by a factor of 14, carried by 11 points out of
  64,000 (0.017 %). Everything else in the run was green: marker,
  `sampled 64000 of 64000 requested`, and P2, P3 and P5 all PASS.
- Options considered:
  (a) **Raise `ON_SURFACE_TOL_M` above 7.387e-05 m.** Rejected. D-152 derived
      the value from the success-depth uncertainty; it is not a knob. Raising
      it to fit a measurement is fitting the criterion to the result, which is
      the failure mode the whole check harness exists to prevent.
  (b) **Accept the red check, the way D-153 accepts the fixture's hole.**
      Rejected on a mechanical ground, not a philosophical one: the D-080
      counter-proof for this probe requires P4 GREEN (`caught = (not p2) and
      (not p5) and p4`), so a permanently failing P4 destroys the mutation run
      that makes every other verdict of this probe worth anything. A check
      that is expected to be red also trains its reader to ignore it, which is
      the RT-77 defect already recorded in `PROBLEMS.md`.
  (c) **Reduce the sample count until the maximum fits.** Rejected: it would
      undo D-154 for a reason that has nothing to do with what D-154 decided,
      and the resolution argument of D-109 (10c) would come back.
  (d) **Judge p99 instead of the maximum, tolerance unchanged.** Chosen.
- Decision: **P4 compares the 99th percentile of the absolute identity
  distances against the unchanged 3e-5 m.** The maximum and the number of
  points above the tolerance are printed on the same line and in the spread
  block, so nothing is hidden. The rule is the pure function `judge_identity`
  in `scripts/check_insertion_sdf.py`, pinned offline by `IDENTITY_CASES`.
- Rationale: The statistic has to match how the checked quantity is CONSUMED.
  P4 guards the pair part-against-part, and that pair is the SDF REWARD term,
  which reduces over every sample point. One point in 64,000 moves a maximum
  and moves nothing the reward computes. At 953 points the distinction was
  invisible because max and bulk agreed; D-154's count made them disagree, and
  the disagreement is a property of the mesh's open edges, which RT-87 measured
  at 380 and D-152 already records as unrepaired.

  This is not the tolerance being loosened, and the offline cases are what make
  that checkable rather than assertable. p99 is compared against the SAME 3e-5.
  A cloud that reads off-surface broadly still fails (case I4, every point at
  1e-4 m), and so does one with five per cent of its points outside the
  tolerance (case I2). What no longer fails is a handful of edge artefacts in a
  cloud whose 99th percentile sits more than an order of magnitude inside the
  bound. Case I3 replays RT-90's distribution and stays green, so the runs that
  were valid before this change remain valid.

  The choice of the 99th percentile rather than, say, the 99.9th is a
  cut nobody measured, and it is stated as such: at 64,000 points p99 excludes
  640 points, which is 58x the 11 outliers RT-98 found, so the rule has room
  before an artefact count would have to grow before it bites. If a later run
  shows hundreds of points outside the tolerance, that is a mesh problem and
  P4 will report it.
- What this does NOT cover: the sign AT the surface, and the SAPU pair. SAPU
  queries the part's points against the FIXTURE mesh and reduces by MAXIMUM
  (`insertion_math.max_interpen_dist`, `-torch.min`), so a single outlier there
  WOULD move a success decision. P4 says nothing about that pair — different
  queried mesh, different edges — and the gap belongs to D-152 (the part) and
  D-153 (the fixture's accepted hole), where it is already recorded. The user
  was shown this split on 2026-08-30 and accepted the change on it.
- Sources: RT-98 (the measurement, `rt_logs/VERDICTS.md`); RT-89 / RT-90 (the
  distribution at 953 points); D-152 (the tolerance and its derivation, the 17
  outliers); D-154 (the count that exposed this, and its own named risk); RT-87
  (380 unpaired edges on the part mesh); `insertion_math.max_interpen_dist`
  (the MAX reduction SAPU uses, which is why the split matters); user decision
  2026-08-30.

## D-156: The workcell block is switched OFF and the Aufnahme stands free — the pocket walls belong to the cut-out, so the block only blocked the fixture noise

- Date: 2026-08-30
- Status: **accepted 2026-08-30** — RT-106 (2026-08-30 22:28:54, git `964ca2f`,
  judged in `rt_logs/VERDICTS.md`): all seven expectation points met, exit 0.
  The `workcell block: *** OFF ***` line printed; `/World/envs/env_0/Block`
  appears NOWHERE in the 505-line log; the fixture prim-vs-buffer read-back is
  `4.34e-07` m; and the pose DIFFERS per env — env 0 `(0.50572, -0.13054)` yaw
  `-0.0961` rad, env 1 `(0.50068, -0.13515)` yaw `+0.0547` rad — which is what
  proves the noise is live, not merely configured. No NaN. Run folder
  `2026-08-30_22-29-04_peg30mm_pocket32mm_offset5mm_yaw+-5.7deg_currOFF_blockOFF`.
  Offline evidence: `check_env_wiring.py` 89 checks + 4 new mutations, exit 0;
  `selftest_checks.py --offline` PASS, exit 0.
- Stream: rl-code
- Context: the robustness tests need the Aufnahme to sit differently every
  episode. D-044 already names fixture offset, yaw and tilt as the variation
  axes, and the machinery exists (`fixture_pos_noise_xy`,
  `fixture_yaw_noise_rad`, `fixture_tilt_noise_rad`, written per reset in
  `_reset_idx`). It could not be switched on: the fixture CAD is a CUT-OUT of
  the block, the block's recess is that cut-out's bounding box with ZERO
  clearance, and only the cut-out has a runtime pose. Any offset therefore
  drives the moving cut-out into standing walls — D-125.
- Decision: stop spawning the block. `InsertionEnvCfg.spawn_workcell_block:
  bool = False` guards the spawn in `_setup_scene`. The Aufnahme then stands
  free and moves with machinery that already exists: no new rigid body, no
  clearance change, no asset change, no D-125.
- Why this is not a workaround, three facts:
  1. **The pocket belongs to the Aufnahme, not the block.** The cut-out is a
     solid 150.0 x 189.1 x 61.0 mm body with the 90.6 x 145.1 mm pocket cut
     into it. Wall left standing on the cut-out itself: -x 39.5 mm, +x 19.9 mm,
     -y 16.1 mm, +y 27.9 mm; floor -36.0 mm, rim +15.0 mm. The block
     contributes NO pocket wall.
  2. **Nothing in the physics or the reward touches the block prim.** Its whole
     runtime role was the spawn and one report line. `BLOCK_TOP_Z`,
     `BLOCK_REAR_WALL_T/H` are constants used by report prints and offline
     checks and stay valid whether or not the prim exists. The SDF and SAPU
     queries run against `aufnahme.obj` and `fuegeteil.obj`; the block is in
     neither mesh.
  3. **It is the published Isaac Lab pattern.** Factory's `_setup_scene`
     (`C:/IsaacLab/.../direct/factory/factory_env.py:85-116`) spawns a ground
     plane, a table and three FREE-STANDING assets; its fixed asset has nothing
     enclosing it, and Factory randomises that asset's position AND yaw every
     reset (`factory_env.py:613-643`).
- **The cost, named: the rear wall is gone.** `block_rueckwand` stood 100 mm
  above the rim and is what made the pocket reachable ONLY FROM THE FRONT.
  Without it the part may approach from any direction, so the task is EASIER
  and less like the real cell. Two rows in `InBachelorErwähnen.md` (rear wall;
  free-standing fixture). The surrounding body and floor are cosmetic only —
  the Aufnahme is kinematic and does not fall.
- Nothing is deleted and it is reversible: the flag, `workcell_block_pos` and
  every block constant stay. **D-125 is PARKED, not closed** — it governs again
  the day `spawn_workcell_block` is `True`. Its worked-out option (c), both
  roots written from ONE sampled pose, is in the plan appendix.
- Made visible, because a run whose scene is not written down is not citable:
  the startup report prints `workcell block: *** OFF -- the Aufnahme stands
  free; no rear wall, the pocket is enterable from any direction ***`, and
  `train.py` appends `blockOFF` to the run folder name, the same rule the
  ladder switch follows (D-110 (3)).
- Folded in, one line of the same edit: the `__init__` guard that refuses a
  deterministic angle together with a per-episode randomisation was blind to
  `fixture_pos_noise_xy`. With xy noise alone `_reset_idx` still runs and, both
  angle noises being 0, writes an IDENTITY quaternion over the static
  `fixture_tilt_rad` — the run trained a task its own config denied. The
  operand is now in `randomised` (not in `_tilt_active`, which asks whether any
  orientation is configured at all, a question an xy offset does not answer).
- Open and NOT settled here: **D-153 reopens the moment
  `fixture_pos_noise_xy > 0`**, independently of the block. See the handoff.
- Sources: D-044 (variation axes); D-125 (the coupling, now parked); D-034
  (kinematic fixture, per-reset pose); D-110 (3) (a switch must be visible in
  the run name); Isaac Lab 2.3.2 `factory_env.py:85-116` and `:613-643`;
  user decision 2026-08-30.

## D-157: The lateral pocket gate moves into the paying predicates — depth alone may not pay, because a descent beside the fixture projects the same depth

- Date: 2026-08-31
- Status: **proposed 2026-08-31**, UNVERIFIED. The DEFECT is measured
  (RT-107, `demo_metrics.json`, 1024 envs, 1500 iterations); the FIX below is
  not built and no run has judged it.
- Stream: rl-code
- Context: RT-107 was the D-108 reward-hack probe. It found one, and not the
  one the probe expected. The run reports `success_rate_recent` **0.9765**
  against `mean_max_depth_mm` **0.0** over the same 2000 episodes, with
  `stage1_lateral_y_mm.mean` **109.0 mm** against `stage1_allows_mm`
  **8.7 mm** and `over_reach` **1991/2000**. `episodes_to_threshold` is
  68960: the policy searched honestly for roughly 69,000 episodes, found the
  exploit, and then held it (`success_rate_cumulative` 0.4287).
- The mechanism, exactly: `depth` is the projection of the part tip onto the
  pocket axis and carries no lateral information. Two consumers read it and
  only one gates it.
  1. The METRIC `_max_depth` (`insertion_env.py:973`) counts depth only while
     `abs(tip_rel.x) < POCKET_WALL_X` and `abs(tip_rel.y) < POCKET_WALL_Y`
     (45.29 / 72.55 mm). Its own comment states the reason: "without it a
     descent BESIDE the fixture would book its height loss as insertion
     depth".
  2. `insertion_math.in_success_region` and `insertion_math.is_engaged` read
     the RAW `depth`. At 109 mm lateral the tip is outside the wall, so the
     metric reads 0 while both predicates fire.
- Why the reward's own exploit analysis missed it: the
  `compute_rewards_insertion` docstring retires "side-parking" with the
  argument that `sdf_dist` grows in that direction. That argument covers the
  `kernel_sum` term and ONLY that term. The two display bonuses do not read
  `sdf_dist` at all — they read `engaged` and `success`, hence `depth` — and
  the same docstring puts them at about two thirds of the episode return. SAPU
  does not close the hole either: beside the fixture there is nothing to
  interpenetrate, so `sapu_reward_scale` returns 1.0.
- Decision, three parts, one hypothesis:
  1. **One home for the containment fact.** A new pure
     `@torch.jit.script` function `in_pocket_cross_section(tip_rel, wall_x,
     wall_y)` in `insertion_math.py`. The env's metric stops computing it
     inline and calls this.
  2. **Both paying predicates take it.** `is_engaged` and `in_success_region`
     gain the containment flag as an argument and AND it in. A pose that is
     axially in the band but laterally outside the walls pays nothing and is
     not a success.
  3. **A cross-consistency instrument.** The invariant `success ==>
     max_depth > 0` is checked per episode and its violation count goes into
     `demo_metrics.json`. This is the general guard, not a patch for this one
     hack: any future predicate that pays for a pose the metric cannot see
     becomes visible at the first metrics dump (iteration ~4) instead of after
     1500 iterations.
- Why the walls and not the four-corner gate: the corner gate is proxy
  geometry (`cfg.held_asset.side`, `cfg.fixed_asset.side`, both marked
  `[proxy]` in `demo_metrics.json`) and assumes a square part in a square
  pocket. `POCKET_WALL_X` / `POCKET_WALL_Y` are the real cut-out's own
  dimensions and are already the metric's rule. Adopting them changes no
  number and introduces no new constant.
- **Named gap, not closed here:** this makes the containment test a BOX, so
  the part's own footprint is still not tested against the pocket contour, and
  a pose inside the box but rotated wrongly is still caught only by SAPU. The
  contour test is the SDF's job (D-109) and is not reopened by this entry.
- Consequence for RT-108: it must NOT run on the pre-fix code. Its 8 hours
  would train the same broken target.
- Sources: RT-107 `demo_metrics.json`; `insertion_env.py:968-978`;
  `insertion_math.py:699-741`; `insertion_math.compute_rewards_insertion`
  docstring; D-106 (success band), D-109 (5) (the two bonuses), D-152/D-153
  (the two exploits that still have no instrument).

## D-158: The force-abort limit drops from 100 N to 60 N — still an invented number, now a conservative one

- Date: 2026-08-31
- Status: **proposed 2026-08-31**, UNVERIFIED. Set by the user.
- Stream: rl-code
- Context: `force_abort_f_max_n` is one of the two entries in
  `RL_PLACEHOLDERS`. It has never been measured. RT-107 shows the policy
  learned to sit just under it: `force_norm_n` p50 79.4 N, p95 87.6 N, p99
  107.0 N, `over_f_max` 24/2000, `force_abort_rate` 0.012. That is a second
  farming pattern beside the depth hack — the limit is being used as a ceiling
  to press against, not as a safety bound.
- Decision: `force_abort_f_max_n = 60.0`.
  **[CORRECTION 2026-09-01: SUPERSEDED IN ITS NUMBER. The field is 300.0 N,
  user-set as an observation window for the first PPO run under the
  reworked reward; still a placeholder, still no measurement. The
  suction-cup datasheet anchor is withdrawn. Home: `docs/decisions_inbox.md`, entry "The tilted scripted insertion is SPENT: its capture depth is its own ramp, the datasheet anchor is withdrawn, F_max goes to 300 N as an observation window" (2026-09-01).]**
  **[CORRECTION 2026-09-06: SUPERSEDED AGAIN. The field is 50.0 N, and for
  the first time it comes off a measurement rather than off a round guess:
  RT-157's per-episode force maxima. Home: D-169.]**
- **What this is NOT.** It is not a measurement and it is not derived. No
  holding force of the suction pair, no permissible contact force of the real
  part, and no fixture load limit has been established. The value stays in
  `RL_PLACEHOLDERS`, the startup report keeps shouting about it, and
  `demo_metrics.json` keeps carrying it. It replaces one invented number with
  a lower invented number, on the grounds that a safety bound should err low
  while it is unmeasured.
- What it will change, stated before the run: under RT-107's force
  distribution a 60 N limit would abort the large majority of episodes. That
  distribution belongs to the hacking policy and does not transfer, so no
  prediction of the post-fix abort rate is made here. `force_abort_rate` is
  the number to read.
- **Two changes, one run — and that is fine here.** CLAUDE.md's "one change
  per run" belongs to the LATER phase: once a working baseline and real
  results exist and every change has to be attributable in the thesis. It does
  not govern the phase this entry is written in, where the task DEFINITION
  itself is known broken and is being repaired (user, 2026-08-31). D-157 and
  D-158 therefore land together before RT-108 and no separating probe run is
  planned. Each still has its own number to read if the question ever comes
  up — D-157 off `mean_max_depth_mm` and the new invariant counter, D-158 off
  `force_abort_rate` and the `force_norm_n` spread.
- Sources: RT-107 `demo_metrics.json` `force_norm_n` block;
  `insertion_env_cfg.py:565`; D-114 (the abort payment and F_max travel
  together).

## D-159: `max_depenetration_velocity = 5.0` on the UR5e — TRIED AND REVERTED

- Date: 2026-08-31
- Status: **REVERTED 2026-08-31** by the user, same day, after RT-113.
  The field is UNSET again (`ur5e_cfg.py:173`), which is the pre-D-159 state.
  **Unset is not `0.0`**: `0.0` would forbid the solver to separate
  overlapping bodies at all, a third setting nobody has measured.
- Stream: rl-code
- Decision as it stood: `UR5E_HOME_CFG.spawn.rigid_props.max_depenetration_velocity = 5.0`.

**WHY IT WAS REVERTED — read this before trying it again.** RT-113
(2026-08-31 18:03:35, git `4cbf9f3`) ran the same `--lateral` counter-proof
with the value in place. The abort force was `[2950.19, 4787.32, 915.92,
786.56]` N against the written-first expectation of "three digits or less".
Two envs fell to ~800-900 N, two did not.

**What that does and does not prove.** The expectation is VIOLATED, so the
change did not do what D-159 claimed. It is NOT a controlled falsification:
the probe poses differ between RT-112 and RT-113 (`pre-step tip_rel xyz mm`,
L224 in each log), because the reset applies +-45 deg of wrist_3 noise. A
clean repeat would need the same poses in both runs. Nobody has run that.
So the honest status is: the hypothesis did not survive its own test, and
the mechanism is still unidentified.

**The observation that forced it.** RT-112 (2026-08-31 17:44:56, git
`0272937`) printed the force on the aborting step for the first time:
2144 / 4228 / 2144 / 3253 N against a 60 N limit, with a measured
interpenetration of 8-16 mm. A force 35 to 70 times the limit is not a
contact force. It is the solver separating two overlapping bodies, and
without a bound it may do that arbitrarily fast.

**Why 5.0 and not a number of my own.** All three official contact-rich
Isaac Lab 2.3.2 tasks set exactly this value on the robot and on every
dynamic asset: `factory/factory_env_cfg.py:128`,
`automate/assembly_env_cfg.py:137`, `automate/disassembly_env_cfg.py:136`,
plus the per-asset copies in `factory_tasks_cfg.py` and
`assembly_tasks_cfg.py`. The meaning is Isaac Lab's own:
"Maximum depenetration velocity permitted to be introduced by the solver
(in m/s)" (`isaaclab/sim/schemas/schemas_cfg.py:96`). Nothing here is
invented; the value is copied and its source is named.

**Where it does NOT need a counterpart.** The part is welded into this
articulation (D-041), so it is a link of the robot and the robot's
`rigid_props` covers it. The fixture is spawned kinematic
(`insertion_env.py:507`); a kinematic body is never depenetrated.

**The alternative that was rejected.** Clamping the force channel in
`_get_observations` at `force_abort_f_max_n`. It has no precedent in the
nearest published work: FORGE does not clip its force observation at all
(`forge/forge_env.py:113-139`) -- it smooths it (which we already do,
D-114) and penalises it above a threshold. Observation clipping IS standard
elsewhere, but on other routes: rl_games via `clip_observations: 5.0` in 22
Isaac Lab task configs (`isaaclab_rl/rl_games/rl_games.py:328`), and
Stable-Baselines3 via `VecNormalize(clip_obs=10.0)`. Our route is rsl_rl,
whose `RslRlVecEnvWrapper` clips ACTIONS only
(`isaaclab_rl/rsl_rl/vecenv_wrapper.py:153`) and whose
`EmpiricalNormalization.forward` applies no clamp at all. Clamping would
also hide the cause instead of removing it.

**Expectation, written before the run (RT-113, `--lateral`).** The force in
the `ABORT at settle step` block falls from 2144-4228 N to three digits or
less. If it does not, the cause is something else and this entry is wrong
rather than merely unverified. — RESULT: it did not. See the status block
above.

**Thesis duty — VOID.** The condition the user attached was "if we set it and
KEEP it, it goes into the concept report". The value is not kept, so nothing
goes into `Gesamtbericht_Konzeptphase_Codestrang.tex`. The
`InBachelorErwähnen` row written on the same day carries an inline correction
instead.

- Sources: RT-112 log (`rt_logs/VERDICTS.md`, 2026-08-31 17:44:56);
  Isaac Lab 2.3.2 local tree `C:\IsaacLab` at the paths named above.

**Correction (2026-08-31, D-160).** RT-112 and RT-113's kN-scale forces are
explained: the `--lateral` probe pose was geometrically impossible (an
upright part buried in fixture material), not a solver defect this setting
could have addressed. This entry's revert stands — the value never applied
to a real defect either way — see D-160 for the actual cause.

## D-160: The RT-112/RT-113 lateral-probe failure had two independent causes — an impossible probe pose, and the force channel carrying the tool's own weight
- Date: 2026-08-31
- Status: verified (RT-114, RT-115; both on the training PC)
- Context: D-159 was reverted after RT-113 still showed kN-scale abort
  forces. `CLAUDE.md` § Change discipline: after two failed attempts, stop
  and rank root causes with one discriminating measurement per cause,
  rather than guess a third time. Two independent causes were found and
  fixed in sequence, each with its own offline proof and its own
  training-PC confirmation.
- Cause 1 — the probe pose was geometrically impossible. `--lateral`
  teleports the part UPRIGHT to a commanded lateral offset at seated depth
  (`check_seated_success.py`). The offset had defaulted to 109.0 mm, RT-107's
  measured POLICY offset (`stage1_lat_y_max_mm`) — a real number, but for a
  part that was free to tilt, not for this script's upright teleport. An
  upright part's footprint (`PART_BBOX_M`) centred at 109 mm still overlaps
  the fixture block's outer material (`POCKET_LOCAL_Y_RANGE[1]` =
  100.45 mm) over the full 34 mm of commanded depth. RT-112 and RT-113
  therefore buried the part in solid geometry and measured correct solver
  physics for an impossible pose (kN forces, 8-16 mm interpenetration) —
  neither run exercised the D-157 success-predicate gate at all.
- Cause 2 — the force channel carried the welded tool's own weight. Fixed
  with cause 1 corrected (RT-114 used a reachable pose, ~191.89 mm, free
  air), the run still read 8.08 N with the part 121.6 mm from the block and
  0.0 mm interpenetration. D-114 imports the FORGE force channel
  (`body_incoming_joint_wrench_b` at the welded-tool link) wholesale, but
  FORGE's own wrench IS the contact force only because Factory/FORGE spawn
  the robot AND the held asset with `disable_gravity=True`
  (`factory_env_cfg.py:127`, `factory_tasks_cfg.py:165`). This repo keeps
  gravity ON at the robot by a separate, earlier decision (supervisor,
  2026-08-25, `ur5e_cfg.py`, superseding p1-gains D-082). The two decisions
  were never checked against each other: with gravity on, the same
  joint-wrench read carries the tool's own weight (0.824 kg * 9.81 =
  8.083 N) on top of any contact. RT-59 had already measured this in
  isolation (8.0861 N at rest, ratio 1.000 against m*g) without it being
  connected to the force channel's role in D-114.
- Options considered (cause 2 only; cause 1's fix — compute the offset from
  geometry instead of typing RT-107's policy number, with an offline guard
  refusing an unreachable offset — had no real alternative once the
  geometry was worked out): (a) tare the tool's weight out of the channel;
  (b) turn `disable_gravity` on at the robot and the held asset, matching
  FORGE exactly; (c) leave the channel as read and treat 8.08 N as a known
  constant the policy learns around.
- Decision: (1) cause 1's fix ships as computed in
  `check_seated_success.py`: `--lateral-y-mm` defaults to
  `POCKET_LOCAL_Y_RANGE[1] + 0.5*hypot(PART_BBOX_M[0], PART_BBOX_M[1]) +
  0.005` (a named 5 mm probe margin) ~= 191.89 mm, yaw-safe; an explicit
  offset below the bare clearance (~172.2 mm) is refused before a sim step
  runs, by a pure function checked offline (`lateral_probe_y_error`).
  `fixture_pos_noise_xy` is pinned to 0.0 for this test, since an identity
  test must reproduce. (2) cause 2: **(a)**. `insertion_math.gravity_tare()`
  subtracts `-mass * gravity` from the raw wrench, ROTATED into the parent
  link's frame before subtraction — the weight is fixed in the world but
  the sensor frame turns with the wrist, so a constant subtraction is wrong
  the instant the wrist is not upright, which is also why RT-59's
  upright-only measurement did not surface the need for a rotation. Applied
  BEFORE the EMA smoothing. The parent link is CHECKED against
  `ee_body_name` at startup rather than assumed (Isaac Lab 2.3.2 has no
  parent-index API on `ArticulationData`). `cfg.force_gravity_tare` defaults
  `True`; `tool_mass_kg = None` is a hard stop, not a silent skip. The
  wrench itself stays RAW in the parent body frame — the 2026-08-28 frame
  decision (`docs/decisions_inbox.md`, "The force wrench is in the parent
  body frame, not the world frame") is untouched; only the known weight
  vector is rotated.
- Rationale: (b) was rejected because it overturns the 2026-08-25
  supervisor decision, which has its own supporting evidence (RT-46:
  0.000026 m of droop under gravity-on with the current gains) — nothing in
  this investigation contradicts that decision, so reopening it would be
  unjustified. (c) was rejected because 8.08 N is not a constant once the
  wrist rotates away from upright; treating it as one would silently bias
  both the observation and the `force_abort_f_max_n` comparison by a
  pose-dependent amount. (a) closes the gap between the imported FORGE
  pattern and this repo's own gravity-on decision without overturning
  either, and payload/gravity compensation on a joint-torque-derived force
  reading is standard practice, not an invented mechanism. Neither
  underlying decision (D-114's force-channel import, or the 2026-08-25
  gravity-on choice) is overturned by this entry; it resolves a conflict
  between them that neither decision's own record had surfaced.
- Verification: RT-114 (`rt_logs/VERDICTS.md`, 2026-08-31 18:42:22) —
  reachable pose, all L-points PASS, no abort, `VERDICT:
  LATERAL_CONTROL_HOLDS`, but force N = 8.08 in free air. RT-115
  (2026-08-31 19:04:44) — same pose with the tare on: tared force 0.002 N,
  raw untared control (printed alongside, specifically so a flipped tare
  sign would read ~16.17 N instead of ~0 N) 8.09 N. Both confirm the sign
  is correct. Offline before either run:
  `python scripts/check_insertion_math.py --self-test` (COUNTER-PROOF
  PASSED, including a 90-degree-wrist-turn case and its own D-080 mutation
  pair), `python scripts/check_seated_success.py --self-test` (37/37,
  including the new L8 force point and its cases),
  `python scripts/selftest_checks.py --offline` (PASS).
- Consequence for the abort limit: `force_abort_f_max_n = 60.0` (D-158,
  still an invented placeholder) now compares against a channel with
  roughly 8 N less baseline than before the tare — the real contact margin
  under that number moved from ~52 N to the full 60 N (the field is 300 N
  since 2026-09-01, see the inbox entry named under D-158's correction;
  the tare argument is unchanged by the number). This is a side
  effect of a bug fix, not a re-derivation of the limit; D-158's number and
  status are unchanged by this entry. One `InBachelorErwähnen` row records
  the shift.
- Known limit (stated in the code and repeated here): the tare removes the
  static weight (`mass * gravity`) only, not the inertial term
  (`mass * acceleration`) that appears under wrist motion. A real F/T
  sensor's payload compensation has the same limit.
- Sources: RT-59 (`rt_logs/VERDICTS.md`, 2026-08-28 14:27:03), RT-112/RT-113
  (2026-08-31 17:44:56 / 18:03:35), RT-114 (2026-08-31 18:42:22), RT-115
  (2026-08-31 19:04:44), RT-46 (droop measurement, cited in `ur5e_cfg.py`),
  Isaac Lab 2.3.2 local tree `C:\IsaacLab`
  (`factory_env_cfg.py:127`, `factory_tasks_cfg.py:165`,
  `forge_env.py:95-114`, `articulation_data.py:731-732` — the
  `body_incoming_joint_wrench_b` docstring), D-114, D-158, D-159 (this
  file), the 2026-08-28 frame-decision inbox entry, the 2026-08-25 gravity
  decision (`ur5e_cfg.py` comment, superseding p1-gains D-082). Commits on
  `p3-rl-code`: 15c3e9f, 6908265, 9ace861, 6767def, 8423e94.

## D-161: The 2026-08-31 training run learned nothing because the D-109 shaping term is numerically zero at the reset pose — the missing piece is the curriculum, not the reward
- Date: 2026-08-31
- Status: verified (RT-119, training PC, git 92fd6d3)
- Stream: rl-code
- Context: the 1024-env run `08-31_19-16-20_offset5mm_currOFF_seed0`
  (96 504 episodes, ~1 h) reached `success_rate_cumulative` 0.0 and
  `mean_max_depth_mm` 0.0. The replay shows the arm retreating UPWARDS to a
  fully extended pose and trembling there — it never moves towards the
  fixture. `force_abort_rate` is 0.001. The behaviour and the metrics agree
  and needed one explanation, not a hyperparameter guess.
- Measured (RT-119, `--reward-curve`, four envs, IK residual < 0.001 mm at
  every point; the counter-proof `--curve-coarse-margin-mm 200` ran in the
  same session and lifts the 165 mm value to 0.0648, so the sweep reads the
  kernel it claims to read):

  | height above entrance | SDF distance | `kernel_sum` (peak 1.0) |
  |---|---|---|
  | +165 mm (the reset pose) | 173.21 mm | 3.05e-23 |
  | +100 mm | 108.42 mm | 8.06e-15 |
  | +50 mm | 58.78 mm | 2.29e-08 |
  | +20 mm | 29.50 mm | 1.46e-04 |
  | +10 mm | 20.69 mm | 2.04e-03 |
  | 0 mm (entrance) | 13.61 mm | 1.64e-02 |
  | -20 mm | 4.30 mm | 1.83e-01 |
  | -34 mm (seat) | 0.46 mm | 5.83e-01 |

- Finding: every episode starts at exactly the same pose,
  `WORKCELL_HOME_TIP_ABOVE_ENTRANCE` = 165 mm above the entrance
  (`reset_joint_noise` and `reset_yaw_noise` are both 0.0). At that pose the
  whole reward is 3e-23. PPO cannot distinguish that from zero, so the
  policy stands on a FLAT landscape. The only quantity in the reward with a
  usable magnitude is `abort_payment` = -1.0 on the force abort, and the one
  behaviour that reliably avoids it is to touch nothing — which is what the
  run learned, and what the replay shows.
- Why the pose drifts as far as it does: the action is a joint-target
  INTEGRATOR clamped only against the joint limits, never against the reset
  pose (`insertion_env.py` `_pre_physics_step`). 256 steps x 0.02 rad allows
  5.12 rad ~ 293 deg of free drift per joint per episode, and no term in the
  reward penalises action magnitude, joint velocity, jerk or leaving the
  workspace (`insertion_math.py`, "Existing instead of solving"). The tremor
  is `init_noise_std = 1.0` exploration around a saturated target.
- The root cause is a MISMATCH, not a wrong reward. D-109 point (9) derives
  the coarse kernel's 10 mm margin from the "start scatter, Block 6" —
  i.e. the reward was solved for a start INSIDE rung 0 of the D-110 ladder.
  D-110 is accepted but `curriculum.py` is "not wired into the env yet", and
  the run used `curriculum_enabled: false`. The reward expects 10 mm; the
  env delivers 165 mm.
- Decision: the kernel widths are NOT touched. They are solved from measured
  tolerances (D-109 point (9)) and widening them to cover a 165 mm approach
  would destroy the argument that carries them. The fix belongs on the
  START side: wire rung 0 (start height inside the coarse margin) into
  `_reset_idx` and train there first, to answer "can this policy learn at
  all" before the ladder itself is built. `rung_step_sizes` stays UNSET
  (RL_PENDING) — the full ladder is its own decision and does not block
  rung 0.
- Open, not decided here: `force_norm_n` averaged 42.9 N (p95 49.6 N) in
  that run while the arm was, by the replay, in free air, where RT-115
  measured 0.002 N. Three candidates, none proven: inertial load from the
  tremor (the tare subtracts `m*g` only, not `m*a`), reaction against the
  arm's own joint limits, or real contact. Its own measurement.
- Sources: RT-119 (`rt_logs/VERDICTS.md`, 2026-08-31 22:54:29); D-109 (the
  reward), D-110 (the ladder), D-113 (no success termination), D-114 (the
  force channel and the abort payment).

## D-162: Rung 0 is a start HEIGHT in the cfg, solved to joints by a dls IK at every reset
- Date: 2026-08-31
- Status: UNVERIFIED (offline checks pass; no training-PC run yet)
- Stream: rl-code
- Context: D-161 measured the cause of the failed 2026-08-31 run -- the whole
  D-109 reward is 3.05e-23 at the 165 mm home pose, so PPO trains on a flat
  landscape. D-161 fixed the direction of the repair (the START moves, the
  kernel widths do not) and left the wiring open. This is the wiring.
- Decision, in four parts:

  1. THE START IS A HEIGHT, NOT A JOINT TABLE.
     `InsertionEnvCfg.start_tip_above_entrance` is the height of the leading
     tool point above the stage-2 opening plane, positive up, the same sign
     convention as `WORKCELL_HOME_TIP_ABOVE_ENTRANCE` (+0.165 m). `None`
     keeps the bare home pose, which is the pre-D-161 behaviour.
     Alternative rejected: a second joint table beside `UR5E_HOME_JOINT_POS`,
     produced by another screener run. It would put the rung in six angles
     that no longer say what height they mean, and every later rung would
     need another training-PC run before it could even be tried.

  2. THE DEFAULT IS DERIVED, NOT TYPED.
     **SUPERSEDED 2026-09-01 BY D-163 -- the value below is wrong and was
     measured wrong the same day (RT-120: force_abort_rate 1.0). Points 1, 3
     and 4 of this entry stand unchanged; only the NUMBER moved.**
     `insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE =
     -(POCKET_SEAT_DEPTH - KERNEL_MARGIN_COARSE)` = -0.026 m: one coarse
     kernel margin above the seat, i.e. the part starts 26 mm inside a 36 mm
     pocket and has 7 mm left to the success band. Both inputs already own
     their numbers (D-106 (1) the seat, D-109 (9) the 10 mm margin), so the
     start height tracks a re-measurement instead of going stale. RT-119
     measured `kernel_sum` 0.183 at -20 mm and 0.583 at the seat, so this
     height sits inside the band where the gradient is readable -- against
     3.05e-23 at the home pose.

  3. HEIGHT -> JOINTS IS THE ALREADY-VERIFIED TELEPORT, RUN AT RESET.
     `insertion_env._solve_start_pose` is the dls `DifferentialIKController`
     loop of `scripts/check_seated_success.py` `_solve_to`, the one RT-119
     drove to eight heights with a residual under 0.001 mm, with the same
     defaults (120 iterations, 0.05 mm tolerance, lambda 0.05) carried into
     the cfg. It aims at the pocket AXIS at the requested height and carries
     a ZERO rotation delta, so the reset's own orientation -- including
     `reset_yaw_noise` when it comes back -- survives.
     It runs LAST in `_reset_idx`, after the fixture pose is sampled and
     written: the goal is THIS episode's pocket, and solving first would aim
     at the previous episode's entrance, which below the opening plane is
     inside a wall. It also seeds `_joint_targets`, because the action is an
     integrator and a stale target would snap the arm back to 165 mm on the
     first step.
     WHY NOT FACTORY'S RESET IK: `factory_env.set_pos_inverse_kinematics`
     calls `step_sim_no_action` between iterations, and its own docstring
     restricts it to resets where every env resets together. Ours must work
     in a partial reset, so the loop must not step the sim -- which is
     exactly the RT-80 lesson `_solve_to` already encodes.

  4. THE RUNG IS RECORDED WHERE RESULTS ARE READ.
     The run folder gains `start<+/-N>mm` (or `startHOME`) beside `currOFF`,
     and `demo_metrics.json` gains `start_tip_above_entrance_mm`, the home
     height for comparison, and the solve's own residual, iteration count and
     unconverged-reset count. With the ladder off the start height IS the
     rung; a run whose rung can only be inferred is one that gets compared
     wrongly.

- Not decided here, on purpose: `rung_step_sizes` stays UNSET and
  `curriculum_enabled` stays False (D-110 (3) sources the step sizes from
  this project's own learning curves, and there are none yet). The joint-
  target integrator still has no clamp against the reset pose -- D-161's
  second finding, left unfixed deliberately, one hypothesis per change.
- Scope of the change: the three teleport runs (`check_seated_success.py`,
  `seat_probe.py`, `scripted_insert.py`) pin `start_tip_above_entrance = None`
  in their env cfg. They do their own teleport and report against the 165 mm
  home stand-off; a rung-0 reset would move the pose they claim to command.
- Verification available offline: eight new wiring checks in
  `scripts/check_env_wiring.py`, each with its own mutation (D-080). 117
  checks and the full counter-proof pass; `scripts/selftest_checks.py
  --offline` PASS. Nothing about the reward magnitude at the new height is
  proved offline -- the height-to-SDF mapping is geometry and needs the sim.
- Sources: D-161 (the measurement), D-109 (9) (the kernel margins), D-110
  (the ladder), RT-119, RT-80; Isaac Lab 2.3.2
  `isaaclab_tasks/direct/factory/factory_env.py` (the rejected reset-IK
  shape).

## D-163: The rung-0 start is +30 mm, above the fixture -- the tilted approach is part of the task and happens above the opening plane
- Date: 2026-09-01
- Status: UNVERIFIED (offline checks pass; no training-PC run yet)
- Stream: rl-code
- Supersedes: D-162 point (2) only. The mechanism of D-162 (a cfg height, an
  IK solved at reset, recorded in the run tag and the metrics file) is
  unchanged and was measured to work -- see "What held" below.
- Context: D-162 set the rung-0 start to
  `-(POCKET_SEAT_DEPTH - KERNEL_MARGIN_COARSE)` = -26 mm, a literal reading of
  D-161's "one coarse kernel margin above the seat". That put the tool point
  26 mm INSIDE a 36 mm pocket.
- Measured (RT-120, training PC, `demo_metrics.json`; the log carried no
  `[rt_log]` header, so the run is identified by its metrics only):

  | quantity | value |
  |---|---|
  | `force_abort_rate` | **1.0** -- every episode aborted |
  | `force_norm_n` p50 / mean / p95 | 93.5 / 102.1 / 169.9 N |
  | `force_abort_f_max_n` (the limit) | 60.0 N (at the time; 300 N since 2026-09-01) |
  | mean failing-episode length | **3.6** of 256 steps |
  | `success_rate_recent` / `_cumulative` | 0.0 / 0.0 |
  | episodes | 255 617 |
  | `success_depth_invariant_violations` recent | 0 |

  The reset teleported the part into contact. The force channel aborted on
  the first steps, paid -1.0, and reset. No episode ever ran long enough to
  learn anything.
- What held, and is NOT re-opened: the start-pose IK itself.
  `start_pose_solve_unconverged_resets` = 0, worst residual 0.0159 mm against
  a 0.05 mm tolerance, and **5** of 120 allowed iterations. The D-162
  mechanism is cheap and lands where it says it does.
- THE TASK ARGUMENT, and it is the real reason (user, 2026-09-01): a straight
  vertical descent is not how this part will be inserted. The intended
  strategy is four phases -- approach TILTED, touch with one edge, align on
  that edge, then push. All of phases 1 to 3 happen ABOVE the stage-2 opening
  plane. A start inside the pocket does not merely make the task easy; it
  deletes the part of the task the thesis is about.
- [2026-09-14, inline: the constant is now 0.020 m -- H_min MEASURED per
  D-179 (4), RT-197a/b (20 mm clean in the full corner; 10 and 5 mm pushed
  out of the rim, RT-198b/RT-199b). The task argument below stands; the
  number moved because D-179 made this field H_min's one home. Entry:
  `docs/decisions_inbox.md` "H_min is MEASURED" (2026-09-14).]
- Decision: `RUNG0_START_TIP_ABOVE_ENTRANCE = 0.030` m. USER-SET, not derived,
  and the entry says so rather than dressing it up: it answers a question
  about the TASK, not about a tolerance, so no tolerance can derive it. It is
  15 mm above the stage-1 top rim (`STAGE1_DEPTH`), so the whole part hangs in
  free air with the fixture below it.
- STATED PLAINLY, NOT HIDDEN: the reward is weak there and its value at
  +30 mm is UNMEASURED. RT-119's sweep read `kernel_sum` 1.46e-04 at +20 mm
  and 2.29e-08 at +50 mm; +30 mm lies between and no run has read it. Against
  the -1.0 abort payment that is small, which is D-161's mechanism again in
  a milder form. This run is therefore expected to be watched, not to be
  quoted as a result.
- The guard that came out of it (D-080): `check_env_wiring.py` now checks
  "the rung-0 start height clears the fixture's stage-1 rim"
  (`RUNG0_START_TIP_ABOVE_ENTRANCE > STAGE1_DEPTH`), with the mutation
  `rung0-start-height-back-inside-the-pocket` putting -0.026 back and
  breaking exactly that check. RT-120 could not have run under this guard.
- Open, and being researched in parallel (2026-09-01): the D-109 reward is ONE
  SDF distance to the seated pose, so position and rotation share one number
  (`insertion_math.kernel_sum`). A deliberate tilt is a LARGER distance and
  therefore LESS reward -- the intended strategy must get worse before it gets
  better, and a purely dense distance reward cannot pay for that detour. A
  research pass on published multi-phase mating rewards is running; the reward
  is NOT touched until it lands.
- INLINE CORRECTION (2026-09-02): the fixed +30 mm start did not learn
  (RT-134, RT-137: the part parks on the outer rim, depth 0). Step C
  SAMPLES the start height Uniform[low, +30 mm] per episode; for the
  episodes drawn below 0 the task argument above is knowingly overridden
  (user, 2026-09-02). `RUNG0_START_TIP_ABOVE_ENTRANCE` stays the UPPER
  bound and this entry's number is unchanged. Home of the change:
  `docs/decisions_inbox.md`, entry "Step C: the rung-0 start height is
  SAMPLED ..." (2026-09-02).
- INLINE CORRECTION 2 (2026-09-11, D-179): under `dr_mode='autodr'` the
  start height is ONE-SIDED, from the measured H_min up to 0.120 m, and
  `start_tip_above_entrance` is then its LOWER edge. Under `dr_mode='off'`
  the 2026-09-02 reading above (upper edge of [low, high]) stands.
- Sources: RT-120 (`demo_metrics.json`), D-161, D-162, D-109 (the reward),
  D-110 (the ladder); `insertion_tasks_cfg.STAGE1_DEPTH` = 0.015 m (user CAD).


## D-164: The force abort becomes `truncated` and is BOOTSTRAPPED, not lump-paid
- Date: 2026-09-02
- Status: proposed (pending verification)
- Stream: rl-code
- Supersedes, in part: D-113's assignment of the force abort to
  `terminated`. Success stays `terminated` — it is a true task terminal.
  Only the abort moves.
- Context: RT-131 (first PPO run under the revised reward, rung 0, start
  +30 mm, 1024 envs, seed 42) did not learn to insert, and the reason is an
  accounting asymmetry rather than a tuning problem.
  MEASURED. TensorBoard, run `09-01_23-46-59_offset5mm_currOFF_start+30mm_seed42`:
  iterations 0–40 `force_abort_rate` climbs to 1.0 while `mean_max_depth_mm`
  peaks at 0.4 mm — the policy is trying; iterations 40–70 both collapse to 0
  and stay there through iteration 299. `Train/mean_reward` climbs to 57.7 by
  iteration ~150 and is then flat. A replay of the checkpoint shows the part
  set down on the OUTER RIM of the fixture, so `mean_max_depth_mm` = 0.0 is a
  correct reading, not a blind gate.
  THE POLICY WAS NOT FROZEN. `Policy/mean_noise_std` reads 0.0051 at iteration
  299 against `init_noise_std = 1.0`, but it was still ~0.2–0.35 during
  iterations 40–70, i.e. 40–70x its final value, exactly when the abort rate
  collapsed. The decay is smooth with no knee. Exploration was alive; the
  policy actively learned to stop making contact. This rules out
  "exploration died first" as the cause and makes the reward the right target.
  THE ARITHMETIC, computed offline from the committed constants
  (`kernel_sum`/`squash`, `KERNEL_A_COARSE`…`KERNEL_B_FINE`, T = 256,
  `time_penalty_per_step` = −1/256). The force abort was `terminated`, not
  `truncated` (`insertion_math.compute_dones`), so rsl_rl added no
  `gamma*V(s)` and the whole remaining return was lost: holding the start
  distance for a full episode pays **58.89**; a force abort at step 20 pays
  **3.60**. One abort cost **−55**, of which only **−1.0** was the named
  `abort_payment` and **−54** the forfeited remainder — the written deterrent
  was understated 55x. Against that, the whole 30 mm → 5 mm approach is worth
  **+5.72** (kernel per step 0.2339 at 30 mm, 0.2563 at 5 mm). The policy
  would have to succeed in over 90 % of attempts before trying pays; it began
  at 0 %. Not trying is the correct answer to the reward as written.
- Options considered: (a) leave the pricing unchanged — the measured outcome
  is a policy that stops attempting insertion by iteration 70;
  (b) pay the existing success lump `step_task * steps_remaining` on the abort
  branch — **REJECTED, and it was this entry's first draft.** The lump is
  UNDISCOUNTED while PPO runs `gamma = 0.99`
  (`agents/rsl_rl_ppo_cfg.py:47`): at step 20 of 256 it pays `236*c` against a
  present value of `90.67*c` for continuing, i.e. **2.60x better than not
  aborting**, and the factor grows the earlier the abort fires (2.69x at step
  10). It would not have made the abort neutral; it would have made it a
  jackpot;
  (b') the same lump, discounted: `c * (1 - gamma^n)/(1 - gamma)` — correct
  arithmetic, but still a hand-computed terminal value and still an own
  construction;
  (c) (b') plus a larger `abort_payment` sized to deter on this reward scale —
  rejected because no source supplies that magnitude;
  (d) leave the abort alone and raise the approach gradient instead — the
  plateau is real but worth +5.72 against a −55 asymmetry, and the kernel
  widths are closed by D-109 point (9) (solved from measured tolerances, not
  tunable), so this is blocked as well as secondary;
  (e) mark the force abort `truncated` instead of `terminated` and let
  rsl_rl bootstrap `gamma*V(s)`;
  (f) CaT's graded termination probability (Alg. 1, two lines) — published,
  tested, PPO-native, keeps the deterrent in expectation, but needs a Delta
  function this task has no measured basis for.
- Decision: **(e)**. `compute_dones` returns the force abort in `truncated`.
  The Isaac Lab rsl_rl wrapper then puts it into `extras["time_outs"]`
  (`isaaclab_rl/rsl_rl/vecenv_wrapper.py:162`, active because
  `is_finite_horizon` is False by default and this env does not override it),
  and rsl_rl's `process_env_step` adds `gamma*V(s)` back — the same machinery
  the timeout already uses. `abort_payment` is still paid on the aborting
  step, so the deterrent is exactly the number D-114 wrote and nothing more.
  No lump, no new constant, no discounting arithmetic of our own.
  NOT changed by this entry: `abort_payment` (still D-114's `[open]`, set to
  −1.0 by the 2026-09-01 inbox entry), `F_max` (raising it is not the fix —
  `force_abort_rate` already reached 1.0 at 300 N), and the kernel widths.
- Rationale: The bootstrap is published practice for exactly this situation.
  Pardo et al. 2018 Eq. (6) gives the rule (`y = r` at environmental
  terminations, `y = r + gamma*v(s')` otherwise) and Sec. 5 extends
  partial-episode bootstrapping beyond time limits to "any early termination
  causes". The force abort is a designer-imposed safety stop, not a terminal
  state of the task's own MDP — the same category as the step cap D-113
  already routes through `truncated`. Bootstrapping also gets the discounting
  right for free, because `V(s)` is already a discounted value; that is the
  decisive advantage over (b) and (b'), which compute a terminal value by hand
  and got it wrong on the first attempt.
  That the diagnosis is real and not our artefact is now citable: a positive
  per-step reward turning a forfeiting termination into a large implicit
  penalty, with a stand-still local optimum, is stated by Kobayashi 2023
  §III-A, ARS §4.1 and DAC §4.1.
- Accepted risk — the literature's PREDICTION, not a speculation. Four
  primary sources say the forfeiture is a useful deterrent and that removing
  it costs: DeepMimic Table 5 (removing early termination roughly halves
  return on a positive-reward task), ET-MDP Prop. 1/Eq. 15 (the forfeiture
  carries a correctness guarantee that `abort_payment` alone would then have
  to bear), Kobayashi 2023 §V (the analogous handling repeats an overcurrent
  failure on real hardware), Beltran-Hernandez 2020 Table II (removing the
  penalty roughly doubles collisions). The pass condition for the next run is
  therefore two-sided: `force_abort_rate` must NOT collapse to 0 and
  `mean_max_depth_mm` must stay above 0, but a rate near 1.0 together with a
  high `force_norm_n` p95 is the failure mode these sources predict.
- Consequence, NOT decided here: the shipped SUCCESS lump has the same
  undiscounted defect. `compute_rewards_insertion`'s docstring calls early
  success "slightly preferred"; at `gamma = 0.99` the factor is ~2.6, not
  slight. That is a separate entry and a separate change.
- Sources: RT-131 TensorBoard (`Policy/mean_noise_std` 0.0051 @299,
  `Loss/entropy` −23.2139 @299, `mean_max_depth_mm`, `force_abort_rate`,
  `success_rate`, `Train/mean_reward`) and `demo_metrics.json`, run
  `09-01_23-46-59_offset5mm_currOFF_start+30mm_seed42`; the arithmetic above,
  computed offline from `insertion_math.kernel_sum`/`squash` and
  `insertion_tasks_cfg.KERNEL_*`;
  `docs/reference/literature_check_abort_return_forfeiture_2026-09-02.md`
  (Pardo Eq. (6) and Sec. 5; CaT Alg. 1; DeepMimic Table 5; ET-MDP Prop. 1;
  Kobayashi 2023 arXiv:2308.12772 §III-A and §V; DAC §4.1/§4.2;
  Beltran-Hernandez 2020 arXiv:2003.00628 v3 Eq. (6) and Table II);
  `isaaclab_rl/rsl_rl/vecenv_wrapper.py:162`;
  `agents/rsl_rl_ppo_cfg.py:47` (`gamma = 0.99`); D-113 (the exits), D-114
  (the force abort and `abort_payment`), D-109 point (9) (kernel widths are
  closed).

## D-165: A depth-progress term is paid beside the single SDF distance
- Date: 2026-09-02
- Status: proposed (pending verification)
- Stream: rl-code
- Amends: D-109 point (3) ("ONE SDF-distance term replaces approach,
  depth, yaw and align") and point (4) ("no progress term, hence no
  gate"). The SDF term stays as it is; a depth term returns BESIDE it.
  D-109 point (9) (kernel widths solved from tolerances) is not touched.
  D-157's lateral gate is reused, not re-argued.
- Context: RT-134 (D-164 A/B against RT-131, rung 0, start +30 mm, seed
  42) measured the policy descending 30 mm into stage 1, landing on the
  stage-2 shoulder **4.88 mm beside the axis (median; max 9.34 mm; play
  0.8 mm per side in y, 0.29 mm in x)**, pressing at 135 N mean (p95
  166 N, never 300 N) and staying there: `mean_max_depth_mm` 0.0 at
  iteration 191, `force_abort_rate` 0.0, reward 55.94 (RT-131: 57.66).
  The user's screen showed the same pose. D-164 had done what it was
  built for — contact is back (RT-131 never touched, force p95 41.9 N) —
  but nothing pays for the last millimetres.
  MEASURED, RT-135 (`check_seated_success.py --reward-curve
  --curve-lateral-y-mm`, 1 mm above the entrance plane, lateral 0 to
  8 mm): `kernel_sum` 0.2462456 at y = 0, 0.2461874 at y = 4.88 mm,
  monotone in between; sdf 14.2557 → 14.3669 mm. **The whole 4.88 mm of
  lateral error is worth 5.82e-5 per step**, i.e. 0.015 over a 256-step
  episode, against a time penalty of 3.9e-3 per step (67x larger). At an
  sdf of 14.3 mm the mid kernel (a = 998 /m) reads ~6e-7 and the fine
  kernel (a = 5094 /m) is numerically zero; the coarse kernel alone is
  alive, with a slope of ~0.5 per metre of sdf. The policy CAN see the
  offset — `tip_rel` x, y are observation channels 12–13 (D-107 (4)) —
  it is unpaid, not blind. The counter-proof (coarse margin narrowed to
  10 mm) moved the same value 18x, so the instrument reads the kernel it
  claims to read; its `< 1e-3` threshold in the expectation was an
  arithmetic error of the expectation (sdf assumed 36 mm), recorded in
  `rt_logs/VERDICTS.md`.
  The proxy's own reward docstring names this pose: "Parking the tip on
  the pocket with the peg lying across it … leaving that pose costs
  before it pays" (`proxytask_env.py:164-173`).
- Options considered: (a) widen the kernels — closed by D-109 point (9),
  restated by D-161 and D-164 (d); not reopened. (b) The proxy's linear
  lateral term `-w * ||tip_rel||` (w = 2.0): the one existing form with a
  gradient at depth 0, worth ~0.01 per step across 4.88 mm, and the term
  the proxy records as having CREATED the parking attractor; rejected.
  (c) A reward for "searching" — moving on the surface: no published
  source pays for search (`literature_check_reward_2026-08-26.md` §2–§7;
  `literature_multiphase_insertion_reward_2026-09-01.md` §2.1, §S1
  verdict); every source pays for getting closer and search emerges; a
  movement term rewards jitter and fights the action-rate penalty
  head-on; rejected as own construction. (d) The proxy's gated
  depth-progress term `w_depth * (new_max_depth - max_depth)`
  (`proxytask_env.py:193-195`, `w_depth = 100`,
  `proxytask_env_cfg.py:130`), which carried the square peg to ≥ 99 %
  at 1 mm clearance and ±5 cm hole scatter (proxy HANDOFF, D-034
  ladder); Inoue 2017 pays its insertion phase the same way (normalised
  remaining depth, `…multiphase…` §T1). (e) Sampling the start height
  from a range instead of a fixed +30 mm (IndustReal §IV.G, SBC): the
  literature's answer to "the good path makes the dense reward worse
  first" (`…multiphase…` §4.3, §S2.2); planned as the NEXT change and
  its own entry, not folded into this one.
- Decision: **(d)**. `compute_reward_terms_insertion` gains the row
  `progress = w_progress * depth_progress * scale`, where
  `depth_progress` is the metres of NEW gated maximum depth the part
  reached this step — the increment of the env's existing `_max_depth`
  buffer, the same buffer `mean_max_depth_mm` reports, D-157's box gate
  included — and `scale` is the SAPU factor. The row sits OUTSIDE
  `step_task`, so the success lump (`step_task * steps_remaining`) never
  multiplies a one-off delta by the steps remaining. Weight
  `w_depth_progress = 100.0`, the proxy's number, carried as `[proxy]`
  in `PROXY_TASK_VALUES` so the startup report and `demo_metrics.json`
  print it as such; the guard refuses a negative weight; zero reproduces
  the pre-D-165 reward bit for bit (`env.w_depth_progress=0`). The row
  is logged as `Episode_Reward/progress` (the 2026-09-02 per-term log).
  The seated identity test pins the weight to 0 — its teleport jumps the
  buffer 0 → 34 mm in one step, which `expected_reward` does not model.
  [UNVERIFIED FOR THIS SETUP, audit 2026-09-13: the term answered the joint_pd
  standstill of RT-131/RT-134. Under OSC no standstill was seen; RT-179
  (AutoDR, old reward) hovered "engaged" instead, and Laufplan step 6b now
  watches for engaged-hover. w_depth_progress 100 stays [proxy] and is a
  hypothesis without a post-boundary measurement.]
- Rationale: The term telescopes over an episode to `w * max_depth`, so
  it cannot be farmed by moving up and down, and it is bounded (full seat
  0.036 m → 3.6). It pays only for depth INSIDE the D-157 box, so depth
  beside the pocket (RT-107) pays nothing; SAPU zeroes it for a
  tunnelled part; outside the lump it cannot be bought out. It bridges
  exactly the band the kernels leave flat: from the entrance plane down
  to where the mid kernel wakes (sdf ≲ 3 mm), roughly the first 30 mm of
  entry, where RT-119/RT-123 measured the kernel sum moving from 0.246
  to 0.295 only. NAMED LIMIT: the term pays nothing until a corner is
  in. From 4.88 mm off, entry needs a 0.8 mm x 0.29 mm window; this term
  does not by itself move the part those millimetres, and the run with
  this change alone is expected to reproduce RT-134's pose — that is
  the A/B it exists to give. The start-distribution change (option (e))
  is what teaches the last millimetres from inside; the two are one
  package and are run as two hypotheses.
- Accepted risk: the weight is the proxy's number in a different reward
  scale (the proxy's per-step approach term was 2 per metre; ours peaks
  at 1.0 per step at the seat). Its sensitivity here is UNKNOWN and is
  not asserted anywhere — the offline checks pin the FORM (per metre,
  SAPU-scaled, outside the lump, zero = old reward), not the size.
- Sources: RT-134 (`rt_logs/VERDICTS.md` 2026-09-02 10:28:39;
  `demo_metrics.json` of run
  `09-02_10-28-39_offset5mm_currOFF_start+30mm_seed42`); RT-135
  (`rt_logs/VERDICTS.md` 2026-09-02 12:05:31 and its Nachtrag;
  `rt135_lateral_curve.json`, `rt135_lateral_curve_counterproof.json` on
  the training PC); RT-119/RT-123 (the height curve);
  `Generalisierung-Angle-Probe/…/proxytask_env.py:164-173, 191-195`,
  `proxytask_env_cfg.py:130`, proxy `HANDOFF.md` (99.65 % at ±5 cm);
  `docs/reference/literature_check_reward_2026-08-26.md` §2–§7;
  `docs/reference/literature_multiphase_insertion_reward_2026-09-01.md`
  §2.1, §4.3, §S1, §S2.1 (Trott et al. 2019, distance shaping and local
  optima), §S2.2 (IndustReal §IV.G), §T1 (Inoue 2017); D-107 (4);
  D-109 (3), (4), (9), (10); D-157; D-164;
  `C:\IsaacLab\…\direct\anymal_c\anymal_c_env.py:35-47, 142-157, 186-193`
  (the per-term log pattern).

## D-166: The search force is 20 N, which DERIVES osc_kp_pos and closes it at 100

- Date: 2026-09-04
- Status: proposed (pending verification — the first PPO run reads
  `force_norm_n` p95 back)
- Stream: rl-code
- Amends: the `osc_kp_pos` placeholder in `RL_PLACEHOLDERS`
  (`insertion_env_cfg.py`), which named this rule and left `F_search`
  open. Decision (4) of the inbox entry "Audit 2026-09-03 (a)" is thereby
  answered. D-158 (the force-abort limit) is NOT touched here; see D-167.
- Context: the rule was already written into the placeholder text and the
  startup report already computes it:

      kp <= F_search / (Lambda_max * step_limit)

  Both inputs are measured or published, and only `F_search` was open.
  `Lambda_max` = 8.82 kg, the task-space inertia the startup report reads
  from PhysX at the reset pose (RT-143a/b, `rt_logs/VERDICTS.md`
  2026-09-03; the report printed the resulting bound as 113.4 in the same
  run). `step_limit` = `osc_pos_step_limit_m` = 0.02 m.
  The published search-phase band is 1-20 N
  (`docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`);
  which number inside it is ours was the open decision, and the user
  settled it on 2026-09-04: **20 N, the top of the band.** The user also
  ruled that this question does NOT go to the supervisor.
  WHY THE FORCE BOUNDS THE GAIN AT ALL, and why this stopped being an
  abstract rule on 2026-09-04: under the Factory-form re-anchoring the
  OSC's steady push is `Lambda * kp * Delta`, because the pose error is
  held at exactly `Delta` every physics step. RT-147 (PASS, 2026-09-04
  11:50:58) measured the velocity half of that same law
  (`v = Delta*sqrt(kp)/(2*zeta)`, rate-independent), and the force half is
  consistent with RT-144c: `8.82 * 500 * 0.0005` = 2.205 N predicted
  against 2.0209 N read, 9 % apart. The gain therefore sets the maximum
  force the robot can apply, directly.
- Options considered: (a) `F_search` = 1 N, the bottom of the band ->
  `kp <= 5.7`, which at the 0.02 m step limit gives a maximum speed of
  `0.02*sqrt(5.7)/2` = 24 mm/s and a maximum push of 1 N; rejected as
  needlessly soft — nothing in the band argues for its bottom, and it
  would make every contact phase slower than the published examples.
  (b) `F_search` = 20 N -> `kp <= 113.4`. (c) Keep 300 N (the current
  `force_abort_f_max_n`) as the reference -> `kp <= 1700`, which is
  outside the published band by two orders and would re-license the
  ramming the controller switch was made to remove; rejected.
  (d) Set `kp` to the bound itself, 113.4; rejected as a number with a
  false air of precision — it is an upper bound computed at ONE pose, not
  a target.
- Decision: **(b), and `osc_kp_pos` STAYS AT 100.0.** The value does not
  change; its STATUS does. It ceases to be a placeholder and becomes the
  largest round number under the derived bound of 113.4 N/m. Its
  consequences at the 0.02 m step limit: maximum steady push
  `8.82 * 100 * 0.02` = **17.6 N**, inside the band; maximum speed
  `0.02 * sqrt(100) / 2` = **100 mm/s**, which crosses a rung-0 episode's
  65 mm of travel in 0.65 s.
  [UNVERIFIED FOR THIS SETUP, audit 2026-09-13: Lambda_max 8.82 kg was read at ONE
  reset pose (RT-143, +30 mm on the pocket axis). Under AutoDR (D-178, D-179)
  the reset pose spans 30 mm lateral, up to 120 mm height and 10 deg tilt;
  Lambda_max there is unmeasured, so the 17.6 N push bound is unproven for
  this setup. F_search 20 N is a user decision and is not touched. Re-read
  Lambda from the startup report at the AutoDR ceilings.]
- Code state, and it is deliberately NOT changed by this entry:
  `osc_kp_pos` still carries its `[placeholder]` row in
  `RL_PLACEHOLDERS`. The attempt to relabel it in place was made on
  2026-09-04 and REVERTED, because two checks refused it and both were
  right: `check_insertion_math.py:1244` requires every row of that table
  to carry the literal "[placeholder]" -- the table means "invented" by
  construction -- and `:1254` forbids a mark from restating the value it
  labels, which the proposed text did (it wrote the gain back into its
  own prose, the `the-value-is-written-back-into-its-own-mark` defect
  class). A DERIVED number therefore does not belong in that table at
  all, and letting one GRADUATE out of it touches four files (the cfg,
  the six-name list in `check_env_wiring.py:1749-1754`, the mark
  contract in `check_insertion_math.py`, and the D-080 counter-proof
  mutations that sit on both). That is its own change, with its own
  review, and it is NOT blocking: the value is unchanged either way and
  THIS entry is the home of the fact. Until it is made, the cfg row is
  stale and DECISIONS.md governs.
- Consequence for kp 500: **excluded.** It would push
  `8.82 * 500 * 0.02` = 88.2 N, more than four times the top of the band.
  RT-144c's apparent win over RT-144b was a travel-time race and not a
  kp finding (HANDOFF-RL § Open NEXT 0, RT-147); nothing else argues for
  it. The two-value measurement the placeholder text called for is
  therefore closed, not pending.
- Accepted risk: **the margin is 13 %, and it rests on one pose.**
  `Lambda` is configuration-dependent and 8.82 kg was read at the reset
  pose only. The push exceeds 20 N as soon as `Lambda` exceeds 10.0 kg
  anywhere along the insertion, and no run has read `Lambda` at any other
  pose. This is not hedged in the code: the first PPO run reads
  `force_norm_n` p95 back, and if it stands above 20 N the bound was
  computed at the wrong pose and `kp` moves, not the band. A second risk
  is that `Lambda_max` is the largest eigenvalue, so the bound is the
  worst-case DIRECTION at that pose; a force along a softer direction
  stays below it, which makes the bound conservative in that one respect.
- Sources: `insertion_env_cfg.py` `RL_PLACEHOLDERS` entry `osc_kp_pos`
  (the rule, quoted verbatim in the RT-144/RT-147 logs);
  RT-143a/b (`rt_logs/VERDICTS.md` 2026-09-03: `Lambda_max` 8.82 kg,
  cond(J) 7.78, kp bound 113.4); RT-147 (`rt_logs/VERDICTS.md`
  2026-09-04 11:50:58, `rt_logs/RT-147_expectation.md`); RT-144c
  (`rt_logs/VERDICTS.md` 2026-09-04 10:51:04, F_max lower bound
  2.0209 N); `docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`
  (the 1-20 N band); `docs/decisions_inbox.md` entry "Audit 2026-09-03
  (a)", Decision (4); user decision 2026-09-04 (F_search = 20 N, and not
  a supervisor question);
  `C:\IsaacLab\source\isaaclab\isaaclab\controllers\operational_space.py:422-424,
  :98-101, :444` (the acceleration law the force expression follows from).

## D-167: Under OSC the force abort is no longer the brake against ramming

- Date: 2026-09-04
- Status: proposed (pending verification — the first PPO run reads
  `force_abort_rate` and `force_norm_n` p95 back)
- Stream: rl-code
- Amends: nothing. It RECORDS what D-166 does to D-114's and D-158's
  force abort. `force_abort_f_max_n` keeps its value; D-164's routing of
  the abort into `truncated` is untouched.
- Context: D-114 built the force abort as the brake against ramming, and
  that was the right instrument under `joint_pd`, where the target
  INTEGRATES and the arm can keep pressing: RT-144a measured an F_max
  lower bound of 250.12 N with all six joints saturated 100 % of the
  descent (`rt_logs/VERDICTS.md` 2026-09-04 10:01:11). Under the OSC of
  D-166 the same run cannot happen. The steady push is
  `Lambda * kp * Delta`, and with `kp` = 100 and the 0.02 m step limit its
  ceiling is 17.6 N — the controller is itself the brake, and it brakes
  17 times lower than the abort threshold. `force_abort_f_max_n` is
  300.0 N, invented 2026-08-28 and re-set by the user on 2026-09-01 as an
  OBSERVATION WINDOW for the first PPO run under the reworked reward, not
  as a measured limit.
- Options considered: (a) lower the threshold to something the OSC can
  reach, e.g. just above the 20 N band; rejected for the baseline run —
  it would turn an unused guard into an active terminator on the very run
  that is supposed to measure the undisturbed force distribution, and the
  number would be invented a second time. (b) Remove the abort;
  rejected — D-164 already made it `truncated` rather than `terminated`,
  so its cost to a policy is small, and transient impact forces are
  UNMEASURED: nothing rules out a spike far above the steady push.
  (c) Keep 300 N for the baseline run and let that run measure what
  forces actually occur.
- Decision: **(c).** `force_abort_f_max_n` stays 300.0 N for the first
  PPO run, and its ROLE is recorded as changed: it is no longer the brake
  against ramming — D-166's gain bound is — but a guard against
  transients whose size nobody has measured. It stays labelled
  `[placeholder]`. What replaces the number is a reading, not an
  argument: `force_norm_n` p95 and `force_abort_rate` off the first PPO
  run. If the abort never fires and p95 sits far under 20 N, the entry to
  write next is the one that lowers or removes it.
  **[CORRECTION 2026-09-06: SUPERSEDED IN ITS NUMBER by D-169 (50.0 N) --
  but NOT by the route this entry laid out. The condition above has TWO
  halves and only the first held. The abort never fired
  (`force_abort_rate` 0.0 at all four RT-157 heights); `force_norm_n` p95
  read 25.79 / 27.70 / 28.01 / 29.07 N, i.e. ABOVE 20 N, not far under it.
  D-169 therefore does NOT cite this trigger. It lowers the limit on a
  different measurement: the per-episode force maximum of every SUCCESSFUL
  episode. Home: D-169.]**
- Accepted risk: a guard that cannot fire looks like a working guard.
  For the length of the baseline run the env has NO force ceiling in
  practice, and the only thing standing between the part and a hard
  contact is D-166's bound — which is itself computed at one pose
  (D-166, Accepted risk). If `Lambda` is much larger somewhere along the
  insertion, both the bound and this guard are wrong together, in the
  same direction.
- Sources: RT-144a (`rt_logs/VERDICTS.md` 2026-09-04 10:01:11, 250.12 N,
  six joints saturated); RT-147 (`rt_logs/VERDICTS.md` 2026-09-04
  11:50:58) and D-166 for the `Lambda * kp * Delta` ceiling;
  `insertion_env_cfg.py` `RL_PLACEHOLDERS` entry `force_abort_f_max_n`
  (the "observation window" wording, user 2026-09-01); D-114; D-158;
  D-164 and `insertion_math.py:1038`
  (`truncated = (timeout | force_abort) & (~success_now)`).

## D-168: The action-rate scale is derived from a noise budget, not imported from FORGE

- Date: 2026-09-05
- Status: proposed (pending verification — RT-154 reads the sigma
  trajectory back; expectation in `rt_logs/RT-154_expectation.md`)
- Stream: rl-code
- Amends: D-109 (8), which introduced the action-rate term with FORGE's
  scale 0.1 and carried the caveat that no measurement of this task stood
  behind that number. This entry executes that caveat. The FORM of the
  term (`-scale * ||a_t - a_(t-1)||_2` per step) is unchanged.
- Context: the term reads the SAMPLED action, not the clamped one
  (`insertion_env.py:870,883`, deliberately — pricing the clamped pair
  would make drift outside [-1, 1] free). Its expected value is therefore
  a tax on the policy's exploration noise sigma and on nothing else. With
  `a_t - a_(t-1) ~ N(0, 2 sigma^2 I_6)`, `E||.|| = sigma sqrt(2) E[chi_6]`
  and `E[chi_6] = 2.3501`, so the per-episode cost is
  `scale * sigma * (1 * 2.3501 + 255 * 3.3235) = scale * sigma * 849.8`
  (the leading term is the first step after a reset, where
  `_prev_actions` is zero). At `scale` 0.1 and the rsl_rl start sigma of
  1.0 (`rsl_rl_ppo_cfg.py:31`) that is **85 per episode**, against a
  TOTAL kernel income of **57** (RT-151c measured 0.2235/step over 256
  steps; the logged `Episode_Reward/kernels` sat at 56.4-57.1). RT-151c
  and RT-153 further showed the kernel is FLAT near the pocket axis, so
  the lateral gradient available to the task is under 1.2 per episode.
  The largest reliable gradient anywhere in the reward pointed at
  "shrink sigma", by roughly a factor of 70. RT-148b measured exactly
  that: sigma 0.99 -> 0.01 by iteration 500, with 97 % of the apparent
  return gain coming from the action-rate term alone. rsl_rl has no
  sigma floor (`rsl_rl/modules/actor_critic.py`, v3.0.1:
  `Normal(mean, self.std.expand_as(mean))`, no clamp), so nothing stopped
  the collapse. The policy optimised correctly; the reward was wrong.
- Options considered: (a) port FORGE's action EMA
  (`factory_env.py:213`, `ema_factor_range` [0.025, 0.1] in
  `forge_env_cfg.py:23`) and keep the scale 0.1. This is the published
  construction and would remove the sigma tax structurally, because the
  smoothed buffer moves far less than the raw draw. Rejected for this
  step: it changes the ACTION path, not the reward, so it belongs to a
  different layer of the differential diagnosis and to its own run; it
  also introduces an unmeasured `ema_factor` for this task, i.e. it
  trades one imported number for another. (b) Remove the term. Rejected —
  it is the only price on the unclamped command, and RT-149 already
  measured `|a|` up to 3.5. (c) Keep the form and DERIVE the scale from a
  stated noise budget for this task.
- Decision: **(c).** `action_rate_scale` 0.1 -> **0.0034**
  (`insertion_env_cfg.py`). The budget was written down before the run
  that tests it: at target sigma **S = 1.0** — `init_noise_std` itself,
  i.e. the sigma the run is meant to keep alive — the noise tax may cost
  at most **X = 5 %** of the 57 kernel income. That gives
  `0.05 * 57 / 849.8 = 0.00335`, rounded to 0.0034 (5.07 % at S).
  CROSS-CHECK against the source the old number came from: referred to
  the raw draw, FORGE's 0.1 on an EMA-smoothed buffer is worth 0.0018
  (ema 0.025) to 0.0073 (ema 0.1). 0.0034 lands inside that band — the
  form stays FORGE's, the number is this task's.
  [UNVERIFIED FOR THIS SETUP, audit 2026-09-13: the budget (5 % of a 57 kernel
  income) was read off RT-148b under the pre-D-184 reward. Under the current
  reward the income is unmeasured; 0.0034 stays in the code as a carried
  number, not a derived one. Re-derive from the per-term log of the first
  post-boundary run.]
- What it fixes and what it does not: it removes the sigma tax as the
  dominant gradient. It does NOT make the task solvable. The kernel is
  still flat (RT-151c, RT-153), so **success is expected to stay 0** in
  RT-154. The potential-based rebuild of the kernel is a SEPARATE
  hypothesis and must not ride in the same run.
- Accepted risk: the action-rate term is the only price on the UNCLAMPED
  command, and this cuts it by a factor of 29. The named failure is that
  the action mean drifts outside [-1, 1], every draw saturates the clamp,
  and exploration is dead again while sigma still reads healthy.
  Instrumented, not assumed: `action_abs_max_mean` and `action_sat_frac`
  are new `extras["log"]` curves (`insertion_env.py`), read back in
  RT-154. Second risk: a weaker rate price permits higher-frequency
  command chatter at the tip clamp wall; the readouts for it are
  `Episode_Reward/action_rate` minus the sigma prediction
  `0.0034 * 849.8 * sigma`, plus `force_norm_n` p95 and
  `force_abort_rate`.
- Falsifier: sigma still collapses to ~0.01 by iteration 500 under the
  reduced scale. Then the action-rate term was not the driver and the
  next suspects are `entropy_coef` (0.005) and the value layer, not the
  reward weights.

## D-169: The force-abort limit drops from 300 N to 50 N, and for the first time the number comes off a measurement

- Date: 2026-09-06
- Status: proposed (pending verification)
  [2026-09-14, inline: KEPT unchanged for the five-seed study (RT-201s1..s5,
  RT-202s1..s5), user decision; the justification and what the limit is
  NOT: `docs/decisions_inbox.md`, entry "The D-169 force-abort limit is kept
  for the five-seed study".]
  [2026-09-15, inline CORRECTION: SUPERSEDED for the restarted seed study.
  The user lowered the limit to 30 N (not off a measurement) and every seed
  restarts with 1500 fixed iterations (raised from 1000 the same day): `docs/decisions_inbox.md`, entry
  "Seed study restart: 1000 fixed iterations, a 30 N force abort, every seed
  from scratch". The value's one home is `insertion_env_cfg.py`.]
- Context: `force_abort_f_max_n` had been an invented number since
  2026-08-28: 50 N, then 100 N the same day, 60 N on 2026-08-31 (D-158),
  and 300 N on 2026-09-01 as an OBSERVATION WINDOW, so that the first PPO
  run under the reworked reward would report the policy's own force
  distribution instead of the limit. D-167 then recorded that under OSC the
  abort is no longer the brake against ramming -- D-166's gain bound is --
  and that the number should be replaced by a reading off the first PPO
  run. That run happened (RT-156), and RT-157 replayed its checkpoint from
  four fixed start heights (+30/+40/+50/+60 mm). The readings are now on
  file, so the observation window has served its purpose and is closed.
- Options considered:
  (a) 20 N, the search-force figure of D-166 and the upper end of the
      published search band (1-20 N). Rejected on measurement: it aborts
      50.5 % of the successful episodes at +40 mm and 36.1 % at +60 mm. A
      limit below the force the task needs does not make the policy gentler,
      it forbids the solution and rewards contact avoidance.
  (b) 30 N, just above the measured p99. Rejected: it still aborts ~3 % of
      the successful episodes, i.e. it prices a solution that works.
  (c) 60 N, D-158's number, reached by rounding up from the largest measured
      episode maximum (43.88 N). Rejected as the weaker of the two safe
      choices: it costs the same zero successes as 50 N while leaving more
      unused headroom, and 50 N is the smallest round number that is still
      free of measured cost.
  (d) Keep 300 N and remove the abort entirely. Rejected for the reason
      D-167 already gave: transient impact forces are unmeasured, and a
      guard that cannot fire looks like a working guard.
- Decision: `force_abort_f_max_n = 50.0`. Its `RL_PLACEHOLDERS` mark is
  rewritten from "INVENTED" to "no longer INVENTED", and the field STAYS in
  `RL_PLACEHOLDERS`, because a bound measured on our own simulation is not
  the same thing as a damage limit of the real part.
  [UNVERIFIED FOR THIS SETUP, audit 2026-09-13: the 50 N came off RT-157, a
  replay of the RT-156 policy with the contact penalty ON, clamp box 0.07 m,
  no observation noise, no AutoDR -- all before the run boundary (audit line
  at the top of this file). The value STAYS in the code, but it is no
  evidence for the D-184 reward: re-read the force maxima of successful
  episodes on the first post-boundary run that shows successes (RT-189 line)
  before treating 50 N as a limit.]
- Rationale: the abort reads the EMA-smoothed force the policy also observes
  (`insertion_env.py:1319`, `_get_dones` docstring), and the RT-157 traces
  record exactly that channel per step. `scripts/trace_outcome_split.py`
  takes the FIRST episode of every env -- one sample per env, none discarded
  by the fixed step cap -- and reports the per-episode maximum of that force
  for the successful ones. Over 1546 successful first episodes at +40 and
  +60 mm: a 50 N limit aborts 0, 40 N aborts 1 at each height, 30 N aborts
  3.0 % / 2.4 %, 20 N aborts 50.5 % / 36.1 %. 50 N is therefore the smallest
  round limit with no measured cost to the solution, and it is the first
  time this field is bounded by data rather than by a round guess.

  This is NOT the trigger D-167 wrote down. That entry made the lowering
  conditional on the abort never firing AND `force_norm_n` p95 sitting far
  under 20 N. Only the first half held: `force_abort_rate` is 0.0 at all
  four heights, but p95 read 25.79 / 27.70 / 28.01 / 29.07 N, above 20 N
  rather than far below it. The conflict is recorded rather than resolved by
  plausibility: D-166's 20 N is the SEARCH force that derives `osc_kp_pos`,
  while the numbers here are the JOINING force of a policy that actually
  seats the part. Which of the two is the reference for the thesis is still
  undecided (`HANDOFF-RL.md`, literature-check line).

  Named limits of the derivation, so no later reading overstates it: it
  describes ONE policy (RT-156, `model_250.pt`) at TWO of the four probed
  start heights, not the task; it is an upper bound inside the simulation,
  not a permissible load of the real part -- the suction pair is modelled as
  a rigid chain (D-041), there is no sim-to-real, and the datasheet anchor
  stays withdrawn; and the 43.88 N maximum it rounds up from was measured
  under `osc_kp_pos = 100`, so a change of gains invalidates it.
  Consequence for every earlier result: each abort and success rate up to
  and including RT-157 must be read under the 300 N limit, not this one.
- Sources: `rt_logs/VERDICTS.md`, entries 2026-09-06 (RT-157h30/h40/h50/h60
  and the RT-157h40t trace findings); `docs/figures/RT-157h{30,40,50,60}_demo_metrics.json`
  (`force_norm_n`, `force_abort_rate`); `scripts/trace_first_episode.py` and
  `scripts/trace_outcome_split.py` (both `--self-test`, both run on
  `RT-157h40t_trace.json` and `RT-157h60_trace.json`);
  `source/insertion/insertion/tasks/direct/insertion/insertion_env.py:1319`
  (the two terminal sources); D-158, D-166, D-167 (superseded in their
  number, corrected inline); D-114 (the abort's form); D-041 (rigid chain);
  `docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`
  (the published 1-20 N search band).
- Numbering note: the project rule says to take the next number from the
  highest on `main`. `main` stands at D-151 while this branch already holds
  D-152...D-168, because the merge to `main` is deliberately deferred
  (`HANDOFF-RL.md`, "Phase cut 2 -> 3 -> 4"). Following the rule literally
  would collide with seventeen existing entries, so the number is taken from
  this branch's highest instead.

## D-170: The start-pose solver gets a lateral offset, because aiming it at the pocket axis silently cancelled D-034's lateral randomisation

- Date: 2026-09-06
- Status: proposed (pending verification)
- Corrected 2026-09-09 by D-176: `start_lateral_offset` is ONE HALF WIDTH
  PER POCKET AXIS now, (x, y). The single number this entry describes is
  still accepted and still means the same half width on both axes, so
  nothing below changes meaning; the shape of the field and of
  `demo_metrics.json` `start_lateral_offset_m` did. Read D-176 before
  quoting the field's form or the hydra command from here.
- Corrected again 2026-09-11 by D-178: the offset is a DISK and the field is
  one number again, the radius. Point 4 below ("the ceiling is not set") is
  closed there: ceiling 0.030 m, gated by D-153's hole reach until the mesh
  is repaired.
- Context: D-034 randomises the fixture pose per episode
  (`fixture_pos_noise_xy`, 0.005 m at rung 0) so that the part does not
  start over the pocket, and `reset_joint_noise` was written to scatter the
  part tip sideways (0.005 rad = p95 3.43 mm, `insertion_env_cfg.py`). D-162
  then added the reset-time IK that puts the tool point at
  `start_tip_above_entrance` above the opening plane, because D-161 had
  measured the whole D-109 reward at 3.05e-23 on the 165 mm home pose. That
  solver aims at the pocket AXIS -- `tip_rel` x = y = 0 -- and it runs LAST
  in `_reset_idx`, after the fixture write. So it CANCELS the lateral half
  of both fields. D-162's own text decides only the HEIGHT ("THE START IS A
  HEIGHT, NOT A JOINT TABLE") and never mentions the side; the cancellation
  was never recorded anywhere. RT-160 is the measurement: over 17780
  episodes at `fixture_pos_noise_xy` = 0.005,
  `start_pose_solve_worst_residual_mm` = 0.0354 and
  `start_pose_solve_unconverged_resets` = 0 -- every episode started on the
  axis to within 0.035 mm. The user asked for a lateral start offset, tried
  the two existing fields, and correctly reported that the part never lands
  on an edge.
- Options considered:
  (a) Raise `fixture_pos_noise_xy` or `reset_joint_noise`. REJECTED: it
      cannot work, for the reason above. The solver overwrites the result of
      both. Raising them changes only where the pocket sits in the workspace
      and which arm configuration reaches it.
  (b) Switch the solver off (`start_tip_above_entrance = null`). REJECTED:
      that restores the 165 mm home pose and with it D-161's flat reward
      landscape, 3.05e-23. It trades the whole start-height decision away to
      get one lateral millimetre.
  (c) A separate off-axis probe script beside the env, in the shape of
      `check_seated_success.py --lateral` (D-157). REJECTED for this
      purpose: that probe teleports ~192 mm aside into FREE AIR to test the
      reward's lateral gate. It is a reward identity test, not a start
      distribution, and it cannot be used for training.
  (d) CHOSEN: put the offset in the solver's own goal. A new field
      `InsertionEnvCfg.start_lateral_offset` (metres, uniform +- per pocket
      axis, drawn per env at every reset) is added to `d_pocket[:, 0]` and
      `d_pocket[:, 1]`; the height command `d_pocket[:, 2]` is untouched, so
      D-162 stands unchanged. Default 0.0.
- Decision:
  1. The lateral start offset is a FIELD OF THE SOLVER, not of the fixture.
     The fixture noise (D-034) keeps its own meaning -- where the pocket
     sits in the workspace -- and the two are logged and tagged separately.
     The run tag calls this one `lat<N>mm`; `offset<N>mm` stays D-034's.
  2. THE DEFAULT IS 0.0 AND NOTHING IS DERIVED. No value is decided here.
     A run opts in per hydra (`env.start_lateral_offset=0.002`). At 0.0 the
     sampling branch does not run and the goal reduces to `0.0 - tip_rel`.
     That is not the same bits as `-tip_rel`: falsified offline over 100008
     doubles, the two differ for an exact `x = +0.0`, where the old form
     gave `-0.0`. They compare equal, carry the same norm and drive the same
     IK command, so the behaviour is unchanged and the word "bit-identical"
     is NOT used.
  3. A LATERAL OFFSET IS REFUSED BELOW THE OPENING PLANE. `__init__` raises
     when `start_lateral_offset != 0.0` and `start_tip_above_entrance_low`
     is negative. Two reasons, both load-bearing: the start IK commands
     position only, so an off-axis goal inside the pocket is the RT-120 wall
     contact at reset; and `_solve_start_pose` seeds the gated max depth
     from the reached pose on the argument that the part is on the axis, so
     off the axis the D-157 lateral gate does not hold and the seed would
     not be a gated depth. The guard makes that argument true by
     construction instead of by luck.
  4. THE CEILING IS NOT SET AND D-153 APPLIES UNCHANGED. The part's +y face
     reaches the fixture mesh's 1.5 mm shoulder hole at
     `HOLE_REACH_OFFSET_Y` = 7.8847 mm. Below that the exposure cannot
     occur; above it, `stage1_lateral_y_mm.over_reach` in
     `demo_metrics.json` must be read before a run is believed. No number
     above that is authorised here.
- Consequences:
  - `start_lateral_offset` is written to `demo_metrics.json`
    (`start_lateral_offset_m`), printed by the startup report in both
    states, and carried in the run-folder tag as `lat<N>mm`. The "off" state
    prints too, because "every episode starts on the axis" is exactly the
    fact that went unnoticed for four days.
  - Every run before 2026-09-06 started on the pocket axis, whatever its
    `fixture_pos_noise_xy` said. Any earlier claim that a run trained
    lateral recovery is wrong, RT-160 included. This is an
    `InBachelorErwähnen.md` line.
  - UNVERIFIED. Nothing here has run on the training PC. The offline
    evidence is `scripts/check_env_wiring.py` (156 checks pass, counter-proof
    passes) and the sign-of-zero test above. The first real reading is a
    replay of the RT-160 checkpoint at rising offsets.
- Sources: D-034 (fixture randomisation), D-162 (the start is a height),
  D-161 (the flat reward at the home pose), D-153 (the shoulder hole and its
  reopening clause), D-157 (the lateral gate and its free-air probe), RT-120
  (wall contact at a negative start), RT-160 (`demo_metrics.json`, the
  0.0354 mm residual).
- Number: taken as the highest on THIS branch plus one, not the highest on
  `main`, for the reason recorded under D-169: `main` stands at D-151 while
  this branch holds D-152...D-169.

## D-171: A potential-based alignment row is added beside the kernel; its margin is the RT-158 mouth-mode tilt, its weight one episode of time
- Date: 2026-09-06
- Status: proposed (pending verification)
- Context: RT-158 measured the failing part TILTED at episode end
  (`rot_beyond_yaw` median: success 2.17 deg, rim mode 5.42 deg, mouth mode
  15.34 deg; `rt_logs/VERDICTS.md` 2026-09-06 15:05:29). RT-168 showed the
  rotation stiffness is not what holds it (a free part rights itself to the
  8.52 deg cone at every gain), and RT-170 BEFUND 2 measured the D-109
  kernel sum nearly blind to tilt: 1.02e-5 per degree against 2.35e-4 per mm
  of depth, i.e. per action step (0.097 rad against 0.02 m) descending is
  worth ~82 times righting -- and that ratio is the KERNEL share alone;
  D-165's progress term pays a further 425 times the kernel share per 20 mm
  step inside the box. Two record corrections made on the same day, read
  off the code: the 8.11 deg `gate_min_alignment` sits only in the
  `_peg_geometry` INSTRUMENT, so there is no reward cliff to ramp; and the
  paying predicates (`in_pocket_cross_section`, `in_success_region`) check
  no angle at all -- alignment is enforced only implicitly (33 mm of depth
  is unreachable at 8.5 deg without >0.2938 mm of interpenetration). The
  specification is therefore sound; what is missing is the GRADIENT while
  tilted. That is a shaping problem. The reward has no potential-based term
  today (`insertion_math.py`, the far-penalty docstring: "NOT
  potential-based, so Ng, Harada & Russell (1999) does not protect it").
  Constraint from the handoff: the three-axis clause ("no reward code until
  all three axes are measured") had one discharge, D-168; yaw (RT-152) is
  unmeasured. The user decided on 2026-09-06 to skip RT-152 and to review
  and build this row instead.
- Options considered:
  (a) RT-152 first (the merge order of the same day). SET ASIDE by the
      user: yaw is not the measured failure, and the row below is
      yaw-invariant, so the yaw slice cannot change its motivation. RT-152
      stays a thesis-must-state limitation.
  (b) Widen the kernel's rotation sensitivity (rebuild `kernel_sum` /
      `mean_outside_distance` without the outside clamp). REJECTED here: it
      is the kernel rebuild the three-axis clause names, and the hypothesis
      of why the clamped SDF mean is a weak rotation sensor is UNMEASURED
      (plan § 1.3 names the cheap falsifier).
  (c) A non-potential alignment bonus (per-step `w * cos theta`). REJECTED:
      it pays for HOLDING an aligned pose in free air, i.e. a hover income,
      and it changes the optimal policy; the hacking checklist's "dense
      proximity term without potential form -> hovering".
  (d) CHOSEN: a potential-based shaping row, `F = gamma * Phi(s') -
      Phi(s)`, `Phi = w_tilt * sech(a * theta)`, theta the angle between the
      part's +z and the pocket's -z, computed from the same quaternion the
      observation carries (D-109 Q3).
  Margin of the kernel: (i) the CAD cone `osc_tilt_clamp_rad` = 8.52 deg
  (the plan's proposal) or (ii) the measured failure tilt 15.34 deg. The
  user chose (ii) after the review: with (i) the pull per 0.097 rad step is
  0.194 w at 8.52 deg but 0.018 w at 15.34 deg -- below the 0.08/step
  contact penalty at the pose the policy actually fails in; and the 8.52 deg
  "rest" of RT-168 was a FREE part's (its own confound). With (ii) the pull
  is 0.108 w at 15.34 deg and 0.370 w at 8.52 deg.
- Decision:
  1. `insertion_math.REWARD_TERMS` gains a last row `tilt_shaping`.
     `tilt_cos_theta(body_quat, pocket_rot) = -axes_in_pocket[:, 2, 2]` --
     the minus sign of `_peg_geometry`'s `alignment`: seated is ANTIPARALLEL
     (the review caught the plan's "+z against +z", which would have paid for
     an upside-down part). `tilt_potential = w * 2 * squash(theta, a, 0)`,
     i.e. `w * sech(a * theta)`: the D-109 (6) curve at b = 0 and the D-109
     (9) width rule `a = arccosh(10) / margin`, `TILT_MARGIN_RAD` =
     15.34 deg, `KERNEL_A_TILT` = 11.1799 /rad (`insertion_tasks_cfg.py`).
  2. The row is `shaping_gamma * Phi(s') - Phi(s)`, with `Phi(s') = 0` on
     the SUCCESS step (the only true terminal, D-164) and the real `Phi(s')`
     on a truncation (bootstrapped, D-116/D-164). It is NOT SAPU-scaled and
     NOT in `step_task`, like `progress`, so the lump does not multiply it.
  3. `Phi(s)` for the reward is the potential of the pose the policy was
     SHOWN: `_get_observations` writes `_tilt_phi_prev` from the very
     quaternion the observation carries, on every call; `_get_dones` caches
     `_last_tilt_phi` of the current pose. The reset commands no orientation
     (`_solve_start_pose`, zero rotation delta), so there is no sampled tilt
     to seed from; the reset OBSERVATION is s_0 by definition, and whatever
     staleness it has is action-independent, so the telescoped sum stays
     action-independent. Masking the first step was rejected: it leaves
     `-Phi(s_1)` in the sum and pays 0.72 w for one 0.097 rad tilt at t = 0.
  4. `w_tilt` = 1.0, DERIVED from a per-episode budget, not typed: the time
     penalty sums to exactly 1.0 over an episode (`-1/T` per step), and 1.0
     says "from the cone edge to upright is worth one episode of time".
     Lower bound from the only extra cost righting has over sitting still,
     the action-rate penalty: `0.0034 * sqrt(6) * 38` control steps (RT-168,
     `tilt_recovery_probe.py --hold-steps`, one `env.step` each) = 0.3164
     over the term's total pull 0.9 w gives w >= 0.3516. Sensitivity decade
     {0.1, 1.0, 10.0} on the success rate (D-109 (9)(iii)); 0.1 is below
     the bound and is the predicted FAIL arm. `w_tilt = 0` reproduces the
     pre-RT-171 reward bit for bit -- the regression switch.
  5. `shaping_gamma` has ONE home, `agents/rsl_rl_ppo_cfg.py`
     (`PPORunnerCfg.algorithm.gamma` = 0.99); `InsertionEnvCfg.shaping_gamma`
     reads it from there at import, the guard refuses a value outside
     (0, 1], and the startup report prints it. NAMED GAP: a hydra override of
     `agent.algorithm.gamma` does not reach the env.
  6. The three-axis clause is DISCHARGED for this one row, as it was for
     D-168: the row does not touch the kernel, targets the axis RT-170
     measured, and is yaw-invariant by construction. The kernel rebuild
     stays blocked; RT-152 stays deferred and thesis-must-state.
- Rationale: Ng, Harada & Russell (1999): for any Phi, adding
  `gamma * Phi(s') - Phi(s)` leaves the set of optimal policies unchanged,
  provided the absorbing state has one fixed potential (hence Phi = 0 on the
  success step; price: the success step pays about -w against a lump of
  ~615, 0.16 %). Discounted over an episode the row is `gamma^T Phi(s_T) -
  Phi(s_0)` whatever the path, so it can neither carry the success-over-
  failure ordering (the lump does: 615 against 0) nor be farmed by tilting
  and righting (the cycle sums to `(gamma^T - 1) Phi(s_0)` < 0). What it
  changes is the per-step gradient while tilted: at the cone edge a righting
  step pays +0.52 w under the CAD margin and +0.37 w under the chosen one,
  against a tilted step's income of about -0.084 today (kernel ~0, progress
  0, time -0.0039, contact -0.08). Reward-ordering test: it MUST be run
  discounted -- summed plainly the row charges every aligned step
  `(1 - gamma) Phi` and mis-ranks a hover; the logged per-episode sum is
  therefore negative by construction (about `-0.01 * L - 1` for an aligned
  success at step L) and is bounded in [-3.56, +0.99] at w = 1, which RT-171
  P4 reads as the phantom detector. Exploits searched and closed: righting
  in free air before descending (Phi saturates at w; the start is already
  near-aligned), the tilt-and-right cycle (telescopes), a wrong reference
  axis (pocket, not world -- checked offline with a tilted pocket), an
  upside-down part (cos theta = -1, Phi ~ 0), and the lump (outside
  `step_task`). Named new risk: the row prices the D-163 edge-first lean
  (8.52 deg costs `1 - Phi(8.52)` = 0.63 w until righted); RT-171 P5's
  third branch reads that. Offline evidence, laptop, numpy stand-in:
  `check_insertion_math.py` 243 checks (214 before) and its D-080
  counter-proof with three new mutations (sign flipped, factor 2 dropped,
  terminal not zeroed) and two cfg mutations; `check_env_wiring.py` 156 and
  its counter-proof; `check_seated_success.py --self-test` 76/76 (it pins
  `w_tilt = 0` like `w_depth_progress`). UNVERIFIED under Isaac until
  RT-171 runs (`rt_logs/RT-171_expectation.md`, written before the run;
  RT-172 is its control).
- Sources: Ng, Harada & Russell, "Policy invariance under reward
  transformations", ICML 1999; Tassa et al. 2018 (dm_control `tolerance()`,
  the D-109 (9) width rule); D-109, D-116, D-164, D-165, D-168; RT-158,
  RT-168, RT-170 and the two 2026-09-06 record corrections
  (`rt_logs/VERDICTS.md`); the plan file
  `weiter-aus-dem-vorigen-squishy-bird.md` § 3-5 and its § 9 review (laptop,
  not in git); `HANDOFF-RL.md` § Open 2d; Isaac Lab 2.3.2
  `direct_rl_env.py:398-410` (no forward between reset and observation).
- Number: highest across every branch is D-170 (`git grep` over all
  branch heads, 2026-09-06); `main` still stands at D-151.
## D-172: The alignment term (RT-171) is dropped; the next measurement is the tilt magnitude ladder under the unchanged reward

- Date: 2026-09-07
- Status: accepted (user decision 2026-09-07 after the RT-172 verdict; the
  ladder run itself is not yet planned and stays UNVERIFIED)
- Context: RT-170 measured the reward kernel nearly blind to tilt
  (`kernel_sum` 1.02e-5 per degree against 2.35e-4 per mm of depth,
  `rt_logs/VERDICTS.md` 2026-09-06 BEFUND 2), and RT-157/RT-158 had shown
  the failing poses of the RT-156 policy at +40 mm to be tilted (mode B
  15.34 deg). From that a potential-based alignment term
  `F = gamma*Phi(s') - Phi(s)`, `Phi = w*sech(a*theta)`, was proposed and
  reviewed (`HANDOFF-RL.md` § Open 2d, seven corrections) and reserved as
  RT-171 on `p5-reward`. Its pre-registered control was RT-172
  (`rt_logs/RT-172_expectation.md`): fine-tune the same checkpoint under
  the reward as it stands with the pocket tilted 0..5 deg, and read the
  three-way branch at P5. RT-172 ran (2026-09-06 18:57:14, git `3612660`)
  and met branch 1: success 1.0 in BOTH tilt bins (0-3 deg n 1224, 3-6 deg
  n 776), cumulative 0.9912 over 281768 episodes, force p95 26.54 N
  against RT-156's 27.27 N, 0 aborts at 50 N, sigma 0.57, curve saturated
  from about iteration 450, user stop at 632/2250. The branch text written
  before the run reads: "the current reward pays enough for a 5 deg pocket
  through the success lump and D-165's progress term. The alignment term
  (RT-171) then needs a motivation other than RT-170's kernel ratio, and
  the next question is the magnitude ladder, not the reward."
- Options considered:
  (a) Run RT-171 as planned (same checkpoint, 0..5 deg, one term more).
      REJECTED: its control already reads 1.0 in both bins; a term cannot
      show an improvement over 1.0, so the run could not discriminate.
  (b) Run RT-171 against a harder rung instead. NOT CHOSEN NOW: it would
      introduce the term and the rung in one run, two changes, while the
      harder rung has never been measured under the unchanged reward. It
      stays available as the fallback if (c) fails.
  (c) CHOSEN: drop the alignment term and climb the tilt magnitude ladder
      under the UNCHANGED reward, one rung per run, same checkpoint
      lineage, same 1024 envs (D-117 (c)).
- Decision:
  1. RT-171 is DROPPED. The number stays RETIRED, not free: it is written
     into `HANDOFF-RL.md`, `rt_logs/RT-172_expectation.md`, `VERDICTS.md`
     and the training report as the alignment-term run, and re-issuing it
     to another run would collide with those references. The next free
     number is RT-173.
  2. The alignment-term proposal and its review are NOT deleted; they stay
     in `HANDOFF-RL.md` § Open 2d as the fallback for the case that a rung
     of the ladder fails under the unchanged reward. Reopening it requires
     that measured failure, not a re-reading of RT-170.
  3. The next measurement is the tilt magnitude ladder: fine-tune the
     RT-172 lineage with a larger `fixture_tilt_noise_rad`, reward
     unchanged. The magnitude of the next rung is NOT set here. The one
     bound of ours is the controller cone `osc_tilt_clamp_rad` = 8.52 deg
     (CAD, `insertion_env_cfg.py:245`); a pocket tilted 8 deg leaves
     0.52 deg of approach tilt, and anything at or above 8.52 deg is
     outside what the controller may command. D-110 (4)'s cell-derived
     range still does not exist, so the rung value stays USER-SET and
     `InBachelorErwähnen.md` keeps its 2026-09-07 row on the missing
     source.
  4. Before the next rung runs, two instrument gaps named by RT-172 are to
     be weighed, not silently carried: the tilt-bin table exists only in
     the end window (P4, no per-block start value), and the height-bin
     table has one open bin above +30 mm (P4b NACHTRAG). Neither blocks
     the ladder; both limit what a rung can claim.
- Rationale: The reward hypothesis was falsifiable and was tested against
  its own pre-registered branch; the branch that fired says the reward is
  sufficient at 5 deg. Adding a shaping term to a task the current reward
  already solves at 1.0 buys no measurement and adds an own construction
  (the potential's margin, its axis, its reset value) that the thesis would
  then have to defend. The ladder asks the cheaper question first: where,
  under the unchanged reward, does the policy stop learning the tilt. Only
  a measured failure there gives the alignment term a target number.
- Sources: `rt_logs/RT-172_expectation.md` (branch text, written before
  the run), `docs/figures/RT-172_demo_metrics.json`, `rt_logs/VERDICTS.md`
  (RT-172 lines 2026-09-06/07; RT-170 BEFUND 2), `docs/runs/RT-172.md`,
  `HANDOFF-RL.md` § Open 2c (no source for the tilt magnitude) and 2d
  (the alignment-term review), D-110 (4), D-117 (c), D-165.
- Number: `main` stands at D-151, this branch at D-170, and `p5-reward`
  holds D-171 (read with `git show p5-reward:DECISIONS.md`); D-172 is the
  first number free across all three.

## D-173: Success is resolved per lateral-offset bin and per yaw bin before any offset or yaw ladder runs

- Date: 2026-09-07
- Status: accepted (laptop; the two tables are wired and offline-checked,
  but no training run has produced one yet, so every claim about their
  CONTENT stays UNVERIFIED)
- Stream: rl-code
- Context: RT-174 (2026-09-07 01:27:10, git `35d91b6`) trained the first
  lateral start offset, `env.start_lateral_offset = 0.006`, on top of tilt
  0..8 deg, and read `success_rate_recent` 1.0 with all three populated
  tilt bins at 1.0 over 768806 episodes (`rt_logs/VERDICTS.md` 2026-09-07).
  Its own expectation file names the hole this leaves, before the run:
  "There is NO `success_by_lateral_bin` (not built); the lateral half is
  read only through the overall rate and the tilt bins"
  (`rt_logs/RT-174_expectation.md`). The offset is drawn per episode and
  uniform over the box, so the overall rate is an AVERAGE over the whole
  draw. A policy that solves the draws near the axis and fails at the
  corner reads the same number as one that solves all of them, up to the
  share of the corner. RT-175 was handed over on the yaw axis
  (`env.fixture_yaw_noise_rad = 0.1047`) with the same hole; its
  expectation states it too. This is the argument D-037 already made for
  tilt and step C already made for start height -- the same instrument was
  simply never built for the other two axes of the user's target set
  (tilt / lateral / yaw, user 2026-09-07).
- Options considered:
  (a) Read the axes off replays instead (`play.py --trace-obs`,
      `trace_outcome_split.py`). REJECTED as the primary readout: the
      replay policy is deterministic (RT-154p NACHTRAG (a)), so a trace
      measures the trained policy at inference, not what the training
      distribution learned, and it costs a separate run per question.
  (b) Run a ladder of single-value offsets (the RT-161..163 shape: 2 / 4 /
      6 mm, one run each). REJECTED as the primary readout: three runs to
      answer what one run's window can answer, and each run then trains a
      DIFFERENT distribution, so the three numbers do not compose into the
      mixed-draw case that is actually being trained.
  (c) CHOSEN: two per-bin tables in the same trailing window that already
      carries tilt and start height, through the one existing binning
      routine, reported in `demo_metrics.json`.
- Decision:
  1. `_success_by_lateral_bin` and `_success_by_yaw_bin` are added to
     `InsertionEnv` and both go through `_rate_by_bin`
     (`insertion_env.py`), the same routine the tilt, start-height and
     force-abort tables use. No second binning implementation exists.
  2. Their value buffers `_recent_lateral_mm` and `_recent_yaw_deg` are
     filled in `_log_finished_episodes`, beside the tilt and start-height
     lines. That method runs at the TOP of `_reset_idx`, BEFORE the
     fixture-noise block redraws `_fixture_yaw` and before
     `_solve_start_pose` redraws `_start_lat_off`. This ordering is the
     correctness condition, not a detail: harvesting later would pair
     every outcome with the NEXT episode's offset and yaw and make both
     tables silently meaningless. `scripts/check_env_wiring.py` pins it.
  3. The lateral value is the HYPOT of the two pocket-axis components in
     mm, not a per-axis figure. Reason: the failure question is "how far
     from the axis did this episode start", which is one distance; a
     per-axis split would need two tables and neither would carry the
     corner, where a +-r box reaches r*sqrt(2).
  4. The yaw value is the MAGNITUDE in degrees. The pocket yaw is drawn
     symmetrically about 0 (D-038), so +4 deg and -4 deg are the same
     task, and a signed table would halve every bin count for nothing.
  5. Bin edges. Lateral: (2, 4, 6, 8) mm -- the three rungs of the
     RT-161..163 ladder plus one bin that catches the corner of the 6 mm
     box (8.49 mm). Yaw: (2, 4, 6, 8) degrees -- the user's stated yaw
     target is "minimal, max. 6 deg" (2026-09-07), so the last edge sits
     one rung above it. Both last bins are open at the top. The edges are
     LADDER RUNGS, not derived from any geometry; they are named here as
     the chosen readout resolution and nothing else rests on them.
     [Corrected 2026-09-11 by D-178 point 7: lateral (6, 12, 18, 24, 30) mm
     for the 30 mm disk, yaw (1, 2, 3, 4, 5) deg for the 5 deg ceiling. N
     edges give N bins (`_rate_by_bin`, `insertion_env.py:3164-3173`), so the
     tables of this entry have FOUR bins, the last one [6, inf).]
  6. Both tables are written to `demo_metrics.json` under
     `success_by_lateral_bin` and `success_by_yaw_bin`. Empty bins are
     kept with rate `null`, exactly like the tilt table: a bin that never
     filled is itself the finding.
  7. No lateral ladder and no yaw ladder is run before these tables are
     in a run's `demo_metrics.json`. RT-174 and RT-175 were handed over
     without them and their per-axis answer is therefore permanently
     missing; that is recorded, not repaired.
- Rationale: The cheapest instrument that answers the question, built from
  the routine that already exists, on the axes the project is about to
  climb. Nothing new is invented: the shape, the docstring argument, the
  empty-bin rule and the wiring check all copy the tilt table (D-037).
- Consequences: `demo_metrics.json` gains two keys, so any consumer that
  enumerates keys sees them; `compare_runs.py` reads named keys and is
  unaffected. Runs before 2026-09-07 have neither key -- a comparison
  across that date must not silently treat a missing key as an empty
  table. The env gains two deques of the same window length as the six
  that are already there.
- Falsified by: a run whose lateral or yaw table shows a rate that
  contradicts the overall rate in a way the ordering argument (2) would
  produce -- i.e. bins that track the NEXT episode's draw. The wiring
  check and its mutation `lateral-buffer-never-filled` /
  `yaw-bin-table-reads-the-tilt-buffer` exist to catch that offline
  (D-080), but only a run can show the numbers are real.
- Verification: `python scripts/check_env_wiring.py` -- five new checks
  green among 161; `python scripts/check_env_wiring.py --self-test` --
  COUNTER-PROOF PASSED with the two new mutations. Laptop, 2026-09-07.
  NOT verified on the training PC.

## D-174: A run is a probe or an acceptance run; only acceptance runs carry a number, and they carry three seeds

- Date: 2026-09-07
- Status: accepted (laptop; the explicit seed line is offline-checked, but no
  acceptance run has been fired, so every claim about seed SPREAD stays
  UNVERIFIED)
  **Corrected 2026-09-11 by `Laufplan_Phase5.md`:** point 3 no longer holds
  for Phase 5 — training seeds 1–5 per condition, stop bar 0.995.
- Stream: rl-code
- Context: `PPORunnerCfg` set no `seed`. Every run RT-104 … RT-175 therefore
  fired the base class default without anyone choosing it: IsaacLab 2.3.2,
  `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:141`, `seed: int = 42`,
  docstring "The seed for the experiment. Default is 42."
  `scripts/rsl_rl/train.py:28` gives `--seed` the default `None`, and
  `scripts/rsl_rl/cli_args.py:71-75` overrides `agent_cfg.seed` only when the
  flag is typed, so omitting it leaves 42 in place. Exactly one run departed
  from it, RT-107 with `--seed 0` (`rt_logs/VERDICTS.md`, folder
  `08-30_23-55-48_offset5mm_currOFF_seed0`). No configuration has ever been
  repeated under a second seed. Two consequences the report cannot carry:
  every statement in `docs/Blockberichte/Bericht_Trainingsdokumentation.tex`
  rests on n = 1, and the RT-156 → RT-172 → RT-173 → RT-174 → RT-175 lineage
  is one `--resume` chain, so no result stands without its predecessors.
  `scripts/rsl_rl/train.py:287` already names a repeated-seed protocol
  (D-053) that this project has never executed.
- Options considered:
  (a) Leave the seed inherited and state n = 1 as a limitation. REJECTED: the
      limitation is real either way, but an inherited value cannot be cited,
      and a reader cannot tell an unset field from a chosen one.
  (b) Repeat every rung of the manual ladder under three seeds. REJECTED on
      cost, not on method: the ladder has five rungs to date and each run
      costs hours on the single training PC. The rungs are the development
      path; only its end state is claimed.
  (c) CHOSEN: name the value explicitly, and split runs into two classes so
      the cheap class stays cheap and the expensive class is the only one
      that is cited.
- Decision:
  1. `seed = 42` is written explicitly in
     `source/insertion/insertion/tasks/direct/insertion/agents/rsl_rl_ppo_cfg.py`.
     The VALUE does not change; only its provenance does. `--seed` still
     overrides it and the run folder still carries it
     (`train.py:301-302`, enforced by `check_env_wiring.py:1072-1073`).
  2. A PROBE RUN (`Sondierlauf`) tests one hypothesis: one seed, `--resume`
     allowed, hand stop allowed. It is not a result of the thesis. Every run
     RT-104 … RT-175 is a probe run under this definition, retroactively.
  3. An ACCEPTANCE RUN (`Abnahmelauf`) closes a curriculum stage: frozen
     configuration, SEEDS 42 / 1 / 2, fixed iteration budget, no hand stop.
     Only acceptance runs carry a number in the thesis.
  4. 42 stays in the set so the acceptance run is comparable with the whole
     development path; 1 and 2 are arbitrary and are named as arbitrary. No
     seed is chosen or replaced after its result is seen.
  5. One seed of the set runs WITHOUT `--resume`, from scratch. This
     separates "the reward solves the task" from "the chain of predecessors
     solves it". A from-scratch failure is itself a finding — the curriculum
     is then load-bearing, not decoration — and RT-140 shows that outcome is
     possible.
  6. With n = 3, all three values plus the range are reported. No mean with
     standard deviation: three samples do not carry one.
  7. Per acceptance run the same six metrics are reported —
     `success_rate_recent`, `mean_max_depth_mm`, `force_norm_n.p95_n`,
     `force_abort_rate`, `stage1_lateral_y_mm.over_reach`, and the per-bin
     table of the axis the stage opened (D-037, D-173).
- Rationale: The value 42 is not the problem; an unread value is. Three seeds
  is the smallest set that shows a spread at all, and it is reported as a
  spread rather than as a mean because a mean over three draws invites a
  precision the sample does not have. The from-scratch seed is the only cheap
  test that separates the reward from the resume chain, and it costs one run
  of the three rather than a separate experiment.
- Consequences: `rsl_rl_ppo_cfg.py` gains one field, so a run started without
  `--seed` behaves exactly as before. Every existing report section keeps its
  numbers but is re-labelled a probe run. The acceptance run for the current
  stage (tilt 0..8 deg + lateral 6 mm + pocket yaw 6 deg) is not yet fired,
  so the stage is documented and not yet closed.
- Falsified by: an acceptance run whose three seeds disagree beyond the
  reporting granularity — e.g. a success rate that spans more than the gap
  between two curriculum rungs. That would mean the single-seed probe runs
  measured the seed and not the reward, and the whole development path would
  need re-reading.
- Verification: `python scripts/check_env_wiring.py` — 161/161 PASS with the
  explicit seed line; `python scripts/selftest_checks.py --offline` — PASS.
  Laptop, 2026-09-07. NOT verified on the training PC: no acceptance run has
  been fired, so points 3 … 7 are UNVERIFIED.
- Numbering note: the highest D-number across ALL branches was D-173
  (`p4-reward`); `main` stands at D-151 and is behind. D-174 is free
  everywhere.

## D-175: Saturation is read, not gated — no expectation file carries a P7

- Date: 2026-09-08
- Status: accepted (laptop; organizational, nothing to verify on the training
  machine — the rule governs how a log is judged, not what the code does)
- Stream: rl-code
- Context: Every expectation file since RT-172 carried a point P7 of the form
  "did the run finish learning?": Episode Reward flattens, entropy and sigma
  stop rising, sigma above 0.30 at the last block, value loss falling. It was
  scored like every other point — ERFUELLT / VERLETZT / NICHT BELEGT — and a
  NICHT BELEGT drags the whole run to UNKLAR under `/rt-check` step 6.
  P7 could never be closed from the console log alone. The trailing blocks it
  asks about live in the TensorBoard scalars, and `scripts/filter_rt_log.py`
  returns only the first matches per search term, so the last block is
  structurally out of reach. The scalars CSV is exported by hand on the
  training PC afterwards (`scripts/export_tb_scalars.py`), which means P7 is
  unanswerable at the moment the verdict is written, every time.
  The record shows exactly that, four runs in a row: RT-172, RT-174, RT-175
  and RT-176 each had P7 open at the first judgement and each needed a second
  `rt_logs/VERDICTS.md` line, days or hours later, to close it (`RT-174 P7
  ERFUELLT`, `RT-175 P7 ERFUELLT`, `RT-176 P7 GESCHLOSSEN`). In none of the
  four did P7 change the finding. RT-177 would have been the fifth: it was
  hand-stopped, so its P7 is unanswerable in principle.
  Meanwhile `CLAUDE.md` § `Training & Log-Analyse` already obliges every log
  analysis to report all six metrics — Episode Reward, Policy Loss, Value
  Loss, Entropy, Explained Variance, KL — in that order and each with its
  trend, before any verdict, and separately to report dying entropy and a
  diverging value loss as red flags unasked. P7 restated that obligation
  inside each run's own file.
- Options considered:
  (a) Keep P7 and accept the second pass. REJECTED: it produced four UNKLAR
      verdicts whose cause was a missing export, not a property of the run.
      A verdict mark that never discriminates is noise in the record.
  (b) Keep P7 but make it non-scoring ("report only"). REJECTED: that is what
      the six-metric rule already is. Two homes for one obligation is the
      failure mode `CLAUDE.md` names as "one home per fact".
  (c) Defer the whole verdict until the scalars CSV arrives. REJECTED: it
      couples the judgement of the run to a manual file transfer and would
      have delayed every verdict of the last week.
  (d) CHOSEN: strike P7 from the expectation protocol. Saturation is read
      from the six metrics and reported; it carries no PASS/FAIL mark.
- Decision:
  1. From RT-178 onward, no expectation file contains a P7 or any other
     saturation gate. Expectation points are reserved for claims a log can
     settle at the moment the verdict is written.
  2. Saturation remains a mandatory READING, unchanged and with no new home:
     `CLAUDE.md` § `Training & Log-Analyse` continues to own it — six
     metrics, in order, each with trend, before the verdict, plus the red
     flags reported unasked.
  3. A reward still climbing at the end of the budget is reported as such and
     is NOT a FAIL of the reward. The answer is more budget. This sentence
     moves from P7 into the branch-3 wording of each expectation file.
  4. In `rt_logs/RT-177_expectation.md`, P7 is marked STRICKEN with its
     original text kept, not deleted. The change was made after the user had
     reported a result for that run, and a silently removed point cannot be
     distinguished afterwards from one tuned to fit the outcome.
  5. Expectation files RT-172 … RT-176 are historical record and are not
     edited. Their P7 lines stand.
- Rationale: A check earns its place by being able to fail informatively at
  the moment it is asked. P7 could not: the evidence it needs arrives after
  the verdict, so its only reachable state at judgement time was NICHT
  BELEGT, which under the `/rt-check` rules is not a pass. Four consecutive
  runs demonstrate the cost — an UNKLAR that had to be walked back — and none
  demonstrates a benefit, because in no case did P7 alter what the run was
  found to show. Removing it loses no information: the six-metric report is
  strictly larger than P7's content and is already mandatory for every log
  analysis, so the reading survives while the misleading mark disappears.
  The asymmetry that makes this safe is the one RT-177 exposed: an unfinished
  run weakens a NEGATIVE result, where "was the budget too small?" is a real
  confound, and leaves a POSITIVE result intact, because further iterations
  cannot unlearn a solved task. Saturation therefore belongs to the
  interpretation of a weak result, not to the acceptance of any result — which
  is precisely a reading, not a gate.
- Falsified by: a run whose six-metric report is judged sufficient, whose
  conclusion is later overturned by the scalars CSV, and where a scoring P7
  would have caught it. That would show saturation needs a gate after all,
  and P7 — or a version of it answerable from the console log — must return.
- Verification: none required; the decision changes no code. Its effect is
  visible in `rt_logs/RT-177_expectation.md` (P7 struck, branch 3 rewritten)
  and in the absence of a P7 from RT-178 onward.
- Numbering note: highest D-number across all branches was D-174
  (`p4-reward`); `main` stands at D-151 and is behind. D-175 is free
  everywhere.
---

## D-176: The lateral start range is one half width per pocket axis, and its cfg default stays a bare number

- Date: 2026-09-09
- Status: proposed (pending verification)
- Superseded 2026-09-11 by D-178 for the SHAPE: the offset is a disk, the
  field is one number (the radius), and the x-versus-y question below is
  withdrawn, not answered. The hydra-merge rationale below still holds and
  is why the default stays a bare number.
- Context: D-170 introduced `start_lateral_offset` as ONE number, applied
  as a symmetric range on both pocket axes. The two axes are not the same
  question. The measured plays differ by a factor of about 2.7 (PLAY_X
  0.588 mm against PLAY_Y 1.600 mm), and D-153's shoulder-hole exposure is
  a +y reach (`HOLE_REACH_OFFSET_Y` = 7.8847 mm). With one number the study
  cannot ask how far the policy tolerates x independently of y. Phase 5
  also needs the reset to write the lateral row unconditionally: under
  `dr_mode='autodr'` the cfg field stays 0.0 for the whole run while the
  AutoDR boundary opens, so a guard that reads the field answers "no" on
  every episode of the run it exists for (plan risk R1).
- Options considered:
  (a) keep one number and scale the two axes by a fixed ratio derived from
  the plays -- rejected: the ratio would be a second, unmeasured decision
  and the axes could never be varied independently;
  (b) two separate cfg fields, `start_lateral_offset_x` and `_y` --
  rejected: two homes for one quantity, and every reader (the refusals, the
  startup report, the metrics file, the run tag, the AutoDR exclusion set)
  would have to name both;
  (c) one field holding a pair, with a default of `(0.0, 0.0)` -- rejected
  after measurement, see Rationale: it breaks every pre-existing hydra
  command;
  (d) one field holding a pair, default a bare `0.0`, with one resolver
  that accepts a number, a list and a tuple -- CHOSEN.
- Decision: `InsertionEnvCfg.start_lateral_offset` carries (x, y) half
  widths in metres. Its DEFAULT stays the bare number `0.0`.
  `insertion_env_cfg.resolve_start_lateral_offset` is the single place that
  reads the field's shape: a number means the same half width on both axes
  (the pre-B3 meaning, so RT-174's and RT-175's `=0.006` keep theirs), a
  list or tuple is (x, y). A wrong length, a negative half width, a string
  and a bool are refused, never repaired. `InsertionEnv.__init__` resolves
  the field once into `self._start_lat_half`; every other reader sees a
  pair of floats. The `lat != 0.0` guard around the reset write is removed,
  so the lateral row is written at every reset, at width 0 too. The run tag
  becomes `lat{x}x{y}mm` (before this a pair got no tag at all, because the
  test was `isinstance(_lat, (int, float))`), and `demo_metrics.json`
  `start_lateral_offset_m` becomes a pair -- a run recorded before this
  entry holds a bare number there, so a reader of both must accept both
  shapes.
- Rationale: the default's type is not cosmetic. Isaac Lab merges hydra
  overrides in `isaaclab/utils/dict.py::update_class_from_dict`. A LIST
  value takes branch 2a (`out_val = tuple(value) if isinstance(obj_mem,
  tuple) else value`) and is assigned wholesale, whatever the default is. A
  SCALAR value falls through to branch 4, `isinstance(value,
  type(obj_mem))`, and branch 5 raises when that fails. With a tuple
  default, `env.start_lateral_offset=0.006` -- the command RT-174 and
  RT-175 ran -- therefore dies at startup with "Incorrect type ...
  Expected: <class 'tuple'>, Received: <class 'float'>", before the
  resolver is reached. With `0.0` both command forms merge. The container
  the merge delivers for a list is a plain `list`, not an omegaconf
  `ListConfig`, because `isaaclab_tasks/utils/hydra.py` runs
  `OmegaConf.to_container` first; the resolver's sequence branch is
  duck-typed all the same, so the list and the tuple route share one line.
  Refusing rather than repairing a bad override follows the same rule as
  the other resolvers in that module: a silently repaired range would run
  the study against a distribution nobody asked for, and the whole point of
  Phase 5 is that the reported distribution is the applied one.
- Verification status: OFFLINE ONLY. `check_env_wiring.py` (191 checks) and
  `check_insertion_math.py` (250 checks) both pass with their D-080
  counter-proofs; the math checks EXECUTE the resolver through all three
  accepted shapes and all four refusals. The hydra merge was read from the
  installed Isaac Lab 2.3.2 tree and its `update_class_from_dict` executed
  against an extracted copy; it was never run against a live hydra, so the
  CLI-parse leg is UNVERIFIED until the Phase-5 smoke run.
- Sources: `IsaacLab/source/isaaclab/isaaclab/utils/dict.py`
  (`update_class_from_dict`, branches 2a and 4/5);
  `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py`
  (`OmegaConf.to_container` before the merge); D-170; D-153;
  `insertion_tasks_cfg.HOLE_REACH_OFFSET_Y`; RT-174 and RT-175
  (`rt_logs/VERDICTS.md`, `docs/figures/RT-174_demo_metrics.json`);
  Phase-5 plan sections 1 and 12 (risk R1); commits `a671aa6`, `dbbad59`.
- Numbering note: highest D-number across all branches was D-175
  (`p4-reward`, `p5-reward`, `p5-robustheit`); `main` stands at D-151 and
  is behind. D-176 was free everywhere.
## D-177: The action is a 6-D pose delta under task-space impedance — the controller switch of 2026-09-03 gets its number

- Date: 2026-09-03 (decided); numbered 2026-09-10
- Status: proposed (pending verification) — the controller RUNS as the cfg
  default and every later entry builds on it (D-166, D-167, D-168, D-171
  to D-176), but the supervisor question of Decision (2) below is still
  UNANSWERED, so the drive-value half of the switch is not fixed in the
  concept. The discriminating measurement named below (RT-129-form
  scripted insertion once under `joint_pd`, once under OSC, peak force of
  clean insertions compared, joint torques logged in both) was never run
  in that paired form: RT-144a measured the `joint_pd` leg alone
  (F_max lower bound 250.12 N, all six joints saturated 100 % of the
  descent, `rt_logs/VERDICTS.md` 2026-09-04 10:01:11) and the OSC leg was
  replaced by the analytic ceiling `Lambda * kp * step` = 17.6 N in D-167.
- Stream: konzept (decided) / rl-code (built)
- Supersedes: D-108 Decision (1). D-108 carries an inline correction
  pointing here. What D-108 called its *fallback* — "task-space delta
  actions + impedance control after the Factory pattern" — is the
  running code. D-108's hypotheses H1 and H3 lapse with it; H2 is the
  reason for the switch.
- Amends the LETTER of D-105 for exactly two values: see Decision (2).
- Context: D-108 kept six joint-delta actions on two legs — the proxy's
  >= 99 % at 2 mm play, and the warm start. The warm start fell one day
  later (D-110 (1)); the scripted gate fell on 2026-08-29 (inbox "The
  D-108 gate drops the scripted descent"). Since then D-108 stood on
  nothing. Measured under the joint-PD chain, no insertion below about
  50 N was ever observed, neither scripted (RT-129: 50-170 N) nor learned
  (RT-138: successes peak just under the 300 N abort, and the target lag
  is HIGHER in successes than in failures). A fresh run with a contact
  penalty reached 0 % (RT-140/141); the same penalty applied to the
  finished policy changed nothing (RT-139). The mechanism is the
  integrating position target without anti-windup: the commanded joint
  position runs away from the achieved one while the part is blocked, and
  the stiff USD drive converts that lag into force without bound. This is
  D-108's own H2, and it is what a compliant task-space controller is for
  (Factory, IndustReal).
- Options considered: (a) keep joint deltas and keep tuning the reward;
  (b) Isaac Lab `OperationalSpaceController` in effort mode with
  `gravity_compensation=True`, gravity kept ON; (c) the Factory pattern
  with robot gravity OFF. (c) was rejected on two grounds: it conflicts
  with D-105, and our part is WELDED into the robot USD (RT-45), so robot
  gravity OFF would make the part weightless and delete the 8.08 N seat
  weight RT-81 measured.
- Decision (user, grill session 2026-09-03), six branches:
  (1) The switch is decided at that moment, not after the measurement.
  The action becomes a 6-D pose delta under a compliant task-space
  impedance controller. The measurement is the proof, not the decision.
  **The thesis must state that the switch rests on indirect evidence** —
  no run of the joint-PD chain logged joint torques; the case is built
  from contact forces and target lag. `InBachelorErwähnen.md` carries the
  row since 2026-09-10 — the source entry ASKED for it on 2026-09-03 and
  it was never written; this numbering session found the gap and filled
  it.
  (2) Option (b). Gravity stays ON (D-105 core holds). Controller =
  Isaac Lab `OperationalSpaceController` with `gravity_compensation=True`,
  `inertial_dynamics_decoupling=True`, partial decoupling off,
  `impedance_mode="fixed"` — the settings of Isaac Lab's own gravity-ON
  unit test. Effort mode requires the PhysX drives to be INERT: stiffness
  and damping of all three actuator blocks go to 0. That breaks the
  LETTER of D-105 ("all drive parameters from the USD") for those two
  values — USD stiffness 9400.5 and damping 0.378 become unused; maxForce
  150/28, joint limits, velocity limits, armature and friction stay
  USD-authored. **SUPERVISOR QUESTION, still open:** "Gravity stays ON.
  The controller becomes task-space impedance in torque mode. For that the
  drive stiffness and damping must be 0; maxForce stays. Agreed?"
  `variable_kp` is deliberately NOT used: if the policy learns the
  stiffness while force is penalised, it goes limp, and a reward ablation
  would then measure the learned controller instead of the reward.
  (3) All three rotation axes stay free — Factory's upright clamp is NOT
  copied, because the real task needs tilt (D-071). The rotation target is
  clamped to at most 8.52 deg from the pocket axis (CAD), the same form as
  Factory's position clamp. Yaw stays free (D-106).
  (4) FORM from Factory, NUMBERS from this task. Target = current pose +
  clamped delta, re-anchored every physics step — that re-anchoring IS the
  anti-windup, and it bounds the commanded force at `F ~ Lambda * kp *
  step`. Hence `kp` is DERIVED, not adopted:
  `kp <= F_search / (Lambda_max * step_limit)`. Closed in D-166 at 100
  with `F_search` = 20 N and `Lambda_max` = 8.82 kg. Rotation stiffness 30
  is Factory's, carried as a labelled placeholder. Damping is critical
  (ratio 1.0), the identical rule in Factory and in the OSC.
  (5) Six joints, six task axes, so the null space is empty and
  `nullspace_control="none"` is the only consistent setting; the OSC
  raises on anything else.
  (6) Build and ask IN PARALLEL. The code is built and the supervisor gets
  the question of (2) together with the numbers. Nothing is fixed in the
  concept until the answer is in — the code runs OSC regardless.
- Where the decision lives in code: `control_mode = "osc"` is the cfg
  default (`insertion_env_cfg.py:210`); the controller is built in
  `_build_task_space_controller` (`insertion_env.py:1433-1442`);
  the drives are zeroed before the `Articulation` is constructed
  (`insertion_env.py:1313-1316`) and the startup report reads the gains
  BACK from PhysX so a silent no-op cannot pass as applied; the per-step
  clamps live in `_apply_osc`. Current numbers: `osc_kp_pos` 100 (D-166),
  `osc_kp_rot` 30, `osc_damping_ratio` 1.0, `osc_pos_step_limit_m` 0.02,
  `osc_rot_step_limit_rad` 0.097, `osc_pos_clamp_m` 0.05,
  `osc_pos_clamp_z_m` 0.10 [both corrected 2026-09-11 by D-180: 0.08 m and
  0.1435 m], `osc_tilt_clamp_rad` 8.52 deg,
  `osc_decimation` 8 (15 Hz policy rate; D-024's 60 Hz becomes the
  controller-only rate). `joint_pd` survives as a per-run override
  (`env.control_mode=joint_pd`).
- Still open, inherited from the source entry and NOT closed here:
  `osc_pos_step_limit_m` and `osc_rot_step_limit_rad` still sit in
  `RL_PLACEHOLDERS` — the tool-speed source that would derive them is
  MISSING, and the values are Factory's action thresholds. `osc_decimation`
  8 is likewise a labelled Factory placeholder, and the 256-step episode
  length is therefore OWED a re-derivation (it now lasts four times as
  long in wall-clock terms). The Jacobian condition number at the home
  pose is printed by the startup report as the open check on the
  decoupling's unit-mass model near singularity and at torque saturation.
- Sources: `docs/decisions_inbox.md` on branch `p4-konzept`, entry
  "Audit 2026-09-03 (a): D-108 wird neu geoeffnet -- der Regel-MODUS,
  nicht die Antriebswerte" (the full six-branch original, with its own
  source list); `docs/reference/konzept_audit_proxy_vs_real_2026-09-03.md`
  (K1); `HANDOFF-RL.md` section Frozen; DECISIONS.md D-024, D-071, D-105,
  D-106, D-108, D-110, D-149, D-166, D-167, D-168; `rt_logs/VERDICTS.md`
  RT-45, RT-81, RT-84, RT-129, RT-138 to RT-141, RT-144a; `C:\IsaacLab`
  `factory_env_cfg.py:160-172`, `factory_env.py:268,292-293`,
  `factory_utils.py:19-23`, `operational_space.py:93-101,478-494`,
  `test/controllers/test_operational_space.py:350-358`,
  `scripts/tutorials/05_controllers/run_osc.py:101-103`;
  `Geometrie_Fuegeteil_Aufnahme.tex:264` (8.52 deg).
- Numbering note: highest D-number across all branches was D-176
  (`p5-robustheit`); `main` stands at D-151 and is behind. D-177 was free
  everywhere.

## D-178: The lateral start offset is a disk of radius 30 mm with one one-sided AutoDR limit; the pose ceilings are the Phase-5 targets (tilt 10 deg, yaw 5 deg)

- Date: 2026-09-11 (disk, radius and tilt ceiling decided in the grill
  session of 2026-09-10; yaw ceiling lowered by the user on 2026-09-11)
- Status: proposed (pending verification) - built on the laptop in 5a7d295,
  offline checks green, UNVERIFIED on the training PC
- Stream: rl-code
- Supersedes: the SHAPE of D-176 (the (x, y) pair). Corrects D-173 point 5
  (bin edges) and D-170 point 4 (ceiling not set). Each of the three carries
  an inline pointer here. For the lateral, yaw and tilt rows it replaces the
  unnumbered inbox entry "The six AutoDR ceilings are DECIDED, not derived"
  (`docs/decisions_inbox.md`, 2026-09-10, ON HOLD).
- Context: D-176 made `start_lateral_offset` an (x, y) half-width pair so
  the study could ask x independently of y, and `autodr.py` `DR_DIMS`
  carries four lateral limits (`lat_x` lo/hi, `lat_y` lo/hi). Two facts in
  the code work against that shape. First, the only lateral readout is a
  RADIUS: `_log_finished_episodes` bins
  `torch.linalg.norm(self._start_lat_off[idx], dim=-1)`
  (`insertion_env.py:2879`, D-173 point 3). Four limits would move
  independently while the table can see only their combined distance.
  Second, the controller-box sufficiency check sizes the box for the square
  band (`scripts/check_env_wiring.py:3956-3971`): the corner reaches
  `r*sqrt(2)` and yaw inflates the sideways reach by `cos(psi) + sin(psi)`.
  The box therefore pays for a corner no table reads. The Phase-5 targets
  were set in the grill session of 2026-09-10 on the rule "the ceiling IS
  the target: the run is finished when every limit stands at 100 % and the
  policy seats": radius 30 mm, tilt 10 deg, start height 120 mm, yaw
  10 deg. On 2026-09-11 the user lowered yaw to 5 deg ("can still be widened
  if needed"). Today's `DR_DIMS` holds the RT-177 settings, all
  `[TESTWERT]`: lateral +-0.006 m per axis, yaw +-6 deg, tilt 8 deg.
- Options considered:
  (a) keep the square band of D-176 with four limits — rejected for the
  two reasons in Context;
  (b) a disk, uniform over its AREA, with one one-sided limit — CHOSEN.
- Decision:
  1. The lateral start offset is drawn on a disk of radius `R`:
     `r = R*sqrt(u)`, `phi = 2*pi*v`, with `u`, `v` independent and uniform
     on [0, 1]. The map from the pre-drawn unit values is a pure function
     `insertion_math.disk_offset(u_r, u_phi, r_max)`; `u_r = 1` gives
     exactly `R`.
  2. AutoDR: `lat_x` and `lat_y` (four limits) become `lat_r`, ONE-SIDED
     like `tilt` (one limit, `lat_r_hi`), and `lat_phi`, a table column
     without a buffer like `tilt_azimuth`. The limit count falls from 11 to
     8 with this entry alone.
  3. `start_lateral_offset` is ONE number again: the radius. A list or a
     tuple is refused with a pointer to this entry. The hydra form
     `env.start_lateral_offset=0.006` of RT-174/RT-175 stays valid (a number
     over a number). The run tag `lat{x}x{y}mm` becomes `lat{r}mm`.
     `demo_metrics.json` `start_lateral_offset_m` is a number again; readers
     keep accepting the pair of the D-176 runs.
  4. **The x-versus-y question of D-176 is WITHDRAWN, not answered.** The
     `lat_phi` column of the evaluation table is the one place where it can
     be asked again later.
  5. Ceilings, decided and not derived: `lat_r` hi_max 0.030 m; `tilt`
     hi_max 0.17453292519943295 rad (10 deg); `yaw` +-0.08726646259971648
     rad (5 deg). ~~No run may draw a radius above `HOLE_REACH_OFFSET_Y` =
     7.8847 mm before the fixture mesh hole is repaired and re-measured.~~
     **LIFTED 2026-09-12 by D-187**, after RT-183 re-measured the hole and
     RT-184 showed it is an order of magnitude below the reward's own
     sampling spacing (0.0845 mm2 against 0.901 mm between sample points).
     A run may draw `lat_r` up to the 0.030 m ceiling. Read D-187 before
     citing this clause.
  6. Tilt 10 deg lies above the cone 8.52 deg: ACCEPTED. The start IK
     commands position only (`insertion_env.py:658-659`), and
     `clamp_tilt_to_cone` measures against the pocket axis
     (`insertion_math.py:1472-1485`). At 10 deg pocket tilt the tool
     therefore starts 1.48 deg outside the cone, and the clamp pulls the
     first target in. Expectation for the AutoDR run, written now: a small
     force spike at step 1 on `tilt_hi` boundary envs. It goes into the
     run record; no rule follows from it.
  7. Bin edges (inline correction of D-173 point 5). `_rate_by_bin` makes
     ONE bin per edge: bin i is [edge[i-1], edge[i]), and the last bin also
     takes everything at or above the last edge
     (`insertion_env.py:3164-3173`). So the LAST edge is set to the ceiling.
     Lateral: (6, 12, 18, 24, 30) mm, five bins of 6 mm. Yaw: (1, 2, 3, 4, 5)
     deg, five bins of 1 deg. TILT: (2, 4, 6, 8, 10) deg, five bins of
     2 deg. [Tilt ADDED 2026-09-12, user. This point re-cut lateral and yaw
     on 2026-09-11 and LEFT THE TILT TABLE on D-037's superseded ladder
     (3, 6, 9, 12, 15) deg. Against point (5)'s 10 deg ceiling the last bin
     -- everything at or above 12 deg -- could never fill, and [9, 12) only
     ever saw 9-10 deg, so one of five bins was dead by construction and one
     was half used. The rule is the same for all three now: last edge = the
     ceiling, ceiling/5 x (1..5). A rung, not a derivation, the standing
     D-173 gave its edges. The reason it was missed has a home: the plan's
     target table carries rows for the lateral, height and yaw bins and NO
     ROW FOR THE TILT BINS. `check_autodr.py` now binds the last tilt edge
     to the `DR_DIMS` ceiling, both sides read from source, so the next
     ceiling change cannot pass silently.] Measured against that rule by
     simulation
     (10^6 draws, 2026-09-11): the old lateral edges (2, 4, 6, 8) mm put
     96.0 % of all draws of a 30 mm disk into the last bin; the old yaw edges
     (2, 4, 6, 8) deg give 40 / 40 / 20 / 0 % under a 5 deg ceiling, one bin
     empty. Equal-AREA lateral edges were named and not chosen, because the
     rungs must stay readable. Both sets are rungs, not derived — the same
     standing D-173 gave its edges.
- Rationale: The disk matches the only readout the study has. It also
  shrinks what the controller box must hold. The disk is rotationally
  symmetric about the pocket axis, so a fixture-yaw rotation maps it onto
  itself and the disk's own geometry does not change with yaw; yaw
  therefore drops out of the sideways reach: world xy reach
  `r + h*sin(theta) + step`, world z reach
  `h*cos(theta) + r*sin(theta) + step` (box numbers: the box decision of
  this series). [Warrant corrected 2026-09-12, critic finding in
  `HANDOFF-RL.md` § Open point 000: the wording written on 2026-09-11
  derived "yaw drops out" from the tool point sitting on the tool axis
  (`peg_tip_offset` = (0, 0, `FLANGE_TO_PART_BOTTOM`)). That is the
  `peg_tip_offset` argument and it carries a different claim; the
  rotational symmetry of the disk is what makes yaw drop out. The
  formulas and every number are unchanged.] `r = R*sqrt(u)` is the uniform
  distribution over the disk area; `r = R*u` would put half of all draws
  inside `R/2`, a quarter of the area. A ceiling that is decided rather than
  derived is the ADR form: under AutoDR the study reports where each
  boundary ENDS, and the ceiling only stops a boundary from walking into the
  physically meaningless. The inbox entry of 2026-09-10 cites Akkaya et al.
  2019, App. C.3, Tab. 15 ("Maximum value of a phi: 4.0") for this; that
  table was not re-read for this entry. The yaw ceiling has an anchor, not a
  derivation: +-5 deg in IndustReal and AutoMate
  (`docs/reference/literature_check_randomisierungsbereiche_2026-09-09.md`).
  The CAD fit limit of 0.2349 deg (`Belege_Streuwerte.md` A3) is the final
  accuracy the policy must reach before seating. It argues neither for 5
  nor for 10 deg.
- Falsified by: at full width (`lat_r` at 0.030 m), the lateral bin shares
  of the DRAWS must be (e_k^2 - e_{k-1}^2) / R^2 = 4 / 12 / 20 / 28 / 36 %
  over the five bins (simulated: 4.0 / 12.0 / 20.0 / 28.0 / 36.0 %), and the
  yaw bins 20 % each. Different shares mean the draw is not uniform.
- Verification status: BUILT on the laptop (5a7d295), offline checks green,
  UNVERIFIED on the training PC -- nothing has run under Isaac. Offline gate of
  the build commit:
  `check_insertion_math.py` proves the area identity
  deterministically (`u_r = linspace(0, 1, n)`: exactly n/2 draws below
  `R/sqrt(2)`; with the `sqrt` removed the share becomes 0.707);
  `check_autodr.py` counts the limits; `check_env_wiring.py` carries the
  disk box formula.
- Sources: grill session 2026-09-10 and user 2026-09-11 (plan
  `distributed-stargazing-deer.md`, context table, rows "Decke", "Versatz",
  "Kipp 10 deg"; predecessor plan `du-planst-einen-baustein-swirling-sprout.md`
  § 1.2 and § 4.1); D-153, D-170, D-173, D-176; `insertion_env.py:2879`,
  `:658-659`; `insertion_math.py:1472-1485`;
  `scripts/check_env_wiring.py:3956-3971`; `insertion_tasks_cfg.py:1516`;
  `docs/decisions_inbox.md` entry of 2026-09-10 (ceilings, ON HOLD);
  `Belege_Streuwerte.md` A3.
- Numbering note: highest D-number across all local and origin branches was
  D-177 (`p5-robustheit` and `p5-robustheit-astra`, same entry); `main`
  stands at D-151. D-178 was free everywhere.

## D-179: The start height is one-sided under AutoDR: its centre is the lowest clean start H_min, measured, and its ceiling is 0.120 m

- Date: 2026-09-11 (decided in the grill session of 2026-09-10)
- Status: proposed (pending verification) - built on the laptop in 5a7d295,
  offline checks green, UNVERIFIED on the training PC. H_min is still NOT
  MEASURED.
- Stream: rl-code
- Corrects: D-163's inline correction of 2026-09-02 (step C, Uniform[low,
  +30 mm]) for `dr_mode='autodr'`. D-163 carries a pointer here.
- Context: `DR_DIMS.start_height` is two-sided today: centre 0.030 m
  (`insertion_env_cfg.start_tip_above_entrance` =
  `RUNG0_START_TIP_ABOVE_ENTRANCE`, D-163), `lo_max` 0.020 m, which the
  table itself calls "a placeholder, nothing else" (`autodr.py:394-398`),
  and `hi_max` 0.050 m, RT-177's run setting. The lower boundary has no
  source. It is also bounded by something other than a range: a start
  inside the pocket is refused (`insertion_env.py:648-662`), because the
  start IK commands position only and an un-commanded orientation there is
  the RT-120 wall contact at reset. The ceiling target of the grill session
  of 2026-09-10 is 0.120 m. `DimSpec.bind` today refuses a one-sided
  dimension whose centre differs from its `lo_max` (`autodr.py:361-365`).
- Options considered:
  (a) keep the dimension two-sided and find a lower edge — rejected: no
  source gives one, and below the lowest clean start there is no harder
  task, only an illegal reset;
  (b) one-sided, centre = the lowest clean start H_min, widened upward only,
  up to 0.120 m, with no second height curriculum beside it — CHOSEN.
- Decision:
  1. `start_height` becomes one-sided in `DR_DIMS`: `one_sided=True`,
     `hi_max` = 0.120 m. `lo_max` stays in the table only as a documented
     placeholder that `bind` overwrites.
  2. `DimSpec.bind` sets `lo_max := centre` for every one-sided dimension
     BEFORE all range checks. The refusal "centre != lo_max"
     (`autodr.py:361-365`) is deleted.
  3. H_min has ONE home: `insertion_env_cfg.start_tip_above_entrance`. The
     cfg comment carries BOTH readings: under `dr_mode='off'` the field
     stays the upper edge of [low, high] (D-163, step C); under
     `dr_mode='autodr'` it is the centre of the one-sided height, i.e.
     H_min. Until the measurement the value is `[TESTWERT]` (today 0.030 m).
  4. H_min is MEASURED, not set. Ladder downward: 30, 20, 10, 5 mm (0 mm
     only if 5 mm is clean), `dr_mode='off'`, radius 0.030 m, tilt 10 deg
     and yaw 5 deg ~~held fixed at their ceilings~~ (D-178). Both criteria are
     [WORDING CORRECTED 2026-09-12, critic round 1 — the NUMBERS 0.030 m,
     10 deg and 5 deg are unchanged, only the words "held fixed at their
     ceilings" were wrong. `dr_mode='off'` does not hold anything at a
     ceiling: `InsertionEnv._live_bounds` returns UNIFORM BANDS —
     `"lat_r": (0.0, R)`, `"yaw": (-h, +h)`, `"tilt": (0.0, max)` — and
     `insertion_math.disk_offset` draws `r = R * sqrt(u)`. The entry already
     said so in its own next clause, "because the disk draws its rim rarely",
     which only holds for a DRAWN radius; the two clauses contradicted each
     other. The three numbers are the BAND CEILINGS the run opens to, not
     values every episode sits at. Tilt and yaw CAN be held fixed, but through
     the deterministic probe fields `fixture_tilt_rad` / `fixture_yaw_rad`
     (`insertion_env_cfg.py:678`, `:684`), which are different fields from the
     noise magnitudes `fixture_tilt_noise_rad` / `fixture_yaw_noise_rad` that
     `_live_bounds` reads — and mixing a probe angle with a noise range is
     refused in `__init__`. A fixed RADIUS field does not exist at all; the
     only lateral field is `start_lateral_offset`, which IS the disk radius.
     WHICH OF THE TWO the H_min ladder is to use is therefore an OPEN
     question, not a settled one — it is not decided here.
     Second correction, same date: the plan
     `distributed-stargazing-deer.md` 4.2 says "Gier 10 deg fest" for this
     same ladder. 10 deg is the SUPERSEDED yaw ceiling; D-178 lowered it to
     5 deg on 2026-09-11 and D-178 owns that number. The plan text is stale,
     not this entry.]
     required: (a) the solver is clean —
     `start_pose_solve_flags_dropped_envs` = 0 over at least 4096 episodes
     per rung, reported as a share, because the disk draws its rim rarely;
     (b) no contact force at reset — `scripts/zero_agent.py
     --start-tip-above-entrance-mm` per rung, force at steps 0-2 about 0 N.
     H_min is the lowest rung that meets (a) AND (b). The radius of
     0.030 m makes the mesh repair of D-178 point 5 a precondition.
  5. The height bins get six edges from H_min to 0.120 m after the
     measurement, not before.
- Named loss, written down so it is not rediscovered: after point 2, every
  centre the env passes for a one-sided dimension silently becomes its lower
  edge, and the loop `bound_max(side) == c` sees only `hi`. What still pins
  the centres is the text pin of the centre dict in
  `scripts/check_env_wiring.py:1963-1972`.
- Rationale: a start below the lowest clean start is not a harder task. It
  is an illegal reset: contact at reset (D-163, RT-120), or an off-axis part
  the position-only solver cannot place inside the pocket (D-170 point 3).
  The lower edge is therefore a MEASUREMENT boundary and not an AutoDR
  boundary. The upper edge is the ceiling target; with it the controller box
  still stays below the measured home tip of 0.145879 m
  (`scripts/check_env_wiring.py:75`, RT-143), with the numbers of D-180. The
  lineage start band of Factory, AutoMate and FORGE ([0.037, 0.057] m above
  the fixed-asset tip, `factory_tasks_cfg.py:112-113`) is an anchor only.
- Falsified by: a ladder in which criterion (b) fails at every rung down
  from 30 mm. Then no clean start exists at radius 0.030 m and tilt 10 deg,
  and the ceiling set of D-178 is infeasible at its lower end.
- Verification status: BUILT on the laptop (5a7d295), offline checks green,
  UNVERIFIED on the training PC -- nothing has run under Isaac. Offline gate of
  the build commit:
  `check_autodr.py` gains "bind sets lo_max = centre for one-sided
  dims" with the mutation `lo_max=float(centre)` -> `lo_max=self.lo_max`;
  the old check "refuses a centre off its lower edge" and its mutation
  `one-sided-centre-check-dropped` are deleted together.
- Sources: grill session 2026-09-10 (plan `distributed-stargazing-deer.md`,
  context table row "Starthöhe", step 4.2); D-163, D-170, D-178;
  `autodr.py:361-365`, `:394-398`; `insertion_env.py:648-662`;
  `insertion_env_cfg.py:368-411`; `scripts/check_env_wiring.py:75`,
  `:1963-1972`; `scripts/zero_agent.py:46-68`; RT-120, RT-143.
- Numbering note: D-179 follows D-178 in the same session; free everywhere.

## D-180: The controller box is sized for the disk — x/y 0.08 m with at most 0.010 m air, z 0.1435 m — and the step limit stays 0.02 m

- Date: 2026-09-11 (decided in the grill session of 2026-09-10)
- Status: proposed (pending verification) - built on the laptop in 5a7d295,
  offline checks green, UNVERIFIED on the training PC
- Stream: rl-code
- Corrects: step B7's box values — x/y "Factory's pos_action_bounds,
  0.05 m" (`insertion_env_cfg.py:239-244`) and z 0.10 m (`:245-268`) — and
  the "Current numbers" list of D-177, which carries a pointer here.
- Context: `scripts/check_env_wiring.py` checks that the tip clamp box holds
  the outermost legal start plus one action step. It computes that reach
  with the square-band form (`:3956-3971`): the corner at `r*sqrt(2)` and a
  yaw factor `cos(psi) + sin(psi)` on the sideways reach. It also pins x/y
  to exactly Factory's 0.05 m (`:4023-4025`). With the ceilings of D-178
  (radius 0.030 m, tilt 10 deg) and D-179 (start height 0.120 m), both boxes
  are too small, and the equality pin fails as soon as x/y moves. The step
  limit `osc_pos_step_limit_m` = 0.02 m is a FORCE value, not a geometry
  value: `kp <= F_search / (Lambda_max * step)` (D-166).
- Options considered:
  (a) lower the step limit to 0.015 m, so that z fits below the home tip
  with the square form (predecessor plan
  `du-planst-einen-baustein-swirling-sprout.md` § 1.2) — rejected: it moves
  D-166's force bound to solve a geometry question, and the disk form
  removes the need;
  (b) the disk form, the step limit unchanged, both half-widths set with
  named air — CHOSEN.
- Decision:
  1. The sufficiency formula changes in its one home,
     `scripts/check_env_wiring.py:3956-3971` (no move to `autodr.py`):
     `reach_xy = r + h*sin(theta)`, `reach_z = h*cos(theta) + r*sin(theta)`,
     with `r` the `lat_r` ceiling. Yaw drops out (D-178, Rationale).
  2. `osc_pos_clamp_m` 0.05 -> **0.08 m**. Required: 0.030 + 0.120*sin(10
     deg) + 0.02 = 0.070838 m. Air 9.16 mm, SET.
  3. `osc_pos_clamp_z_m` 0.10 -> **0.1435 m**. Required: 0.120*cos(10 deg)
     + 0.030*sin(10 deg) + 0.02 = 0.143386 m. Air above the requirement
     0.11 mm; air below the RT-143 home tip 0.145879 m: 2.38 mm. Both SET.
  4. `osc_pos_step_limit_m` stays 0.02 m.
  5. The equality check "the x/y half-width is still Factory's own 0.05 m"
     is replaced by a two-sided rule with set air:
     `reach_xy + step <= half_xy <= reach_xy + step + XY_CLAMP_AIR_MAX_M`,
     with `XY_CLAMP_AIR_MAX_M = 0.010` typed as a check constant beside
     `RT143_HOME_TIP_M` (`scripts/check_env_wiring.py:75`). The mutation
     `the-xy-clamp-grows-past-the-set-air-ceiling` (`:7331`) is retargeted to
     0.08 -> 0.12, which must break the new rule.
- Rationale: x/y needs an upper bound for the reason the same file gives for
  z (`:4015-4022`): every sideways rule is a `>=`, so without a ceiling any
  value passes while the comment goes on naming a number. The typed constant
  is a check limit, not a second home of the box value. The z air below the
  requirement is 0.11 mm: a later raise of the start ceiling by about 0.1 mm
  turns the sufficiency check red, which is the check doing its job. The z
  air of 2.38 mm below the home tip matters because a box that reaches the
  home pose ends the reset pull-down RT-143 measured; the predecessor plan
  judged 0.3 mm unusable. Whether 2.38 mm is enough is NOT measured.
- Consequence for the thesis: the x/y box is no longer Factory's number.
  `InBachelorErwähnen.md` gets that row.
- Verification status: BUILT on the laptop (5a7d295), offline checks green,
  UNVERIFIED on the training PC -- nothing has run under Isaac. Offline gate of
  the build commit:
  `check_env_wiring.py` and its counter-proof with the retargeted
  mutation.
- Sources: grill session 2026-09-10 (plan `distributed-stargazing-deer.md`,
  context rows "Schrittlimit" and target rows "Box xy", "Box z");
  predecessor plan § 1.2; D-166, D-177, D-178, D-179;
  `scripts/check_env_wiring.py:75`, `:3956-3971`, `:4015-4025`, `:7331`;
  `insertion_env_cfg.py:236-268`; RT-143.
- Numbering note: D-180 follows D-179 in the same session; free everywhere.

## D-181: Contact friction is set from the DuPont PA66-on-POM tables — band 0.08-0.20, centre 0.14 — for an assumed PA6-on-POM pairing

- Date: 2026-09-11
- Status: proposed (pending verification) - built on the laptop in 5a7d295,
  offline checks green, UNVERIFIED on the training PC (the [TESTWERT] marker was
  lifted in d0fe39b).
- Stream: rl-code
- Corrects: D-111 (2) (mu = 0.4 as the midpoint of a same-polymer span),
  which carries a pointer here. Changes the BAND of the unnumbered inbox
  entry "Contact friction is randomised PER RESET, static = kinetic, on both
  bodies" (`docs/decisions_inbox.md`, 2026-09-10); its FORM stays.
- Context: D-111 (2) assumed that part and fixture are the same DEHNvap
  housing thermoplast and set mu = 0.4, the midpoint of a dry
  polymer-polymer span of about 0.2-0.8. That span has no primary source
  (research 2026-09-11, conflict C7). `DR_DIMS.friction` holds 0.2-0.6 about
  0.4 and says "band ... not measured" (`autodr.py:399-402`). The datasheet
  names only "Thermoplast, Farbe rot, UL 94 V-0", and no public DEHN
  statement names the polymer of the housing or of the test adapter
  (research, Q4). On 2026-09-11 the user set the working assumption: two
  different thermoplastics — the part is the thermoplastic housing of the
  combined arrester, the fixture a test-adapter station of a softer plastic —
  and chose PA6 (part) against POM (fixture). The user's first figures
  (sliding 0.20-0.28, static 0.25-0.32) appear on neither of the two pages
  they were cited from (C1).
- Options considered:
  (a) the user's first figures, 0.20-0.32 — rejected by the user after the
  research: they are not on the cited pages and have no primary source;
  (b) the DuPont primary values for PA66 on POM — CHOSEN.
- Decision:
  1. Pairing: PA6 (part) against POM (fixture). A named assumption, not a
     measurement.
  2. `DR_DIMS.friction`: `lo_max` 0.08, `hi_max` 0.20 — the full span of
     every primary PA66/POM value the research found: DuPont Zytel/Minlon
     Design Guide Module II, Table 39, p. 101 (Zytel on Delrin, no
     lubricant: static 0.13-0.20, dynamic 0.08-0.11), and DuPont Delrin
     Design Guide, Table 9, p. 24 (Delrin 500 on Zytel 101: static 0.10,
     dynamic 0.20). No own +-% half-width.
  3. Centre 0.14, the midpoint of the band, in its one home
     `insertion_tasks_cfg.CONTACT_FRICTION` (0.4 -> 0.14).
  4. The form stays: one value per env per reset, static = kinetic, written
     to both bodies.
- Rationale: the band is the span of the sources, not a construction around
  a nominal. Static and kinetic share one number because the sim writes one
  value into both columns (Factory `set_friction`,
  `factory_utils.py:31-37`). The two DuPont tables disagree on which of the
  two is larger (C2), so the band covers both. PhysX raises an effective
  static coefficient below the dynamic one to the dynamic value
  (`PxsMaterialCombiner.h:161-169`, as read by the research; not run), so a
  static value below the kinetic one could not be represented anyway.
- Named gaps: (i) every primary value is PA66; none is PA6. (ii) The two
  tables differ by about 15x in pressure and 9.6x in speed, and neither
  condition is an insertion. (iii) The combine mode actually carried by the
  shapes has never been read back from the stage (inbox entry of
  2026-09-10). (iv) "The adapter is softer" is the user's statement; whether
  POM is softer than PA6 was not checked. (v) The two DuPont tables were read
  by the research agent on page images; this entry did not re-read them.
- Consequence: `CONTACT_FRICTION` also applies to every `dr_mode='off'` run.
  From the build commit on, every episode runs with a different friction
  than RT-172 to RT-179; a comparison across that commit carries friction as
  a confounder.
- Sources: `docs/reference/literature_check_reibung_paarung_2026-09-11.md`
  (questions 1-5, conflicts C1-C7); DuPont Zytel/Minlon Design Guide Module
  II, Tab. 39, p. 101; DuPont Delrin Design Guide, Tab. 9, p. 24; DEHNvap
  datasheet 900 360; D-111; `docs/decisions_inbox.md` 2026-09-10 friction
  entry; `autodr.py:399-402`; `insertion_tasks_cfg.py:1378-1391`;
  `factory_utils.py:31-37`; user, 2026-09-11.
- Numbering note: D-181 follows D-180 in the same session; free everywhere.

## D-182: Observation noise runs through Isaac Lab's `observation_noise_model` — pocket-position bias sigma 2.5 mm per episode, force sigma 3.5 N per step — and it is switched ON

- Date: 2026-09-11 (the hook: grill session 2026-09-10; values and
  activation: user 2026-09-11)
- Status: proposed (pending verification) - built on the laptop in c6e8ed8,
  offline checks green, UNVERIFIED on the training PC (the stopper it left behind
  is D-185).
- Stream: rl-code
- Corrects: D-111 (3), which carries a pointer here. Its FORM stands (a
  per-episode constant offset on the pose channels); a per-step force
  channel is added, and the activation no longer waits for supervisor
  question F1. The own form `insertion_math.add_obs_offset`
  (`insertion_math.py:919`) and the fields `obs_noise_enabled` /
  `obs_noise_amplitude` (`insertion_env_cfg.py:973-975`) are deleted.
- Context: D-111 (3) fixed the form and tied the activation to F1; nothing
  was ever switched on. Isaac Lab 2.3.2 provides a hook:
  `DirectRLEnvCfg.observation_noise_model` (`direct_rl_env_cfg.py:172`),
  built in `direct_rl_env.py:210-213`, applied at `:414-415` AFTER
  `_get_observations` and only to `obs_buf["policy"]`, and reset in the base
  `_reset_idx` (`:624-625`), which `InsertionEnv._reset_idx` calls
  (`insertion_env.py:2082`). Reward, termination and success read the env's
  own buffers and therefore see the clean values by construction. The
  library class `NoiseModelWithAdditiveBias` cannot carry a per-channel bias
  from the first reset: `_bias` is created as `(num_envs, 1)`
  (`noise_model.py:157`) and widened to the channel count only at the first
  `__call__` (`:186-191`), but `reset()` (`:174`) runs earlier. And `reset`
  writes `_bias = func(_bias, cfg)` while `NoiseCfg.operation` defaults to
  `"add"` (`noise_cfg.py:29`): each reset would ADD a new draw onto the old
  bias. Both facts are read from the source, not run.
- Options considered:
  (a) keep the own form `add_obs_offset` inside `_get_observations` —
  rejected: the noisy value would sit in the same method whose buffers the
  reward reads, kept apart only by hand, while the library already provides
  the hook, the reset schedule and both noise functions;
  (b) the library hook with a small subclass — CHOSEN.
- Decision:
  1. Two scalar cfg fields, ON by default: `obs_noise_pocket_pos_std_m` =
     0.0025 and `force_obs_noise_std_n` = 3.5. Both at 0.0 give
     `observation_noise_model = None`. No tensor enters the cfg, so hydra's
     `to_dict` and `params/env.yaml` carry scalars only.
  2. A module `obs_noise.py` beside `insertion_math.py` holds a scalar cfg
     class and `InsertionObsNoise(NoiseModelWithAdditiveBias)`. The class
     builds the two (28,) masks through a free function
     `insertion_math.noise_masks(*, pocket_pos_std_m, force_std_n, device)`
     from
     `OBS_SLICES`, and passes the library `GaussianNoiseCfg(std=step_std)`
     as the per-step noise and `GaussianNoiseCfg(mean=0.0, std=bias_std,
     operation="abs")` as the bias noise. It then sets `_bias` to
     `zeros(num_envs, OBS_DIM)` and `_num_components` to `OBS_DIM`.
  3. `InsertionEnv.__init__` sets `cfg.observation_noise_model` BEFORE
     `super().__init__`, the pattern of `resolve_control_mode`
     (`insertion_env.py:130-131`).
  4. Channels: pocket-position bias, Gaussian sigma 2.5 mm, drawn per
     episode, on `tip_rel` 12:15. Force, Gaussian sigma 3.5 N, drawn per
     step, on `force` 25:28. The same distribution in every run arm; no
     AutoDR buffer.
  5. `_write_metrics` writes both scalars.
- Sources of the two numbers:
  - 2.5 mm: FORGE, arXiv:2408.04587v2, Appendix A, Table II, "Pos-Est
    Noise: 2.5 mm"; the fixed asset position is randomised once per episode
    by Gaussian noise (HTML fetched 2026-09-11). NAMED CONFLICT: FORGE's
    code sets no `fixed_asset_pos` of its own (`forge_env_cfg.py:31-35`) and
    inherits Factory's 0.001 m (`factory_env_cfg.py:46`). The paper is cited.
  - 3.5 N: UR5e datasheet, "Force sensing, tool flange/torque sensor",
    Force x-y-z "Precision ± 3.5 N", "Accuracy ± 4.0 N"
    (`docs/reference/Datenblaetter/ur5e-datasheet.pdf`). Reading ±3.5 N as a
    standard deviation is an ASSUMPTION; the datasheet does not define it.
    For comparison FORGE uses 1 N per timestep (Table II; code
    `forge_env.py:112-114`).
- Rationale: the hook is the library's, it acts after the env has computed
  everything, and it leaves the clean channel for reward and termination
  intact by construction. Own code is limited to the pre-sized bias buffer
  and the mask. With a 2.5 mm pocket bias against `PLAY_X` = 0.5876 mm the
  observation alone can no longer place the part; contact has to close the
  gap (curriculum check § 8.9).
- Named side effects: (i) the very first observation is UNNOISED in three
  places — `DirectRLEnv.reset()` returns `_get_observations()` directly
  (`direct_rl_env.py:331`), `RslRlVecEnvWrapper.get_observations()` calls
  `_get_observations()` (`vecenv_wrapper.py:148`), and `play.py
  --trace-obs` reads raw slices. (ii) `randn_like` per step shifts the
  global RNG stream, so a run with noise draws different reset conditions
  than a run without; the no-DR reference therefore runs WITH noise.
  (iii) F1 stays open as a supervisor question: the activation is the
  user's call, not the supervisor's answer.
- Named unknown: whether a scalar cfg class that leaves the base
  `NoiseModelCfg.noise_cfg` unset passes Isaac Lab's cfg handling is
  UNVERIFIED; the build checks it first.
  **ANSWERED 2026-09-12 by D-185, and the answer is NO.** It does not pass:
  `DirectRLEnv.__init__` opens with `cfg.validate()` and `_validate` raises on
  the inherited `MISSING`, so every run died in the constructor. Decision
  point 2 above is superseded on this one field — `InsertionObsNoiseCfg` now
  binds `noise_cfg: NoiseCfg | None = None`. Everything else in point 2 stands.
- Verification status: BUILT on the laptop (c6e8ed8), offline checks green,
  UNVERIFIED on the training PC -- nothing has run under Isaac. Offline gate: `noise_masks`
  checks (only `tip_rel` carries `bias_std`, only `force` carries
  `step_std`; mutation `"force"` -> `"joint_pos"` must break), a check that
  the bias uses `operation == "abs"` (mutation to `"add"` must break), and
  the assign-before-`super().__init__` check in `check_env_wiring.py`.
  Training-PC expectation: `params/env.yaml` is written and
  `compare_runs.read_config_yaml` does not return `{}`.
- Sources: D-111; `C:\IsaacLab` `direct_rl_env_cfg.py:172`,
  `direct_rl_env.py:210-213, 331, 414-415, 624-625`,
  `utils/noise/noise_model.py:96-97, 157, 174, 186-191`,
  `utils/noise/noise_cfg.py:29`, `isaaclab_rl/rsl_rl/vecenv_wrapper.py:148`,
  `direct/forge/forge_env_cfg.py:31-35`, `direct/forge/forge_env.py:112-114`,
  `direct/factory/factory_env_cfg.py:46`; FORGE arXiv:2408.04587v2 App. A
  Tab. II; UR5e datasheet; `insertion_env.py:130-131, 2082`;
  `insertion_math.py:919`; `insertion_env_cfg.py:973-975`; user 2026-09-11.
- Numbering note: D-182 follows D-181 in the same session; free everywhere.

## D-183: Grasp scatter returns as an observation-only offset — uniform ±3 mm along the part's short axis per episode, hidden in `tip_rel`

- Date: 2026-09-11
- Status: proposed (pending verification) - built on the laptop in c6e8ed8,
  offline checks green, UNVERIFIED on the training PC
- Stream: rl-code
- Supersedes: D-126 Decision (d) ("per-reset grasp scatter is DROPPED") and
  its sentence "(a) and (c) stay unmeasured and are not the fallback".
  Corrects D-107 (4): the grasp offset is no longer visible to the policy.
  D-070's intent is now realised through D-126 option (c). All three entries
  carry a pointer here.
- Context: D-070 decided that grasp uncertainty enters as a per-reset
  flange-to-part offset (Factory `held_asset_pos_noise`). D-126 measured that
  a runtime write of `physics:localPos0` on `tool_weld` does not reach PhysX
  (RT-51/RT-52), dropped the feature, and named route (b) — drop the weld,
  spawn the part as a separate rigid body — as the only way back. On
  2026-09-11 the user asked for grasp scatter again. The gripper can slip
  along the part's SHORT axis; along the long axis it has no room and centres
  itself (user, with a CAD side view of the gripped part, 96.41 mm across).
  Magnitude ±3 mm. Offered routes (b) and (c), the user chose (c) and
  accepted that the physics stays nominal, stating that current decisions
  outrank old ones.
- Options considered:
  (b) drop the weld and spawn the part as a separate rigid body on a
  runtime-writable joint, Factory's shape — real physics, but it reopens
  D-076 and rebuilds the articulation;
  (c) perturb the part-anchored observation instead of the body — CHOSEN.
- Decision:
  1. Per episode, `e_x ~ U[-a, a]`, `a` = 0.003 m, along the part's short
     axis, which is x of the part and tool frames
     (`insertion_tasks_cfg.py:1488-1489`; `PART_WELD_ROT_COLUMNS`,
     `:1296`, maps part x to tool -x, which a symmetric draw does not
     notice). y and z stay 0.
  2. The offset enters ONLY the observation: `_get_observations` computes
     `tip_rel` with `_tip_offset_local + e` (`insertion_env.py:1722-1729`).
     `_get_dones` (`:1850-1857`), the controller box in `_apply_osc`
     (`:1498-1506`) and every other reader of `_tip_offset_local` stay
     nominal.
  3. **[Corrected 2026-09-12, user.]** The offset is column `grasp_obs_x`,
     the SIXTEENTH of `autodr.TABLE_COLUMNS`. `_draw_reset_conditions` draws
     the whole row; `_reset_idx` maps that column with Factory's form
     `(2*u - 1)*a` (`factory_env.py:763-769`) before the reset observation.
     No separate `torch.rand`. The same in every run arm; NO `DimSpec`, no
     AutoDR boundary and no success buffer -- the boundary count stays seven,
     like `lat_phi`, `tilt_azimuth` and the six joint-noise columns.
     TWO reasons, and the first version of this point had neither: (i) the
     evaluation replays a table row so two policies meet the identical
     episode, and a belief error drawn beside the row is re-drawn on every
     replay; (ii) a separate draw consumes a random number even at
     `grasp_obs_offset_x_m = 0.0`, so "off" was not off. A fixed-width
     column consumes the same count whatever the magnitude is -- therefore
     NO AMPLITUDE GUARD MAY BE ADDED, since a guard would make "on" and
     "off" consume different counts again.
     The superseded wording claimed a column was impossible because "the
     evaluation replays that table as a set of POSES". That is false:
     nothing in the table is a pose. `RowTable` validates plain [0, 1]
     numbers, `insertion_env.py` says "each consumer maps its own column",
     and `GridTable` calls the non-raster ones "the disturbance columns".
  4. Cfg field `grasp_obs_offset_x_m` = 0.003, written to
     `demo_metrics.json`.
  5. Own code, named: the offset turns with the part, so the library's
     additive bias (fixed in observation coordinates) cannot carry it.
- Rationale: with a real grasp error, the robot's belief of the tip differs
  from the truth by a fixed tool-frame vector turned by the part's
  orientation. Adding the offset to the tool point in the observation path
  gives the same belief-to-truth relation, while the body stays where PhysX
  has it. This is an argument, not a measurement.
- Named physics gap: mass, centre of mass and lever arms stay at the nominal
  weld, and contact happens on the nominal body. The thesis now states grasp
  scatter as "modelled in the observation only, physics nominal"; this
  replaces D-126's thesis sentence "grasp uncertainty is not simulated".
- Sources of the number: Factory `held_asset_pos_noise` for the peg,
  `[0.003, 0.0, 0.003]` (`factory_tasks_cfg.py:123`), uniform
  (`factory_env.py:763-769`); FORGE arXiv:2408.04587v2, Appendix A, Table
  II, "Held: x,y (rel): [−3, 3] mm". The axis: the user, 2026-09-11.
- Verification status: BUILT on the laptop (c6e8ed8), offline checks green,
  UNVERIFIED on the training PC -- nothing has run under Isaac. Offline gate: an AST check
  that the offset appears in `_get_observations` and in no reward,
  termination or controller path (mutation "offset in `_get_dones`" must
  break); the RNG inventory LOSES a draw, it does not gain one -- the
  offset comes off the reset row since the sixteenth-column correction in
  point 3, so `_reset_idx` calls `rand` nowhere, and the inventory check
  now requires exactly that; `yaw_cos_sin` does not depend on the offset; `part_tip_pose` is
  documented for a (3,) offset (`insertion_math.py:204`), so the
  (N, 3) form is checked offline first. **DONE 2026-09-12:**
  `check_insertion_math.py` compares a (3,) offset against its (N, 3)
  broadcast, and the (3,) path against the pre-D-183 bare `matmul`, both
  with `==` and no tolerance; both read max |d| 0.000e+00. This is not
  academic: `insertion_env.py:1757` always passes (N, 3) (the buffer is
  `zeros(num_envs, 3)` even with the scatter off) while `:1884` passes the
  shared (3,), so the bit-identity is what makes a scatter-off run equal to
  the pre-D-183 env. FLOAT64 ONLY -- the offline stand-in forces it -- so
  whether torch's float32 CUDA kernels round the vector dispatch and the
  matrix dispatch to the same bits stays the training PC's answer.
- Sources: D-070, D-076, D-107, D-126; RT-51, RT-52 (`rt_logs/VERDICTS.md`);
  `C:\IsaacLab` `direct/factory/factory_tasks_cfg.py:123`,
  `direct/factory/factory_env.py:763-769`; FORGE arXiv:2408.04587v2;
  `insertion_env.py:1498-1506, 1722-1729, 1850-1857`;
  `insertion_tasks_cfg.py:1296, 1488-1489`; user 2026-09-11.
- Numbering note: D-183 follows D-182 in the same session; free everywhere.

## D-184: The contact penalty and the distance leash leave the reward

- Date: 2026-09-11
- Status: proposed (pending verification) — built on the laptop, offline
  checks green, UNVERIFIED on the training PC
- Stream: rl-code
- Supersedes: the inbox candidates "The contact force is priced per step
  (FORGE form) beside the abort; ..." and "The reward pays for the flight: a
  dense distance leash beside the contact penalty" (both 2026-09-03; both
  carry a withdrawal line). Closes the `[REOPENED 2026-09-03]` note on D-114
  point (3): the abort and its payment stand alone again. Closes the penalty
  half of the question in `HANDOFF-RL.md` § Open item 00 (asked 2026-09-10).
- Context: Both terms entered the reward on 2026-09-03, before the controller
  moved to OSC (D-166, 2026-09-04): a FORGE-form relu on the smoothed contact
  force above 15 N (scale 0.2), and a relu on the SDF distance beyond
  `KERNEL_MARGIN_COARSE` (scale `1/KERNEL_MARGIN_COARSE`). Their cfg defaults
  stayed ON, so every later command had to switch them off by hand. The
  training commands of RT-156 and RT-171 to RT-179 carry
  `env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0`
  (`rt_logs/RT-<N>_expectation.md`; for RT-176 `docs/runs/RT-176.md`).
  Commands without the two overrides ran the defaults: the RT-156p play trace
  (contact ON, leash 5.773, `rt_logs/VERDICTS.md`) and the RT-157 replays
  (`far_penalty_scale` 5.773338721782807,
  `docs/figures/RT-157h30_demo_metrics.json`).
- Options considered:
  (a) drop the two overrides — both terms back ON at their cfg defaults;
  (b) keep both 0.0 overrides on every command, code unchanged;
  (c) remove both terms from the code — CHOSEN.
  (a) and (b) are the two answers the open question in `HANDOFF-RL.md` allowed.
- Decision:
  1. `insertion_math.REWARD_TERMS` loses `contact` and `far`.
     `compute_reward_terms_insertion` and `compute_rewards_insertion` lose
     `force_norm` and the four scale / threshold / leash floats.
  2. `insertion_env_cfg.py` loses the fields `contact_penalty_scale`,
     `contact_penalty_threshold_n`, `far_penalty_leash_m`, `far_penalty_scale`,
     their two `RL_PLACEHOLDERS` rows and their four `validate_rl_config`
     guards. `insertion_env.py` loses the `_last_force_norm` cache and the two
     startup-report lines. `force_norm` in `_get_dones` stays; the force abort
     reads it.
  3. A Hydra override of one of the four fields now stops the run: Isaac Lab's
     `update_class_from_dict` raises `KeyError` for an unknown key
     (`C:\IsaacLab\source\isaaclab\isaaclab\utils\dict.py:167`; source read,
     not run). The recorded commands of RT-156 and RT-171 to RT-179, and the
     commands copied from them into `docs/runs/RT-17x.md` and
     `docs/Blockberichte/Bericht_Laufkatalog.tex`, do not run on this code as
     written; the two overrides must be deleted first. Those files describe
     past runs and are not changed.
- Rationale: The simplest reward that specifies the task wins, and every
  added term is a hypothesis that needs evidence (project rule). No run with
  either term ON has a positive result: RT-139 (contact ON, force unchanged,
  status line of the inbox entry), RT-140 (contact ON, FAIL) and RT-141
  (contact 0.02 plus the leash, FAIL, success 0.0) (`rt_logs/VERDICTS.md`).
  Force is now bounded by two mechanisms that did not exist when the contact
  penalty was added: the OSC command cap (D-166) and the force abort at 50 N
  (D-169). The tip target is bounded by the controller box
  (`osc_pos_clamp_m`, `osc_pos_clamp_z_m`; Phase-5 size in D-180). Option (b)
  keeps a switched-off code path whose defaults are ON, and RT-156p and
  RT-157 show that a command without the overrides runs it. Option (a)
  restores two terms whose measured runs all failed, without a new hypothesis.
- Expectation, noted before the run:
  1. Offline (checked): the discounted ordering test stays green without the
     JAM trajectory's contact cost — at `w_tilt` 0.1 expert 401.27 > hover
     21.15 / jam 23.21 / idle 3.75 / tilt-cycle 21.15
     (`check_insertion_math.py`, laptop 2026-09-11).
  2. Training PC: the per-step reward equals that of a command with both 0.0
     overrides; each removed row was exactly zero there. Bit identity is NOT
     claimed: `torch.sum(terms, dim=0)` now reduces 9 rows instead of 11, and
     its reduction order is not documented.
  3. Newly allowed, against the cfg defaults: pressing between 15 N and the
     50 N abort costs nothing per step, and standing far from the fixture
     costs only through the kernels. Against the RT-156 / RT-171..RT-179
     commands nothing changes.
  4. Falsified if the budget run (`Laufplan_Phase5.md` step 5) shows a rising
     `force_abort_rate`, or a policy that parks away from the fixture (depth
     stays 0, success stalls).
- Named consequence for the thesis: the reward does not price force; force is
  bounded only by the abort (50 N) and the command cap (D-166). Row in
  `InBachelorErwähnen.md`.
- Sources: D-114, D-166, D-169, D-180; `docs/decisions_inbox.md`, the two
  2026-09-03 entries named above; `rt_logs/VERDICTS.md` (RT-139, RT-140,
  RT-141, RT-156p); `rt_logs/RT-156_expectation.md`,
  `rt_logs/RT-171_expectation.md` to `RT-179_expectation.md`,
  `docs/runs/RT-176.md`; `docs/figures/RT-157h30_demo_metrics.json`;
  `C:\IsaacLab\source\isaaclab\isaaclab\utils\dict.py:167`;
  `Laufplan_Phase5.md` step 1; plan `commit-a0-strafterme-raus.md`; user
  2026-09-11.
- Verification status: laptop, offline, 2026-09-11. `check_insertion_math`
  254 checks / 92 mutations (was 265 / 99), `check_env_wiring` 299 / 295
  (was 305 / 300), `check_autodr` 115 / 91 (unchanged), all three
  counter-proofs PASSED; `check_mutation_anchors` 401 anchors / 0 stale (was
  413); `selftest_checks.py --offline` PASS. The sweep of the A0 plan reads 0
  hits. `check_seated_success.py` (the
  teleport identity test) had to drop the same arguments from its call; its
  training-PC run is owed. UNVERIFIED on the training PC.
- Numbering note: the highest number on every local and origin branch was
  D-183 (`p5-robustheit`); D-184 is free everywhere (checked 2026-09-11).

## D-185: `InsertionObsNoiseCfg.noise_cfg` is bound to `None`, because `DirectRLEnv.__init__` validates the cfg before anything else

- Date: 2026-09-12
- Status: accepted (offline-verified; UNVERIFIED on the training PC)
- Stream: rl-code
- Corrects: D-182, Decision point 2 and its "Named unknown". The unknown it
  named — "whether a scalar cfg class that leaves the base
  `NoiseModelCfg.noise_cfg` unset passes Isaac Lab's cfg handling" — is now
  ANSWERED, and the answer is no. D-182 carries a pointer here.
- Context: `obs_noise.py` shipped with `InsertionObsNoiseCfg(NoiseModelCfg)`
  leaving the inherited `noise_cfg` at `MISSING` on purpose, on the reading
  that the field is never dereferenced. That reading is correct — the model
  builds its own inner `NoiseModelWithAdditiveBiasCfg` in `__init__`, and
  `NoiseModel.__init__` stores THAT one as `_noise_model_cfg`
  (`noise_model.py:116`) — and it is beside the point. The field does not have
  to be read to kill the run:
  * `NoiseModelCfg.noise_cfg: NoiseCfg = MISSING` (`noise_cfg.py:78`);
  * `DirectRLEnv.__init__` opens with `cfg.validate()`
    (`direct_rl_env.py:90`), before the sim, the scene or any buffer;
  * `_validate` recurses through `obj.__dict__` over the WHOLE cfg tree and
    raises `TypeError: Missing values detected in object ... noise_cfg` at the
    top-level call (`configclass.py:246-300`).
  Both sigmas default to non-zero (`0.0025`, `3.5`), so `resolve_obs_noise_model`
  returns a cfg on every run and EVERY run died in the constructor. The module
  docstring had read `_validate` where it is DEFINED and never asked who calls
  it. Isaac Lab's own example passes a real `noise_cfg` every time
  (`shadow_hand_env_cfg.py:283-286`); there is no escape hatch in `_validate`
  (the one special case is `MeshConverterCfg`, `configclass.py:265`).
- Options considered:
  (a) leave it inherited — refuted by measurement, see below;
  (b) bind a stand-in `noise_cfg = GaussianNoiseCfg(std=0.0)`, the shape of
      Isaac Lab's own examples — silences `validate()`, but see the rationale;
  (c) bind `noise_cfg: NoiseCfg | None = None` — CHOSEN;
  (d) drop the subclass and put the two (28,) mask TENSORS into a plain
      `NoiseModelWithAdditiveBiasCfg` — rejected: that is exactly the object
      hydra dumps into `params/env.yaml`, which is why D-182 built the
      scalar-only cfg in the first place.
- Decision: `InsertionObsNoiseCfg` binds `noise_cfg: NoiseCfg | None = None`.
  The field stays dead — nothing reads it — and the class docstring says so
  and says why it is bound anyway.
- Rationale: (b) and (c) both silence `validate()`; they differ in what
  happens the day somebody drops the inner-cfg construction in
  `InsertionObsNoise.__init__` and lets the base class fall back on this
  field. `None` raises `AttributeError: 'NoneType' object has no attribute
  'func'` at the first observation of the run. `GaussianNoiseCfg(std=0.0)`
  passes the observation through UNNOISED and says nothing — a silent loss of
  the whole D-182 feature that no log line would show. The loud failure is
  the right one. `| None` is also the library's own idiom for "there is no
  model here" (`DirectRLEnvCfg.observation_noise_model`,
  `direct_rl_env_cfg.py:172`).
- Measurement (laptop, 2026-09-12): the real `_validate` was executed out of
  the installed tree (`C:\IsaacLab\source\isaaclab\isaaclab\utils\configclass.py`,
  lines 246-300, extracted by AST and run — the full package cannot be
  imported here because Python 3.14 breaks `_add_annotation_types`, which is
  an offline-tooling limit, not an Isaac Lab defect). Three candidates, one
  command:
    A `noise_cfg = MISSING` (what shipped) -> raises
       `TypeError: Missing values detected ... - noise_cfg`
    B `noise_cfg = None`                   -> passes
    C `noise_cfg = GaussianNoiseCfg(...)`  -> passes
- Named side effect: none in the observation, the reward or the RNG stream.
  The field is not read by any path; only `cfg.validate()` ever looks at it.
- Verification status: laptop, offline, 2026-09-12. `check_env_wiring.py`
  ALL 308 CHECKS PASSED (one new check, "the inherited MISSING noise_cfg is
  bound, so cfg.validate() passes"), `--self-test` COUNTER-PROOF PASSED with
  two new mutations: `the-inherited-missing-noise-cfg-comes-back` (the
  binding deleted) and `the-bound-noise-cfg-is-itself-missing` (the name
  bound, to `MISSING`, which walks past any check that only asks whether the
  name appears). The check reads the VALUE, not the name, for that reason.
  UNVERIFIED on the training PC: no run has started an env yet.
- Sources: D-182; `C:\IsaacLab\source\isaaclab\isaaclab\envs\direct_rl_env.py:90`,
  `:210-213`; `.../envs/direct_rl_env_cfg.py:172`;
  `.../utils/configclass.py:246-300`; `.../utils/noise/noise_cfg.py:78`;
  `.../utils/noise/noise_model.py:116`;
  `C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\shadow_hand\shadow_hand_env_cfg.py:283-286`.
- Numbering note: the highest number on `main` is D-151 and the highest on
  `p5-robustheit` is D-184; D-185 is free (checked 2026-09-12).

## D-186: the `--lateral` probe of `check_seated_success.py` widens the OSC tip clamp box to its own commanded offset

- Date: 2026-09-12
- Status: accepted (verified on the training PC, RT-180c, 2026-09-12)
- Stream: rl-code
- Context: the lateral leg is the PAID COUNTER-PROOF of the seated identity
  test — it teleports the part to free air beside the fixture and demands
  that the success predicate does NOT fire there. The offset has to clear the
  block, so `lateral_probe_min_y_m` puts the floor at
  `POCKET_LOCAL_Y_RANGE[1]` + half the part width = 172.2 mm, and the yaw-safe
  default is 191.889 mm. The OSC tip clamp box is 0.08 m half-width
  (`insertion_env_cfg.py:252`) and is applied AFTER the target is built as
  current pose + delta (`insertion_env.py:1521-1550`), so a zero action
  outside the box is not "stay put" — it is a pull back toward the entrance.
  Every legal lateral offset lies outside the shipped box. Two runs measured
  both ends of the settle count and neither works:
  * RT-180, 30 settle steps: the tip was dragged from 191.889 mm to
    84.3 mm / -50.1 mm. L2 and L3 FAIL.
  * RT-180b, 1 settle step: the pose stands (L3 and D-157 HOLD), but the part
    is still moving — 1.569 N tared against a 1.0 N gate, raw 8.28 N against
    the tool's own 8.083 N weight. L8 FAILS. (CORRECTED 2026-09-12, after
    the entry was pushed: this bullet first read "raw force 11.6 N". That
    number belongs to a SECOND probe at two settle steps, not to RT-180b —
    `HANDOFF-RL.md`, the RT-180b paragraph, names both. The argument is
    unchanged: raw above the tool weight at either count means motion.)
  No settle count satisfies L3 and L8 at once, because the pull is present at
  every step. The curve mode already guards this by REFUSING offsets outside
  the box (`curve_lateral_error`, D-080 cases and mutation); the lateral mode
  cannot refuse, because refusing every legal offset would delete the
  counter-proof.
- Options considered:
  (a) keep 0.08 m and tune the settle count — refuted by RT-180 and RT-180b,
      the two ends of the range;
  (b) move the probe inside the box — impossible: 80 mm < 172.2 mm buries the
      part in block material, which is what RT-112/RT-113 measured;
  (c) skip the settle steps entirely — rejected: the L8 force reading and the
      reward identity need at least one physics step, and a pose that was
      never stepped is not a pose the env holds;
  (d) widen `osc_pos_clamp_m` for this run only, to the commanded offset plus
      the probe margin — CHOSEN;
  (e) raise the shipped `osc_pos_clamp_m` for every run — rejected: the box is
      a control constant of the TASK (D-180 names it a placeholder pending the
      disk-reach measurement), and a probe script must not move a task
      constant for everybody.
- Decision: a pure function `lateral_probe_clamp_m(lateral_y_m, clamp_m,
  margin_m=LATERAL_PROBE_MARGIN_M)` returns
  `max(clamp_m, |lateral_y_m| + margin_m)`, and `main` assigns it to
  `env_cfg.osc_pos_clamp_m` inside `if args_cli.lateral:`, beside the existing
  pins and before `gym.make`. With the current geometry the box goes
  0.08 m -> 0.196889 m for the default offset. The z half-width
  `osc_pos_clamp_z_m` is NOT touched; 0.1435 m already covers the 34 mm depth.
  `abs` because the box is symmetric — a negative offset needs the same
  half-width as its mirror. The function never NARROWS a box that already
  covers the offset.
- Rationale: the counter-proof's claim is "the predicate does not fire beside
  the pocket". That claim is only testable if the run can HOLD the pose it
  commands. The clamp box is not part of the claim; it is the reason the pose
  could not be held. Widening it for this one run removes the obstacle and
  changes nothing the run asserts. The seated and shallow legs keep the
  shipped 0.08 m, so the identity test's main leg is untouched.
- Named side effect: the lateral leg no longer runs at the training box's
  half-width, so it says nothing about how the box behaves in training. It
  never did — RT-180 is the proof that it could not.
- Named unknown (CLOSED by RT-180c, see Verification status): the 18.66 mm
  gap RT-180 reported between the commanded 191.889 mm and the observed
  173.227 mm at step 2 was NOT explained by this decision when it was
  written. RT-180c read it again at the widened box and it disappeared.
- Reporting: the metrics dict now carries `osc_pos_clamp_m` and
  `settle_steps`, both READ from the cfg and the parsed argument. RT-180b's
  file did not say how many settle steps had run, so the number had to be
  taken from the command line in prose; this repo judges runs from files.
  The mode line prints the widened half-width as well.
- Verification status: **VERIFIED on the training PC, 2026-09-12, RT-180c.**
  Laptop, offline: `check_seated_success.py --self-test` 81/81 (four new
  clamp cases plus one mutation: a 1.0 m box handed in must come back
  UNCHANGED for every case, or the rule is not reading `clamp_m`);
  `selftest_checks.py --offline` PASS. Marker bumped
  `check_seated_success-2026-09-05c` -> `check_seated_success-2026-09-12a`.
  Training PC, RT-180c, all eight points PASS and 22 of 22 pins met, read
  from `rt_logs/RT-180c/RT-180c_lateral.json` (git-ignored):
  `osc_pos_clamp_m` 0.1968892059209824, `settle_steps` 30, marker
  `check_seated_success-2026-09-12a`; `lateral_y_mm` 191.88926 mm against a
  commanded 191.88921 mm, a gap of 0.00007 mm; depth 34.0002 mm; `force_n`
  0.00036 N with `force_n_raw_untared` 8.08344 N. LIMIT: the `[rt_log]` head
  was not handed over, so the run's `git:` line is not evidenced — the marker
  is what pins the script version.
- The named unknown is CLOSED (2026-09-12, RT-180c): the 18.66 mm gap was
  the clamp pull, not a different reference point. At the widened box the
  same command lands 0.00007 mm from its target, and the 18.66 mm is gone.
  One settle step under the 0.08 m box therefore already moved the tip
  18.66 mm, which is also why RT-180b's raw force stood above the tool
  weight.
- Sources: `rt_logs/VERDICTS.md` (RT-180, RT-180b); `rt_logs/RT-180_expectation.md`,
  `rt_logs/RT-180b_expectation.md`; `scripts/check_seated_success.py`;
  `source/insertion/insertion/tasks/direct/insertion/insertion_env.py:1521-1550`;
  `source/insertion/insertion/tasks/direct/insertion/insertion_env_cfg.py:252`;
  D-157, D-180, D-080.
- Numbering note: the highest number on `main` is D-151 and the highest on
  every branch is D-185; D-186 is free (checked 2026-09-12).

## D-187: the fixture mesh hole stays unrepaired and the lateral radius ban is lifted, because the hole lies an order of magnitude below the reward's own sampling resolution

- Date: 2026-09-12
- Status: accepted (verified on the training PC, RT-183 and RT-184)
- Stream: rl-code
- Supersedes: D-178 (5), the clause "no run may draw a radius above
  `HOLE_REACH_OFFSET_Y` = 7.8847 mm before the fixture mesh hole is repaired
  and re-measured". D-178 carries a pointer here. D-153 keeps its finding and
  gains a pointer here for its reopening condition.
- Context: run-plan step 4.1 (`Laufplan_Phase5.md`) asked for a mesh repair as
  a precondition for the 0.030 m lateral start radius, and D-178 (5) banned any
  radius above 7.8847 mm until that repair. Neither the defect's CURRENT
  existence nor its effect had been measured on this setup; the evidence was
  RT-94 and D-153, both 2026-08-30, and the asset lives outside git. Rule L-09
  (`docs/reference/pruefregeln.md`) forbids reading that as today's state.
- Measurement, two runs on the training PC, both 2026-09-12:
  1. **RT-183** (`check_fixture_mesh.py`, git `379aab2`): the hole IS still
     there, at the same address. G1 903/757 PASS, G2 frame guard PASS with
     every error 0.0000 mm, 903 authored boundary edges welding to 9,
     `ZERO-EDGE COUNT 3` at x 35.9863..37.4512, y 79.6347..79.7500,
     z 0.0000 mm, verdict `OPEN_EDGES_NOT_CLEAR`, exit 1. Both code constants
     recompute from today's numbers.
  2. **RT-184** (`check_seated_success.py --reward-curve`, expectation
     `rt_logs/RT-184_expectation.md` written before the run): a grid of four
     heights (0, 5, 10, 15 mm) x six lateral offsets (0, 4, 7, 7.8847, 8.2,
     8.7 mm) — three below the reach offset, three inside the 0.8153 mm
     exposure window. `sdf_mean_outside_mm` rises SMOOTHLY across the boundary
     at every height: the slope change across it is 1.27x to 1.39x, while the
     change between the two CLEAN support points below it (0->4 against 4->7)
     is 1.98x. `interpen_max_mm` is exactly 0.0 in all 24 rows, and that is a
     MAX over sample points (`insertion_math.max_interpen_dist`), not a mean,
     so one wrong-signed point would have set it. All pins met: worst
     `pose_drift_mm` 0.00077 against a 0.05 mm tolerance, `all_poses_held` and
     `all_solves_converged` true, `verdict_ok` true.
- THE ARGUMENT, and it is a resolution argument, not a harmlessness claim:
  * the hole is a triangle 1.4649 x 0.1153 mm, area **0.0845 mm2**;
  * the reward samples the part surface with `sdf_num_sample_points = 64000`
    (`insertion_env_cfg.py:1268`);
  * over the part's bounding-box area of 51907 mm2 that is 1.233 points per
    mm2, a mean spacing of **0.901 mm**;
  * expected sample points over the hole per pose: **0.10**.
  The hole is an order of magnitude below the resolution of the very term it
  could corrupt. RT-184 therefore does NOT show the hole is harmless; it shows
  the reward cannot resolve it.
- Options considered:
  (a) repair the mesh in CAD, re-export, re-measure — rejected: it spends the
      scene stream's effort to remove an exposure that the reward's own
      sampling cannot see, and the repair would then be unfalsifiable at this
      density;
  (b) measure again at a far higher sample count — rejected: it would measure
      a sampling density the training never uses, which is rule L-09's error
      in a new coat;
  (c) drop the repair from step 4.1, lift the radius ban, and record the
      resolution argument as a named limitation — CHOSEN (user, 2026-09-12).
- Decision:
  1. The mesh is NOT repaired. Step 4.1 of `Laufplan_Phase5.md` loses its
     mesh-repair half; the teleport identity test before and after it goes
     with it, because there is no "after".
  2. D-178 (5)'s radius ban is LIFTED. A run may draw `lat_r` up to its
     0.030 m ceiling.
  3. `check_fixture_mesh.py` keeps its gate sharp and keeps exiting 1 on this
     asset, exactly as D-153 decided. The failure stays the record.
  4. `stage1_lateral_y_mm.over_reach` stays in the metrics and stays worth
     reading, but it is NOT a gate: its own condition is the z band alone, so
     at a 30 mm radius it fires on the first step of every episode and says
     nothing about harm.
- Named side effect: runs above 7.8847 mm now happen without the repair, so
  any future SDF anomaly in that band must be checked against this hole first.
  The address is in D-153.
- Named unknown, unchanged by this entry: whether `wp.mesh_query_point` pairs
  edges by INDEX or rebuilds adjacency by POSITION. D-153 records it; no run
  has measured it.
- Consequence for the thesis: `InBachelorErwähnen.md` gets the row — the
  fixture mesh ships with a 1.5 mm open triangle on the stage-1 shoulder, and
  it is kept because it lies below the SDF term's 0.901 mm sampling spacing,
  not because it was shown to be harmless.
- Verification status: verified on the training PC by RT-183 and RT-184, both
  2026-09-12. LIMIT: neither run handed over its `[rt_log]` head for RT-184,
  so that run's `git:` line is not evidenced; the marker
  `check_seated_success-2026-09-12a` pins the script version.
- Sources: RT-183 and RT-184 verdict lines in `rt_logs/VERDICTS.md`;
  `rt_logs/RT-184_expectation.md`; `rt_logs/RT-184/sdf_table.csv`
  (git-ignored); D-153, D-178 (5), rule L-09 in
  `docs/reference/pruefregeln.md`; `insertion_env_cfg.py:1268`, `:479`;
  `insertion_tasks_cfg.py` (`POCKET_HOLE_MIN_Y`, `HOLE_REACH_OFFSET_Y`,
  `STAGE1_DEPTH`); `insertion_sdf.py:487-515`; `insertion_math.py:1245`.
- Numbering note: the highest number on `main` is D-151 and the highest on
  every branch is D-186; D-187 is free (checked 2026-09-12).

## D-188: the three torque channels enter the observation as the OPTIONAL mode `wrench` (31 wide); `force` (28, D-114) stays the default, and the RT-158 "ruled out" line is narrowed to the tilt

- Date: 2026-09-13
- Status: accepted (UNVERIFIED — nothing has run under Isaac; RT-192 is the
  implementation check, `rt_logs/RT-192_expectation.md`)
  [2026-09-13, inline: RT-192 PASS (a7, b..e), RT-193 SAME as RT-191, RT-194
  0.999 under noise. The "torque sigma 0, open" point is CLOSED for wrench
  runs from RT-196 on: sigma 0.2 N m as a run override, inbox entry "The
  torque channels get the datasheet precision as their noise".]
- Stream: rl-code, branch `p5-kraftsensor` (off `p5-robustheit` at
  `d9205bf0`); plan `Pläne/PlanKraft plus Drehmoment in der Beobachtung.md`,
  Auftrag `Pläne/Kontaktbeobachtung_Implementierungsplan_Claude.md`
- Context: RT-189s1 parks the part above the pocket. RT-189s1pf
  (`rt_logs/VERDICTS.md`, last line) shows motion AND rotation braked
  13-15x with a tared contact force of 0.075-0.241 N and interpenetration
  ~0, and no channel that says WHERE the part is held. RT-157 BEFUND 4
  said the same of the three force channels: a rim pose 15.9 mm off reads
  -0.49 N, a clean insertion -0.41 N. The user asked for the three torque
  components of the same joint wrench in the observation. A moment is
  `r x F`: it carries the lever arm, i.e. a hint at the contact location;
  the force alone does not, and `ee_quat` shows the tilt, not the lever.
- Conflict with the record, resolved: `HANDOFF-RL.md` (RT-158 block) says
  "THE MOMENT CHANNELS ARE RULED OUT, measured", because the commanded
  lateral action already tracks the tilt direction, so `ee_quat` carries
  that information. That measurement is about the TILT and holds; it says
  nothing about the CONTACT LOCATION, which is what the moment adds. The
  line carries an inline correction pointing here. D-114 is NOT reversed:
  the `force` layout stays the default and the abort still reads
  `_force_smooth` only.
- Options considered: (a) a fixed 31-wide layout for every run — breaks
  every existing checkpoint and every `OBS_SLICES["force"]` reader by
  index; (b) a mode switch that APPENDS the block — nothing moves;
  (c) 32 wide with a `wrench_valid` flag channel — the flag is 0 on one
  step in 256, the normaliser (eps 0.01) turns that into -13.7 on row 0
  and +0.05 otherwise, a channel that says "step 0" and nothing else;
  (d) the tilt direction in the pocket frame instead of the moment
  (HANDOFF's own suggestion) — a form of what `ee_quat` already carries,
  not the lever arm.
- Decision: (b). `InsertionEnvCfg.obs_wrench_mode: str = "force"`, hydra
  `env.obs_wrench_mode=wrench`. `insertion_math.obs_slices(mode)` /
  `obs_dim(mode)` / `obs_version(mode)`; `wrench` = `OBS_SLICES` +
  `torque (28, 31)`. `resolve_obs_layout(cfg)` sets
  `cfg.observation_space` BEFORE `super().__init__`, like
  `resolve_control_mode`. The torque is read from columns 3:6 of the
  same `body_incoming_joint_wrench_b` row as the force, tared with
  `gravity_tare_torque` (`lever_w x hold_force_w`, lever = authored tool
  COM minus the PARENT link origin, rotated into the parent frame — the
  same frame the force is in, nothing rotated or shifted afterwards),
  EMA-smoothed with the same alpha 0.25, appended UNSCALED (rsl_rl's
  `actor_obs_normalization` owns the scaling; N m may stand beside N).
  The torque buffers are filled in EVERY mode, so a `force`-mode replay
  still records them in the trace's truth block. No flag channel:
  `_wrench_valid` is a buffer, in the log and in the trace.
- The torque noise sigma `torque_obs_noise_std_nm` is its OWN number in
  N m (the 3.5 N are NOT carried over) and is **0.0 and OPEN** (user
  2026-09-13: observation noise is a suspect for the parking, so no
  provisional value; a later widening starts from 0). Candidate when
  widened: 0.2 N m, the UR5e datasheet's torque precision, read as a
  sigma the way B4b reads the 3.5 N — `Belege_Streuwerte.md` B4c. A
  torque sigma in `force` mode is REFUSED by `noise_masks`, not dropped.
- Named assumptions, and where each is paid: (1) [MEASURED 2026-09-13,
  RT-192a2: NOT the origin -- the moment is taken about a point 0.1032 m up
  the tool axis from it; `cfg.torque_ref_offset_parent_m`, RT-192
  expectation Nachtrag] the moment's reference point is the PARENT link origin — Isaac Lab's own test
  (`test_articulation.py:1871-1882`) uses it; RT-192 point 1 measures it
  against the child origin too; (2) the SIGN of the weight moment follows
  the measured force sign (raw ≈ +hold, RT-115/RT-180c), not the Isaac
  test's `m*g` — RT-192 points 1 and 2 measure it; (3) the rsl_rl 3.0.1
  checkpoint layout (`model_state_dict`, `actor.0.weight`) — the width
  guard `scripts/tools/checkpoint_width.py` refuses a mismatch with a
  readable line naming the fitting mode, and leaves an unknown layout to
  `runner.load`. No zero-padding, no weight transfer.
- What this does NOT decide: whether `wrench` learns better — a separate
  experiment. And the torque does not explain the parking: 0.2 N at a
  ~0.15 m lever is ~0.03 N m; whatever holds the part stays open
  (controller/arm is suspect 3 of the differential diagnosis, before the
  observation).
- Coupled change, its own entry: the wrench EMA now skips the reset step
  and advances once per step (`docs/decisions_inbox.md`, 2026-09-13,
  "The wrench EMA skips the reset step"). That changes row 0 of the
  `force` channel for EVERY policy; the `force` channel after that commit
  is not RT-189s1's.
- Verification: laptop — `check_insertion_math.py` 285/285 and
  COUNTER-PROOF PASSED (Prüfblock A: lever (0.1, 0, 0.2) m x 8.0834 N
  = (0, -0.80834, 0) N m by hand, upright and turned 90 deg);
  `check_env_wiring.py`, `check_mutation_anchors.py`,
  `selftest_checks.py --offline`; `checkpoint_width.py --self-test` 16;
  `trace_outcome_split.py --self-test` 43. Training PC — RT-192
  (Prüfblock B, `probe_wrench_bodies.py --tare-check`, then a 2-iteration
  `wrench` training, a `wrench` replay, a `force` replay of
  `09-12_21-58-11` and its refused `wrench` twin).
- Sources: `insertion_math.py` (`OBS_MODES`, `obs_slices`,
  `gravity_tare_torque`, `assemble_observation_wrench`, `noise_masks`);
  `insertion_env.py` (`resolve_obs_layout` call, the 0:6 read,
  `_torque_smooth`, `_wrench_valid`); `insertion_env_cfg.py`
  (`obs_wrench_mode`, `torque_obs_noise_std_nm`, `resolve_obs_layout`);
  `obs_noise.py`; `scripts/tools/checkpoint_width.py`;
  `scripts/probe_wrench_bodies.py --tare-check`; Isaac Lab 2.3.2
  `articulation_data.py:731-747` (wrench, parent body frame, `time_stamp`
  typo), `:624-638` (`body_com_pose_w`), `articulation.py:1003-1041`
  (`set_external_force_and_torque`, local link frame),
  `test_articulation.py:1817-1895`; UR5e datasheet, torque row;
  `HANDOFF-RL.md` RT-158 block (inline correction); RT-157 BEFUND 4;
  `rt_logs/VERDICTS.md` RT-189s1pf.
- Numbering note: the highest number on `main` is D-151 and the highest on
  every branch is D-187; D-188 is free (checked 2026-09-13 over
  p5-robustheit, p5-reward, p5_robustheit_astra, p5-kraftsensor).

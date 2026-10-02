# Model

## Prompt

One decision is one prompt for the Qwen-Drive-1.0 VLM (`drivejev/compiler.py`):

```
<|im_start|>user
<FRONT VIEW> relative_time=-0.5s        [240 vision tokens, 640×384]
<FRONT VIEW> relative_time=0s           [240 vision tokens, 640×384]
<FRONT TELE VIEW 15deg> relative_time=0s [84 vision tokens, 384×224]
Student road, ego and perception observation:
{"ego":{...},"maneuver":{...},"nav":{...},"recent_actions":[...],"traffic":{...}}      ← state token
Choose the next executable driving behavior. Candidate conditions are part of the behavior.
Candidate 1: {"action_type":"continue_current","target_id":"j2-1","speed_profile":"retain","description":"..."}   ← candidate token
Candidate 2: {...}
...
<|im_end|>
<|im_start|>assistant
Decision:                                                                                 ← decision token
```

* JSON is serialized with sorted keys; numbers are rounded by the executor (2 decimals for speeds, 1 for distances).
* Candidates are sorted by their semantic body (`action_type`, `target_id`, `speed_profile`, `description`), so the
  order carries no information about the label; candidate IDs are used only to map scores back.
* Unknown fields anywhere in the state or the candidates raise an error (no silent truncation either; the prompt
  is ~1,000 tokens against a 2,048 limit).
* The JSON schema of the state is in [simulator.md](simulator.md#student-observation).

## Encoder

The VLM body of Qwen-Drive-1.0-4B (`vlm.model`; the planning expert and LM head are not loaded) runs once per
decision in BF16 with explicit multimodal RoPE positions. Batched calls are right-padded, which leaves every real
token unchanged (causal attention and left-to-right linear-attention layers).

## Decision heads

| Head | Formula | Parameters |
|---|---|---|
| `pointer_mlp` (released) | `q = MLP([LN(h_dec); LN(h_state)])`, `k = MLP(LN(h_cand))`, logit = `q·k/√256` | 4.2 M |
| `pointer` | `q = W_q h_dec`, `k = W_k h_cand`, logit = `q·k/√256` | 1.3 M |

Both run in FP32 on top of the frozen BF16 hidden states; softmax over the offered behaviours gives the
probabilities. The executed behaviour is the argmax (`serve.py --decision-rule group` optionally decides
"move vs. slow/stop" by summed probability first, because behaviours with identical outcomes share the
teacher's probability mass).

## Training (summary; code not released yet)

* Labels: the privileged teacher's soft distribution over offered behaviours (mass spread over
  behaviours whose world rollouts are indistinguishable). DriveJev 1.1 labels come from the *observable* teacher
  ([simulator.md](simulator.md#observable-teacher-11)), whose rollouts only contain road users the student's
  perception has seen.
* Data: teacher-driven episodes (5 % of decision slots perturbed for 1 s to visit recoverable mistakes),
  hazard-dense episodes, and DAgger rounds in which the model drives and the teacher labels every visited state.
  Labels that depended on information the student cannot observe were removed.
* Objective: soft-target cross-entropy with square-root inverse-frequency class weights; selection on validation seeds only.
* DriveJev 1.1: the 1.0 data plus teacher-driven interaction episodes (all nine hazard and interaction kinds) and
  interaction DAgger rounds; decisions that the teacher changed because of a predicted interaction conflict (wait for
  a gap, yield, brake) are weighted ×3 (they are ~6 % of the labels but decide most interaction outcomes).

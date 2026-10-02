# Model

## Prompt

One decision is one prompt for the vision-language backbone (`drivejev/compiler.py`):

```
<|im_start|>user
<FRONT VIEW> relative_time=-0.5s        [240 vision tokens, 640×384]
<FRONT VIEW> relative_time=0s           [240 vision tokens, 640×384]
<FRONT TELE VIEW 15deg> relative_time=0s [84 vision tokens, 384×224]
Student road, ego and perception observation:
{"ego":{...},"maneuver":{...},"nav":{...},"recent_actions":[...],"traffic":{...}}      (state token: last token of this block)
Choose the next executable driving behavior. Candidate conditions are part of the behavior.
Candidate 1: {"action_type":"continue_current","target_id":"j2-1","speed_profile":"retain","description":"..."}   (candidate token)
Candidate 2: {...}
...
<|im_end|>
<|im_start|>assistant
Decision:                                                                                 (decision token)
```

* JSON is serialized with sorted keys; numbers are rounded by the executor (2 decimals for speeds, 1 for distances).
* Candidates are sorted by their semantic body (`action_type`, `target_id`, `speed_profile`, `description`), so the order carries no information about the label; candidate IDs are used only to map scores back.
* Unknown fields anywhere in the state or the candidates raise an error, and nothing is silently truncated (the prompt is about 1,000 tokens against a 2,048 limit).
* The JSON schema of the state is in [simulator.md](simulator.md#student-observation).

## Encoder

The backbone body (vision encoder and language model, without the LM head) runs once per decision in BF16 with explicit multimodal RoPE positions and is loaded with plain `transformers` (`drivejev/backbone.py`); camera frames are patched by `drivejev/vision.py`. Batched calls are right-padded, which leaves every real token unchanged (causal attention and left-to-right linear-attention layers). One decision takes 111 ms on an RTX 5090 (about 9 decisions per second at batch size 1: 94 ms backbone, 17 ms prompt compilation including image decoding, 0.3 ms head).

## Decision head

| Head | Formula | Parameters |
|---|---|---|
| `pointer_mlp` | `q = MLP([LN(h_dec); LN(h_state)])`, `k = MLP(LN(h_cand))`, logit = `q·k/√256` | 4.2 M |

The head runs in FP32 on the BF16 hidden states; a softmax over the offered behaviours gives the probabilities, so the same head handles any set of 2 to 8 behaviours. The executed behaviour is the argmax (`serve.py --decision-rule group` optionally decides "move vs. slow/stop" by summed probability first, because behaviours with identical outcomes share the teacher's probability mass).

## Training (summary; the training code is not part of this release)

* Labels: the soft distribution of the *observable* privileged teacher over the offered behaviours ([simulator.md](simulator.md#observable-teacher)), with the mass spread over behaviours whose world rollouts are indistinguishable. The teacher's rollouts contain only road users the student's perception has seen, so every label can be explained from the model's inputs; labels that depended on information the student cannot observe were removed.
* Data: teacher-driven episodes in the whole JevPilot world (5 % of decision slots perturbed for 1 s to visit recoverable mistakes), hazard-dense episodes, interaction episodes covering all nine hazard and interaction kinds, and DAgger rounds in which DriveJev drives and the teacher labels every visited state. 119 k labels in total.
* Interaction weighting: decisions that the teacher changed because of a predicted interaction conflict (wait for a gap, yield, brake) are weighted ×3; they are about 6 % of the labels but decide most interaction outcomes.
* Objective: soft-target cross-entropy with square-root inverse-frequency class weights. Model selection used the validation seeds only.

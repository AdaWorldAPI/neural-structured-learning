# Persistent structure at inference: provenance without label leakage

Status: RESEARCH PROPOSAL, unimplemented.

## Reference/contrast
TensorFlow NSL deliberately uses graph structure only during training; inference remains the base model's ordinary prediction path. The NSL wrapper's `graph_regularization.py` delegates evaluate/predict to base_model. That design is a speed/serving feature, NOT a defect.

## Hypothesis
Retaining a *separate optional structural evidence channel* beside inference (not forcibly inside every neural layer) allows retrieval, explanation, scope-aware replay, and conflict detection after model execution. The channel should degrade gracefully to plain NSL inference, and must not claim graph edges are facts.

## Proposed data contract
- Prediction: model and quantization version, input digest, output, numerical contract, inference timestamp.
- Graph evidence: edge identity, type, weight and provenance; SourceId/index seal, branch, scope; independence-group or shared-source marker.
- Reasoning record: rule/program revision, candidate vs certified conclusion, premise IDs, proof reference if independently checked, logical time and causal parent IDs.
- Explicit epistemic states: observation, hypothesis, contradiction, refused, and externally certified claim; probabilistic confidence is not a proof.
- No personal medical data in reproducible test fixtures.

## Experiment
1. Baseline: reproduce graph-regularized NSL toy model and verify identical output when evidence channel disabled.
2. Produce a deterministic reason trace for 3-node and 3-hop graphs with one irrelevant branch.
3. Replay with source revision or rule digest changed: refuse stale traces or recompute. Verify two repeated observations of the same source never become two independent pieces of evidence.
4. Compare inference time, trace bytes, replay time and explanation completeness against plain inference.
5. Negative oracle: two correlated neighbors and a duplicate source cannot inflate epistemic confidence via repeated counting.

## Admission
- Unchanged inference output when disabled (bit exact when same evaluator, otherwise declared tolerance).
- Deterministic, replayable dependency references, including branch and scope.
- Independent checker required before labeling an outcome 'certified'; a hash alone is insufficient.
- Missing evidence or invalid seal must produce explicit refusal, not guessed provenance.

## Non-goals
No claim of AGI, causal identification from correlation, or clinical explanation validity. Evidence retained *alongside* inference is distinct from forcing inference through NSL's training-time regularizer.

References: https://github.com/tensorflow/neural-structured-learning ; https://github.com/AdaWorldAPI/lance-graph/pull/1762

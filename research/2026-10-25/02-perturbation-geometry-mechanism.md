# Structured perturbations: adversarial gradients vs geometry vs interventions

Status: RESEARCH PROPOSAL, unimplemented. Distinct perturbation operators must never be conflated.

## Existing reference points
- NSL `keras/adversarial_regularization.py` and `lib/adversarial_neighbor.py` use labeled-loss gradients and selected feature constraints to generate adversarial neighbors.
- NSL `lib/regularizer.py` also has virtual adversarial regularization, with perturbation and embedding-distance objectives.
- Lance-graph relational and temporal ideas are research directions, not validated clinical or physical simulations.

## Hypothesis
A common *experiment descriptor* can carry three intentionally different transformations: (A) adversarial feature-gradient step, (B) geometric coordinate/angle perturbation with explicit chart and units, (C) mechanistic do-style intervention with structural dependencies. All may be evaluated using the same graph/focus machinery, but only C can have a causal interpretation if its model assumptions hold.

## Probe
1. Reproduce the NSL adversarial gradient example unchanged in Python as oracle.
2. Build a geometrically controlled toy: point and direction on a sphere, Cartesian (x,y,z), radius, theta/phi with seams, poles and optional local tangent coordinates. Test angle-to-Cartesian-to-angle round trips without claiming lossless discretization.
3. Introduce analytic toy dynamics (linear diffusion or spring network). Compare finite differences, autodiff Jacobian and exact reference; record validity radius.
4. Add a counterfactual intervention to a tiny causal DAG, explicitly separate from adversarial examples, and detect when a correlated edge is mistakenly treated as mechanistic.
5. Carry provenance: baseline ID, perturbation operator, physical units, coordinate chart, random seed, scope, and measured error.

## Falsification / gates
- NSL baseline loss parity with documented tolerance, including masked features and gradient zero.
- Pole, branch-cut, periodicity and signed-zero fixtures; Jacobian checked against finite differences away from singularities.
- Conservation constraints checked where appropriate, but not asserted for arbitrary learned correction.
- Explicit negative test: graph proximity alone must NOT become a causal edge.
- Compare performance vs direct numerical reference; no implied climate/clinical validity.

## Non-goals
Not a claim of true distributed thought, proof of mechanistic reasoning, or a replacement for PDE solvers. No changes to mainline numerical kernels until the scalar oracle, semantics, and cross-platform tests exist.

References: https://www.tensorflow.org/neural_structured_learning ; https://github.com/tensorflow/neural-structured-learning

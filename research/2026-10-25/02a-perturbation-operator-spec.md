# H2 / Perturbation types: shared execution, distinct semantics

**Status:** proposed design and falsifiers; companion to [PR #2](https://github.com/AdaWorldAPI/neural-structured-learning/pull/2). Never interpret adversarial examples as causal interventions.

## Three explicitly tagged operators

**AdversarialGradient**
- Fields: model_digest, loss_digest, feature-mask digest, norm (L1/L2/Linf), step_size, epsilon_ball, PGD_steps, projection_rule, optional clipping bounds, seed.
- Reference: [NSL adversarial_neighbor.py](../../neural_structured_learning/lib/adversarial_neighbor.py) implements projected gradient-style neighbors and supports constraints and multiple iterations; reproduce the *configured* source behavior, not an unspecified FGSM approximation.
- Interpretation: perturbation constructed to increase a model loss, not an observed physical force.

**GeometricPerturbation**
- Fields: reference frame, units, surface/chart identifier, radius, base position, tangent components, angular normalization convention, discretization/codebook, step and uncertainty.
- Reference point: sphere x=r sin(theta)cos(phi), y=r sin(theta)sin(phi), z=r cos(theta); theta is polar angle [0,pi], phi azimuth modulo 2pi. Surface position and direction are distinct quantities.
- Interpretation: transform a geometric state; result is an input to physics, not an intervention by itself.

**MechanisticIntervention**
- Fields: structural-model/version digest, target variable, do-value, declared parents, exogenous variables, equation identifiers, time-index and scope.
- Interpretation: replace one structural assignment equation under the stated model and propagate to descendants. Requires independently justified structural assumptions. Graph similarity is not a structural equation.

The common envelope is a **research proposal**:
~~~text
PerturbationEnvelope {
   perturbation_id, baseline_state_digest, operator_kind,
   operator_parameters_digest, source_snapshot, scope,
   random_seed?, physical_units?, generated_at?, parent_event?
}
~~~
Names here define a potential DTO, not Rust types present on main.

## Core math and ways it can fail

For a differentiable system F(x), local first-order update: delta_F ≈ J_F(x) delta_x. Compare with F(x+delta_x)-F(x) and measure relative residual.

Sphere coordinate differential (off singularities):
- d p = e_r d r + e_theta r d theta + e_phi r sin(theta) d phi.
- Near theta in {0,pi}, azimuth phi is not a unique coordinate; use Cartesian/tangent reference rather than asserting round-trip coordinate identity.
- Angles phi and phi+2pi are equivalent; a naive numeric equality fails the seam. Use circular angular distance.
- A discretized direction codebook loses information. A LUT may replace a specific **discrete** operation, never automatically continuous differential geometry.

## Toy causal counterexample

Observed graph: Smoking <-> Stress (correlated); Smoking -> LungRisk is a proposed causal relation. A similarity edge Stress~Smoking must not be elevated to Stress causes LungRisk.
Better mechanistic unit fixture: X := Ux; Y := 2X + Uy. Replacing X with do(X=3) produces Y=6+Uy; merely conditioning X=3 need not be identical when Ux and Uy are correlated.

## Independent reference / falsification suite

| ID | Positive check | Negative control |
| --- | --- | --- |
| P01 | one PGD step matches unmodified NSL gradient and mask (declared tolerance) | remove feature mask |
| P02 | projection satisfies configured epsilon norm ball | remove projection |
| P03 | 8 PGD steps reproduce source with pinned seed/gradient rules | accidentally recompute only initial gradient |
| P04 | sphere point norm equals radius within measured roundoff | chart singularity yields spurious divergence |
| P05 | Cartesian->spherical->Cartesian agrees outside poles | phi=0 vs 2pi compared naively |
| P06 | finite-difference derivative converges to Jacobian off singularities | perturbation exceeds claimed linear validity region |
| P07 | analytic spring-network / heat-step reference agrees | nonconservative learned residual claimed to obey conservation |
| P08 | do(X=x) uses structural equation replacement | substitute conditioning for intervention |
| P09 | provenance distinguishes generated adversarial neighbor, geometric deformation and intervention | relabel a gradient edge as causal and assert REFUSE |

## End-to-end experiment

Start with 4 point masses on a ring and deterministic toy network logits. Record:
1. adversarial perturbation of the model input under NSL;
2. angular/tangent perturbation of one point and its Cartesian displacement;
3. intervention replacing one physical force equation;
4. effects after one and many time steps, with source evidence and numerical errors.

**Metrics:** relative state error, angular drift, Jacobian residual vs delta magnitude, loss change, conservation residual for the appropriate model, replay determinism, source/provenance completeness.

**Stop rule:** any conflation of edge similarity with causal force, unhandled chart singularity, or untraceable perturbation means no unified perturbation operator API is admitted. These can still share typed storage and execution primitives.

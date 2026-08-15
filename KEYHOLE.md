# Keyhole welding — theory and results

A complete account of the keyhole problem: the physics, the method from first
principles, and every number the run produces, with the reasoning that makes
each number meaningful.

Everything here is reproduced by two scripts in this directory:

```bash
python discover_symmetry.py --data dataset_keyhole.csv     # -> output/run.log
python rho_and_transform.py --data dataset_keyhole.csv     # -> output/rho_and_transform.log
```

Both are deterministic given a fixed environment (seed 42, fixed BLAS and library
versions). There is no training loop and no stochastic optimiser — but the GP
hyper-parameters *are* fitted by marginal-likelihood optimisation with six
restarts, and the pipeline selects a family, an active dimension and a lift.
Everything downstream of the gradient field is an eigenvalue, an eigenvector, or
an interval derived from one. **[revised]**

> **Revision note.** This document was revised after a technical review raised
> several well-founded objections. Six claims in the first version were wrong or
> overstated; each correction is marked **[revised]** where it occurs, and
> [Part V](#part-v--open-theoretical-questions) collects the questions that
> remain open. Every number in Part III was re-verified and stands.

**Contents**

1. [The physics](#part-i--the-physics)
2. [The theory](#part-ii--the-theory)
3. [Results](#part-iii--results)
4. [Limits](#part-iv--what-is-solid-and-what-is-not)
5. [Open theoretical questions](#part-v--open-theoretical-questions)
6. [Notation and file map](#appendix)

---

# Part I — The physics

## 1.1 What is happening in the weld

In laser welding a focused beam melts metal. Above a power threshold the metal
does not merely melt — it **vaporises**, and the recoil pressure of the escaping
vapour pushes open a narrow, deep cavity called a **keyhole**. The keyhole is
what makes deep welds possible: the beam deposits energy down the entire depth
of the cavity instead of only on the surface.

It is also the main failure mode. If the keyhole is too deep or unstable, its
tip collapses and traps vapour as a **pore**. Porosity is the dominant defect in
laser welding, so predicting keyhole depth from process settings is the
practical question the dataset is about.

## 1.2 The variables

`dataset_keyhole.csv` — 90 experiments, three materials (71 / 12 / 7 samples),
seven physical inputs:

| symbol | meaning | units |
|---|---|---|
| `etaP` | absorbed laser power (η × P) | W = kg·m²/s³ |
| `Vs` | scanning speed | m/s |
| `r0` | laser beam radius | m |
| `alpha` | thermal diffusivity | m²/s |
| `rho` | density | kg/m³ |
| `cp` | specific heat | m²/(s²·K) |
| `Tl-T0` | melting temperature rise | K |

The response is `e*` = keyhole depth ÷ beam radius, a dimensionless aspect
ratio. It ranges **0.209 to 12.773** across the dataset.

## 1.3 The known answer

The literature answer is that all seven variables collapse into a single
dimensionless group, the **Keyhole number**:

```
Ke = etaP · Vs^(-1/2) · r0^(-3/2) · alpha^(-1/2) · rho^(-1) · cp^(-1) · (Tl-T0)^(-1)
```

**[revised] Note the constant.** The published definition carries a factor of π,
`Ke = ηP / (π ρ c_p (T_l−T_0) √(α V_s r₀³))`, which the code's comment at
`discover_symmetry.py:62` omits. I verified that the `Ke` column of the CSV
*does* include it (`Ke_csv / Ke_no-π = 0.31884 ≈ 1/π`). The omission has **no
effect on any result in this document**: everything reported is an exponent
direction, and a multiplicative constant shifts `log Ke` without rotating it. It
does change the fitted slope and intercept below, which are quoted on the
π-inclusive CSV values.

Physically it is a ratio: energy delivered per unit thermal-diffusion length,
against the energy needed to raise that much material to melting. Measured
directly on this CSV, the empirical law is close to linear:

```
e* ≈ 0.383 · Ke − 0.305        R² = 0.978        (Ke spans 1.55 → 37.77)
```

## 1.4 The task

Recover that structure from `(X, e*)` alone. Not the coefficients — the
**structure**:

* how many independent dimensionless combinations does `e*` depend on?
* which combinations are they?
* what transformations of the inputs leave `e*` unchanged?
* how certain is each of those answers?

Nothing about units, materials, or welding physics is supplied to the method.

---

# Part II — The theory

## 2.1 Buckingham-Π gets you partway, then stops

Seven variables minus four independent physical dimensions (mass, length, time,
temperature) gives **3 dimensionless groups**. The code finds a valid basis from
the null space of the dimension matrix:

```
Pi1 = Vs^-1 · r0^-1 · alpha
Pi2 = etaP^-1 · Vs^3 · r0^2 · rho
Pi3 = Vs^-2 · cp · (Tl-T0)
```

with exact dimensional consistency, `||D·B||_inf = 0.0`.

Two things this does **not** give you:

* **Uniqueness.** Any invertible mixing of the exponents yields another equally
  valid basis. There are infinitely many, and the Π theorem prefers none.
* **Relevance.** It says nothing about which groups the response actually
  depends on. `Ke` sits inside this basis at exponents `[-0.5, -1, -1]`
  (residual `1.3e-15`), but nothing in the theorem singles it out.

So the real question is: *within the 3-dimensional space of Π-combinations,
which directions does `e*` respond to, and which leave it alone?*

## 2.2 The active subspace — one matrix answers it

Suppose the response has **ridge** structure, `y = f(Aᵀx)`: it depends on only a
few linear combinations of its inputs. Form the average outer product of the
gradient with itself,

```
C = E_rho[ grad_f(x) · grad_f(x)ᵀ ]  =  W Λ Wᵀ ,    λ1 ≥ λ2 ≥ … ≥ λn ≥ 0
```

and read the spectrum:

* **large λ** → moving along that eigenvector changes `y` substantially. These
  span the **active subspace**: the combinations that matter, i.e. the
  dimensionless groups.
* **λ = 0** → moving along that eigenvector never changes `y`. And here is the
  identity the whole method rests on:

```
λ = 0 for g  ⟺  E[(grad_f · g)²] = 0  ⟺  grad_f ⊥ g everywhere  ⟺  y(x + εg) = y(x)
```

So, **subject to regularity conditions stated below**:

| question | answer |
|---|---|
| how many response-relevant combinations? | `rank(C)` |
| which ones? | the leading eigenvectors |
| what is invariant? | the trailing eigenvectors |

**[revised] Two precision points that the first version of this document ran
together.**

*The null space is an almost-everywhere statement, not automatically a global
symmetry.* `gᵀCg = 0` gives `∇f·g = 0` only `ρ`-almost everywhere. Concluding
that `f` is constant along every orbit additionally needs: continuity of the
directional derivative, a density positive on a connected open region, and orbit
segments that stay inside the supported domain. On this dataset the support is a
union of three material sheets, so the connectedness condition is *not* met in
the raw variables — see §3.9.

*It is the null space **within the chosen catalogue**, not "the complete Lie
algebra of symmetries".* That stronger phrase would require proving the
candidate family contains the relevant algebra and is closed under Lie brackets.
Neither is claimed here.

*And `rank(C)` is not the number of dimensionless groups.* Buckingham-Π supplies
three independent dimensionless coordinates; `rank(C)` measures how many of them
the response actually varies through, under the chosen lift and density. The
accurate statement is "the data are consistent with approximately one
response-relevant monomial combination of the three", not "there is one
dimensionless group".

## 2.3 Why logarithms — scaling becomes translation

The identity above delivers **translational** symmetry, `x → x + εg`. A power
law is not that. It is a **scaling** symmetry, `x → x ⊙ exp(εs)`: multiply each
coordinate by its own factor.

Take logarithms and multiplication becomes addition:

```
x → x ⊙ exp(ε s)          becomes          u → u + ε s        with u = log x
```

A scaling symmetry **is** a translation, viewed in the right coordinates. That
coordinate map is the **lift** `k(x)`, and the response takes the form

```
f(x) = G( Aᵀ k(x) )        with k = log
```

which is exactly the *linearizable* symmetry class this project targets. The
chain rule carries the gradient across, `grad_u f = x ⊙ grad_x f`.

**Consequence worth stating plainly:** if `e*` obeys a power law, then in log
coordinates it is an ordinary ridge function, Constantine's machinery applies
verbatim, and **the exponents of the power law are the components of an
eigenvector**.

## 2.4 One matrix per symmetry family

Rather than substituting coordinates — which for the quadratic case divides by
`x` and blows up wherever a variable crosses zero — the implementation uses the
equivalent vector-field form. A *family* is a linear space of candidate
infinitesimal motions `ξ_θ(x) = Σ_a θ_a ξ_a(x)`. The response is invariant under
its flow exactly when the Lie derivative vanishes everywhere:

```
L_ξ y (x) = grad_f(x) · ξ_θ(x) = θᵀ c(x) = 0     for all x
c_a(x) := grad_f(x) · ξ_a(x)          ("symmetry-defect features")
```

Averaging over the data gives one symmetric matrix per family,
`C = E_rho[ c cᵀ ] ∈ R^{p×p}`:

| family | `ξ_a(x)` | `c_a(x)` | p (here) | equivalent lift |
|---|---|---|---|---|
| translation | `e_a` | `∂_a f` | 3 | `x` |
| scaling | `x_a e_a` | `x_a ∂_a f` | 3 | `log x` |
| rotation | `(e_i e_jᵀ − e_j e_iᵀ)x` | `x_j ∂_i f − x_i ∂_j f` | 3 | `x²/2` |

Two consequences:

* **The three "symmetry types" are not three methods.** They are three choices
  of basis `{ξ_a}` fed to one eigen-decomposition, on one shared gradient field.
  That is what makes them directly comparable.
* **The catalogue is extensible.** Any Lie subalgebra whose action is linear in
  its parameters — boosts, shears, affine maps, physics-constrained subgroups —
  drops in by supplying `ξ_a`.

## 2.5 The defect — why eigenvalues cannot rank families

Eigenvalues carry units, and different families carry *different* units, so `λ`
cannot compare across families. The scale-free replacement is the **defect**:

```
defect(θ) = sqrt(  E[(grad_f · ξ_θ)²] / E[||grad_f||² · ||ξ_θ||²]  )   ∈ [0, 1]
```

This is the RMS cosine of the angle between the gradient and the flow:

* `0` — the flow runs exactly along a level set: perfect invariance;
* `1` — the flow points straight up the gradient.

It is bounded in `[0,1]` by Cauchy–Schwarz, and it is a **ratio of two
expectations under the same `ρ`**, so a density that up-weights a high-gradient
region inflates numerator and denominator together.

**[revised] It is not, however, coordinate-free.** The numerator is the
invariant pairing between a differential and a vector field, but the Euclidean
norms in the denominator depend on a choice of metric on the input space. Two
consequences, both measured on this dataset:

* **A normalisation is required for the quantity to exist at all.**
  `discover_symmetry.py:127` centres the log-Π coordinates
  (`log_pi -= log_pi.mean(0)`, i.e. geometric-mean normalisation of each group).
  Without it the raw Π values span 18 orders of magnitude
  (`4.2e-01`, `5.9e-08`, `3.6e+06`) and every defect collapses to `0.00000` in
  floating point. This convention was previously undocumented; it is part of the
  method, not a formatting detail.
* **The defect *value* is basis-dependent; the *ranking* is not.** Re-running the
  screen under four different valid Π bases `Π' = Π^M` (§3.9a) moves the scaling
  defect over `0.00056–0.00965`, a factor of 9 — so it is **not** comparable
  across datasets as a bare number. But scaling wins in every basis, by margins
  of 14.5×–99×, and the recovered exponent direction is stable to four decimals.

So: read the defect as a within-analysis comparator under a stated metric, not as
a universal constant.

## 2.6 Two ways a zero eigenvalue can lie

`λ ≈ 0` has three possible causes, and only one is a symmetry:

1. **True symmetry** — the samples move along the direction and `y` does not.
2. **Design degeneracy** — the samples never move along it at all, so nothing
   about `y` there was ever measured. Guard: **excitation**,

   ```
   exc(θ) = E[ ||P_U F||² ] / E[ ||F||² ]  ∈ [0, 1]
   ```

   with `P_U` projecting onto the span the centred data actually cover.
   `exc ≈ 0` means the "symmetry" is an artefact of the experiment.
3. **Response degeneracy** — an *inert* coordinate the response ignores. This is
   an exact invariance of **every family at once**, and its excitation is 1,
   because the design may vary it perfectly well. Guard: **response coverage**,

   ```
   cov(θ) = E[ ||P_live ξ_θ||² ] / E[ ||ξ_θ||² ]  ∈ [0, 1]
   ```

   with `P_live` projecting onto coordinates that carry gradient signal.

Without both guards the family screen is decided by an `argmin` tie-break rather
than by the data. Both matter here: the keyhole dataset contains only three
materials, so the design-degeneracy path is live.

## 2.7 The Gaussian process — and why it is not just smoothing

Everything above consumes `grad f`, which is not measured. A GP supplies it, and
supplies more than a point estimate: **the derivative of a GP is a GP**, in
closed form. With an ARD squared-exponential kernel,

```
grad_mu(a)   = K_d(a) K_y^-1 y
Cov[grad_f]  = sigma_f² diag(1/l²) − K_d(a) K_y^-1 K_d(a)ᵀ
```

Three things follow that a point-estimate surrogate cannot provide:

**Model uncertainty.** Draw gradient fields from the posterior and push each
through the eigen-decomposition → credible intervals on every eigenvalue, on the
subspace, and on each exponent. This answers *would a different plausible
response surface give a different symmetry?*

**A resolution scale — `C_noise`. [revised]** Along a *true* symmetry the
response is exactly zero, so the entire eigenvalue seen in `C` is estimation
error. The GP quantifies that error: its posterior covariance is the expected
squared error of the posterior mean. Pushed into the family's parameter space,

```
C_noise[a,b] = E_x[ ξ_a(x)ᵀ Cov[grad_f(x)] ξ_b(x) ]
```

gives the scale at which an eigenvalue stops being distinguishable from zero.

**The first version of this document said `C − C_noise` "is the spectrum the
exact gradient field would have produced, to first order". That is not
justified, and the claim is withdrawn.** Under the ordinary Bayesian reading the
posterior second moment of the gradient is `μμᵀ + Σ`, so

```
E[C_true | data] = C_plugin + C_noise        (PLUS, not minus)
```

The posterior mean of `C` is the plug-in *plus* the floor. A subtraction would
need a separately derived frequentist de-biasing result — an unbiased gradient
estimator with sampling covariance `Σ` — and the GP posterior mean and posterior
covariance do not automatically supply that pair.

**What is true, measured rather than asserted.** The posterior mean is also the
wrong estimator for this question: it is strictly positive definite, so it can
never detect a null direction. The relevant question is whether the plug-in's
positive value *along a direction where the truth is zero* is explained by the
floor. On a synthetic problem where the true null direction `s` is known exactly
(`sᵀC_true s = 0` by construction):

| noise | median `sᵀC_plug s / sᵀC_noise s` | range over seeds |
|---|---|---|
| 2 % | 1.47 | 0.41 – 3.00 |
| 10 % | 0.99 | 0.28 – 4.17 |

So the floor is the **right order of magnitude** — the plug-in along a true null
direction really is inflated by roughly `Σ` — but it is calibrated only to
within a factor of ~3 either way. A ratio below 1 means a subtraction
over-corrects and the clip to zero is manufactured.

**Therefore the honest use is a ratio, not a subtraction:** report
`λ_i / (floor along eigenvector i)` as a signal-to-noise statement. On this
dataset that gives ≈1.4 for `λ₃` and ≈1.9 for `λ₂` — "both sit at the noise
floor", which is the same conclusion without needing a de-biasing theorem.
An eigenvalue reaching zero after subtraction is **not** evidence of an exact
invariance, and this document no longer claims it is.

*Second caveat, independent of the above:* `C_noise` is not diagonal in the
eigenbasis of `C`, so subtracting it rotates the eigenvectors. When the floor is
comparable to the structure it is meant to expose, that rotation is noise-driven.
Both spectra are printed side by side below for exactly this reason.

A second, independent error bar comes from a non-parametric **bootstrap** over
samples: *would a different dataset of the same size give a different subspace?*

## 2.8 The input density `ρ`, and the discrete part

`C = ∫ c cᵀ ρ dx` is an integral against a density. In a simulation `ρ` is a free
choice; in an experiment it is whatever the experiment was. Here it is awkward:
four of the seven inputs are material properties, and there are three materials.

Three results make this tractable.

**The exact algebra does not depend on `ρ`.** For any `ρ` with support `S`,

```
θ ∈ null(C_rho)  ⟺  E_rho[(grad_f · ξ_θ)²] = 0  ⟺  grad_f · ξ_θ = 0  rho-a.e. on S
```

Two densities with the same support give the same null space — the same
generators, exactly.

**`ρ` changes the conditioning, not the answer.** The gap `λ_k/λ_{k+1}` is
`ρ`-dependent, and `||P_Ŵ − P_W|| ≲ ||Ĉ − C|| / (λ_k − λ_{k+1})`. `ρ` moves the
error bar. What `ρ` genuinely changes is *approximate* symmetries, where
`grad_f · ξ` is small but non-zero and the best direction is a `ρ`-weighted
average — so the honest report is a **spread over `ρ`**, not a single number.

**Discrete `ρ` needs its own estimator.** If a coordinate takes `m` values, the
support is a union of `m` sheets and no flow that moves that coordinate maps the
support into itself. The answer along it comes from the surrogate's
*interpolation between levels* rather than from measurements — so its eigenvalue
is exactly as uncertain as the noise floor says. The right estimator is to
**stratify**:

```
C = Σ_g p_g C_g          exact for a mixture rho = Σ_g p_g rho_g
```

Nothing is approximated; what it buys is that no direction's estimate rests on
the surrogate interpolating across a material boundary. It also answers a
question the pooled matrix cannot: *one symmetry seen three times, or three
symmetries averaged into a plausible-looking one?*

There is one purely numerical hazard worth naming. On a design with few distinct
levels, an unconstrained ARD fit will drive that coordinate's length-scale
*below the level spacing*; the GP then fits each level as an isolated bump, the
posterior mean is flat **at every sample**, and the gradient field is identically
zero — with `R²(LOO)` still near 1, so the usual diagnostic misses it entirely.
`density.resolution_bounds` floors the length-scales at half the design
resolution. (On this dataset the fitted length-scales sit far above that floor,
so the guard is currently a no-op here — but it is the failure mode to check for
on any discrete design.)

---

# Part III — Results

All numbers from `output/run.log` and `output/rho_and_transform.log`, seed 42.

## 3.1 Stage 0 — Buckingham-Π reduction

```
7 variables − 4 dimensions = 3 independent Pi groups
  Pi1 = Vs^-1 · r0^-1 · alpha
  Pi2 = etaP^-1 · Vs^3 · r0^2 · rho
  Pi3 = Vs^-2 · cp · (Tl-T0)
  dimensional consistency ||D B||_inf = 0.0e+00

  Known Ke in the discovered Pi basis: exponents [-0.5, -1.0, -1.0]  (residual 1.3e-15)
```

This is bookkeeping. The discovery has not started — `Ke` is located in the
basis only so the result can be scored later.

## 3.2 Step 1 — the surrogate and its gradient field

```
GP surrogate:  sigma_f² = 139.8   sigma_n² = 0.01526   logML = 17.54
  R² (in-sample)     = 1.0000
  R² (leave-one-out) = 0.9784        <- read this before anything downstream
  ARD length-scales (standardised; large = inert coordinate):
      log Pi1   14.3228
      log Pi2    3.6964
      log Pi3    4.6092
```

`R²(LOO) = 0.978` on 90 points with no training loop.

The length-scale of **14.3 on `log Π1`** is already a finding: the response
barely moves along Π1. That is a free by-product of ARD — a per-coordinate
relevance measure — and it returns later as the reason Π1's transform exponent
cannot be pinned down.

## 3.3 Step 2 — the symmetry class

All three families, built from the *same* gradient field:

| family | φ | k\* | #gen | inactive E | gap | **defect** | R²(1 coord) |
|---|---|---|---|---|---|---|---|
| translation | id | 1 | 2 | 9.621e-03 | 106.6 | 0.0181 | 0.261 |
| **scaling** | **log** | **1** | **2** | **6.560e-03** | **188.1** | **0.0011** | **0.979** |
| rotation | x²/2 | 2 | 1 | 2.344e-04 | 65.6 | 0.0154 | 0.136 |

**Scaling wins by a factor 14.5 in defect.** Two details carry more weight than
the headline:

* **The rotation row has the smallest inactive energy** (`2.344e-04`). Ranking by
  inactive energy would pick rotation — and be wrong. Rotation needs *two* active
  directions to reach that number, and its best invariance is 14× worse. This is
  precisely why the criterion is the defect and not an energy fraction.
* **`R²(1 coord)` is an independent, matched-complexity check**: how well does
  `e* ≈ h(w1ᵀ φ(x))` predict using a single coordinate, cross-validated? 0.979
  for scaling against 0.261 and 0.136. The same winner under a criterion that
  shares nothing with the defect but the gradient field.

### The spectrum, in log-Π coordinates

`C = E[(Π ⊙ ∇f)(Π ⊙ ∇f)ᵀ]`:

| i | λ (plug-in) | λ/λ₁ | λ (noise floor removed) | λ/λ₁ | defect | excitation |
|---|---|---|---|---|---|---|
| 1 | 3.894e+01 | 1.000 | 3.872e+01 | 1.000 | 0.0365 | 1.000 |
| 2 | 2.071e-01 | 5.32e-03 | 9.331e-02 | 2.41e-03 | 0.0015 | 1.000 |
| 3 | 5.009e-02 | 1.29e-03 | **0.000e+00** | **0** | 0.0011 | 1.000 |

**`k* = 1`. [revised]** The data support a one-dimensional approximation: one
monomial combination of the three Π groups accounts for 99.34 % of the gradient
energy, and two directions are left over.

This is a **model-selection decision, not an exact rank.** All three plug-in
eigenvalues are strictly positive, and so is `λ₂` after the noise correction, so
the literal numerical rank is 3. The rule is `select_active_dimension` with
`energy_tol = 0.01`: `k` is the smallest dimension whose complement carries at
most 1 % of `Σλ`. The first version of this document said "no threshold tuning",
which was wrong — 1 % is a threshold. What supports the choice is that three
independent criteria agree: the energy rule, the spectral gap (188×), and the
matched-complexity `R²(1 coord) = 0.979`.

The third row is the direction the review flagged. After subtracting the noise
floor it reaches zero — but per §2.7 that subtraction is not a calibrated
de-biasing operation, so **zero here is not evidence of an exact invariance**.
The defensible reading is the ratio: `λ₃ / floor ≈ 1.4` and `λ₂ / floor ≈ 1.9`,
i.e. both trailing eigenvalues sit at the level of the surrogate's own gradient
error, and the data cannot distinguish them from zero.

### The discovered group

```
discovered exponents  [0.5241, 1.0000, 0.9568]
known Ke              [0.5000, 1.0000, 1.0000]
cos = 0.999540        (angle 1.74 deg)
```

**That is the Keyhole number**, recovered from 90 experiments with no physics
supplied.

## 3.4 Step 3 — uncertainty

| i | λ (plug-in) | bootstrap 95% (sampling) | GP posterior 95% (model) |
|---|---|---|---|
| 1 | 3.894e+01 | [3.106e+01, 4.770e+01] | [3.758e+01, 4.078e+01] |
| 2 | 2.071e-01 | [1.490e-01, 2.680e-01] | [2.471e-01, 4.070e-01] |
| 3 | 5.009e-02 | [3.819e-02, 6.162e-02] | [7.209e-02, 1.448e-01] |

The two intervals answer different questions and are not interchangeable: the
bootstrap asks *would another dataset of this size move the answer?*, the
posterior asks *would another plausible response surface move it?*

**Subspace stability** (distance, 0 = identical):

```
sampling (bootstrap)     median 0.0057   95% 0.0148
model    (GP posterior)  median 0.0061   95% 0.0151
```

The active direction is essentially fixed under both.

**Exponents with 95% credible intervals** (normalised so the largest is 1):

| group | discovered | 95% CI | known Ke |
|---|---|---|---|
| Π₁ | +0.5241 | [+0.5132, +0.5418] | +0.5 |
| Π₂ | +1.0000 | (reference) | +1.0 |
| Π₃ | +0.9568 | [+0.9419, +0.9855] | +1.0 |

Note the **Π₁ interval excludes 0.5**, narrowly. The correct reading is "the data
prefer 0.53 ± 0.02", not "the textbook value is wrong". A method that silently
rounded to 0.5 would be discarding information the data contain.

## 3.5 Step 4 — the generators, and a real test

Generators are read straight off the trailing eigenvectors — no separate
extraction step and no clustering tolerance:

| # | invariant rescaling | λ | defect | excitation |
|---|---|---|---|---|
| 1 | `Π₂^(+0.716) · Π₁^(−0.522) · Π₃^(−0.462)` | 0 (below floor; raw − floor = −6.37e-03) | 0.0010 | 1.000 |
| 2 | `Π₁^(+0.776) · Π₃^(−0.607) · Π₂^(+0.174)` | 9.331e-02 (λ/λ₁ = 2.41e-03) | 0.0015 | 1.000 |

Excitation `1.000` on both: the design genuinely explores these directions, so
the near-zero eigenvalues are invariances and not design artefacts.

### The orbit check

Take real experiments, push them along the **exact finite flow**
`Π → Π ⊙ exp(ε g)` with `|ε| ≤ 0.5`, and evaluate the surrogate end-to-end:

| direction | peak-to-peak Δ`e*` | as % of sd(`e*`) |
|---|---|---|
| generator 1 | 0.236 | **8.5 %** |
| generator 2 | 0.364 | **13.1 %** |
| Ke direction (contrast) | 4.269 | **153 %** |

A factor of 12–18 between the invariant directions and the active one.

**[revised] What this does and does not establish.** The GP was never told about
these directions, so nothing in the construction forces flatness — it is a real
test *of the surrogate's internal consistency*. It is **not** independent
physical validation: the same gradient field both discovers the generator and is
evaluated along its orbit. Independent validation would require new experiments
or simulations placed on the predicted orbits, or nested cross-validation in
which the GP, lift, family, dimension and active vector are all re-estimated per
fold.

**Support check.** At the `|ε| ≤ 0.5` used here, 80–93 % of flowed points remain
inside the bounding box of the data, with a worst excursion of 9.3 % of a
coordinate range; at `|ε| = 1` that falls to 50–76 %. So the current setting is
defensible, but the bounding box is a generous notion of support — the true
support is a union of three material sheets, and a flow that moves between
sheets is interpolating, not measuring.

## 3.6 Step 5 — the Π theorem, recovered from data

Now run the *same* scaling family on the **raw seven variables**, with no
dimensional analysis at all. If the method works, unit rescalings must appear as
invariances of a dimensionless target.

```
GP on raw log-variables: R²(LOO) = 0.9832
rank of the centred log-design = 5 of 7  -> 2 directions were never varied
                                            (only 3 materials in the dataset)
```

Gradients are projected onto the explored span, so unmeasured directions appear
as **exact zeros** rather than as fabricated symmetries.

| i | λ | λ/λ₁ | defect | excitation |
|---|---|---|---|---|
| 1 | 6.327e+01 | 1.000 | 0.0000 | 1.0000 |
| 2 | 1.635e+00 | 2.58e-02 | 0.0000 | 1.0000 |
| 3 | 1.960e-01 | 3.10e-03 | 0.0000 | 1.0000 |
| 4 | 6.477e-02 | 1.02e-03 | 0.0000 | 1.0000 |
| 5 | 1.682e-02 | 2.66e-04 | 0.0000 | 1.0000 |
| 6 | 2.732e-16 | 4.32e-18 | 0.0000 | **0.0000** |
| 7 | 0.000e+00 | 0 | 0.0000 | **0.0000** |

The last two carry excitation **exactly 0** — correctly reported as *never
measured* rather than as symmetries. That is §2.6 doing its job on real data.

**The unit-rescaling directions** — the rows of the dimension matrix, which the
method was never shown:

| unit | explored fraction | defect |
|---|---|---|
| Mass | 0.690 | 0.0000 |
| Length | 0.803 | 0.0000 |
| Time | 0.755 | 0.0000 |
| Temperature | 0.687 | 0.0000 |

Principal angles to the inactive subspace:

```
at k = 2:  [ 0.00, 0.00, 43.45, 90.00 ] degrees
at k = 1:  [ 0.00, 0.00,  0.00, 43.50 ] degrees
```

Three of the four unit directions sit inside the recovered symmetry algebra; the
fourth is at 43.5°. The active direction in raw log-space matches the known `Ke`
exponents on the explored part at **cos = 0.9975**.

**[revised] How much this actually establishes.** The first version called this
"the Buckingham-Π structure recovered from data." That is too strong, for three
reasons:

* **A unit change is not an experimental perturbation.** The data were recorded
  in one unit system; they do not contain repeated unit-rescaling experiments.
  What is being checked is algebraic covariance, not an observed physical
  response.
* **The target is dimensionless by construction.** `e* = e/r₀` is a function of
  the Π groups, so any log-space direction leaving all Π invariant leaves `e*`
  invariant *identically*. The GP recovering that is a consistency check on the
  surrogate, not a discovery about the physics.
* **Some of the zeros are forced by design rank.** The centred log-design has
  rank 5 of 7, and gradients are projected onto the explored span. Directions
  outside that span are guaranteed zero. They should be read as
  **non-identifiable**, which the excitation column correctly reports as `0.0000`
  — not as recovered symmetries.

The 43.5° fourth angle is the honest signal here. The defensible claim is:
*on the directions this design explores, the response behaves consistently with
depending only on dimensionless combinations* — partial alignment, not complete
recovery.

## 3.7 Step 6 — local symmetry (a negative control)

Clustering the gradient field over 1–3 regimes selects **one** regime: a single
global power law fits all 90 points; two- and three-way splits are rejected
because the resulting regimes are too small.

```
#regimes   sizes           k per regime   pooled inact.E   max subsp.dist
       1   [90]            [1]            6.560e-03        0.000
       2   [8, 82]         (regime too small)
       3   [72, 5, 13]     (regime too small)
```

This is the right answer for a single scaling law, and a useful check that the
clustering does not manufacture regimes that are not there.

## 3.8 Does the answer depend on `ρ`?

Same 90 rows, same gradient field, three densities by importance reweighting in
log-Π coordinates:

| `ρ` | ESS | k\* | gap | best defect | subspace shift | cos(w, Ke) |
|---|---|---|---|---|---|---|
| as-sampled | 90.0 | 1 | 188 | 0.0011 | 0.0000 | 0.999569 |
| uniform | 23.9 | 1 | 181 | 0.0006 | 0.0359 | 0.997981 |
| gaussian | 59.5 | 1 | 185 | 0.0011 | 0.0205 | 0.998874 |

`k*` and the generator are unchanged under every density, and the Keyhole number
comes back every time. What moves is the conditioning and the defect — and the
range **0.0006–0.0011 is the `ρ`-uncertainty**, a third error bar alongside the
sampling and model intervals of §3.4.

The cost is visible in the ESS column: forcing a uniform density drops the
effective sample size from 90 to **24**. Any `ρ`-study that pushes the ESS far
below `N` is measuring its own weights rather than the physics.

## 3.9 What `ρ` actually is here

`design_report` on the seven physical inputs, told nothing about the experiment:

```
coordinate    levels   levels/N   kind
etaP              90      1.000   continuous
Vs                11      0.122   discrete
r0                 4      0.044   discrete
alpha              3      0.033   discrete
rho                3      0.033   discrete
cp                 3      0.033   discrete
Tl-T0              3      0.033   discrete

rank of the centred design      = 4 of 7   -> 3 directions never varied
rank of the centred log-design  = 5 of 7
discrete coordinates: Vs, r0, alpha, rho, cp, Tl-T0
they span rank 3 between them (not 6)
```

That last line is the finding. Four of these are material properties and the
dataset has three materials, so they are **one three-level factor**, not four
independent discrete coordinates — and the rank line says so without being told
anything about metallurgy.

Note what dimensional analysis does *not* fix: in Π coordinates the three groups
have 19, 90 and 16 levels — continuous, because a product of powers of a discrete
and a continuous variable is continuous. **The discreteness is hidden, not
removed.** The support is still a union of three sheets, which is what §3.10
confronts.

## 3.9a [new] Does the answer depend on the choice of Π basis?

`ρ` is not the only arbitrary choice. The Buckingham-Π basis is non-unique: any
invertible integer matrix `M` gives another valid basis `Π' = Π^M`. Since the
defect's denominator uses a Euclidean norm (§2.5), the basis is part of the
metric — so the screen must be re-run in several of them.

| basis | translation | scaling | rotation | winner | 2nd/best | cos(w, Ke) |
|---|---|---|---|---|---|---|
| identity | 0.01806 | **0.00106** | 0.01535 | scaling | 14.5× | 0.999569 |
| shear | 0.15739 | **0.00965** | 0.15484 | scaling | 16.0× | 0.999155 |
| mix | 0.12370 | **0.00056** | 0.05547 | scaling | 99.1× | 0.999621 |
| heavy | 0.13443 | **0.00391** | 0.22801 | scaling | 34.4× | 0.999304 |
| permutation | 0.01806 | **0.00106** | 0.01535 | scaling | 14.5× | 0.999569 |

Three separate readings, and they do not all point the same way:

* **The defect value is basis-dependent** — 0.00056 to 0.00965, a factor of 9.
  A defect quoted without its basis and normalisation means little.
* **The class ranking is basis-invariant** — scaling wins in every basis, and the
  margin over the runner-up never falls below 14×.
* **The physical answer is basis-invariant** — mapping each recovered direction
  back through `w_orig = Mᵀv` gives `cos(w, Ke)` between 0.9992 and 0.9996
  everywhere.

So the conclusion is robust even though the score is not. That distinction should
be carried into any write-up.

## 3.10 The discrete part, handled as a mixture

| stratum | n | p_g | k\* | best defect | cos(w, Ke) |
|---|---|---|---|---|---|
| Mat1 | 71 | 0.789 | 1 | 0.0128 | 0.999820 |
| Mat2 | 12 | 0.133 | 1 | 0.0004 | 0.996978 |
| Mat3 | 7 | 0.078 | 1 | 0.0006 | 0.999458 |
| **pooled** | 90 | — | **1** | **0.0011** | **0.999569** |

Max subspace distance between strata: **0.106**. The pooled subspace is identical
to the global one (distance `0.0000`), which is the mixture identity
`C = Σ_g p_g C_g` holding numerically. Pooled generator:
`Π₂^(+0.718) · Π₁^(−0.518) · Π₃^(−0.464)` invariant.

### [revised] These are NOT three independent estimates

The first version of this document called the table above "one symmetry seen
three times ... each material recovers the Keyhole number independently." **That
is wrong, and the retraction matters.** Every row uses gradients from the *same*
globally fitted GP, so they are conditional checks of one shared surrogate.

Refitting a separate GP on each material shows how much the shared surrogate was
carrying:

| stratum | n | cos(w, Ke) — global GP | cos(w, Ke) — own GP | R²(LOO), own GP |
|---|---|---|---|---|
| Mat1 | 71 | 0.999820 | 0.954818 | 0.9896 |
| Mat2 | 12 | 0.996978 | **0.666677** | 0.8798 |
| Mat3 | 7 | 0.999458 | **0.666667** | 0.7113 |

`0.6667` is the degenerate answer — the direction `e₂`, "only Π₂ matters." And
the reason is structural, not statistical:

```
Mat1  n=71   centred log-Pi rank = 3 of 3
Mat2  n=12   centred log-Pi rank = 2 of 3   <- one direction never varied
Mat3  n= 7   centred log-Pi rank = 2 of 3   <- one direction never varied
```

Two of the three materials are **rank-deficient in Π space**. They cannot
identify a three-component exponent vector at all, with any estimator. Seven
samples spanning a 2-D subspace is not independent evidence for a 3-D direction.

**What survives.** The mixture decomposition is still exact, and it still does
the job it was introduced for: no direction's estimate rests on the GP
interpolating *across* a material boundary. The between-stratum distance of 0.106
still says the three materials are consistent with one shared symmetry rather
than three different ones. What it cannot support is the word *independent*.

## 3.11 The transform, discovered rather than selected

Everything above was *told* to consider three families. This step withholds the
catalogue and estimates the exponent `p` in `φ'(x) = x^p` from the geometry of
the gradient field alone. `p = 0` would mean the level sets are parallel planes
(translation); `p = −1` that the normal falls off like `1/Π` (scaling); `p = +1`
that it grows like `Π` (rotation).

| p | φ(x) | k\* | inactive E | min defect | |
|---|---|---|---|---|---|
| −2.0 | −1/x | 3 | 3.017e-01 | 0.00213 | |
| −1.6 | −x^-0.6/0.6 | 3 | 2.034e-01 | 0.00182 | |
| −1.2 | −x^-0.2/0.2 | 3 | 4.625e-02 | 0.00145 | |
| **−1.0** | **log x** | **1** | **6.560e-03** | **0.00106** | **scaling — best** |
| −0.8 | x^0.2/0.2 | 2 | 3.951e-02 | 0.00164 | |
| 0.0 | x | 1 | 9.621e-03 | 0.01806 | translation |
| +1.0 | x²/2 | 1 | 3.111e-05 | 0.11657 | rotation |
| +2.0 | x³/3 | 1 | 1.140e-07 | 0.21251 | |

Selection is by **min defect**; `k_active` and inactive energy are diagnostics
only. The reason is visible in the table: at large `|p|` the field concentrates
on the smallest `x`, `C` loses rank for reasons unrelated to symmetry, and the
inactive energy would happily select that artefact (`1.14e-07` at `p = +2`).

The scaling law is recovered without the catalogue being consulted. The honest
caveat is printed alongside: **`p` within 5% of the best spans `[−2.0, −0.2]`** —
on 90 points the defect curve is shallow, and a range of `p` is compatible with
the data.

## 3.12 The lift `k(x)`, searched over a primitive library

The full search scores every candidate lift on the same gradient field by
`L(k) = λ_min(X_N[k])` with per-sample normalisation, so `tr X_N = 1` and
different lifts are directly comparable. The threshold comes from the GP's own
gradient-noise floor rather than a constant — along a true symmetry the whole
eigenvalue *is* estimation error, so the floor is the resolution limit.

```
tau = 0.1126   [surrogate gradient-noise floor]

k(x)                                        L = lam_min   #sym   r   margin  cplx
--------------------------------------------------------------------------------
k = log x                                     7.788e-03      2   1      8.4      3   <== selected
the same, snapped to named exponents          2.316e-02      2   1      8.1      3
k from the log|grad| vs log|x| regression     1.365e-02      2   1      8.3      6
k = x                                         4.860e-02      1   2      2.2      0
k: Pi3 -> Pi3/Pi2                             1.578e-02      1   2      2.8      2
```

Ranking is lexicographic — fewest effective variables `r`, then largest margin,
then simplest expression. `L = λ_min` alone cannot rank these: it is 0 for every
lift as soon as any symmetry exists.

```
SELECTED: k(x) = (log Pi1, log Pi2, log Pi3)
  f(x) = G(Aᵀ k(x))   with r = 1 effective variable and 2 independent generators

Active direction from the selected lift: [0.4966, 1.0000, 0.9876]
Known Ke in the Pi basis                : [0.5,    1.0,    1.0   ]
cos = 0.999983
```

**cos = 0.999983** — the sharpest recovery in the whole run. The lift is
*discovered*, not selected from a menu: the search is handed `(Π, grad e*)` and
returns `k = log Π` with one effective variable, which is the statement that
`e*` obeys a power law in the Π groups.

---

# Part IV — What is solid, and what is not

## 4.1 Solid

| claim | evidence |
|---|---|
| the data support a **one-dimensional approximation** | 99.34 % of gradient energy in one direction; gap 188×; `R²(1 coord) = 0.979`; stable across 3 densities, 4 Π bases, bootstrap and GP posterior |
| the active direction is the Keyhole number | cos = 0.99954 (Π basis), 0.999983 (lift search), 0.9975 (raw variables), 0.9992–0.9996 across Π bases |
| the symmetry class is scaling | defect 0.0011 vs 0.0181 / 0.0154; `R²(1 coord)` 0.979 vs 0.261 / 0.136; winner unchanged in all 4 bases at ≥14× margin |
| 2 approximately invariant directions | defects 0.0010 / 0.0015, excitation 1.00, orbit response 8.5 % / 13.1 % against 153 % |
| `ρ` moves conditioning, not the answer | `k* = 1` under all three densities, max subspace shift 0.0359 |
| the three materials are **consistent with one** symmetry | between-stratum subspace distance 0.106, pooled = global to 0.0000 |
| the run is reproducible in a fixed environment | no stochastic optimisation after the GP fit; seeded; **not** bit-for-bit across BLAS implementations **[revised]** |

## 4.2 Not solid — and the run says so from the inside

**Per-coordinate transform exponents.** Estimating one `p` per coordinate is more
than 90 points support:

| coordinate | p | 95% CI | gradient weight | φ(x) |
|---|---|---|---|---|
| Π₁ | −0.091 | [−0.497, +0.220] | 0.108 | `x` (weak: little gradient energy) |
| Π₂ | −1.159 | [−1.192, −1.108] | 0.426 | `log x` |
| Π₃ | −0.657 | [−0.884, −0.462] | 0.466 | `x^0.343/0.343` |

Π₂ comes back at `−1` cleanly. Π₁ carries about a tenth of the gradient energy
with an ARD length-scale of 14.3 — it is nearly inert, so its exponent is
estimated from almost nothing, and the interval says so rather than hiding it.

**The fully self-discovered run does not converge.** Starting from raw Π, with
the coordinate system and the transform estimated together, six iterations reach
`R²(LOO) = 0.9729` and stop improving. The scaling active direction from that
field gives **cos = 0.805** against `Ke` — much worse than the 0.99954 obtained
when the GP is fitted in log-Π.

```
iter   R2(LOO)              p (estimated)     moved
   0    0.9114              [+0. -1. -1.]    0.4197
   1    0.9660              [+0. -1. -1.]    0.1679
   2    0.9717              [+0. -1. -1.]    0.0672
   3    0.9727     [+0.   -1.   -0.658]     0.1890
   4    0.9729     [+0.   -1.   -0.633]     0.0907
   5    0.9728     [+0.   -1.   -0.619]     0.0447
DID NOT CONVERGE — falling back to the iterate with the best R2(LOO)
```

**This is the correct behaviour.** The method reports non-convergence and falls
back instead of returning a confident transform the data do not support. On the
synthetic problems, where the surrogate is accurate, the same code converges to
the exact exponent in two or three iterations. The limiting factor here is 90
data points, and the run identifies that itself.

**The two transform estimators disagree** in that run (global scan `p = −1.10`
vs local median `p = −0.63`), and the printed verdict is the right one: the
invariance is probably not component-wise in those coordinates, and the `gl(n)`
linear family — which searches the mixing directly — is the tool to reach for.

**[new] Four further limits, added after review:**

* **The one-coordinate `R²` is not nested.** The active direction is estimated
  once from all 90 observations; only the final 1-D regression is
  cross-validated. That leaks. An unbiased figure would repeat GP fitting,
  gradient estimation, family and lift selection, dimension selection and
  direction estimation inside every fold. Leave-one-material-out would be the
  informative version — and given the rank-2 strata of §3.10, it should be
  expected to fail.
* **The per-material results are conditional on one surrogate**, not independent
  (§3.10).
* **Posterior draws are pointwise-independent.** `sample_gradients` draws each
  evaluation point independently, ignoring posterior correlation between points.
  For `C = (1/N)Σ gᵢgᵢᵀ`, independent errors average down as `1/N` while
  spatially coherent errors do not average at all — so the reported model
  intervals are **anti-conservative**, i.e. too narrow, for coherent surrogate
  error. (`gradients.py` currently documents this as "conservative"; that is
  backwards and should be fixed.)
* **Hyper-parameter uncertainty is not propagated.** ARD length-scales, `σ_f²`
  and `σ_n²` are fixed at their marginal-likelihood maxima, so the intervals in
  §3.4 are conditional on those estimates rather than fully Bayesian. A
  squared-exponential kernel also imposes infinite smoothness; a Matérn
  comparison would test how much the gradient field depends on that choice.

## 4.3 What the exercise establishes

1. **The structure is recoverable.** The response-relevant dimension, the group
   itself, the approximate invariant directions and the symmetry class, all from
   one gradient field and one eigen-decomposition per family — and stable under
   changes of density and of Π basis.
2. **`ρ` degrades it in a predictable way.** A discrete design has fewer distinct
   positions to fit a slope through and leans on the surrogate to interpolate
   between levels, so the intervals widen there — which is exactly what the
   theory predicts, and not a claim that the method fails on discrete data.
3. **The failures are visible from inside the run.** Non-convergence is reported;
   unmeasured directions come out as exact zeros with excitation 0; a nearly
   inert coordinate gets a wide interval; the noise floor is printed next to the
   spectrum. A method that gets things wrong *loudly* is worth more than one that
   gets them wrong silently.

## 4.4 [revised] The defensible summary

> The data provide strong evidence for an approximately one-dimensional monomial
> ridge structure in log-Π coordinates. The estimated active direction is closely
> aligned with the established Keyhole number, and that alignment is stable under
> resampling, under the surrogate posterior, under three input densities and
> under four Π bases. The corresponding scaling directions show approximate
> invariance *within the fitted surrogate and the sampled region*. Additional
> theoretical assumptions, nested validation, and a revised treatment of GP
> gradient uncertainty are required before interpreting the trailing eigenvectors
> as an exact global symmetry algebra, or before claiming that Buckingham-Π
> structure has been recovered from raw data alone.

---

# Part V — Open theoretical questions

These are unresolved. They are recorded here rather than hidden because each one
bounds how strongly a result above may be stated.

| # | question | status |
|---|---|---|
| 1 | Is `C_noise` a Bayesian posterior variance or the sampling covariance of a frequentist gradient estimator? The subtraction needs the second; the GP supplies the first. | **open** — §2.7 replaces the subtraction with a ratio, which needs neither |
| 2 | What theorem licenses subtracting it from the plug-in matrix? | **none known**; calibration measured at a factor ~3 either way |
| 3 | What regularity, domain and connectedness assumptions turn `null(C)` into a global symmetry algebra? | **stated but unproven** in §2.2; the connectedness condition provably fails in the raw variables |
| 4 | How should `k*` be selected when all eigenvalues are positive? | currently a 1 % energy threshold, corroborated by gap and `R²`; no formal test |
| 5 | Is the one-coordinate `R²` nested over the whole pipeline? | **no** — known leakage, §4.2 |
| 6 | Are the per-material results independent? | **no** — refuted, §3.10 |
| 7 | How is local excitation assessed on the discrete material sheets? | globally via `excitation`; no per-sheet tangent analysis |
| 8 | Why is the quadratic lift labelled "rotation"? | the two share **orbits** on positive data but are different families; the naming is misleading and should change — see below |
| 9 | How sensitive is the defect to metric, basis and normalisation? | **measured**, §2.5 and §3.9a: value swings 9×, ranking and answer do not |
| 10 | Can the invariances be tested against new observations on predicted orbits? | **not yet attempted** — the strongest available next test |

**On question 8**, to be precise about what is and is not claimed. The
`RotationFamily` in the code uses `ξ = (eᵢeⱼᵀ − eⱼeᵢᵀ)x`, i.e. genuinely `Ax`
with `A` skew — not the coordinate-wise field `ẋᵢ = θᵢ/xᵢ` that the quadratic
lift induces. Those are different families of different dimension
(`n(n−1)/2` versus `n`). What coincides, on strictly positive data, is their
**orbits** in a single `(i,j)` plane (`xᵢ² + xⱼ² = const`), which is enough for
the invariance question but not for identifying the families. The synthetic
benchmarks already separate the cases — `rotation`, `boost` and `rotation-w`
share the `x²/2` lift while only the first lies in `so(n)`, with `so(n)` defects
of 0.0000, 0.0873 and 0.0493 respectively. The mathematics is handled; the label
is not. Calling it the **quadratic-lift family** would remove the ambiguity —
and would also explain why the `p = +1` row of §3.11 and the `rotation` row of
§3.3 report different active dimensions: they are different families that happen
to share `p = 3` at `n = 3`.

---

# Appendix

## A.1 Notation

| symbol | meaning |
|---|---|
| `k(x)` | the **lift** — the coordinate map that linearises the symmetry (here `log`) |
| `r` | the **active dimension** — number of dimensionless groups (here 1) |
| `k_active` | the same quantity in the older code; `active_subspace.lifts.notation()` prints the mapping |
| `p` | dimension of a symmetry family, or the transform exponent in `φ'(x) = x^p` — context distinguishes them |
| `C` | `E_rho[c cᵀ]`, the active-subspace matrix |
| `ξ_a` | a basis vector field of a symmetry family |
| `c_a` | `grad_f · ξ_a`, the symmetry-defect feature |
| `exc` | excitation — did the design move along this direction? |
| `cov` | response coverage — does the response respond to what this direction moves? |

## A.2 Files

| file | what it is |
|---|---|
| `dataset_keyhole.csv` | 90 experiments, 7 inputs, `e`, `Ke`, `e*` |
| `discover_symmetry.py` | Stages 0–6 above → `output/run.log` |
| `rho_and_transform.py` | §3.8–3.12 → `output/rho_and_transform.log` |
| `output/asm_artifacts.npz` | Π, `y`, gradients, `C`, spectrum, bootstrap samples |
| `output/summary.png` | spectrum, exponents with CIs, orbit check, collapse plot |
| `../../METHOD.md` | the method in full, with the general theory |
| `../synthetic_benchmarks/` | the same pipeline on problems with known answers |

## A.3 Reproducing

```bash
python discover_symmetry.py --data dataset_keyhole.csv
python rho_and_transform.py --data dataset_keyhole.csv
```

Seed 42, single CPU core, seconds per script.

**[revised] What "reproducible" means here.** The pipeline contains no stochastic
optimisation after the GP hyper-parameter fit, and that fit is seeded — so a
rerun in the same environment reproduces these tables exactly. It is *not*
bit-for-bit across machines: floating-point linear algebra, eigensolver
implementations, BLAS versions and eigenvector sign conventions all differ.
Reproducibility should be checked to a tolerance on: eigenvalues, principal
angles or projector distances, exponent vectors **up to sign and scale**, and
predictions with their intervals.

**Sign convention.** An eigenvector's sign is arbitrary. Throughout this document
the recovered direction is normalised so that its Π₂ component is positive
before comparison with `Ke`; `cos` is reported as `|cos|`, so the comparison is
against the *line* spanned by `Ke`, not a ray. Reporting `Ke` versus `1/Ke` is
the same statement under this convention.

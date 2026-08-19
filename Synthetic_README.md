# Four known symmetries, recovered from data

A minimal, self-contained demonstration of the active-subspace method on
synthetic data where **the answer is known in advance**. Four datasets:
translation, scaling, rotation — and then the Ergun equation, which has *two
different symmetries in two different regimes* and does not tell you where the
boundary is.

For every one of them we wrote down the response function ourselves, so the
dominant directions and the generators are not estimates to be argued about but
numbers to be checked against.

```bash
python datasets.py       # writes data/{translation,scaling,rotation}.csv
python run_analysis.py   # writes output/report.txt
python make_plots.py     # writes output/{translation,scaling,rotation}.png

python ergun.py          # writes data/ergun.csv
python run_ergun.py      # writes output/ergun_report.txt and output/ergun.png
```

No training loop anywhere. Every number below is reproduced verbatim from
`output/report.txt` and `output/ergun_report.txt`.

**Contents.** §1 ridge functions and the lift · §2 **the protocol, all steps in
order** · §3 the four datasets · §4 results, datasets A–C · §5 results, local
symmetry · §6 summary and caveats.

---

## 1. Ridge functions: the object the whole method is about

A **ridge function** is a function of many inputs that only ever looks at a few
combinations of them:

```
y = h( w₁ᵀx , … , w_rᵀx ),        r ≪ n
```

The `w`'s are the **dominant directions** — the combinations the response
actually depends on. Everything perpendicular to them is invisible to `y`: move
`x` along any direction `g` with `wᵢᵀg = 0` for all `i`, and `y` does not
change. Those `g` are the **generators** of the symmetry.

So for a ridge function, "find the symmetry" and "find the dominant directions"
are the *same question asked from opposite ends*. There are `n` directions in
total; `r` of them move the response and `n − r` of them do not.

### 1.1 The generalisation this project is built on

Most interesting symmetries are not translations, so the plain ridge form is
too narrow. The fix is to allow a **lift** `k(x)` — a change of coordinates —
and ask for a ridge function *there*:

```
y = G( Aᵀ k(x) )
```

`k` is what turns the group action into a plain translation. Three choices
cover the classical cases, and they are datasets A, B and C:

| lift `k(x)` | `Aᵀk(x)` is linear in | symmetry that fixes `y` | name |
|---|---|---|---|
| `k(x) = x` | `x` | `x → x + εg` | translation |
| `k(x) = log x` | `log x` | `x → x ⊙ exp(εs)` | scaling / power law |
| `k(x) = x²/2` | `x²` | `x → exp(εA)x`, `Aᵀ = −A` | rotation |

Read the middle column as the point of the whole exercise: **a scaling symmetry
is a translation symmetry, once you measure `x` on a log ruler.** A rotation is
a translation once you measure `x²`. That is what "linearizable" means in the
project title.

### 1.2 When one ridge function is not enough

Dataset D breaks the assumption on purpose. The Ergun equation is a *sum* of
two power laws with different exponents, so no single `w` works everywhere.
What it has instead is a **local** symmetry: one power law in the viscous
regime, another in the inertial regime, and a crossover between them. The
regimes are not labelled in the data. Step 6 of the protocol below is the part
that handles this.

---

## 2. The protocol, all steps in order

This is the complete method, in the order the code executes it. Steps 1–5 and 7
are `run_analysis.py`; steps 6 and 8 are `run_ergun.py`; step 9 is the
validation both scripts print.

### Step 1 — collect `(X, y)` and nothing else

`X` is `(N, n)` inputs, `y` is `(N,)` scalar responses. No regime labels, no
candidate symmetry, no derivative information. Everything else is derived.

### Step 2 — choose the analysis coordinates

The method is applied in whatever coordinate the data is handed over in. For
datasets A–C that is `x` itself. For Ergun it is `U = log x` with
`Y = log(dP/L)`, for two reasons: the design spans several decades, so `log` is
the only coordinate in which the GP sees an even spread of points; and in log
coordinates a scaling symmetry *is* a translation symmetry, so every recovered
direction reads directly as a vector of **exponents**.

This is a modelling choice and it is the one place physical knowledge enters.

### Step 3 — fit a GP surrogate and take its gradient field

`∇f` is not given, so it is read off a Gaussian process fitted to `(X, y)`. The
derivative of a GP is a GP in closed form, so the gradient at every sample is
available analytically — no finite differences, no separately trained
derivative model.

**Check `r2_cv` before anything else.** It is the surrogate's leave-one-out R².
If it is not near 1, the gradient field is meaningless and so is every step
below it. It is printed first on every dataset (0.9999+ throughout here).

### Step 4 — build `C` for each candidate family and diagonalise it

For the plain ridge case, Constantine's construction is one expectation and one
eigen-decomposition:

```
C = E_ρ[ ∇f(x) ∇f(x)ᵀ ] = W Λ Wᵀ,      λ₁ ≥ λ₂ ≥ … ≥ λₙ ≥ 0
```

Every eigenvalue has a direct reading, because for a unit direction `g`

```
gᵀ C g = E[ (∇f · g)² ]
```

is the mean-square rate of change of `y` when you move along `g`. So:

* **large `λ`** → moving along that eigenvector changes `y` a lot → it is a
  dominant direction;
* **`λ = 0`** → `∇f · g = 0` at every sample → `y(x + εg) = y(x)` for all `ε`
  → `g` is a **generator of an exact symmetry**.

That last line is the whole method. The null space of `C` *is* the Lie algebra
of translations that fix `y`. Nothing is trained, nothing is optimised; the
symmetry is an eigenvector.

For a general lift, `∇f` is replaced by the derivative of `y` along that
family's basis flows, `c(x)`, and `C = E[c cᵀ]` is assembled the same way. The
three families differ in exactly one line of code each:

| family | flow `ξ_θ(x)` | feature `c(x)` | why |
|---|---|---|---|
| translation | `θ` | `∇f` | `d/dε f(x+εθ) = ∇f·θ` |
| scaling | `x ⊙ θ` | `x ⊙ ∇f` | chain rule with `∇_u f = x ⊙ ∇_x f`, `u = log x` |
| rotation | `Ax`, `A` antisym. | `c_{ij} = x_j ∂_i f − x_i ∂_j f` | `d/dε f(exp(εA)x)` |

The parameter space has dimension `n` for translation and scaling, and
`n(n−1)/2 = 6` for rotation. **The eigen-decomposition happens in that
parameter space**, which is why the rotation ground truth is written as
6-vectors.

A caution on what `θ` is. For translation and scaling, `θ` *is* the generator —
a direction in ℝⁿ. For rotation it is not: the generator there is an
antisymmetric matrix `A ∈ so(n)`, and `θ` holds its `n(n−1)/2` independent
entries. The distinction matters when reading the output, and §4.5 prints the
recovered generators both ways. It does not affect the algebra, because
`⟨A, B⟩_F = 2 θ_A · θ_B` makes the coordinate representation an isometry up to
scale.

### Step 5 — split the spectrum, and score every direction

Two numbers decide how many generators there are.

**Active dimension.** A direction is called inactive only if everything from
there down carries less than `energy_tol` of `Σλ`. Datasets A–C use
`energy_tol = 1e-3`; Ergun needs `0.05`, for a reason given in §5.3. **This
threshold is a choice, and it is the one place in the pipeline where a choice
changes the reported number of generators.** The full spectrum is always
printed so the choice can be second-guessed.

**Defect.** The scale-free quality score for a candidate generator `θ`:

```
defect(θ) = RMS cosine between ∇f and the flow direction ξ_θ
```

`0` = exact invariance, `1` = the flow points straight up the gradient. It is
dimensionless and bounded, so it is comparable across families, datasets and
units — this single number replaces a neural pipeline's validation-MSE gap.

### Step 6 — if no single symmetry fits, cluster the gradient field

Steps 4–5 assume one symmetry holds over the whole dataset. When it does not,
the tell is that `C` has more active directions than the physics should need.

The fix is *not* to cluster the inputs. Two samples belong to the same symmetry
regime when their **normalised defect features point the same way**, because
the local invariance algebra is the orthogonal complement of `span{c(x)}`.
Since `c` and `−c` describe the same subspace, the object to cluster is the
rank-one projector `ĉĉᵀ`; k-means on its vectorisation is a sign-invariant
spherical clustering. Then repeat steps 4–5 inside each cluster.

**How many regimes?** Take the fewest that reach the smallest within-regime
active dimension `k_pooled`. Splitting is worth it only when the response is
more reducible *inside* a regime than across all of them. Adding regimes past
that point buys no new symmetry.

This matters because the regimes are curved, overlapping slabs in input space —
k-means on `x` would not find them — but they are two tight clusters of
*directions*.

### Step 7 — select the family

All candidate families are scored **on the same gradient field**, then:

1. smallest minimum defect wins;
2. families within a factor 1.5 of the best are treated as tied, and the tie
   goes to the one with **fewer dominant directions** — the one claiming more
   symmetry.

Rule 2 is needed because on a clean dataset every family's defect bottoms out
at the surrogate's noise floor, where the ranking is meaningless. This is the
same lexicographic rule the lift search uses (`lifts.sort_key`: fewest
effective variables, then margin, then simplicity).

### Step 8 — read the answer out

* **dominant directions** = active eigenvectors. In log coordinates these are
  exponent vectors; in `x` coordinates they are the ridge directions.
* **generators** = inactive eigenvectors, weakest response first. Convert them
  back to the objects they represent before reading them: a translation or
  scaling generator is a vector, a rotation generator is the antisymmetric
  matrix `A = Σ θ_k (e_i e_jᵀ − e_j e_iᵀ)`. The finite symmetry is `exp(εξ_θ)`
  in every case — `x + εθ`, `x ⊙ exp(εθ)`, or `exp(εA)x`.
* **the invariants** follow: a generator `s` of the scaling family means
  `∏ xᵢ^{sᵢ}` is a dimensionless group the response ignores.

### Step 9 — validate

Four checks, all reported:

1. **subspace distance** between the recovered and true subspaces (synthetic
   data only — this is the check that requires an answer key);
2. **defect of the true generators**, evaluated on the recovered gradient
   field: this says how exact the known symmetry looks *through the surrogate*,
   and it is the floor the recovered generators should be at;
3. **sufficient summary**: does `y` collapse onto a curve when plotted against
   the recovered dominant coordinate?
4. **orbits**: does flowing a point along the recovered generator keep it on a
   level set of `y`? This is the picture in panel (a) of every figure.

---

## 3. The four datasets

Datasets A–C use `n = 4` inputs, `N = 600` samples, and 1 % Gaussian noise on
`y`. Each is deliberately built with the same shape — one (or two) dominant
directions, the rest generators, and one input that is completely inert — so
that the three results are directly comparable. Files:
`data/{translation,scaling,rotation}.csv`, columns `x1,x2,x3,x4,y,y_clean`,
answer key `data/ground_truth.json`.

### Dataset A — translation

```
x ~ Uniform[1, 3]⁴
t = w·x            w = (1, 2, −1, 0)/√6
y = t + 0.1 t²     + noise
```

*Known answer:* 1 dominant direction `w ∝ (1, 2, −1, 0)`; 3 generators spanning
its orthogonal complement in ℝ⁴. Note `w₄ = 0`, so `e₄` is exactly a generator —
`x4` is inert.

### Dataset B — scaling

```
x ~ Uniform[1, 3]⁴
y = x1² · x3 / x2  + noise
```

A power law. In log coordinates it is `y = exp(2 log x1 − log x2 + log x3)`,
a ridge function with `w ∝ (2, −1, 1, 0)`.

*Known answer:* 1 dominant direction `w ∝ (2, −1, 1, 0)` — the **exponent
vector** — and 3 generators. A generator `s` is any rescaling `x → x⊙exp(εs)`
with `w·s = 0`; the corresponding invariant is the dimensionless group
`∏ xᵢ^{sᵢ}`.

### Dataset C — rotation

```
x ~ Uniform[0.5, 2.5]⁴
y = √(x1² + x2²) + 0.5(x3² + x4²)  + noise
```

*Known answer:* in `u = x²/2` this is `y = H(u1+u2, u3+u4)`, so there are **two**
dominant directions there, `(1,1,0,0)` and `(0,0,1,1)`, and **two** generators:
rotation of the `(x1, x2)` plane and rotation of the `(x3, x4)` plane. In the
`so(4)` parametrisation (6 free parameters, one per coordinate pair, ordered
`(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)`) those two generators are the basis
vectors `e₁` and `e₆`, leaving 4 active directions.

### Dataset D — Ergun, two local symmetries

`N = 800` samples, log-uniform over `(u, Dp, ρ, μ)`, porosity fixed at 0.40,
1 % multiplicative noise. File `data/ergun.csv`, answer key
`data/ergun_ground_truth.json`.

```
dP/L = 150 μ (1−e)² u / (e³ Dp²)  +  1.75 ρ (1−e) u² / (e³ Dp)
       \___ viscous (Blake-Kozeny) ___/  \___ inertial (Burke-Plummer) ___/
```

Each term alone is a pure power law with its own scaling symmetry, but they are
*different* power laws:

| regime | `dP/L ∝` | exponents of `(u, Dp, ρ, μ)` |
|---|---|---|
| viscous (low Re) | `μ u / Dp²` | `( 1, −2, 0, 1)` |
| inertial (high Re) | `ρ u² / Dp` | `( 2, −1, 1, 0)` |

Because the response is a *sum*, its log-gradient is always a convex
combination of those two vectors,
`∇_{log x} log(dP/L) = f_v·w_visc + f_i·w_iner` with `f_v + f_i = 1`. Three
consequences, and all three are checked:

1. **Globally the gradient field has rank exactly 2.** A global analysis finds
   2 dominant directions and 2 generators — and those 2 generators are *exact*
   symmetries valid in both regimes. They are the dimensionless groups of the
   full Ergun equation.
2. **Inside either asymptotic regime the rank is 1.** So each regime has 3
   generators: the 2 global ones, plus one more that holds only there. That
   extra generator is the local symmetry.
3. The two local dominant directions are `w_visc` and `w_iner`.

The CSV also carries `Re_diagnostic` and `f_inertial_diagnostic` columns. These
are **never given to the method** — they are used only in step 9 to score
clusters that were found without them.

---

## 4. Results — datasets A, B, C

### 4.1 Family selection: all three correct

Minimum defect for each family on each dataset (same gradient field across a
row); the selected family is in bold.

| dataset | translation | scaling | rotation | selected | truth |
|---|---|---|---|---|---|
| A: translation | **0.0006** (1 active) | 0.0006 (3 active) | 0.0007 (3 active) | **translation** | ✔ |
| B: scaling | 0.0017 (3 active) | **0.0017** (1 active) | 0.0588 (5 active) | **scaling** | ✔ |
| C: rotation | 0.0949 | 0.1861 | **0.0019** | **rotation** | ✔ |

Dataset C separates on defect alone by a factor of 50. Datasets A and B are
ties on defect, broken by step 7's active-dimension rule.

### 4.2 How to read the figures

**(a) Orbits vs. level sets.** The dots are the 600 data points, projected onto
the `(x1, x2)` plane and coloured by the measured `y`. The blue shading and thin
blue lines are the level sets of the noise-free response on the slice
`x3 = x4 = mean` — drawn as a backdrop only. The red curves are the **orbits of
the recovered generator**: take a point, apply the recovered flow
`exp(εξ_θ)` for `ε` from `−0.9` to `+0.9`, and draw where it goes. *If the
generator is correct, the red curves lie on the blue level sets*, because
moving along a symmetry does not change `y`. That coincidence is the whole
claim of this example, drawn rather than tabulated.

The red curves are drawn from the recovered eigenvector only; the true answer
is used nowhere in panel (a) except for the backdrop shading. Note also that
the scatter colours look noisy — that is correct, not a defect. `y` also
depends on `x3`, so two points at the same `(x1, x2)` can have different `y`;
the projection cannot show that and does not pretend to.

**(b) Sufficient summary.** `y` plotted against the recovered dominant
coordinate `w·k(x)`. A ridge function collapses onto a single curve here, and
the tightness of that collapse *is* the statement "`y` depends on nothing
else". This is the classical active-subspace diagnostic.

**(c) Spectrum.** The eigenvalues of `C` on a log axis, red above the
active/generator cut and green below it. The size of the gap at the dashed line
is how confident the split is.

### 4.3 Dataset A — translation

![translation](output/translation.png)

The orbits are straight lines of slope `−w1/w2 = −0.5`, lying along the level
sets. The summary plot collapses to the quadratic `h(t) = t + 0.1t²` we built
in, and the spectrum drops six decades in one step.

```
    i     lambda_i  lam_i/lam_1  cum.energy    defect
    1  1.77702e+00   1.0000e+00    0.999996    1.0000   A
    2  4.08909e-06   2.3011e-06    0.999999    0.0015   .
    3  1.75248e-06   9.8619e-07    1.000000    0.0010   .
    4  6.86955e-07   3.8658e-07    1.000000    0.0006   .
```

| quantity | recovered | true |
|---|---|---|
| dominant directions | 1 | 1 |
| generators | 3 | 3 |
| `w` | `(+0.4084, +0.8164, −0.4083, −0.0004)` | `(+0.4082, +0.8165, −0.4082, 0)` |
| angle to true `w` | **0.024°** | — |
| subspace distance, generators | **4.2e-4** | 0 |

Defect of the three *true* generators, evaluated on the recovered gradient
field: `1.2e-3`, `1.4e-3`, `7.7e-4` — all at the noise floor, which is the
correct answer for an exact symmetry measured through a 1 %-noise surrogate.

### 4.4 Dataset B — scaling

![scaling](output/scaling.png)

The orbits are no longer straight: they are the power-law curves
`x1^{w1} x2^{w2} = const`, which is exactly what "a translation in `log x`"
looks like when you draw it in `x`. Same picture as dataset A, different ruler.
The summary plot collapses onto `exp(w·log x)`.

```
    i     lambda_i  lam_i/lam_1  cum.energy    defect
    1  2.15422e+02   1.0000e+00    0.999837    0.8791   A
    2  3.15786e-02   1.4659e-04    0.999984    0.0140   .
    3  2.92402e-03   1.3573e-05    0.999997    0.0032   .
    4  5.65190e-04   2.6236e-06    1.000000    0.0017   .
```

| quantity | recovered | true |
|---|---|---|
| dominant directions | 1 | 1 |
| generators | 3 | 3 |
| exponent vector `w` | `(+0.8169, −0.4069, +0.4087, −0.0002)` | `(+0.8165, −0.4082, +0.4082, 0)` |
| angle to true `w` | **0.084°** | — |
| subspace distance, generators | **1.5e-3** | 0 |

Rescaled to integers, the recovered `w` is `(2.00, −1.00, 1.00, 0.00)` — the
exponents of `y = x1²x3/x2`, read off an eigenvector. The inert coordinate `x4`
comes back as a generator in its own right: the recovered generator closest to
`e₄` is `(+0.015, +0.019, −0.012, +0.9996)` with defect `1.7e-3`.

Note `λ₁`'s own defect is 0.879, not 1.0. The defect of a *dominant* direction
has no reason to be 1 — it is 1 only when the flow points exactly up the
gradient everywhere.

### 4.5 Dataset C — rotation

![rotation](output/rotation.png)

The orbits close into circular arcs sitting on the circular level sets — a
rotation of the `(x1, x2)` plane, recovered as one basis vector of a
6-dimensional eigenproblem. Panel (b) is the one place the `r = 2` structure
shows: the points nearly collapse onto a line against dominant coordinate 1,
and the residual spread is explained by dominant coordinate 2 (the colour), not
by noise. Panel (c) shows why this dataset is the hardest of the three — the
four dominant eigenvalues span only two decades among themselves, so the cut
has far less contrast than the single six-decade cliff of dataset A.

```
    i     lambda_i  lam_i/lam_1  cum.energy    defect
    1  7.62353e+00   1.0000e+00    0.863808    0.3462   A
    2  6.15073e-01   8.0681e-02    0.933501    0.1395   A
    3  5.43468e-01   7.1288e-02    0.995080    0.1218   A
    4  4.18793e-02   5.4934e-03    0.999825    0.0958   A
    5  1.40721e-03   1.8459e-04    0.999985    0.0067   .
    6  1.33308e-04   1.7486e-05    1.000000    0.0019   .
```

4 active, 2 generators in the 6-dimensional `so(4)` parameter space — the truth.

**A rotation generator is a matrix, not a vector.** The eigen-decomposition
returns 6 numbers because `so(4)` is a 6-dimensional vector space, but those
numbers are the *coordinates* of the generator in the basis of coordinate-pair
rotations. The generator itself is the antisymmetric matrix

```
A = Σ_k θ_k ( e_i e_jᵀ − e_j e_iᵀ ),      (i, j) = k-th pair
```

and the symmetry it generates is the finite flow `x → exp(εA)x`. Written out,
the two recovered generators are unmistakable:

```
  A1  [coords (−0.031, −0.010, +0.014, +0.010, −0.013, −0.999)]   defect = 1.9e-3
        [ +0.000  −0.031  −0.010  +0.014]         [ 0  0  0  0]
        [ +0.031  +0.000  +0.010  −0.013]   vs    [ 0  0  0  0]
        [ +0.010  −0.010  +0.000  −0.999]         [ 0  0  0  1]
        [ −0.014  +0.013  +0.999  +0.000]         [ 0  0 −1  0]

  A2  [coords (+0.999, +0.012, −0.013, −0.015, +0.015, −0.031)]   defect = 6.7e-3
        [ +0.000  +0.999  +0.012  −0.012]         [ 0  1  0  0]
        [ −0.999  +0.000  −0.015  +0.015]   vs    [−1  0  0  0]
        [ −0.012  +0.015  +0.000  −0.031]         [ 0  0  0  0]
        [ +0.012  −0.015  +0.031  +0.000]         [ 0  0  0  0]
```

`A1` is the rotation of the `(x3, x4)` plane and `A2` the rotation of the
`(x1, x2)` plane, each to three decimal places, with every off-block entry
below 0.032. Subspace distance to the true generator plane: **3.6e-2**.

**Why it is legitimate to eigendecompose vectors when the objects are
matrices.** The Frobenius inner product on antisymmetric matrices is
proportional to the Euclidean one on the coordinates,
`⟨A, B⟩_F = 2 θ_A · θ_B`. So orthogonal eigenvectors really are orthogonal
matrices, and every subspace distance quoted here is a Frobenius distance
between spaces of matrices. Nothing is lost by working in coordinates — but the
matrix is the object, and the report prints both.

**Cross-check in `u = x²/2`.** METHOD.md §2.2 says a rotation of the `(x1, x2)`
plane is exactly the shift `u → u + ε(e₁ − e₂)`. Running the plain
`C = E[∇_u f ∇_u fᵀ]` in that coordinate instead of the `so(4)` family:

```
  lambda1 =  2.4931e+00  [−0.3184 −0.3170 −0.6315 −0.6319]  active
  lambda2 =  4.3161e-02  [−0.6479 −0.6151 +0.3205 +0.3148]  active
  lambda3 =  7.2880e-04  [−0.6918 +0.7219 +0.0043 −0.0179]  GENERATOR
  lambda4 =  3.2390e-05  [−0.0136 +0.0087 −0.7061 +0.7080]  GENERATOR
```

The two generators come out as `(1,−1,0,0)/√2` and `(0,0,1,−1)/√2` — precisely
the predicted shifts. The two active eigenvectors are *mixtures* of `(1,1,0,0)`
and `(0,0,1,1)` rather than those vectors individually, which is expected and
not an error: eigenvectors are only determined up to rotation within a
subspace, and it is the **plane** that is the invariant object. Its distance to
the true `(u1+u2, u3+u4)` plane is **2.4e-2**.

---

## 5. Results — dataset D, automatic local symmetry discovery

![ergun](output/ergun.png)

GP surrogate: leave-one-out R² = **0.99998** in log coordinates.

### 5.0 How to read this figure

Datasets A–C have orbit plots (§4.2); this one cannot, because its generators
live in a 4-dimensional log space with no informative 2-D slice. The four
panels answer four different questions instead.

**(a) Where did the split land?** The classic Ergun collapse — friction factor
against Reynolds number — with each point coloured by the cluster it was
assigned to. The two dashed asymptotes `150/Re` and `1.75` are drawn from the
known groups as a backdrop; the *colours* were found without them. The
discovered boundary lands exactly on the knee where one asymptote stops
dominating the other. That is the whole claim of the local step, in one panel.

**(b) Are the recovered exponents right?** Grouped bars, per input: grey is the
known exponent, colour is the recovered one, for each of the two regimes. The
viscous regime should read `(1, −2, 0, 1)` and the inertial one `(2, −1, 1, 0)`;
note in particular that `ρ` is absent from the viscous law and `μ` is absent
from the inertial one, and both near-zero bars come out near zero.

**(c) Why two regimes rather than one?** The eigenvalue spectra, global versus
per-regime, on one axis. The global curve has *two* large eigenvalues; both
per-regime curves have *one*, and their second eigenvalue drops by an order of
magnitude. Splitting the gradient field is what converts rank 2 into rank 1,
and that conversion is the local symmetry.

**(d) Is the split actually correct?** A histogram of the true inertial
fraction — the quantity the method never saw — coloured by discovered cluster.
Clean separation at 0.5 means the clusters are the physical regimes and not
some other partition that happened to reduce the rank.

### 5.1 The global analysis, and why it is not wrong

```
    i     lambda_i  lam_i/lam_1  cum.energy    defect
    1  4.98766e+00   1.0000e+00    0.874916    0.9354   A
    2  7.12576e-01   1.4287e-01    0.999914    0.3535   A
    3  2.68442e-04   5.3821e-05    0.999961    0.0069   .
    4  2.23823e-04   4.4875e-05    1.000000    0.0063   .
```

Rank 2, exactly as predicted: 2 dominant directions and 2 generators, with a
subspace distance of **1.4e-3** to the truth on both. Those two generators are
*exact* symmetries of the full Ergun equation, defects `6.3e-3` and `6.9e-3` —
the dimensionless groups that survive in both regimes.

What a global analysis cannot say is that each individual *sample* sits on a
line inside that plane, and which line. That is what step 6 recovers.

### 5.2 Choosing the number of regimes

```
  #regimes                sizes     k per regime  pooled inact.E  max subsp.dist
  -----------------------------------------------------------------------------
         1                [800]              [2]       8.635e-05           0.000
         2           [371, 429]           [1, 1]       1.262e-02           0.641
         3      [367, 310, 123]        [1, 1, 1]       4.361e-03           0.698
         4   [97, 357, 272, 74] (regime too small)
```

With one regime the response needs 2 active directions; forcing it down to one
leaves **12.5 %** of the spectrum unexplained. Splitting into two regimes brings
that leftover down to **1.4 %** and **1.2 %**, so one direction per regime is
now enough. Splitting further does not reduce `k_pooled` again, so the extra
regimes buy no new symmetry. **Selected: 2 regimes**, and the subspace distance
of 0.641 between them says the two local symmetries really are different
objects, not one symmetry found twice.

### 5.3 The threshold, and why this dataset needs a looser one

Datasets A–C use `energy_tol = 1e-3`; this one uses `0.05`, and the reason is
physical rather than numerical. Ergun is a *sum*, so between the asymptotes
there is a crossover band where the local exponent vector is genuinely
intermediate. A hard 2-way split has to put those samples in one cluster or the
other, which leaves each cluster with a real ~1 % second eigenvalue. At `1e-3`
that residue is called "active", every regime looks 2-dimensional, and **the
local symmetry is never declared at all.**

The window that works is wide and visible in the numbers above: the leftover is
12.5 % with one regime and ~1.3 % with two, so any threshold between those two
values gives the same answer. `0.05` sits in the middle of it. This is the
clearest illustration in the whole example of the caveat from step 5 — the
threshold does real work, and it should be chosen by looking at the gap rather
than by inheriting a default.

### 5.4 The two recovered regimes

| | regime 0 → **inertial** | regime 1 → **viscous** |
|---|---|---|
| samples | 371 | 429 |
| Re range | 7.5e+1 … 4.3e+6 | 1.4e−3 … 8.5e+1 |
| mean inertial fraction | 0.892 | 0.088 |
| dominant directions | 1 (true 1) | 1 (true 1) |
| generators | 3 (true 3) | 3 (true 3) |
| recovered exponents | `(+1.96, −1.14, +0.92, +0.11)` | `(+1.11, −1.97, +0.09, +0.94)` |
| true exponents | `( 2, −1, 1, 0)` | `( 1, −2, 0, 1)` |
| angle to truth | **4.61°** | **3.72°** |
| subspace distance, generators | 8.0e-2 | 6.5e-2 |

Each regime went from 2 generators (global) to **3** — the two global
dimensionless groups plus one more that holds only in that regime. That extra
generator *is* the local symmetry, and it is different in the two regimes.

### 5.5 Validation against the label the method never saw

Assigning each sample to viscous or inertial by which term actually carries
more of the pressure drop:

* agreement with the discovered clusters, **all samples: 99.5 %**
* agreement, samples away from the crossover (`f_inertial` outside `[0.1, 0.9]`,
  562 of 800): **100.0 %**

Panel (d) of the figure shows this directly: the two histograms separate at
`f_inertial = 0.5` with essentially no overlap. Panel (a) shows the same split
drawn on the classic Ergun collapse — friction factor against Reynolds number —
and the discovered boundary lands exactly on the knee where `150/Re` stops
dominating `1.75`.

### 5.6 What three regimes does, which is also not wrong

```
    regime 0: n =  367, exponents = (+1.05, −1.98, +0.04, +0.98),  1.69° from viscous
    regime 1: n =  310, exponents = (+1.98, −1.07, +0.96, +0.05),  2.25° from inertial
    regime 2: n =  123, exponents = (+1.63, −1.65, +0.54, +0.56), 23.62° from viscous
```

The third regime is the **crossover**, isolated on its own, with a genuine
intermediate power law. With it quarantined, the other two land at 1.7° and
2.3° from the asymptotic exponents — noticeably better than the 4.6° and 3.7°
the 2-way split achieves. That gap is the cost of forcing a hard two-way cut on
a response that really does interpolate. The `k_pooled` rule prefers 2 regimes
because 3 does not reduce the within-regime dimension any further, which is the
right call for "how many symmetries are there"; 3 is the better call if what
you want is the sharpest estimate of each asymptote.

---

## 6. Summary

| dataset | true symmetry | selected | dominant dirs (found / true) | generators (found / true) | subspace error |
|---|---|---|---|---|---|
| A | translation | translation ✔ | 1 / 1 | 3 / 3 | 4.2e-4 |
| B | scaling | scaling ✔ | 1 / 1 | 3 / 3 | 1.5e-3 |
| C | rotation | rotation ✔ | 4 / 4 | 2 / 2 | 3.6e-2 |
| D | global (2 regimes) | — | 2 / 2 | 2 / 2 | 1.4e-3 |
| D | local, viscous | scaling ✔ | 1 / 1 | 3 / 3 | 6.5e-2 |
| D | local, inertial | scaling ✔ | 1 / 1 | 3 / 3 | 8.0e-2 |

**Can we discover the dominant directions and generators from data? Yes** — to
0.02–0.08° on the dominant direction and 4e-4 to 4e-2 subspace error on the
generators for a single global symmetry, from 600 noisy samples. And when the
symmetry is *local* and the regimes are unlabelled, both regimes and both
symmetries come out of the same machinery, agreeing with the hidden ground
truth on 99.5 % of samples and 100 % away from the crossover — with one
eigen-decomposition per family per cluster, and no network anywhere.

### Honest caveats

1. **The rotation case is the least accurate of A–C**, by two decades. Its
   dominant directions span 4 dimensions with eigenvalues within one decade of
   each other, so the active/inactive cut has far less contrast than the
   six-decade gap in dataset A. This is a property of the problem, not of the
   method.
2. **`energy_tol` is a real choice**, and dataset D is the proof: at the value
   that is right for A–C, the local symmetry is never declared at all (§5.3).
   The spectrum is printed precisely so this can be checked by eye rather than
   taken on faith.
3. **Defect ties are the normal case on clean data.** When every family reaches
   the surrogate's noise floor, the defect ranking carries no information and
   the active-dimension tie-break does all the work.
4. **The lift here was selected from a catalogue of three**, not discovered.
   Discovering `k` from data is `active_subspace.discover` and the lift search
   in `Examples/keyhole_symmetry`; this example is the controlled check that
   the machinery downstream of `k` is correct.
5. **The surrogate is the weak link, not the algebra.** Every error reported
   above is dominated by the GP's gradient error, which is why `r2_cv` is
   printed first on every dataset.
6. **sklearn warns that the `x4` length-scale hits its upper bound.** That is
   the *correct* fit, not a failure: `x4` is inert, so the GP wants an infinite
   length-scale along it. The warning is worth reading rather than silencing —
   a length-scale pinned at the *lower* bound would mean the opposite and much
   worse thing (the GP interpolating noise, leaving `∇μ ≈ 0` at the samples).
7. **Panel (a) of the A–C figures is a 2-D slice of a 4-D problem.** The orbits
   shown keep `x3` and `x4` fixed, chosen from the recovered generator subspace
   precisely so they can be drawn. The other generators — including the inert
   direction `e₄` — are invisible in that projection and are only checked
   numerically.
8. **Dataset D's sampling box is wider than any real fluid**, spanning
   gas-to-liquid in `ρ` and `μ` together. That is deliberate: every exponent
   needs something to be estimated from, and both asymptotic regimes need to be
   populated. On a narrow real dataset the poorly-excited exponents would come
   back with wide intervals — which is what `identifiability` and the
   `excitation` column exist to report.
9. **A hard clustering is the wrong model for a smooth crossover.** §5.6 shows
   the cost. A soft/mixture assignment, or reporting the local exponent vector
   as a continuous function of `Re`, would describe this response better; the
   two-regime answer is the right *symmetry* answer, not the best *fit*.

## Files

| file | what it is |
|---|---|
| `datasets.py` | datasets A–C: response functions, sampling, answer key |
| `run_analysis.py` | read CSV → GP gradients → `C` → eigen-decomposition → compare |
| `make_plots.py` | figures for A–C: orbits, sufficient summary, spectrum |
| `ergun.py` | dataset D: the Ergun equation, sampling, answer key |
| `run_ergun.py` | global analysis, gradient clustering, per-regime analysis, figure |
| `data/*.csv` | the datasets |
| `data/ground_truth.json`, `data/ergun_ground_truth.json` | answer keys, machine-readable |
| `output/report.txt`, `output/ergun_report.txt` | full run output |
| `output/*.png` | one figure per dataset |

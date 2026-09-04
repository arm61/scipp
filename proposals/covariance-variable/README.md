# Proposal: a `scipp.Variable` subclass carrying a full covariance matrix

This directory contains a design study and a **working reference prototype** for a
subclass of `scipp.Variable` that stores and propagates a full covariance matrix
instead of only the diagonal (`variances`).

* `covariance_variable.py` — the prototype (`CovarianceVariable`)
* `test_covariance_variable.py` — 64 tests, checked against an independent
  numerically-differentiated `J C Jᵀ` reference

```
pip install scipp pytest
cd proposals/covariance-variable && python -m pytest test_covariance_variable.py
```

The prototype is deliberately kept outside `src/scipp/`: it is a proposal, not an
API commitment.

---

## 1. What the codebase constrains

`Variable` is a C++ class exposed through pybind11 in `lib/python/variable.cpp`.
Three properties of that binding decide the whole design. Each was verified
against scipp 26.8.0 rather than assumed.

| Property | Where | Consequence |
|---|---|---|
| `py::class_<Variable> variable(m, "Variable", py::dynamic_attr(), ...)` | `lib/python/variable.cpp:78` | Not `py::is_final`, so **Python subclassing works**; `dynamic_attr` gives instances a `__dict__`, so **extra Python state can be attached**. |
| `bind_init` uses a *factory* `py::init([](...) { ... return var; })` | `lib/python/variable_init.cpp:273` | The class has no alias type, so factory-init is compatible with Python subclasses. `super().__init__(dims=..., values=..., variances=..., unit=...)` constructs the C++ object normally. |
| Every bound method and free function returns `Variable` **by value** | throughout | pybind11 casts the result to the *registered* Python type. **Every C++ operation returns a plain `Variable` and drops the subclass.** |

The third row is the crux. Empirically, on an instance of a trivial subclass:

```python
class Sub(sc.Variable): pass
s = Sub(dims=['x'], values=[1., 2., 3.], variances=[.1, .2, .3])

type(s + s)          # Variable   (not Sub)
type(s['x', 0])      # Variable
type(s.copy())       # Variable
type(sc.sum(s))      # Variable
type(sc.DataArray(data=s).data)  # Variable
```

Two further findings shape the design:

* **Python-level operator overrides do win.** Defining `__add__`/`__radd__` etc.
  on the subclass takes precedence over the pybind11 slots, *including* Python's
  reflected-operand priority for subclasses, so `plain_variable + cov_variable`
  correctly dispatches to `CovarianceVariable.__radd__`. Overriding a
  `def_property` such as `variances` also works through the normal MRO.
* **`Variable::is_same` is not exposed to Python.** Scipp's own correlation
  special-cases (`correlated()` in `lib/variable/arithmetic.cpp:31`, which makes
  `x + x` yield `4·var` rather than `2·var`) rely on it. A Python subclass must
  reimplement aliasing detection, e.g. via `np.shares_memory` on `.values`.

### Why scipp wants this

This is not a speculative feature. ADR 0015
(`docs/development/adr/0015-disable-support-for-broadcasting-variances.rst`)
disables broadcasting of variances outright, because scipp cannot represent the
correlations a broadcast introduces:

```python
data / norm   # VariancesError: Cannot broadcast object with variances as this
              # would introduce unhandled correlations.
```

The ADR cites *Systematic underestimation of uncertainties by widespread
neutron-scattering data-reduction software* (doi:10.3233/JNR-220049). A full
covariance matrix is precisely the missing representation: with it, that
expression becomes well defined rather than forbidden. **The subclass is best
motivated as lifting the ADR 0015 restriction, not as a general-purpose
enhancement.**

---

## 2. Three possible architectures

| | A. Subclass `Variable` | B. Wrapper (composition) | C. C++ `VariableConcept` |
|---|---|---|---|
| `isinstance(x, sc.Variable)` | ✅ true | ❌ false | ✅ true |
| Existing scipp functions accept it | ✅ | ❌ needs shims | ✅ |
| Survives C++ operations | ❌ silently degrades | ✅ (nothing to lose) | ✅ |
| Storable in `DataArray`/`Coords` | ❌ stripped | ❌ | ✅ |
| Effort | days | days | months |

Option B is what scipp already does for `DataGroup` (`src/scipp/core/data_group.py`)
and is the safest, but it is not what was asked for and it loses the
duck-typing that makes scipp code compose. Option C is the only complete
answer, and it is a large change to `lib/variable/`.

**Recommendation: Option A, built so that its one real weakness — silent
degradation — is neutralised by an invariant plus explicit failure.** That is
what the prototype implements.

---

## 3. The recommended design

### 3.1 The invariant

```
self.variances == diag(self.covariance)
```

The subclass keeps the *inherited C++ variances buffer* populated with the
diagonal at all times. This is what makes Option A safe rather than dangerous:

* When an operation falls through to C++ and returns a plain `Variable`, the
  correlations are lost but the **marginal variances are still correct**. The
  result is a valid, conservative scipp object — not corrupt data.
* Anything that reads `.variances`, plots error bars, or writes to HDF5 keeps
  working with no changes.

### 3.2 Storage layout: mirrored dimensions

Scipp forbids repeated dimension labels (`DimensionError: Duplicate dimension`),
so the two index axes of the matrix need distinct names. The prototype gives the
second axis a primed label:

```
variable    dims ('x', 'y')          shape (2, 3)
covariance  dims ('x', 'y', "x'", "y'")   shape (2, 3, 2, 3)   unit = unit**2
```

The covariance is itself an ordinary `sc.Variable`, so it inherits units,
slicing and dtype handling for free — the unit is `unit**2`, and scipp's unit
algebra then validates the propagation automatically.

Keeping labelled axes (rather than flattening to a single `(N, N)` index) is
what makes the algebra expressible in scipp itself, and lets slicing and
transposition reuse scipp's own machinery.

### 3.3 The algebra: one primitive does almost everything

Every elementwise operation has a *diagonal* Jacobian, so `J C Jᵀ` collapses to
an outer-product scaling:

```python
@staticmethod
def _sandwich(jacobian, cov):
    rename = dict(zip(jacobian.dims, _mirror(jacobian.dims)))
    return jacobian * cov * jacobian.rename_dims(rename)
```

Everything else is a table of partial derivatives:

| operation | ∂/∂a | ∂/∂b |
|---|---|---|
| `a + b` | `1` | `1` |
| `a - b` | `1` | `-1` |
| `a * b` | `b` | `a` |
| `a / b` | `1/b` | `-a/b²` |
| `a ** n` | `n·aⁿ⁻¹` | — |
| `sqrt`, `exp`, `log`, `sin`, `cos` | chain rule | — |

Two consequences fall out of `_sandwich` for free, and they are the reason to
prefer this formulation:

* **Broadcasting is handled automatically and correctly.** If the Jacobian
  carries dimensions the covariance lacks, the product gains those dimensions on
  *both* index axes — which is exactly the correlation structure a broadcast
  introduces. `data / scalar_with_uncertainty` produces a fully populated
  off-diagonal block instead of raising.
* **Unit correctness is checked by scipp.** `(∂f/∂a)² · unit(a)²` must equal
  `unit(f)²` or scipp raises. This caught a real bug during development: scipp's
  `cos()` returns a dimensionless value, but `d(sin x)/dx` must carry `rad⁻¹`,
  otherwise the `rad²` of the covariance survives into a dimensionless result.

Reductions are where the payoff shows:

```python
def sum(self, dim):
    return wrap(base.sum(dim), cov.sum(dim).sum(dim + "'"))   # both index axes
```

`mean` divides the covariance by `n²`. Slicing selects on both axes; `transpose`
reorders both halves; `concat` builds a block-diagonal matrix.

### 3.4 Aliasing

Since `is_same` is unavailable in Python, the prototype detects aliasing with
`np.shares_memory` on `.values`. When both operands are the same buffer, the
partial derivatives are *summed* before the sandwich (the total derivative),
which reproduces scipp's own `x + x → 4·var` special case and generalises it to
the full matrix — `x - x` comes out exactly zero.

### 3.5 Explicit failure instead of silent degradation

Operations that cannot propagate a covariance (`flatten`, `fold`, `hist`,
`cumsum`, `max`, `median`, …) are overridden to raise `CovarianceError` pointing
at `.to_variable()`. This follows the ADR 0015 reasoning directly: the ADR chose
an exception over a warning because *"past experience shows that warnings tend to
get ignored"*. The same argument applies here — a silently dropped correlation
is the exact failure mode the ADR exists to prevent.

`.to_variable()` is the documented, explicit opt-out, and it is lossy only in
the correlations, never in the marginals.

### 3.6 Worked example

```python
data = covariance_array(dims=['x'], values=[10., 20.],
                        covariance=np.diag([10., 20.]), unit='counts')
norm = covariance_scalar(2.0, variance=0.04)

result = data / norm          # plain scipp: VariancesError

result.covariance             # off-diagonal is non-zero: the shared
                              # denominator correlates the two outputs
result.sum('x').variance      # larger than the naive uncorrelated total
```

---

## 4. Known limits of Option A

These are inherent to subclassing a C++ type, not defects of the prototype. Both
are covered by tests that assert the current behaviour, so they cannot regress
unnoticed.

1. **C++ free functions strip the subclass.** `sc.sum(cv, 'x')` returns a plain
   `Variable` with the sum of the *diagonal* — correct as a marginal, but an
   underestimate when the summands are correlated. The method form
   (`cv.sum('x')`) is correct; only the free-function form degrades.

   This bites hardest inside a `DataGroup`, where overriding methods cannot help:
   `call_func` (`src/scipp/core/_cpp_wrapper_util.py:25`) hands the **raw C++
   function** to `data_group_nary`, so the per-item call goes straight to C++ and
   never reaches the subclass.

   `install_dispatch()` is the prototype's opt-in answer: it patches the listed
   functions in the `scipp` namespace so they route to the covariance-aware
   implementation, or raise when there is none, for both direct and
   `DataGroup`-wrapped arguments. `uninstall_dispatch()` reverses it.

   ```python
   sc.sum(dg, 'x')['a']            # plain Variable — correlations lost
   install_dispatch()
   sc.sum(dg, 'x')['a']            # CovarianceVariable, off-diagonals included
   sc.max(dg)                      # CovarianceError instead of a silent strip
   ```

   It is a stand-in, not the fix: functions outside its two lists are untouched
   and still degrade silently. The real fix is a `__scipp_dispatch__` protocol
   honoured by `src/scipp/core/reduction.py` and friends — see §6.2.
2. **C++ containers strip the subclass.** `sc.DataArray(data=cv).data` is a
   plain `Variable`. `DataArray` holds `std::shared_ptr<Variable> m_data`
   (`lib/dataset/include/scipp/dataset/data_array.h:105`), and the binding's
   setter takes `const Variable &` while its getter returns by value under
   `return_value_policy::copy` (`lib/python/bind_data_array.h:399-402`) — so the
   covariance is dropped on *insertion*, and every read builds a fresh plain
   wrapper (`ds['a'].data is ds['a'].data` is `False`). There is no Python-side
   fix; covariance data cannot round-trip through `DataArray`, `Dataset` or
   `Coords`.

   The strip happens on **insertion**, so wrapping first and grouping later
   does not help — this looks safe but is not:

   ```python
   sc.DataGroup({'real': sc.DataArray(data=cv, coords={'omega': freq})})
   #                    ^^^^^^^^^^^^^^^^^^^^^ covariance already gone here
   ```

   Nor can the covariance ride along as a coord: a coord carrying the mirror
   dimension is rejected with `DimensionError`. There is no way to keep a
   covariance inside a `DataArray`, as data or as metadata. The
   `DataArray`-shaped alternative is a nested `DataGroup`, which slices as a
   unit and preserves everything:

   ```python
   sc.DataGroup({'real': sc.DataGroup({'data': cv, 'omega': freq})})
   ```

   **`sc.DataGroup` is the exception and the recommended container**: it is a
   pure-Python dict, so construction, `copy`, `deepcopy`, nesting, slicing and
   arithmetic all preserve the subclass, and its methods dispatch to the
   *item's* method (`operator.methodcaller` in `src/scipp/core/data_group.py`),
   so `dg.sum('x')` uses this class's override. Only the free-function form
   (`sc.sum(dg)`, via `data_group_nary`) degrades, per limitation (1).

   That same method dispatch is why the unsupported-operation stubs must be
   *generated* from `Variable`'s API rather than listed by hand: a hand-written
   denylist omitted `squeeze`, `astype`, `round`, `rename_dims`, `all` and
   `any`, and `dg.squeeze()` then silently dropped the covariance.

Cost: memory and time are **O(N²)** in the number of elements. A 1000-element
variable needs an 8 MB covariance; 10⁶ elements is infeasible. Practical use is
therefore limited to small variables — fit parameters, normalisation
monitors, binned summaries — which is also where correlations matter most. If
the idea graduates, the representation should become pluggable: dense,
block-diagonal, low-rank (`J Jᵀ`), or diagonal-plus-low-rank.

Also unsupported by the prototype, and straightforward to add: binned/event data,
non-float dtypes, and `DataArray`-valued operands.

---

## 5. Beyond a single variable

A per-object covariance tracks correlations **between the elements of one
variable**. It does *not* track correlations **between two different variables**.
After

```python
b = f(a)
c = g(a)
b - c        # treated as independent: wrong
```

`b` and `c` are both derived from `a` and are strongly correlated, but neither
object records that. The prototype's aliasing check only catches the case where
the same buffer appears literally on both sides.

The general fix is to change what is stored. Instead of a covariance, store a
**Jacobian with respect to a set of independent sources**:

```
value:   x            shape (N,)
jac:     J            shape (N, M)   w.r.t. M independent source elements
covariance = J · C_src · Jᵀ          (derived, computed on demand)
```

Every operation composes Jacobians; correlations between *any* two derived
quantities are then recoverable, and `cov(b, c) = J_b C_src J_cᵀ` is available
for free. This is the representation used by the `uncertainties` package, and it
subsumes the covariance design: the covariance becomes a derived property rather
than stored state. It is also cheaper when `M ≪ N`.

The `_sandwich` API generalises to it directly, so the prototype is a reasonable
stepping stone rather than a dead end. I would recommend prototyping this second
tier before committing to any C++ work.

---

## 6. If this were to become a first-class scipp feature

Ordered by cost:

1. **Keep it a separate package layered on scipp** (e.g. `scipp-covariance`).
   No changes to scipp; accept the two escape hatches above. This is the right
   first step.
2. **Add a Python dispatch hook.** Have the Python wrappers in
   `src/scipp/core/{reduction,math,shape}.py` check for a `__scipp_dispatch__`
   attribute before calling into C++, mirroring NumPy's `__array_function__`.
   Small, additive, and it closes limitation (1). Note `Variable.__array_ufunc__`
   is already set to `None` in `src/scipp/core/__init__.py:95`, with a comment
   contemplating exactly this kind of protocol.
3. **Represent the covariance in C++.** A `VariableConcept` variant carrying a
   covariance buffer, plumbed through `transform`. This is the only route to
   storing covariance-carrying data in a `DataArray`, and the only route to
   relaxing ADR 0015 in the core. Large, and it should not be attempted before
   the question in §5 is settled — the answer determines what needs storing.

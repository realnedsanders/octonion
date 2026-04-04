# Octonionic Algebra

## The four normed division algebras

There are exactly four number systems where multiplication preserves norms ($|ab| = |a||b|$) and every non-zero element has an inverse. Each is built by doubling the previous one (the Cayley-Dickson construction):

| Algebra | Dim | Lost property |
|---------|-----|---------------|
| Reals $\mathbb{R}$ | 1 | -- |
| Complex $\mathbb{C}$ | 2 | Ordering |
| Quaternions $\mathbb{H}$ | 4 | Commutativity |
| **Octonions $\mathbb{O}$** | **8** | **Associativity** |

Hurwitz's theorem (1898) proves no further normed division algebra exists.

## Non-associativity

The defining feature of octonions is that $(ab)c \neq a(bc)$ in general. The **associator** measures this:

$$[a, b, c] = (ab)c - a(bc)$$

Key properties:

- **Totally antisymmetric**: swapping any two arguments flips the sign
- **Alternativity**: $[a, a, b] = [a, b, b] = 0$ (vanishes when any two arguments are equal)
- **Flexibility**: $[a, b, a] = 0$
- **Norm bound**: $\|[a,b,c]\| \leq 2\|a\|\|b\|\|c\|$

Despite non-associativity, octonions satisfy the **Moufang identities**, which constrain how parenthesization affects results.

## The Fano plane

The multiplication table of the 7 imaginary octonion units is encoded by the Fano plane $\mathrm{PG}(2,2)$ -- a finite projective plane with 7 points and 7 lines. Each line $(e_i, e_j, e_k)$ satisfies $e_i e_j = e_k$ (with cyclic permutations positive and anti-cyclic negative).

The 7 lines define 7 quaternionic subalgebras $\mathcal{S}_\ell = \mathrm{span}\{1, e_i, e_j, e_k\}$. Within each subalgebra, the algebra is associative.

## API

::: octonion.Octonion

::: octonion.associator

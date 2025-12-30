# Higher-Order Functions & Nested Procedures

## Overview

This compiler extends the BX language with **higher-order function parameters** and **nested procedure definitions**, while preserving **lexical scoping** using **static links**.  
The implementation follows the project specification closely and avoids full closures by representing function values as **fat pointers**.

Compilation pipeline: `BX source → AST → Type Checking & Capture Analysis → TAC → x86-64 Assembly`

---

## Design Choices

### 1. Semantic Analysis & Capture Detection

- **Unique Variable IDs (VIDs)**  
  Every variable and parameter is assigned a unique integer `vid`.  
  This avoids ambiguity under shadowing and allows precise identification of captured variables.

- **Capture Sets**  
  During type checking (`check_program`), the compiler computes the set of variables captured by each nested procedure.  
  If a variable is accessed from an outer lexical scope, its VID is added to `ProcDecl.captures`.

- **Scope Management**  
  A stack of function environments (`fun_env_stack`) is used to:
  - enforce block-level scoping of nested `def`s,
  - prevent redeclaration in the same scope,
  - disallow recursion among inner procedures (as required by the specification).

- **Specification Constraints**  
  Function values are restricted to **parameters only**.  
  Functions are neither returned nor stored in variables, which enables a static-link–based implementation without full closures.

---

## Intermediate Representation (TAC)

### Fat Pointers

- Function values are represented as **fat pointers**:
- Fat pointers are always **16 bytes** and stored on the stack.
- They are created only when a function is passed as a parameter.

### TAC Extensions

Minimal extensions were added: `(code pointer, static link)`

- **`TacMakeClosure`**  
Constructs a fat pointer by pairing a procedure label with the appropriate static link.

- **`TacCall`**  
Supports both direct calls (known labels) and indirect calls via fat pointers, with explicit static-link passing.

- **`TacGetVar` / `TacSetVar`**  
Access variables using a static-link hop count, allowing nested procedures to reach captured variables.

All higher-order behavior is handled during lowering; the TAC itself remains simple.

---

## Backend & Calling Convention (x86-64)

### Static Links

- Each activation record stores a **static link** to its lexical parent.
- Static links are passed explicitly at every call.
- The helper `_walk_static_link` generates assembly to traverse the static-link chain.

### Stack Layout

The compiler implements the stack layout required by the specification:

- Arguments 1–6 are passed in registers.
- Arguments beyond 6 are pushed in reverse order.
- A **static link** is pushed explicitly.
- A **padding word** is pushed to preserve 16-byte stack alignment.

In the callee:
- Padding is at `16(%rbp)`
- Static link is at `24(%rbp)`

### Variable Storage

- **Captured variables** are always assigned stack slots (indexed by VID) so they are reachable via static links.
- **Global variables** are allocated in the `.data` section and accessed using RIP-relative addressing.
- Local, captured, and global variables are unified through the same `TacGetVar` / `TacSetVar` mechanism.

### How to Run

```bash
python3 bxc.py <source_file.bx>
```
This produces `<source_file.s>`
---



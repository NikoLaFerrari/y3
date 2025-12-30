# Higher-Order Functions & Nested Procedures

## Design Choices

### 1. Semantic Analysis & Capture Detection
* **Unique VIDs**: To handle variable shadowing and nested scopes correctly, every declared variable is assigned a unique integer `vid`. The type checker resolves names to these VIDs immediately.
* **Capture Sets**: During type checking (`check_program`), the compiler computes the set of captured variables for each nested procedure. If a variable is accessed from an outer scope, its VID is added to the procedure's `captures` set.
* **Scope Management**: The type checker uses a stack of environments (`fun_env_stack`) to correctly enforce the rule that inner functions cannot be redeclared in the same scope.
* Function values are restricted to parameters only. Functions are neither returned nor stored in variables, enabling a static-link–based implementation without full closures.


### 2. Intermediate Representation (TAC)
* **Fat Pointers**: Function values are treated as "Fat Pointers" (Code Pointer + Static Link).
* **`TacMakeClosure`**: A specific TAC instruction was added to construct fat pointers. It allocates 16 bytes on the stack to store the procedure's label and the current frame pointer (static link).
* **`TacCall` Extension**: The call instruction was updated to handle both direct calls (labels) and indirect calls (fat pointers). It supports passing an explicit static link.

### 3. Backend & Calling Convention (x86-64)
* **Stack Layout**: The compiler implements the specific stack layout required by the specification (Figure 1 in the PDF).
    * **Arguments > 6**: Pushed in reverse order.
    * **Static Link**: Pushed at `24(%rbp)` (relative to callee).
    * **Padding**: A dummy 8-byte value is pushed at `16(%rbp)` to maintain 16-byte alignment.
* **Static Link Chaining**: The `_walk_static_link` function generates assembly to traverse the chain of static links (`movq 24(%rbp), %reg`) to reach the correct stack frame for captured variables.
* **Variable Storage**: Captured variables are forced into stack slots (identified by their VID) to ensure they are addressable via static links, even if they would normally be register-allocated.
* **Global Variables**: Globals are allocated in the `.data` section and accessed via RIP-relative addressing. They are integrated into the same `TacGetVar` / `TacSetVar` mechanism using a special global access mode.

## Implementation Guide

* **`bxc.py`**: The main driver and implementation file.
    * **`check_program`**: Contains the type checker and capture analysis logic.
    * **`TacGenerator`**: Lowers AST to TAC, handling closure creation (`TacMakeClosure`) and identifying global vs. local variable access.
    * **`AsmGen`**: The backend.
        * `gen_proc`: Calculates stack offsets and ensures space is reserved for captured VIDs.
        * `TacCall` (in `gen_proc`): Implements the critical push order: Arguments -> Static Link -> Padding.

* **Global Variables**: Globals are allocated in the `.data` section and accessed via RIP-relative addressing. They are integrated into the same `TacGetVar` / `TacSetVar` mechanism using a special global access mode.

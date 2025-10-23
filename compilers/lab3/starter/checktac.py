#!/usr/bin/env python3
import sys
import json
import argparse
from typing import Any, Dict, List, Optional

INFIX = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "shl": "<<",
    "shr": ">>",
    "and": "&",
    "or":  "|",
    "xor": "^",
    # comparisons
    "cmpeq": "==",
    "cmpne": "!=",
    "cmplt": "<",
    "cmple": "<=",
    "cmpgt": ">",
    "cmpge": ">=",
    # logical (if ever materialized)
    "land": "&&",
    "lor":  "||",
}

PREFIX = {
    "neg": "-",     # arithmetic unary minus
    "not": "~",     # bitwise not
    "lnot": "!",    # boolean not
}

# opcodes that should be printed as raw instructions
CONTROL = {
    "label",
    "br",
    "br_if_true",
    "br_if_false",
    "print",
}

def fmt_val(v: Any) -> str:
    """Format an operand: temps, labels, constants, special registers."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if v == "%zero":
            return "0"
        return v
    return repr(v)

def pretty_instr(ins: Dict[str, Any]) -> str:
    op = ins.get("opcode")
    args = ins.get("args", [])
    res  = ins.get("result")

    # Control flow / I/O
    if op in CONTROL:
        if op == "label":
            # args: [label]
            return f"{fmt_val(args[0])}:"
        if op == "br":
            # args: [label]
            return f"    br {fmt_val(args[0])}"
        if op == "br_if_true":
            # args: [cond, label]
            return f"    br_if_true {fmt_val(args[0])}, {fmt_val(args[1])}"
        if op == "br_if_false":
            # args: [cond, label]
            return f"    br_if_false {fmt_val(args[0])}, {fmt_val(args[1])}"
        if op == "print":
            # args: [value]
            return f"    print {fmt_val(args[0])}"
        # Fallback
        return f"    {op} " + ", ".join(fmt_val(a) for a in args)

    # Moves / const
    if op == "const":
        # args: [imm]
        return f"    {res} = {fmt_val(args[0])}"
    if op == "copy":
        # args: [src]
        return f"    {res} = {fmt_val(args[0])}"

    # Unary operators (prefix)
    if op in PREFIX:
        # args: [src]
        return f"    {res} = {PREFIX[op]}{fmt_val(args[0])}"

    # Binary operators (infix)
    if op in INFIX:
        # args: [lhs, rhs]
        a, b = args
        return f"    {res} = {fmt_val(a)} {INFIX[op]} {fmt_val(b)}"

    # Unknown — print raw for debugging
    return "    " + " ".join([
        str(op),
        *(fmt_val(a) for a in args),
        f"-> {res}" if res is not None else ""
    ]).rstrip()

def pretty_proc(proc: Dict[str, Any], show_header: bool = True) -> str:
    name = proc.get("proc", "@main")
    body = proc.get("body", [])
    lines: List[str] = []
    if show_header:
        lines.append(f"proc {name} {{")
    for ins in body:
        lines.append(pretty_instr(ins))
    if show_header:
        lines.append("}")
    return "\n".join(lines)

def pretty_tac(obj: Any) -> str:
    """
    Accepts either:
      - a list of procs: [{"proc": "@main", "body": [...]}, ...]
      - a single proc dict
    """
    if isinstance(obj, dict) and "body" in obj:
        return pretty_proc(obj)
    if isinstance(obj, list):
        parts = []
        for i, p in enumerate(obj):
            parts.append(pretty_proc(p, show_header=True))
            if i != len(obj) - 1:
                parts.append("")  # blank line between procs
        return "\n".join(parts)
    raise ValueError("Unexpected TAC JSON structure. Expect a proc dict or a list of procs.")

def main():
    ap = argparse.ArgumentParser(description="Pretty-print TAC JSON to readable code")
    ap.add_argument("path", help="Path to TAC JSON file, or '-' for stdin")
    args = ap.parse_args()

    data: Any
    if args.path == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(pretty_tac(data))

if __name__ == "__main__":
    main()


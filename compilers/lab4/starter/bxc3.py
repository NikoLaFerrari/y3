#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import argparse
import dataclasses as dc
from typing import List, Optional, Dict, Tuple
from ply import lex, yacc

# -----------------------------------------------------------------------------
# Debug / Error helpers
# -----------------------------------------------------------------------------
DEBUG = False


def dbg(msg: str):
    if DEBUG:
        print(f"[Debug] {msg}")


def err(where: str, msg: str):
    print(f"[Error][{where}] {msg}")


# =============================================================================
# LEXER
# =============================================================================

# Lab 4: procedures + types + globals.
# - 'main' and 'print' are *not* keywords anymore: they are just identifiers.
# - new keywords: bool, void, return.

reserved = {
    'def': 'DEF',
    'var': 'VAR',
    'int': 'INT',
    'bool': 'BOOL',
    'void': 'VOID',
    'true': 'TRUE',
    'false': 'FALSE',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'return': 'RETURN',
}

tokens = (
    # identifiers / literals
    'IDENT', 'NUMBER',

    # punctuation
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'EQUAL', 'COMMA',

    # arithmetic / bitwise
    'PLUS', 'MINUS', 'STAR', 'SLASH', 'MOD',
    'BAND', 'BOR', 'BXOR',
    'LSHIFT', 'RSHIFT',
    'BNOT',

    # logical & comparisons
    'LNOT', 'LAND', 'LOR',
    'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
) + tuple(reserved.values())

t_PLUS = r'\+'
t_MINUS = r'-'
t_STAR = r'\*'
t_SLASH = r'/'
t_MOD = r'%'
t_BAND = r'&'
t_BOR = r'\|'
t_BXOR = r'\^'
t_RSHIFT = r'>>'
t_LSHIFT = r'<<'
t_BNOT = r'~'

t_EQUAL = r'='
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_COLON = r':'
t_SEMI = r';'
t_COMMA = r','

t_LNOT = r'!'
t_LAND = r'&&'
t_LOR = r'\|\|'
t_EQ = r'=='
t_NE = r'!='
t_LE = r'<='
t_LT = r'<'
t_GE = r'>='
t_GT = r'>'

t_ignore = ' \t'


def t_COMMENT(t):
    r'//[^\n]*'
    pass


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')


def t_IDENT(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, 'IDENT')
    return t


def t_NUMBER(t):
    r'0|[1-9][0-9]*'
    try:
        v = int(t.value)
    except ValueError:
        err("Lexer", f"bad integer literal {t.value!r}")
        t.lexer.error = True
        return None
    if not (0 <= v < (1 << 63)):
        err("Lexer", f"integer {v} out of range [0, 2^63)")
        t.lexer.error = True
        return None
    t.value = v
    return t


def t_error(t):
    err("Lexer", f"line {t.lineno}, pos {t.lexpos}: illegal char {t.value[0]!r}")
    t.lexer.skip(1)


lexer = lex.lex()

# =============================================================================
# AST
# =============================================================================


class AST:
    pass


class Expr(AST):
    pass


@dc.dataclass
class ENum(Expr):
    n: int
    ty: Optional[str] = None  # "int"


@dc.dataclass
class EBool(Expr):
    b: bool
    ty: Optional[str] = None  # "bool"


@dc.dataclass
class EVar(Expr):
    name: str
    ty: Optional[str] = None  # "int" or "bool"


@dc.dataclass
class EUn(Expr):
    op: str
    e: Expr
    ty: Optional[str] = None


@dc.dataclass
class EBin(Expr):
    op: str
    l: Expr
    r: Expr
    ty: Optional[str] = None


@dc.dataclass
class ECall(Expr):
    name: str
    args: List[Expr]
    ty: Optional[str] = None  # return type


class Stmt(AST):
    pass


@dc.dataclass
class SVar(Stmt):  # used both for globals and locals
    name: str
    init: Expr
    ty: str  # "int" or "bool"


@dc.dataclass
class SAssign(Stmt):
    name: str
    rhs: Expr


@dc.dataclass
class SExprStmt(Stmt):
    e: Expr  # usually procedure calls


@dc.dataclass
class SBlock(Stmt):
    ss: List[Stmt]


@dc.dataclass
class SIfElse(Stmt):
    cond: Expr
    thenb: SBlock
    elsep: Optional[Stmt]  # either SBlock or nested SIfElse


@dc.dataclass
class SWhile(Stmt):
    cond: Expr
    body: SBlock


@dc.dataclass
class SBreak(Stmt):
    pass


@dc.dataclass
class SContinue(Stmt):
    pass


@dc.dataclass
class SReturn(Stmt):
    expr: Optional[Expr]  # None for "return;" in void procs


@dc.dataclass
class Param(AST):
    name: str
    ty: str  # "int" or "bool"


@dc.dataclass
class ProcDecl(AST):
    name: str
    params: List[Param]
    ret_ty: Optional[str]  # None => "void"
    body: SBlock


@dc.dataclass
class Program(AST):
    globals: List[SVar]      # top-level var decls
    procs: List[ProcDecl]    # all procedures (including main)


# =============================================================================
# PARSER
# =============================================================================

precedence = (
    ('left', 'LOR'),
    ('left', 'LAND'),
    ('left', 'BOR'),
    ('left', 'BXOR'),
    ('left', 'BAND'),
    ('nonassoc', 'EQ', 'NE'),
    ('nonassoc', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'LSHIFT', 'RSHIFT'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'MOD'),
    ('right', 'LNOT', 'BNOT', 'UMINUS'),
)

# ---- program / top-level ----------------------------------------------------


def p_program(p):
    'program : toplevel_list'
    globs: List[SVar] = []
    procs: List[ProcDecl] = []
    for d in p[1]:
        if isinstance(d, SVar):
            globs.append(d)
        elif isinstance(d, ProcDecl):
            procs.append(d)
        else:
            err("Parser", f"internal: unknown toplevel node {type(d).__name__}")
            raise SyntaxError
    p[0] = Program(globals=globs, procs=procs)


def p_toplevel_list_empty(p):
    'toplevel_list : '
    p[0] = []


def p_toplevel_list_cons(p):
    'toplevel_list : toplevel_list toplevel'
    p[1].append(p[2])
    p[0] = p[1]


def p_toplevel_vardecl(p):
    'toplevel : VAR IDENT EQUAL expr COLON type SEMI'
    # Global variable
    p[0] = SVar(p[2], p[4], p[6])


def p_toplevel_proc(p):
    'toplevel : DEF IDENT LPAREN params_opt RPAREN rettype_opt block'
    p[0] = ProcDecl(name=p[2], params=p[4], ret_ty=p[6], body=p[7])


# ---- types / params ---------------------------------------------------------


def p_type_int(p):
    'type : INT'
    p[0] = "int"


def p_type_bool(p):
    'type : BOOL'
    p[0] = "bool"


def p_rettype_opt_empty(p):
    'rettype_opt : '
    p[0] = None   # void


def p_rettype_opt_typed(p):
    'rettype_opt : COLON rettype'
    p[0] = p[2]


def p_rettype_int(p):
    'rettype : INT'
    p[0] = "int"


def p_rettype_bool(p):
    'rettype : BOOL'
    p[0] = "bool"


def p_rettype_void(p):
    'rettype : VOID'
    p[0] = "void"


def p_params_opt_empty(p):
    'params_opt : '
    p[0] = []


def p_params_opt_some(p):
    'params_opt : param_list'
    p[0] = p[1]


def p_param_list_one(p):
    'param_list : param'
    p[0] = [p[1]]


def p_param_list_many(p):
    'param_list : param_list COMMA param'
    p[1].append(p[3])
    p[0] = p[1]


def p_param(p):
    'param : IDENT COLON type'
    p[0] = Param(name=p[1], ty=p[3])


# ---- blocks / statements ----------------------------------------------------


def p_block(p):
    'block : LBRACE stmt_list RBRACE'
    p[0] = SBlock(p[2])


def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = []


def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    p[1].append(p[2])
    p[0] = p[1]


def p_stmt_vardecl(p):
    'stmt : VAR IDENT EQUAL expr COLON type SEMI'
    p[0] = SVar(p[2], p[4], p[6])


def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = SAssign(p[1], p[3])


def p_stmt_break(p):
    'stmt : BREAK SEMI'
    p[0] = SBreak()


def p_stmt_continue(p):
    'stmt : CONTINUE SEMI'
    p[0] = SContinue()


def p_stmt_if(p):
    'stmt : IF LPAREN expr RPAREN block ifrest'
    p[0] = SIfElse(p[3], p[5], p[6])


def p_ifrest_empty(p):
    'ifrest : '
    p[0] = None


def p_ifrest_else(p):
    'ifrest : ELSE stmt'
    # stmt can be block or nested if
    p[0] = p[2]


def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN block'
    p[0] = SWhile(p[3], p[5])


def p_stmt_block(p):
    'stmt : block'
    p[0] = p[1]


def p_stmt_return_value(p):
    'stmt : RETURN expr SEMI'
    p[0] = SReturn(p[2])


def p_stmt_return_void(p):
    'stmt : RETURN SEMI'
    p[0] = SReturn(None)


def p_stmt_expr(p):
    'stmt : expr SEMI'
    # Expression statement (typically procedure calls like foo(...);)
    p[0] = SExprStmt(p[1])


# ---- expressions ------------------------------------------------------------


def p_expr_num(p):
    'expr : NUMBER'
    p[0] = ENum(p[1])


def p_expr_true(p):
    'expr : TRUE'
    p[0] = EBool(True)


def p_expr_false(p):
    'expr : FALSE'
    p[0] = EBool(False)


def p_expr_var(p):
    'expr : IDENT'
    p[0] = EVar(p[1])


def p_expr_call(p):
    'expr : IDENT LPAREN args_opt RPAREN'
    p[0] = ECall(p[1], p[3])


def p_args_opt_empty(p):
    'args_opt : '
    p[0] = []


def p_args_opt_some(p):
    'args_opt : arg_list'
    p[0] = p[1]


def p_arg_list_one(p):
    'arg_list : expr'
    p[0] = [p[1]]


def p_arg_list_many(p):
    'arg_list : arg_list COMMA expr'
    p[1].append(p[3])
    p[0] = p[1]


def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]


def p_expr_uminus(p):
    'expr : MINUS expr %prec UMINUS'
    p[0] = EUn('-', p[2])


def p_expr_bnot(p):
    'expr : BNOT expr'
    p[0] = EUn('~', p[2])


def p_expr_lnot(p):
    'expr : LNOT expr'
    p[0] = EUn('!', p[2])


def p_expr_binop(p):
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr STAR expr
            | expr SLASH expr
            | expr MOD expr
            | expr BAND expr
            | expr BOR expr
            | expr BXOR expr
            | expr LSHIFT expr
            | expr RSHIFT expr
            | expr EQ expr
            | expr NE expr
            | expr LT expr
            | expr LE expr
            | expr GT expr
            | expr GE expr
            | expr LAND expr
            | expr LOR expr'''
    p[0] = EBin(p[2], p[1], p[3])


def p_error(p):
    if p is None:
        err("Parser", "unexpected end of input")
    else:
        err("Parser", f"line {getattr(p,'lineno','?')}, pos {getattr(p,'lexpos','?')}: "
                      f"unexpected token {p.type} ({p.value!r})")
    raise SyntaxError


parser = yacc.yacc(start='program')

# =============================================================================
# TYPE CHECKING (Lab 4: globals, procedures, bools, returns)
# =============================================================================


class TypeErrorBX(Exception):
    pass


class SemErrorBX(Exception):
    pass


def check_program(prog: Program):
    # -------------------------------------------------------------------------
    # 1) Collect global variables (name -> type)
    # -------------------------------------------------------------------------
    global_vars: Dict[str, str] = {}

    for gv in prog.globals:
        if gv.name in global_vars:
            raise SemErrorBX(f"Global variable '{gv.name}' redeclared")

        if gv.ty not in ("int", "bool"):
            raise TypeErrorBX("Global variables must be of type int or bool")

        # For Lab 4 we typically only allow literal initializers at top level.
        if not isinstance(gv.init, (ENum, EBool)):
            raise TypeErrorBX("Global initializers must be literal integers or booleans")

        init_ty = "int" if isinstance(gv.init, ENum) else "bool"
        if init_ty != gv.ty:
            raise TypeErrorBX(f"Initializer for global '{gv.name}' has type {init_ty}, expected {gv.ty}")

        global_vars[gv.name] = gv.ty

    # -------------------------------------------------------------------------
    # 2) Collect procedure signatures (name -> (param_types, ret_type))
    # -------------------------------------------------------------------------
    proc_sigs: Dict[str, Tuple[List[str], str]] = {}

    for proc in prog.procs:
        if proc.name in proc_sigs:
            raise SemErrorBX(f"Procedure '{proc.name}' redeclared")
        if proc.name in global_vars:
            raise SemErrorBX(f"Name '{proc.name}' used for both global variable and procedure")

        param_types: List[str] = []
        param_names: set[str] = set()

        for param in proc.params:
            if param.name in param_names:
                raise SemErrorBX(f"Duplicate parameter '{param.name}' in procedure '{proc.name}'")
            if param.ty not in ("int", "bool"):
                raise TypeErrorBX(f"Parameter '{param.name}' in '{proc.name}' must be int or bool")
            param_names.add(param.name)
            param_types.append(param.ty)

        ret_ty = proc.ret_ty or "void"
        if ret_ty not in ("int", "bool", "void"):
            raise TypeErrorBX(f"Invalid return type '{ret_ty}' in procedure '{proc.name}'")

        proc_sigs[proc.name] = (param_types, ret_ty)

    # Ensure there is a main
    if "main" not in proc_sigs:
        raise SemErrorBX("No 'main' procedure defined")

    main_params, main_ret = proc_sigs["main"]
    if main_params:
        raise TypeErrorBX("main must not take parameters")
    # main return type can be void or int/bool; we don't enforce more here.

    # -------------------------------------------------------------------------
    # 3) Check each procedure body
    # -------------------------------------------------------------------------

    def check_proc(proc: ProcDecl):
        ret_ty = proc.ret_ty or "void"

        # local environments: stack of name -> type
        locals_stack: List[Dict[str, str]] = [{}]

        # push parameters as an inner scope
        param_env: Dict[str, str] = {p.name: p.ty for p in proc.params}
        locals_stack.append(param_env)

        def lookup_var(name: str) -> str:
            for env in reversed(locals_stack):
                if name in env:
                    return env[name]
            if name in global_vars:
                return global_vars[name]
            raise SemErrorBX(f"Use of undeclared variable '{name}'")

        def declare_local(name: str, ty: str):
            if name in locals_stack[-1]:
                raise SemErrorBX(f"Variable '{name}' redeclared in the same block")
            locals_stack[-1][name] = ty

        def enter_block():
            locals_stack.append({})

        def exit_block():
            locals_stack.pop()

        def chk_e(e: Expr) -> str:
            if isinstance(e, ENum):
                e.ty = "int"
                return "int"
            if isinstance(e, EBool):
                e.ty = "bool"
                return "bool"
            if isinstance(e, EVar):
                t = lookup_var(e.name)
                e.ty = t
                return t
            if isinstance(e, EUn):
                t = chk_e(e.e)
                if e.op == '!':
                    if t != "bool":
                        raise TypeErrorBX("operator ! expects bool")
                    e.ty = "bool"
                    return "bool"
                if e.op in ('-', '~'):
                    if t != "int":
                        raise TypeErrorBX(f"operator {e.op} expects int")
                    e.ty = "int"
                    return "int"
                raise TypeErrorBX(f"unknown unary operator {e.op}")
            if isinstance(e, EBin):
                if e.op in ('&&', '||'):
                    tl = chk_e(e.l)
                    tr = chk_e(e.r)
                    if tl != "bool" or tr != "bool":
                        raise TypeErrorBX(f"{e.op} expects bool && bool")
                    e.ty = "bool"
                    return "bool"
                if e.op in ('==', '!=', '<', '<=', '>', '>='):
                    tl = chk_e(e.l)
                    tr = chk_e(e.r)
                    if tl != "int" or tr != "int":
                        raise TypeErrorBX(f"{e.op} compares ints")
                    e.ty = "bool"
                    return "bool"
                # arithmetic / bitwise
                tl = chk_e(e.l)
                tr = chk_e(e.r)
                if tl != "int" or tr != "int":
                    raise TypeErrorBX(f"{e.op} expects ints")
                e.ty = "int"
                return "int"
            if isinstance(e, ECall):
                # built-in: print(x) is allowed for int/bool and has type void
                if e.name == "print":
                    if len(e.args) != 1:
                        raise TypeErrorBX("print expects exactly one argument")
                    aty = chk_e(e.args[0])
                    if aty not in ("int", "bool"):
                        raise TypeErrorBX("print expects int or bool")
                    e.ty = "void"
                    return "void"

                if e.name not in proc_sigs:
                    raise SemErrorBX(f"Call to undeclared procedure '{e.name}'")

                param_tys, rty = proc_sigs[e.name]
                if len(param_tys) != len(e.args):
                    raise TypeErrorBX(
                        f"Procedure '{e.name}' expects {len(param_tys)} argument(s), "
                        f"got {len(e.args)}"
                    )
                for i, (arg, expected) in enumerate(zip(e.args, param_tys), start=1):
                    aty = chk_e(arg)
                    if aty != expected:
                        raise TypeErrorBX(
                            f"Argument {i} of '{e.name}' has type {aty}, expected {expected}"
                        )
                e.ty = rty
                return rty

            raise TypeErrorBX("unknown expression")

        def chk_block(b: SBlock) -> bool:
            """
            Returns True if this block *guarantees* a return on all paths.
            """
            enter_block()
            must_return = False
            for st in b.ss:
                if must_return:
                    # unreachable, but we don't complain
                    break
                if chk_s(st):
                    must_return = True
            exit_block()
            return must_return

        def chk_s(s: Stmt) -> bool:
            """
            Returns True if this statement definitely returns on all paths.
            """
            # var declaration (local)
            if isinstance(s, SVar):
                if s.ty not in ("int", "bool"):
                    raise TypeErrorBX("variables must be int or bool")
                init_ty = chk_e(s.init)
                if init_ty != s.ty:
                    raise TypeErrorBX(
                        f"Initializer for '{s.name}' has type {init_ty}, "
                        f"expected {s.ty}"
                    )
                declare_local(s.name, s.ty)
                return False

            # assignment
            if isinstance(s, SAssign):
                vty = lookup_var(s.name)
                rhs_ty = chk_e(s.rhs)
                if rhs_ty != vty:
                    raise TypeErrorBX(
                        f"assignment to '{s.name}' mismatched type: "
                        f"lhs is {vty}, rhs is {rhs_ty}"
                    )
                return False

            # expression statement
            if isinstance(s, SExprStmt):
                _ = chk_e(s.e)
                # We allow ignoring value even if non-void.
                return False

            # return
            if isinstance(s, SReturn):
                if ret_ty == "void":
                    if s.expr is not None:
                        raise TypeErrorBX(
                            f"Procedure '{proc.name}' returns void "
                            f"but 'return' has a value"
                        )
                    return True
                else:
                    if s.expr is None:
                        raise TypeErrorBX(
                            f"Non-void procedure '{proc.name}' must return a value"
                        )
                    rty = chk_e(s.expr)
                    if rty != ret_ty:
                        raise TypeErrorBX(
                            f"Return type mismatch in '{proc.name}': "
                            f"expected {ret_ty}, got {rty}"
                        )
                    return True

            # block
            if isinstance(s, SBlock):
                return chk_block(s)

            # if/else
            if isinstance(s, SIfElse):
                cty = chk_e(s.cond)
                if cty != "bool":
                    raise TypeErrorBX("if expects bool condition")
                then_ret = chk_block(s.thenb)
                if s.elsep is None:
                    return False
                else_ret = chk_s(s.elsep)
                return then_ret and else_ret

            # while
            if isinstance(s, SWhile):
                cty = chk_e(s.cond)
                if cty != "bool":
                    raise TypeErrorBX("while expects bool condition")
                # Loop may not execute, so cannot guarantee return
                _ = chk_block(s.body)
                return False

            # break / continue – only checked for presence in codegen, but
            # here we just allow them.
            if isinstance(s, (SBreak, SContinue)):
                return False

            raise TypeErrorBX("unknown statement")

        all_return = chk_block(proc.body)
        if ret_ty != "void" and not all_return:
            raise TypeErrorBX(
                f"Procedure '{proc.name}' may fall off the end "
                f"without returning a value"
            )

    for proc in prog.procs:
        check_proc(proc)

    dbg("Type/Semantic OK")


# =============================================================================
# DRIVER
# =============================================================================


def parse_text(src: str) -> Optional[Program]:
    lexer.lineno = 1
    try:
        ast = parser.parse(src, lexer=lexer)
        dbg("Parse OK")
        return ast
    except SyntaxError:
        return None
    except Exception as e:
        err("Parser", f"internal exception: {e}")
        return None


def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="BX Lab 4: Procedures and Types (front-end)")
    ap.add_argument("source", help=".bx file")
    ap.add_argument("--debug", action="store_true", help="enable debug prints")
    args = ap.parse_args()
    DEBUG = args.debug or (os.environ.get("BX_DEBUG", "") not in ("", "0", "false", "False"))

    try:
        src = open(args.source, "r", encoding="utf-8").read()
    except OSError as e:
        err("IO", f"cannot read {args.source}: {e}")
        sys.exit(1)

    ast = parse_text(src)
    if ast is None:
        err("Driver", "parse failed")
        sys.exit(1)

    try:
        check_program(ast)
    except (TypeErrorBX, SemErrorBX) as e:
        err("Type/Semantic", str(e))
        sys.exit(1)
    except Exception as e:
        err("Type/Semantic", f"internal exception: {e}")
        sys.exit(1)

    # Lab 4: we stop after type checking (no TAC / x64 yet).
    # Success ⇒ exit 0, no extra output required.
    sys.exit(0)


if __name__ == "__main__":
    main()


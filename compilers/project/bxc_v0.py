import os 
import sys
import argparse
from dataclasses import dataclass as dc, field
from typing import List, Tuple, Dict, Optional, Set, Union

import ply.lex as lex
import ply.yacc as yacc


# =============================================================================
# ERROR TYPES
# =============================================================================

class BXError(Exception):
    pass


class TypeErrorBX(Exception):
    pass


class SemErrorBX(Exception):
    pass


# =============================================================================
# LEXER
# =============================================================================

reserved = {
    'def':   'DEF',
    'var':   'VAR',
    'int':   'INT',
    'bool':  'BOOL',
    'void':  'VOID',
    'function': 'FUNCTION',
    'true':  'TRUE',
    'false': 'FALSE',
    'if':    'IF',
    'else':  'ELSE',
    'while': 'WHILE',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'return': 'RETURN',
    'ret':   'RET',      # short form for "return;"
}

tokens = (
    'IDENT', 'NUM',
    'PLUS', 'MINUS', 'TIMES', 'DIV', 'MOD',
    'BAND', 'BOR', 'BXOR',
    'RSHIFT', 'LSHIFT', 'BNOT',
    'EQUAL',
    'LPAREN', 'RPAREN',
    'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'COMMA',
    'ARROW',
    'LNOT', 'LAND', 'LOR',
    'EQ', 'NEQ', 'LT', 'LE', 'GT', 'GE',
) + tuple(reserved.values())

t_PLUS   = r'\+'
t_MINUS  = r'-'
t_TIMES  = r'\*'
t_DIV    = r'/'
t_MOD    = r'%'

t_BAND   = r'&'
t_BOR    = r'\|'
t_BXOR   = r'\^'

t_RSHIFT = r'>>'
t_LSHIFT = r'<<'
t_BNOT   = r'~'

t_EQUAL  = r'='
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_COLON  = r':'
t_SEMI   = r';'
t_COMMA  = r','
t_ARROW  = r'->'

t_LNOT = r'!'
t_LAND = r'&&'
t_LOR  = r'\|\|'

t_EQ   = r'=='
t_NEQ  = r'!='
t_LT   = r'<'
t_LE   = r'<='
t_GT   = r'>'
t_GE   = r'>='


def t_IDENT(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENT')
    return t


def t_NUM(t):
    r'-?[0-9]+'
    t.value = int(t.value)
    return t


t_ignore = ' \t\r'


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_comment_line(t):
    r'//[^\n]*'
    pass


def t_error(t):
    raise BXError(f"Illegal character {t.value[0]!r} at line {t.lineno}")


lexer = lex.lex()


# =============================================================================
# AST
# =============================================================================

class AST:
    pass


# --- Types -------------------------------------------------------------------

Ty = Union[str, 'FunTy']   


@dc(frozen=True)
class FunTy(AST):
    param_tys: Tuple[Ty, ...]
    ret_ty: str  


# --- Expressions -------------------------------------------------------------


class Expr(AST):
    ty: Ty  


@dc
class ENum(Expr):
    n: int


@dc
class EBool(Expr):
    b: bool


@dc
class EVar(Expr):
    name: str


@dc
class EUn(Expr):
    op: str
    e: Expr


@dc
class EBin(Expr):
    op: str
    l: Expr
    r: Expr


@dc
class ECall(Expr):
    name: str
    args: List[Expr]


# --- Statements --------------------------------------------------------------


class Stmt(AST):
    pass


@dc
class SBlock(Stmt):
    ss: List[Stmt]


@dc
class SIfElse(Stmt):
    cond: Expr
    thenb: Stmt
    elsep: Optional[Stmt]


@dc
class SWhile(Stmt):
    cond: Expr
    body: Stmt


@dc
class SBreak(Stmt):
    pass


@dc
class SContinue(Stmt):
    pass


@dc
class SVar(Stmt):
    name: str
    init: Expr
    ty_annot: Ty          
    vid: int = -1         


@dc
class SAssign(Stmt):
    name: str
    e: Expr


@dc
class SExpr(Stmt):
    e: Expr


@dc
class SReturn(Stmt):
    e: Optional[Expr]  


@dc
class SProcDef(Stmt):
    """Nested procedure definition as a statement."""
    proc: 'ProcDecl'


# --- Procedures & Program ----------------------------------------------------


@dc
class Param(AST):
    name: str
    ty: Ty
    vid: int = -1  


@dc
class ProcDecl(AST):
    name: str
    params: List[Param]
    ret_ty: Ty
    body: SBlock
    captures: Set[int] = field(default_factory=set)


@dc
class Program(AST):
    procs: List[ProcDecl]


# =============================================================================
# PARSER
# =============================================================================

precedence = (
    ('left', 'LOR'),
    ('left', 'LAND'),
    ('left', 'EQ', 'NEQ'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIV', 'MOD'),
    ('right', 'LNOT', 'BNOT'),
    ('right', 'ELSE'),
)


def p_program(p):
    'program : procs'
    p[0] = Program(p[1])


def p_procs_single(p):
    'procs : proc'
    p[0] = [p[1]]


def p_procs_many(p):
    'procs : procs proc'
    p[1].append(p[2])
    p[0] = p[1]


def p_proc(p):
    'proc : DEF IDENT LPAREN params RPAREN ret_annot block'
    p[0] = ProcDecl(p[2], p[4], p[6], p[7])


# --- params / types ----------------------------------------------------------

def p_params_empty(p):
    'params : '
    p[0] = []


def p_params_nonempty(p):
    'params : param_list'
    p[0] = p[1]


def p_param_list_one(p):
    'param_list : IDENT COLON type'
    p[0] = [Param(p[1], p[3])]


def p_param_list_cons(p):
    'param_list : param_list COMMA IDENT COLON type'
    p[1].append(Param(p[3], p[5]))
    p[0] = p[1]


def p_type_int(p):
    'type : INT'
    p[0] = 'int'


def p_type_bool(p):
    'type : BOOL'
    p[0] = 'bool'


def p_type_fun(p):
    'type : FUNCTION LPAREN type_list_opt RPAREN ARROW funrettype'
    p[0] = FunTy(tuple(p[3]), p[6])


def p_type_list_opt_empty(p):
    'type_list_opt : '
    p[0] = []


def p_type_list_opt_list(p):
    'type_list_opt : type_list'
    p[0] = p[1]


def p_type_list_one(p):
    'type_list : type'
    p[0] = [p[1]]


def p_type_list_cons(p):
    'type_list : type_list COMMA type'
    p[1].append(p[3])
    p[0] = p[1]


def p_funrettype_int(p):
    'funrettype : INT'
    p[0] = 'int'


def p_funrettype_bool(p):
    'funrettype : BOOL'
    p[0] = 'bool'


def p_funrettype_void(p):
    'funrettype : VOID'
    p[0] = 'void'


def p_ret_annot_void(p):
    'ret_annot : '
    p[0] = 'void'


def p_ret_annot_ty(p):
    'ret_annot : COLON funrettype'
    p[0] = p[2]


# --- blocks / statements -----------------------------------------------------

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


def p_stmt_procdef(p):
    'stmt : DEF IDENT LPAREN params RPAREN ret_annot block'
    p[0] = SProcDef(ProcDecl(p[2], p[4], p[6], p[7]))


def p_stmt_vardecl(p):
    'stmt : VAR IDENT EQUAL expr COLON type SEMI'
    p[0] = SVar(p[2], p[4], p[6])


def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = SAssign(p[1], p[3])


def p_stmt_expr(p):
    'stmt : expr SEMI'
    p[0] = SExpr(p[1])


def p_stmt_block(p):
    'stmt : block'
    p[0] = p[1]


def p_stmt_if(p):
    'stmt : IF LPAREN expr RPAREN stmt %prec ELSE'
    p[0] = SIfElse(p[3], p[5], None)


def p_stmt_if_else(p):
    'stmt : IF LPAREN expr RPAREN stmt ELSE stmt'
    p[0] = SIfElse(p[3], p[5], p[7])


def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN stmt'
    p[0] = SWhile(p[3], p[5])


def p_stmt_break(p):
    'stmt : BREAK SEMI'
    p[0] = SBreak()


def p_stmt_continue(p):
    'stmt : CONTINUE SEMI'
    p[0] = SContinue()


def p_stmt_return_void(p):
    'stmt : RETURN SEMI'
    p[0] = SReturn(None)


def p_stmt_return_val(p):
    'stmt : RETURN expr SEMI'
    p[0] = SReturn(p[2])


def p_stmt_ret_short(p):
    'stmt : RET SEMI'
    p[0] = SReturn(None)


# --- expressions -------------------------------------------------------------

def p_expr_num(p):
    'expr : NUM'
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


def p_expr_parens(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]


def p_expr_unary(p):
    '''expr : LNOT expr
            | MINUS expr %prec LNOT
            | BNOT expr %prec LNOT'''
    p[0] = EUn(p[1], p[2])


def p_expr_binary(p):
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr
            | expr DIV expr
            | expr MOD expr
            | expr BAND expr
            | expr BOR expr
            | expr BXOR expr
            | expr RSHIFT expr
            | expr LSHIFT expr
            | expr EQ expr
            | expr NEQ expr
            | expr LT expr
            | expr LE expr
            | expr GT expr
            | expr GE expr
            | expr LAND expr
            | expr LOR expr'''
    p[0] = EBin(p[2], p[1], p[3])


def p_expr_call(p):
    'expr : IDENT LPAREN arglist RPAREN'
    p[0] = ECall(p[1], p[3])


def p_arglist_empty(p):
    'arglist : '
    p[0] = []


def p_arglist_nonempty(p):
    'arglist : expr_list'
    p[0] = p[1]


def p_expr_list_one(p):
    'expr_list : expr'
    p[0] = [p[1]]


def p_expr_list_many(p):
    'expr_list : expr_list COMMA expr'
    p[1].append(p[3])
    p[0] = p[1]


def p_error(p):
    if p is None:
        raise BXError("[Parser] Syntax error at EOF")
    raise BXError(f"[Parser] Syntax error at token {p.type} (value={p.value})")


parser = yacc.yacc(start='program')


# =============================================================================
# TYPE CHECKER WITH CAPTURE SETS
# =============================================================================

VarInfo = Tuple[Ty, int]


def check_program(prog: Program) -> None:
    def is_base_type(ty: Ty) -> bool:
        return isinstance(ty, str) and ty in ('int', 'bool')

    def is_first_order_type(ty: Ty) -> bool:
        return isinstance(ty, str) and ty in ('int', 'bool', 'void')

    def is_valid_param_type(ty: Ty) -> bool:
        if isinstance(ty, str):
            return ty in ('int', 'bool')
        if isinstance(ty, FunTy):
            return all(is_valid_param_type(pt) for pt in ty.param_tys) and is_first_order_type(ty.ret_ty)
        return False

    def ty_repr(ty: Ty) -> str:
        if isinstance(ty, str):
            return ty
        if isinstance(ty, FunTy):
            params = ", ".join(ty_repr(t) for t in ty.param_tys)
            return f"function({params}) -> {ty.ret_ty}"
        return str(ty)

    # --- Global function environment (top-level procs) ------------------------
    fun_env_global: Dict[str, FunTy] = {}

    for pd in prog.procs:
        if pd.name in fun_env_global:
            raise SemErrorBX(f"Procedure '{pd.name}' redeclared")

        param_tys = [p.ty for p in pd.params]
        for t in param_tys:
            if not is_valid_param_type(t):
                raise TypeErrorBX(f"Invalid parameter type '{ty_repr(t)}' in '{pd.name}'")

        if not is_first_order_type(pd.ret_ty):
            raise TypeErrorBX(f"Invalid return type '{ty_repr(pd.ret_ty)}' in '{pd.name}'")

        fun_env_global[pd.name] = FunTy(tuple(param_tys), pd.ret_ty)

    if 'main' not in fun_env_global:
        raise SemErrorBX("Entry point 'main' missing")

    next_var_id = 0

    def fresh_var_id() -> int:
        nonlocal next_var_id
        vid = next_var_id
        next_var_id += 1
        return vid

    # -------------------------------------------------------------------------
    # Procedure type-checking (top-level and nested) with capture sets
    # -------------------------------------------------------------------------
    def typecheck_proc(pd: ProcDecl,
                       fun_ty: FunTy,
                       outer_var_env: List[Dict[str, VarInfo]],
                       fun_env_stack: List[Dict[str, FunTy]]) -> bool:
        """
        Type-check a procedure (top-level or nested).

        outer_var_env: lexical variable env from surrounding scopes (name -> (ty, vid)).
        fun_env_stack: stack of function environments (for call resolution).
        Returns True iff all control-flow paths return.
        """
        pd.captures.clear()

        var_env_stack: List[Dict[str, VarInfo]] = [dict(frame) for frame in outer_var_env]

        param_env: Dict[str, VarInfo] = {}
        local_ids: Set[int] = set()  

        for param in pd.params:
            if param.name in param_env:
                raise SemErrorBX(f"Duplicate parameter '{param.name}' in '{pd.name}'")
            if not is_valid_param_type(param.ty):
                raise TypeErrorBX(
                    f"Invalid parameter type '{ty_repr(param.ty)}' in '{pd.name}'"
                )
            vid = fresh_var_id()
            param.vid = vid
            param_env[param.name] = (param.ty, vid)
            local_ids.add(vid)

        var_env_stack.append(param_env)

        ret_ty: str = fun_ty.ret_ty

        # --- lookup helpers ---------------------------------------------------

        def lookup_var(name: str) -> Optional[VarInfo]:
            for frame in reversed(var_env_stack):
                if name in frame:
                    return frame[name]
            return None

        def lookup_fun(name: str) -> Optional[FunTy]:
            for fenv in reversed(fun_env_stack):
                if name in fenv:
                    return fenv[name]
            return None

        def add_local_var(name: str, ty: Ty) -> VarInfo:
            frame = var_env_stack[-1]
            if name in frame:
                raise SemErrorBX(f"Identifier '{name}' redeclared in the same block")
            vid = fresh_var_id()
            frame[name] = (ty, vid)
            local_ids.add(vid)
            return frame[name]

        def enter_block():
            var_env_stack.append({})

        def exit_block():
            var_env_stack.pop()

        # --- expression checking (also updates captures) ----------------------

        def chk_e(e: Expr) -> Ty:
            if isinstance(e, ENum):
                e.ty = 'int'
                return 'int'
            if isinstance(e, EBool):
                e.ty = 'bool'
                return 'bool'

            if isinstance(e, EVar):
                vi = lookup_var(e.name)
                if vi is not None:
                    ty, vid = vi
                    if vid not in local_ids:
                        pd.captures.add(vid)
                    e.ty = ty
                    return ty
                fty = lookup_fun(e.name)
                if fty is None:
                    raise SemErrorBX(f"Use of undeclared identifier '{e.name}'")
                e.ty = fty
                return fty

            if isinstance(e, EUn):
                t = chk_e(e.e)
                if e.op == '!':
                    if not (isinstance(t, str) and t == 'bool'):
                        raise TypeErrorBX("operator ! expects bool")
                    e.ty = 'bool'
                    return 'bool'
                if e.op in ('-', '~'):
                    if not (isinstance(t, str) and t == 'int'):
                        raise TypeErrorBX(f"operator {e.op} expects int")
                    e.ty = 'int'
                    return 'int'
                raise TypeErrorBX(f"unknown unary {e.op}")

            if isinstance(e, EBin):
                if e.op in ('&&', '||'):
                    tl = chk_e(e.l)
                    tr = chk_e(e.r)
                    if not (isinstance(tl, str) and tl == 'bool' and
                            isinstance(tr, str) and tr == 'bool'):
                        raise TypeErrorBX(f"{e.op} expects bool && bool")
                    e.ty = 'bool'
                    return 'bool'
                if e.op in ('==', '!=', '<', '<=', '>', '>='):
                    tl = chk_e(e.l)
                    tr = chk_e(e.r)
                    if not (isinstance(tl, str) and tl == 'int' and
                            isinstance(tr, str) and tr == 'int'):
                        raise TypeErrorBX(f"{e.op} compares ints")
                    e.ty = 'bool'
                    return 'bool'
                tl = chk_e(e.l)
                tr = chk_e(e.r)
                if not (isinstance(tl, str) and tl == 'int' and
                        isinstance(tr, str) and tr == 'int'):
                    raise TypeErrorBX(f"{e.op} expects ints")
                e.ty = 'int'
                return 'int'

            if isinstance(e, ECall):
                if e.name == 'print':
                    if len(e.args) != 1:
                        raise TypeErrorBX("print expects exactly one argument")
                    aty = chk_e(e.args[0])
                    if isinstance(aty, str) and aty == 'int':
                        e.name = '__bx_print_int'
                    elif isinstance(aty, str) and aty == 'bool':
                        e.name = '__bx_print_bool'
                    else:
                        raise TypeErrorBX("print expects int or bool")
                    e.ty = 'void'
                    return 'void'

                vi = lookup_var(e.name)
                if vi is not None:
                    vty, vid = vi
                    if not isinstance(vty, FunTy):
                        raise TypeErrorBX(
                            f"Attempting to call non-function variable '{e.name}' "
                            f"of type {ty_repr(vty)}"
                        )
                    if vid not in local_ids:
                        pd.captures.add(vid)
                    callee_ty = vty
                else:
                    fty = lookup_fun(e.name)
                    if fty is None:
                        raise SemErrorBX(f"Call to unknown function '{e.name}'")
                    callee_ty = fty

                if len(e.args) != len(callee_ty.param_tys):
                    raise TypeErrorBX(
                        f"Function '{e.name}' expects {len(callee_ty.param_tys)} "
                        f"args, got {len(e.args)}"
                    )
                for i, (a, pt) in enumerate(zip(e.args, callee_ty.param_tys)):
                    at = chk_e(a)
                    if at != pt:
                        raise TypeErrorBX(
                            f"Argument {i+1} type mismatch in call to '{e.name}': "
                            f"expected {ty_repr(pt)}, got {ty_repr(at)}"
                        )
                e.ty = callee_ty.ret_ty
                return e.ty

            raise TypeErrorBX("unknown expr")

        # --- statement checking (also contributes captures) --------------------

        def chk_s(s: Stmt) -> bool:
            """
            returns True if this statement definitely returns on all paths.
            """
            if isinstance(s, SVar):
                if not (isinstance(s.ty_annot, str) and s.ty_annot in ('int', 'bool')):
                    raise TypeErrorBX(
                        "Local variables must be of type int or bool (no function types)"
                    )
                t_init = chk_e(s.init)
                if t_init != s.ty_annot:
                    raise TypeErrorBX(
                        f"Initializer for '{s.name}' has type {ty_repr(t_init)}, "
                        f"expected {ty_repr(s.ty_annot)}"
                    )
                ty, vid = add_local_var(s.name, s.ty_annot)
                s.vid = vid
                return False

            if isinstance(s, SAssign):
                vi = lookup_var(s.name)
                if vi is None:
                    raise SemErrorBX(f"Assignment to undeclared variable '{s.name}'")
                ty, vid = vi
                t_rhs = chk_e(s.e)
                if t_rhs != ty:
                    raise TypeErrorBX(
                        f"assignment type mismatch to '{s.name}': "
                        f"expected {ty_repr(ty)}, got {ty_repr(t_rhs)}"
                    )

                if vid not in local_ids:
                    pd.captures.add(vid)
                return False

            if isinstance(s, SExpr):
                chk_e(s.e)
                return False

            if isinstance(s, SBlock):
                enter_block()
                must_return = False
                for st in s.ss:
                    if chk_s(st):
                        must_return = True
                exit_block()
                return must_return

            if isinstance(s, SIfElse):
                tcond = chk_e(s.cond)
                if not (isinstance(tcond, str) and tcond == 'bool'):
                    raise TypeErrorBX("if expects bool condition")
                r1 = chk_s(s.thenb)
                r2 = False
                if s.elsep is not None:
                    r2 = chk_s(s.elsep)
                return r1 and (s.elsep is not None) and r2

            if isinstance(s, SWhile):
                tcond = chk_e(s.cond)
                if not (isinstance(tcond, str) and tcond == 'bool'):
                    raise TypeErrorBX("while expects bool condition")
                chk_s(s.body)
                return False

            if isinstance(s, SBreak): return False

            if isinstance(s, SContinue): return False

            if isinstance(s, SReturn):
                if ret_ty == 'void':
                    if s.e is not None:
                        raise TypeErrorBX("void function cannot return a value")
                else:
                    if s.e is None:
                        raise TypeErrorBX(f"non-void function must return a {ret_ty}")
                    t = chk_e(s.e)
                    if not (isinstance(t, str) and t == ret_ty):
                        raise TypeErrorBX(
                            f"return type mismatch: expected {ret_ty}, got {ty_repr(t)}"
                        )
                return True

            if isinstance(s, SProcDef):
                inner = s.proc

                if inner.name in fun_env_stack[-1]:
                    raise SemErrorBX(
                        f"Procedure '{inner.name}' redeclared in the same scope"
                    )

                inner_param_tys = [p.ty for p in inner.params]
                for t in inner_param_tys:
                    if not is_valid_param_type(t):
                        raise TypeErrorBX(
                            f"Invalid parameter type '{ty_repr(t)}' in nested "
                            f"procedure '{inner.name}'"
                        )
                if not is_first_order_type(inner.ret_ty):
                    raise TypeErrorBX(
                        f"Invalid return type '{ty_repr(inner.ret_ty)}' in nested "
                        f"procedure '{inner.name}'"
                    )

                inner_fun_ty = FunTy(tuple(inner_param_tys), inner.ret_ty)

                outer_frames_for_inner = [dict(frame) for frame in var_env_stack]

                inner_fun_env_stack = [dict(fenv) for fenv in fun_env_stack]

                all_ret_inner = typecheck_proc(
                    inner,
                    inner_fun_ty,
                    outer_frames_for_inner,
                    inner_fun_env_stack
                )

                if inner_fun_ty.ret_ty != 'void' and not all_ret_inner:
                    raise TypeErrorBX(
                        f"non-void procedure '{inner.name}' might not return a value "
                        f"on all paths"
                    )

                fun_env_stack[-1][inner.name] = inner_fun_ty
                return False

            raise TypeErrorBX("unknown stmt")

        all_paths_return = chk_s(pd.body)
        return all_paths_return

    # --- drive checking for each top-level procedure -------------------------
    for pd in prog.procs:
        fun_ty = fun_env_global[pd.name]
        outer_var_env: List[Dict[str, VarInfo]] = []
        fun_env_stack: List[Dict[str, FunTy]] = [dict(fun_env_global)]

        all_paths = typecheck_proc(pd, fun_ty, outer_var_env, fun_env_stack)
        if fun_ty.ret_ty != 'void' and not all_paths:
            raise TypeErrorBX(
                f"non-void procedure '{pd.name}' might not return a value on all paths"
            )


# =============================================================================
# DRIVER
# =============================================================================

def parse_text(src: str) -> Program:
    return parser.parse(src, lexer=lexer)

from typing import Dict  


def _build_vid_name_map(prog: Program) -> Dict[int, str]:
    """
    Build a mapping from variable IDs to their source names,
    by walking all parameters and local declarations.
    """
    vid2name: Dict[int, str] = {}

    def walk_stmt(s: Stmt):
        if isinstance(s, SVar):
            vid2name[s.vid] = s.name
        elif isinstance(s, SBlock):
            for st in s.ss:
                walk_stmt(st)
        elif isinstance(s, SIfElse):
            walk_stmt(s.thenb)
            if s.elsep is not None:
                walk_stmt(s.elsep)
        elif isinstance(s, SWhile):
            walk_stmt(s.body)
        elif isinstance(s, SProcDef):
            walk_proc(s.proc)

    def walk_proc(pd: ProcDecl):
        for param in pd.params:
            vid2name[param.vid] = param.name
        walk_stmt(pd.body)

    for pd in prog.procs:
        walk_proc(pd)

    return vid2name

def dump_captures(prog: Program) -> None:
    """
    Print, for each function (top-level and nested), which variables it captures,
    using source names rather than raw IDs.
    """
    vid2name = _build_vid_name_map(prog)

    def show_proc(pd: ProcDecl, indent: int = 0):
        pad = "  " * indent
        vids = sorted(pd.captures)
        names = [vid2name.get(v, f"vid{v}") for v in vids]
        if names:
            cap_str = ", ".join(names)
        else:
            cap_str = ""
        print(f"{pad}def {pd.name} captures {{{cap_str}}}")

        def walk_stmt(s: Stmt):
            if isinstance(s, SProcDef):
                show_proc(s.proc, indent + 1)
            elif isinstance(s, SBlock):
                for st in s.ss:
                    walk_stmt(st)
            elif isinstance(s, SIfElse):
                walk_stmt(s.thenb)
                if s.elsep is not None:
                    walk_stmt(s.elsep)
            elif isinstance(s, SWhile):
                walk_stmt(s.body)

        walk_stmt(pd.body)

    for pd in prog.procs:
        show_proc(pd)


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='input BX source file')
    ap.add_argument('--dump-ast', action='store_true')
    ap.add_argument('--dump-captures', action='store_true')
    args = ap.parse_args(argv)

    with open(args.file) as f:
        src = f.read()

    prog = parse_text(src)
    check_program(prog)

    base, _ = os.path.splitext(args.file)
    out_path = base + ".s"

    if args.dump_captures:
        dump_captures(prog)
    elif args.dump_ast:
        print(prog)
    else:
        with open(out_path, "w") as out:
            out.write(
                "    .text\n"
                "    .globl main\n"
                "main:\n"
                "    retq\n"
            )

if __name__ == '__main__':
    main(sys.argv[1:])


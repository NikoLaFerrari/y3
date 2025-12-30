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
    'ret':   'RET',
    'extern': 'EXTERN',
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
    globals: List[SVar]
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
    'program : decls'
    globals = []
    procs = []
    for d in p[1]:
        if isinstance(d, SVar):
            globals.append(d)
        else:
            procs.append(d)
    p[0] = Program(globals, procs)

def p_decls_one(p):
    'decls : decl'
    if isinstance(p[1], list): p[0] = p[1]
    else: p[0] = [p[1]]

def p_decls_many(p):
    'decls : decls decl'
    if isinstance(p[2], list): p[1].extend(p[2])
    else: p[1].append(p[2])
    p[0] = p[1]

def p_decl_proc(p):
    'decl : proc'
    p[0] = p[1]

def p_decl_global_vardecl(p):
    'decl : VAR var_inits COLON type SEMI'
    # identical to local, just stored as SVar statements at top-level
    p[0] = [SVar(name, init, p[4]) for (name, init) in p[2]]

def p_decl_extern(p):
    'decl : EXTERN DEF IDENT LPAREN params RPAREN SEMI'
    p[0] = ProcDecl(p[3], p[5], 'void', SBlock([]))

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
    'params : param_groups'
    p[0] = p[1]

# param_groups : one or more param_group separated by commas
def p_param_groups_one(p):
    'param_groups : param_group'
    p[0] = p[1]

def p_param_groups_cons(p):
    'param_groups : param_groups COMMA param_group'
    p[0] = p[1] + p[3]

# param_group : IDENT ("," IDENT)* ":" type
def p_param_group(p):
    'param_group : ident_list COLON type'
    p[0] = [Param(name, p[3]) for name in p[1]]

def p_ident_list_one(p):
    'ident_list : IDENT'
    p[0] = [p[1]]

def p_ident_list_cons(p):
    'ident_list : ident_list COMMA IDENT'
    p[1].append(p[3])
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


def p_stmt_procdef(p):
    'stmt : DEF IDENT LPAREN params RPAREN ret_annot block'
    p[0] = SProcDef(ProcDecl(p[2], p[4], p[6], p[7]))


def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    # p[2] can be a single Stmt or a list of Stmt (for multi-var decls)
    if isinstance(p[2], list):
        p[1].extend(p[2])
    else:
        p[1].append(p[2])
    p[0] = p[1]


def p_stmt_vardecl(p):
    'stmt : VAR var_inits COLON type SEMI'
    # p[2] is a list of (name, init_expr)
    vars = [SVar(name, init, p[4]) for (name, init) in p[2]]
    p[0] = vars   # NOTE: returns a *list* of statements


def p_var_inits_single(p):
    'var_inits : IDENT EQUAL expr'
    p[0] = [(p[1], p[3])]


def p_var_inits_many(p):
    'var_inits : var_inits COMMA IDENT EQUAL expr'
    p[1].append((p[3], p[5]))
    p[0] = p[1]

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

    # --- Global variable environment (top-level vars) ------------------------
    global_var_frame: Dict[str, VarInfo] = {}

    for gv in prog.globals:
        if gv.name in global_var_frame:
            raise SemErrorBX(f"Global variable '{gv.name}' redeclared")
        # Only int/bool globals for now
        if not (isinstance(gv.ty_annot, str) and gv.ty_annot in ('int', 'bool')):
            raise TypeErrorBX(
                f"Global variable '{gv.name}' must be of type int or bool"
            )
        vid = fresh_var_id()
        gv.vid = vid
        global_var_frame[gv.name] = (gv.ty_annot, vid)

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

    for pd in prog.procs:
        fun_ty = fun_env_global[pd.name]
        outer_var_env: List[Dict[str, VarInfo]] = []
        
        if global_var_frame:
            outer_var_env.append(global_var_frame)

        fun_env_stack: List[Dict[str, FunTy]] = [dict(fun_env_global)]

        all_paths = typecheck_proc(pd, fun_ty, outer_var_env, fun_env_stack)
        if fun_ty.ret_ty != 'void' and not all_paths:
            raise TypeErrorBX(
                f"non-void procedure '{pd.name}' might not return a value on all paths"
            )

# =============================================================================
# TAC IR (Extended for Higher-Order Functions)
# =============================================================================

class TacInstr:
    pass

@dc
class TacLabel(TacInstr):
    label: str

@dc
class TacBinOp(TacInstr):
    dst: str
    op: str
    lhs: str
    rhs: str

@dc
class TacUnOp(TacInstr):
    dst: str
    op: str
    src: str

@dc
class TacCopy(TacInstr):
    dst: str
    src: str

@dc
class TacJmp(TacInstr):
    target: str

@dc
class TacCJump(TacInstr):
    cond: str
    target_true: str
    target_false: str

@dc
class TacGetVar(TacInstr):
    """
    Load a variable into a temporary.
    dst = var(vid) found 'hops' levels up the static chain.
    If hops=0, it's a local access.
    """
    dst: str
    vid: int
    hops: int

@dc
class TacSetVar(TacInstr):
    """
    Store a value into a variable.
    var(vid) found 'hops' levels up = src.
    """
    vid: int
    hops: int
    src: str

@dc
class TacMakeClosure(TacInstr):
    """
    Create a fat pointer (closure).
    dst = (proc_label, static_link_ptr)
    
    The 'static_link_ptr' is usually the frame pointer of the CURRENT function,
    because the nested function being created is defined HERE.
    """
    dst: str
    proc_label: str
    hops: int  # Hops to find the frame that serves as the static link (usually 0)

@dc
class TacCall(TacInstr):
    """
    Function call.
    dst = call func_label(args) with static_link
    OR
    dst = call_indirect func_reg(args) with static_link attached to func_reg
    """
    dst: Optional[str]
    func: str          # Label (string) or Temp (variable)
    static_link: str   # Temp holding the static link to pass to the callee
    args: List[str]
    is_indirect: bool  # True if func is a temp (Fat Pointer)

@dc
class TacRet(TacInstr):
    val: Optional[str]

@dc
class TacProc:
    name: str  # Mangled Global Label
    params: List[str]
    body: List[TacInstr]
    is_main: bool = False
    
    # Metadata for backend
    vid_to_temp: Dict[int, str] = field(default_factory=dict)

# =============================================================================
# TAC GENERATOR (FIXED NESTED SCOPES)
# =============================================================================

class TacGenerator:
    """
    Lowers a typed Program to TAC with:
      - lexical scoping via static links,
      - fat pointers (codeptr, static_link) stored in 16-byte stack slots,
      - higher-order parameters passed as pointers to fat pointers,
      - indirect calls through fat pointers.

    Design choices:
      - Static link is passed *on the stack* as a hidden argument (8 bytes),
        followed by a dummy 8-byte word to preserve 16-byte alignment.
      - Every nested function has a mangled, globally-unique label.
      - Function name resolution is scope-aware (no global name collisions).
    """
    def __init__(self, prog: Program):
        self.prog = prog
        self.procs: List[TacProc] = []
        self.temp_counter = 0
        self.label_counter = 0

        # --- Analysis metadata (keyed by ProcDecl object id) ---
        self.proc_mangled_by_id: Dict[int, str] = {}
        self.proc_depth_by_id: Dict[int, int] = {}
        self.proc_parent_depth_by_id: Dict[int, int] = {}  # lexical parent depth (-1 for top-level)
        self.vid_depth: Dict[int, int] = {}                # vid -> defining procedure depth

        # Current procedure context (during generation)
        self.current_depth: int = 0

        # Variable environment stack: name -> vid (shared across nested compilation)
        self.env_stack: List[Dict[str, int]] = []

    # --- fresh names ----------------------------------------------------------

    def fresh_temp(self) -> str:
        t = f"%t{self.temp_counter}"
        self.temp_counter += 1
        return t

    def fresh_label(self, suffix: str = "L") -> str:
        l = f".{suffix}{self.label_counter}"
        self.label_counter += 1
        return l

    # --- analysis -------------------------------------------------------------

    def run_analysis(self) -> None:
        """
        Assign:
          - unique mangled labels to every ProcDecl node,
          - lexical depth for procedures,
          - defining depth for every vid (params + locals).
        """
        def walk(pd: ProcDecl, depth: int, prefix: str):
            pid = id(pd)
            mangled = "main" if pd.name == "main" and depth == 0 else (prefix + pd.name)
            self.proc_mangled_by_id[pid] = mangled
            self.proc_depth_by_id[pid] = depth
            self.proc_parent_depth_by_id[pid] = depth - 1  # lexical parent depth

            # params belong to this procedure depth
            for p in pd.params:
                self.vid_depth[p.vid] = depth

            def scan_stmt(s: Stmt):
                if isinstance(s, SBlock):
                    for sub in s.ss:
                        scan_stmt(sub)
                elif isinstance(s, SVar):
                    self.vid_depth[s.vid] = depth
                elif isinstance(s, SIfElse):
                    scan_stmt(s.thenb)
                    if s.elsep:
                        scan_stmt(s.elsep)
                elif isinstance(s, SWhile):
                    scan_stmt(s.body)
                elif isinstance(s, SProcDef):
                    # nested procedure: depth+1, prefix extends with current mangled + "$"
                    walk(s.proc, depth + 1, mangled + "$")

            scan_stmt(pd.body)

        for top in self.prog.procs:
            walk(top, 0, "")

    # --- variable env helpers -------------------------------------------------

    def _env_push(self):
        self.env_stack.append({})

    def _env_pop(self):
        self.env_stack.pop()

    def _env_decl(self, name: str, vid: int):
        self.env_stack[-1][name] = vid

    def _env_lookup(self, name: str) -> Optional[int]:
        for frame in reversed(self.env_stack):
            if name in frame:
                return frame[name]
        return None

    # --- public entry ---------------------------------------------------------

    def gen_program(self) -> List[TacProc]:
        self.run_analysis()

        # Global function environment: source-name -> ProcDecl (top-level only)
        global_fun_env: Dict[str, ProcDecl] = {pd.name: pd for pd in self.prog.procs}

        for pd in self.prog.procs:
            self.gen_proc(pd, fun_env_stack=[global_fun_env])
        return self.procs

    # --- core compilation -----------------------------------------------------

    def gen_proc(self, pd: ProcDecl, fun_env_stack: List[Dict[str, ProcDecl]]):
        """
        Compile one procedure and recursively compile nested procedures.
        fun_env_stack obeys lexical scoping and top-down (no recursion for inner defs).
        """
        pid = id(pd)
        mangled_name = self.proc_mangled_by_id[pid]
        self.current_depth = self.proc_depth_by_id[pid]

        body_instrs: List[TacInstr] = []

        def emit(i: TacInstr):
            body_instrs.append(i)

        # Params are represented as stack slots (by vid) and are accessed with TacGetVar/TacSetVar.
        # We record them in TacProc.params so the backend can move ABI args into those slots.
        tac_params = [f"%v_{p.vid}" for p in pd.params]

        # Enter parameter scope for variables
        self._env_push()
        for p in pd.params:
            self._env_decl(p.name, p.vid)

        # Local loop stack (does not cross procedure boundaries)
        loop_stack: List[Tuple[str, str]] = []

        # Local variable ids (for hop computation only)
        def get_hops_for_vid(vid: int) -> int:
            def_depth = self.vid_depth[vid]
            return self.current_depth - def_depth

        # --- function lookup (lexical) ---------------------------------------

        def lookup_fun(name: str) -> Optional[ProcDecl]:
            for fenv in reversed(fun_env_stack):
                if name in fenv:
                    return fenv[name]
            return None

        def proc_depth(pd_node: ProcDecl) -> int:
            return self.proc_depth_by_id[id(pd_node)]

        def proc_parent_depth(pd_node: ProcDecl) -> int:
            return self.proc_parent_depth_by_id[id(pd_node)]

        def proc_mangled(pd_node: ProcDecl) -> str:
            return self.proc_mangled_by_id[id(pd_node)]

        # --- closure creation -------------------------------------------------

        def make_closure_for(pd_node: ProcDecl) -> str:
            """
            Return a temp holding the address of a 16-byte fat pointer (code, static_link).
            The static_link is a pointer to the frame of pd_node's lexical parent *at this point*.
            """
            dst = self.fresh_temp()
            d = proc_depth(pd_node)
            if d == 0:
                emit(TacMakeClosure(dst, proc_mangled(pd_node), -1))
                return dst

            parent_d = proc_parent_depth(pd_node)  # = d-1
            hops = self.current_depth - parent_d
            emit(TacMakeClosure(dst, proc_mangled(pd_node), hops))
            return dst

        # --- expression compiler ---------------------------------------------

        def compile_expr(e: Expr) -> str:
            if isinstance(e, ENum):
                t = self.fresh_temp()
                emit(TacCopy(t, str(e.n)))
                return t

            if isinstance(e, EBool):
                t = self.fresh_temp()
                emit(TacCopy(t, "1" if e.b else "0"))
                return t

            if isinstance(e, EVar):
                # variables (including higher-order params) live in var env
                vid = self._env_lookup(e.name)
                if vid is not None:
                    t = self.fresh_temp()
                    emit(TacGetVar(t, vid, get_hops_for_vid(vid)))
                    return t

                # otherwise, treat as a function value (fat pointer)
                fn = lookup_fun(e.name)
                if fn is None:
                    raise SemErrorBX(f"Unknown identifier '{e.name}'")
                return make_closure_for(fn)

            if isinstance(e, EUn):
                src = compile_expr(e.e)
                dst = self.fresh_temp()
                emit(TacUnOp(dst, e.op, src))
                return dst

            if isinstance(e, EBin):
                l = compile_expr(e.l)
                r = compile_expr(e.r)
                dst = self.fresh_temp()
                emit(TacBinOp(dst, e.op, l, r))
                return dst

            if isinstance(e, ECall):
                # compile args first
                args = [compile_expr(a) for a in e.args]
                dst = self.fresh_temp() if e.ty != 'void' else None

                # If callee resolves to a variable (only possible for higher-order params),
                # then do indirect call through fat pointer.
                vid = self._env_lookup(e.name)
                if vid is not None:
                    func_ptr = self.fresh_temp()
                    emit(TacGetVar(func_ptr, vid, get_hops_for_vid(vid)))  # value is addr of fat ptr
                    emit(TacCall(dst, func_ptr, "0", args, is_indirect=True))
                    return dst

                # Otherwise, direct call to a known procedure name in scope
                fn = lookup_fun(e.name)
                if fn is None:
                    raise SemErrorBX(f"Call to unknown function '{e.name}'")

                callee_mangled = proc_mangled(fn)
                d = proc_depth(fn)
                if d == 0:
                    sl = "0"
                else:
                    parent_d = proc_parent_depth(fn)
                    hops = self.current_depth - parent_d
                    sl = f"@sl{hops}"
                emit(TacCall(dst, callee_mangled, sl, args, is_indirect=False))
                return dst

            raise NotImplementedError(f"Expr lowering not implemented for {type(e).__name__}")

        # --- statement compiler ----------------------------------------------

        def compile_stmt(s: Stmt, local_fun_env: Dict[str, ProcDecl]):
            if isinstance(s, SBlock):
                self._env_push()
                # new lexical scope for functions too
                block_fun_env: Dict[str, ProcDecl] = dict(local_fun_env)
                for sub in s.ss:
                    compile_stmt(sub, block_fun_env)
                self._env_pop()
                return

            if isinstance(s, SVar):
                val_t = compile_expr(s.init)
                self._env_decl(s.name, s.vid)
                emit(TacSetVar(s.vid, 0, val_t))
                return

            if isinstance(s, SAssign):
                vid = self._env_lookup(s.name)
                if vid is None:
                    raise SemErrorBX(f"Assignment to unknown variable '{s.name}'")
                val_t = compile_expr(s.e)
                emit(TacSetVar(vid, get_hops_for_vid(vid), val_t))
                return

            if isinstance(s, SExpr):
                compile_expr(s.e)
                return

            if isinstance(s, SProcDef):
                inner = s.proc

                # Top-down / no recursion: compile inner with current function env (without itself)
                inner_fun_env_stack = fun_env_stack + [dict(local_fun_env)]
                self.gen_proc(inner, fun_env_stack=inner_fun_env_stack)

                # After its definition, it becomes visible in this scope
                local_fun_env[inner.name] = inner
                return

            if isinstance(s, SIfElse):
                l_then = self.fresh_label("then")
                l_else = self.fresh_label("else")
                l_end = self.fresh_label("endif")
                c = compile_expr(s.cond)
                emit(TacCJump(c, l_then, l_else))
                emit(TacLabel(l_then))
                compile_stmt(s.thenb, dict(local_fun_env))
                emit(TacJmp(l_end))
                emit(TacLabel(l_else))
                if s.elsep is not None:
                    compile_stmt(s.elsep, dict(local_fun_env))
                emit(TacLabel(l_end))
                return

            if isinstance(s, SWhile):
                l_start = self.fresh_label("while")
                l_body = self.fresh_label("body")
                l_end = self.fresh_label("endwhile")
                emit(TacLabel(l_start))
                c = compile_expr(s.cond)
                emit(TacCJump(c, l_body, l_end))
                emit(TacLabel(l_body))
                loop_stack.append((l_start, l_end))
                compile_stmt(s.body, dict(local_fun_env))
                loop_stack.pop()
                emit(TacJmp(l_start))
                emit(TacLabel(l_end))
                return

            if isinstance(s, SBreak):
                if not loop_stack:
                    raise SemErrorBX("break outside of loop")
                _, l_end = loop_stack[-1]
                emit(TacJmp(l_end))
                return

            if isinstance(s, SContinue):
                if not loop_stack:
                    raise SemErrorBX("continue outside of loop")
                l_start, _ = loop_stack[-1]
                emit(TacJmp(l_start))
                return

            if isinstance(s, SReturn):
                v = compile_expr(s.e) if s.e else None
                emit(TacRet(v))
                return

            raise NotImplementedError(f"Stmt lowering not implemented for {type(s).__name__}")

        # Compile procedure body with a fresh local fun env (siblings become visible as we go)
        local_fun_env: Dict[str, ProcDecl] = {}
        compile_stmt(pd.body, local_fun_env)

        # Ensure a return exists
        if not body_instrs or not isinstance(body_instrs[-1], TacRet):
            emit(TacRet(None))

        tp = TacProc(mangled_name, tac_params, body_instrs, is_main=(pd.name == "main" and self.current_depth == 0))
        self.procs.append(tp)

        # Exit parameter scope
        self._env_pop()
# ASSEMBLY GENERATION (x86_64) - REFACTORED & OPTIMIZED
# =============================================================================

class AsmGen:
    def __init__(self, procs: List[TacProc]):
        self.procs = procs
        self.output: List[str] = []
        self.slots: Dict[str, int] = {}
        self.current_stack_size = 0
        self.vid_offsets: Dict[int, int] = {}

    def emit(self, line: str):
        self.output.append(line)

    def gen_program(self) -> str:
        self.emit(f"    .text")
        for p in self.procs:
            if p.is_main:
                self.emit(f"    .globl main")
        
        for p in self.procs:
            self.gen_proc(p)
            
        #self.gen_runtime()
        return "\n".join(self.output) + "\n"

    def _walk_static_link(self, hops: int, reg: str):
        """Helper to walk up the static link chain 'hops' times into 'reg'."""
        if hops == 0:
            self.emit(f"    movq %rbp, {reg}")
        else:
            self.emit(f"    movq 16(%rbp), {reg}")
            for _ in range(hops - 1):
                self.emit(f"    movq 16({reg}), {reg}")

    def gen_proc(self, proc: TacProc):
        self.emit(f"\n{proc.name}:")
        self.emit(f"    pushq %rbp")
        self.emit(f"    movq %rsp, %rbp")
        
        # --- Stack Analysis ---
        # We allocate stack slots for:
        #   1) BX parameters (one 8-byte slot each, addressed by a temp name %v_<vid>)
        #   2) BX locals (each vid with hops==0 in TacGetVar/TacSetVar)
        #   3) TAC temporaries
        #   4) Fat pointers (16-byte slots) for TacMakeClosure
        self.slots = {}
        offset = 0

        # 1) Map parameters (and record their VID offsets for later TacGetVar/TacSetVar)
        for param in proc.params:
            offset += 8
            self.slots[param] = -offset
            if param.startswith("%v_"):
                try:
                    vid = int(param.split("_")[-1])
                    self.vid_offsets.setdefault(vid, -offset)
                except:
                    pass

        # 2) Map local variable VIDs (hops==0 means "this frame")
        local_vids: List[int] = []
        seen_vids: Set[int] = set()
        for instr in proc.body:
            if isinstance(instr, (TacGetVar, TacSetVar)) and instr.hops == 0 and instr.vid >= 0:
                if instr.vid not in seen_vids:
                    seen_vids.add(instr.vid)
                    local_vids.append(instr.vid)

        for vid in local_vids:
            # allocate only if not already allocated (params)
            if vid not in self.vid_offsets:
                offset += 8
                self.vid_offsets[vid] = -offset
            # ensure frame reserves enough space even if precomputed earlier
            else:
                need = -self.vid_offsets[vid]
                if need > offset:
                    offset = need

        # 3) Map temporaries
        closure_count = 0
        for instr in proc.body:
            if hasattr(instr, 'dst') and instr.dst and instr.dst not in self.slots:
                offset += 8
                self.slots[instr.dst] = -offset
            if isinstance(instr, TacMakeClosure):
                closure_count += 1

        # 4) Allocate closure slots (16 bytes each)
        closure_base_offsets = []
        for _ in range(closure_count):
            offset += 16
            closure_base_offsets.append(-offset)

        # 5) Align frame size to 16 bytes
        if offset % 16 != 0:
            offset += (16 - (offset % 16))

        self.current_stack_size = offset
        self.emit(f"    subq ${offset}, %rsp")
        # --- Move Arguments ---
        args_regs = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
        for i, param in enumerate(proc.params):
            off = self.slots[param]
            if i < len(args_regs):
                self.emit(f"    movq {args_regs[i]}, {off}(%rbp)")
            else:
                src_off = 32 + (i - 6) * 8  # +16 for (static_link,dummy)
                self.emit(f"    movq {src_off}(%rbp), %rax")
                self.emit(f"    movq %rax, {off}(%rbp)")

        # --- Body Generation ---
        closure_idx = 0
        
        for instr in proc.body:
            if isinstance(instr, TacLabel):
                self.emit(f"{instr.label}:")
                
            elif isinstance(instr, TacCopy):
                self.load_operand(instr.src, '%rax')
                self.store_operand(instr.dst, '%rax')
                
            elif isinstance(instr, TacBinOp):
                self.load_operand(instr.lhs, '%rax')
                self.load_operand(instr.rhs, '%rcx')
                
                # Dictionary mapping for simple arithmetic
                arith_ops = {
                    '+': 'addq', '-': 'subq', '*': 'imulq',
                    '&': 'andq', '|': 'orq', '^': 'xorq'
                }
                
                if instr.op in arith_ops:
                    self.emit(f"    {arith_ops[instr.op]} %rcx, %rax")
                    
                elif instr.op in ('<<', '>>'):
                    op_asm = 'shlq' if instr.op == '<<' else 'sarq'
                    self.emit(f"    {op_asm} %cl, %rax")
                    
                elif instr.op in ('/', '%'):
                    self.emit(f"    cqto")
                    self.emit(f"    idivq %rcx")
                    if instr.op == '%':
                        self.emit(f"    movq %rdx, %rax")
                        
                elif instr.op in ('==', '!=', '<', '<=', '>', '>='):
                    self.emit(f"    cmpq %rcx, %rax")
                    cc_map = {
                        '==': 'e', '!=': 'ne', '<': 'l', 
                        '<=': 'le', '>': 'g', '>=': 'ge'
                    }
                    self.emit(f"    set{cc_map[instr.op]} %al")
                    self.emit(f"    movzbq %al, %rax")
                
                self.store_operand(instr.dst, '%rax')

            elif isinstance(instr, TacUnOp):
                self.load_operand(instr.src, '%rax')
                if instr.op == '-':
                    self.emit(f"    negq %rax")
                elif instr.op == '!':
                    self.emit(f"    xorq $1, %rax")
                elif instr.op == '~':
                    self.emit(f"    notq %rax")
                self.store_operand(instr.dst, '%rax')
                
            elif isinstance(instr, TacJmp):
                self.emit(f"    jmp {instr.target}")
                
            elif isinstance(instr, TacCJump):
                self.load_operand(instr.cond, '%rax')
                self.emit(f"    testq %rax, %rax")
                self.emit(f"    jnz {instr.target_true}")
                self.emit(f"    jmp {instr.target_false}")

            elif isinstance(instr, (TacGetVar, TacSetVar)):
                # Unified logic for accessing variables via static links
                self._walk_static_link(instr.hops, '%rax')
                off = self.get_vid_offset(instr.vid)
                
                if isinstance(instr, TacGetVar):
                    self.emit(f"    movq {off}(%rax), %rcx")
                    self.store_operand(instr.dst, '%rcx')
                else: # TacSetVar
                    self.load_operand(instr.src, '%rcx')
                    self.emit(f"    movq %rcx, {off}(%rax)")

            elif isinstance(instr, TacMakeClosure):
                base_off = closure_base_offsets[closure_idx]
                closure_idx += 1

                # code pointer
                self.emit(f"    leaq {instr.proc_label}(%rip), %rax")
                self.emit(f"    movq %rax, {base_off}(%rbp)")

                # static link pointer
                if instr.hops == -1:
                    self.emit(f"    movq $0, {base_off + 8}(%rbp)")
                else:
                    # walk up 'instr.hops' static links from current frame to reach
                    # the callee's lexical parent frame.
                    self._walk_static_link(instr.hops, '%r11')
                    self.emit(f"    movq %r11, {base_off + 8}(%rbp)")

                # dst gets the address of the fat pointer on the stack
                self.emit(f"    leaq {base_off}(%rbp), %rax")
                self.store_operand(instr.dst, '%rax')

            elif isinstance(instr, TacCall):
                # Resolve callee and its static link into %rax (code) and %r10 (static link)
                if instr.is_indirect:
                    # instr.func is a temp holding the *address* of a fat pointer
                    self.load_operand(instr.func, '%r11')
                    self.emit(f"    movq 0(%r11), %rax")   # code ptr
                    self.emit(f"    movq 8(%r11), %r10")   # static link
                else:
                    self.emit(f"    leaq {instr.func}(%rip), %rax")
                    if instr.static_link == "0":
                        self.emit(f"    movq $0, %r10")
                    elif isinstance(instr.static_link, str) and instr.static_link.startswith("@sl"):
                        hops = int(instr.static_link[3:])
                        self._walk_static_link(hops, '%r10')
                    else:
                        self.load_operand(instr.static_link, '%r10')

                # Push stack arguments beyond 6 (right-to-left) first
                regs = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
                stack_args = instr.args[6:]
                for arg in reversed(stack_args):
                    self.load_operand(arg, '%r11')
                    self.emit(f"    pushq %r11")

                # Push dummy (alignment) then static link (callee sees it at 16(%rbp))
                self.emit(f"    pushq $0")
                self.emit(f"    pushq %r10")

                # Move first 6 args into registers
                for i, arg in enumerate(instr.args[:6]):
                    self.load_operand(arg, regs[i])

                # Call
                self.emit(f"    call *%rax")

                # Pop (static_link + dummy + stack args)
                cleanup = 16 + 8 * len(stack_args)
                if cleanup:
                    self.emit(f"    addq ${cleanup}, %rsp")

                if instr.dst:
                    self.store_operand(instr.dst, '%rax')

            elif isinstance(instr, TacRet):
                if instr.val:
                    self.load_operand(instr.val, '%rax')
                elif proc.is_main:
                    self.emit(f"    movq $0, %rax")
                self.emit(f"    leave")
                self.emit(f"    ret")

    def load_operand(self, op: str, reg: str):
        if op[0].isdigit() or op[0] == '-':
            self.emit(f"    movq ${op}, {reg}")
        else:
            off = self.slots.get(op)
            if off is None: raise ValueError(f"Missing slot for {op}")
            self.emit(f"    movq {off}(%rbp), {reg}")

    def store_operand(self, op: str, reg: str):
        off = self.slots.get(op)
        if off is None: raise ValueError(f"Missing slot for {op}")
        self.emit(f"    movq {reg}, {off}(%rbp)")

    def get_vid_offset(self, vid: int) -> int:
        if vid not in self.vid_offsets:
            raise ValueError(f"Missing stack offset for vid {vid}")
        return self.vid_offsets[vid]
        
    def precompute_offsets(self):
        """Pre-pass: assign a stable stack offset for every VID in its defining procedure frame."""
        for proc in self.procs:
            off = 0
            # params
            for p in proc.params:
                off += 8
                if p.startswith("%v_"):
                    try:
                        vid = int(p.split("_")[-1])
                        self.vid_offsets.setdefault(vid, -off)
                    except:
                        pass

            # locals (VIDs with hops==0 in this procedure)
            seen: Set[int] = set()
            for instr in proc.body:
                if isinstance(instr, (TacGetVar, TacSetVar)) and instr.hops == 0 and instr.vid >= 0:
                    if instr.vid in seen: 
                        continue
                    seen.add(instr.vid)
                    if instr.vid not in self.vid_offsets:
                        off += 8
                        self.vid_offsets[instr.vid] = -off

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
    Debug helper: prints capture sets for every procedure (including nested),
    using VIDs (unique ids) plus source names when available.
    """
    vid2name = _build_vid_name_map(prog)

    def ty_str(ty: Ty) -> str:
        if isinstance(ty, str):
            return ty
        if isinstance(ty, FunTy):
            ps = ", ".join(ty_str(t) for t in ty.param_tys)
            return f"function({ps}) -> {ty.ret_ty}"
        return str(ty)

    def walk(pd: ProcDecl, indent: int = 0):
        pad = "  " * indent
        params = ", ".join(f"{p.name}:{ty_str(p.ty)}(vid={p.vid})" for p in pd.params)
        print(f"{pad}{pd.name}  depth?  params[{params}]  ret:{ty_str(pd.ret_ty)}")
        if pd.captures:
            caps = ", ".join(f"{vid2name.get(v, '?')}#{v}" for v in sorted(pd.captures))
        else:
            caps = ""
        print(f"{pad}  captures: {{{caps}}}")

        def scan_stmt(s: Stmt):
            if isinstance(s, SBlock):
                for st in s.ss:
                    scan_stmt(st)
            elif isinstance(s, SProcDef):
                walk(s.proc, indent + 1)
                scan_stmt(s.proc.body)
            elif isinstance(s, SIfElse):
                scan_stmt(s.thenb)
                if s.elsep:
                    scan_stmt(s.elsep)
            elif isinstance(s, SWhile):
                scan_stmt(s.body)

        scan_stmt(pd.body)

    for top in prog.procs:
        walk(top, 0)

# =============================================================================
# DRIVER
# =============================================================================

def parse_text(src: str) -> Program:
    return parser.parse(src, lexer=lexer)

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='input BX source file')
    ap.add_argument('--dump-ast', action='store_true')
    ap.add_argument('--dump-captures', action='store_true')
    ap.add_argument('--dump-tac', action='store_true')
    args = ap.parse_args(argv)

    try:
        with open(args.file) as f:
            src = f.read()

        # Phase 1: parse + typecheck
        prog = parse_text(src)
        check_program(prog)

        # Phase 2: TAC
        vid_map = _build_vid_name_map(prog)
        tac_gen = TacGenerator(prog)
        tac_procs = tac_gen.gen_program()

        # Phase 3: output
        if args.dump_captures:
            dump_captures(prog)
        elif args.dump_ast:
            print(prog)
        elif args.dump_tac:
            for proc in tac_procs:
                print(f"PROC {proc.name}:")
                for instr in proc.body:
                    print(f"  {instr}")
                print()
        else:
            asm_gen = AsmGen(tac_procs)
            asm_gen.precompute_offsets()
            asm_code = asm_gen.gen_program()

            base, _ = os.path.splitext(args.file)
            with open(base + ".s", "w") as out:
                out.write(asm_code)

    except Exception as e:
        # Make all compiler errors show up as "Error: ..."
        print(f"Error: {e}")
        return

if __name__ == '__main__':
    main(sys.argv[1:])

#!/usr/bin/env python3
import sys, os, argparse, dataclasses as dc, abc
from typing import List, Tuple, Dict, Optional, Set
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

reserved = {
    'def':   'DEF',
    'var':   'VAR',
    'print': 'PRINT',
    'int':   'INT',
    'bool':  'BOOL',
    'true':  'TRUE',
    'false': 'FALSE',
    'if':    'IF',
    'else':  'ELSE',
    'while': 'WHILE',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'return': 'RETURN',
}

tokens = (
    # id/lits
    'IDENT', 'NUMBER',
    # punct
    'LPAREN','RPAREN','LBRACE','RBRACE',
    'COLON','SEMI','COMMA','EQUAL','ARROW',
    # bitwise / arith
    'PLUS','MINUS','STAR','SLASH','MOD',
    'BAND','BOR','BXOR',
    'LSHIFT','RSHIFT',
    'BNOT',
    # logical & comparisons
    'LNOT', 'LAND', 'LOR',
    'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
) + tuple(reserved.values())

t_PLUS   = r'\+'
t_MINUS  = r'-'
t_STAR   = r'\*'
t_SLASH  = r'/'
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

t_LNOT = r'!'
t_LAND = r'&&'
t_LOR  = r'\|\|'
t_EQ   = r'=='
t_NE   = r'!='
t_LE   = r'<='
t_LT   = r'<'
t_GE   = r'>='
t_GT   = r'>'

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
        val = int(t.value)
    except ValueError:
        err("Lexer", f"bad integer literal {t.value!r}")
        t.lexer.error = True
        return None
    if not (0 <= val < (1<<63)):
        err("Lexer", f"integer {val} out of range [0, 2^63)")
        t.lexer.error = True
        return None
    t.value = val
    return t

def t_error(t):
    err("Lexer", f"line {t.lineno}, pos {t.lexpos}: illegal char {t.value[0]!r}")
    t.lexer.skip(1)

lexer = lex.lex()

# =============================================================================
# AST
# =============================================================================

@dc.dataclass
class AST(abc.ABC):
    pass

# Types
Ty = str  # 'int' | 'bool' | 'void'

# Expressions
class Expr(AST): pass

@dc.dataclass
class ENum(Expr):
    n: int
    ty: Ty = "int"

@dc.dataclass
class EBool(Expr):
    b: bool
    ty: Ty = "bool"

@dc.dataclass
class EVar(Expr):
    name: str
    ty: Optional[Ty] = None  # after typecheck: 'int'

@dc.dataclass
class EUn(Expr):
    op: str
    e: Expr
    ty: Optional[Ty] = None

@dc.dataclass
class EBin(Expr):
    op: str
    l: Expr
    r: Expr
    ty: Optional[Ty] = None

@dc.dataclass
class ECall(Expr):
    name: str
    args: List[Expr]
    ty: Optional[Ty] = None

# Statements
class Stmt(AST): pass

@dc.dataclass
class SVar(Stmt):
    name: str
    init: Expr
    ty_annot: Ty  # still 'int' in Lab4 for vars

@dc.dataclass
class SAssign(Stmt):
    name: str
    rhs: Expr

@dc.dataclass
class SExpr(Stmt):
    e: Expr  # expression-as-statement (e.g., call with void return)

@dc.dataclass
class SBlock(Stmt):
    ss: List[Stmt]

@dc.dataclass
class SIfElse(Stmt):
    cond: Expr
    thenb: SBlock
    elsep: Optional[Stmt]

@dc.dataclass
class SWhile(Stmt):
    cond: Expr
    body: SBlock

@dc.dataclass
class SBreak(Stmt): pass

@dc.dataclass
class SContinue(Stmt): pass

@dc.dataclass
class SReturn(Stmt):
    e: Optional[Expr]  # None for 'return;'

# Procedures & Program
@dc.dataclass
class Param(AST):
    name: str
    ty: Ty

@dc.dataclass
class ProcDecl(AST):
    name: str
    params: List[Param]
    ret_ty: Ty
    body: SBlock

@dc.dataclass
class Program(AST):
    procs: List[ProcDecl]

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

def p_program(p):
    'program : procs'
    p[0] = Program(p[1])

def p_procs_one(p):
    'procs : proc'
    p[0] = [p[1]]

def p_procs_many(p):
    'procs : procs proc'
    p[1].append(p[2]); p[0] = p[1]

def p_proc(p):
    'proc : DEF IDENT LPAREN params RPAREN ret_annot block'
    p[0] = ProcDecl(name=p[2], params=p[4], ret_ty=p[6], body=p[7])

def p_params_empty(p):
    'params : '
    p[0] = []

def p_params_list(p):
    'params : param_list'
    p[0] = p[1]

def p_param_list_one(p):
    'param_list : IDENT COLON sigty'
    p[0] = [Param(p[1], p[3])]

def p_param_list_cons(p):
    'param_list : param_list COMMA IDENT COLON sigty'
    p[1].append(Param(p[3], p[5])); p[0] = p[1]

def p_sigty_int(p):
    'sigty : INT'
    p[0] = 'int'

def p_sigty_bool(p):
    'sigty : BOOL'
    p[0] = 'bool'

def p_ret_annot_void(p):
    'ret_annot : '
    p[0] = 'void'

def p_ret_annot_ty(p):
    'ret_annot : COLON sigty'
    p[0] = p[2]

def p_block(p):
    'block : LBRACE stmt_list RBRACE'
    p[0] = SBlock(p[2])

def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = []

def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    p[1].append(p[2]); p[0] = p[1]

# Variable declaration (Lab4: keep int locals; allow : int only)
def p_stmt_vardecl(p):
    'stmt : VAR IDENT EQUAL expr COLON INT SEMI'
    p[0] = SVar(p[2], p[4], 'int')

def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = SAssign(p[1], p[3])

# "print(expr);" sugar → we keep it; typechecker will rewrite to ECall(__bx_print_*)
def p_stmt_print(p):
    'stmt : PRINT LPAREN expr RPAREN SEMI'
    p[0] = SExpr(ECall('print', [p[3]]))

def p_stmt_expr(p):
    'stmt : expr SEMI'
    p[0] = SExpr(p[1])

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

def p_ifrest_else_stmt(p):
    'ifrest : ELSE stmt'
    p[0] = p[2]

def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN block'
    p[0] = SWhile(p[3], p[5])

def p_stmt_return_void(p):
    'stmt : RETURN SEMI'
    p[0] = SReturn(None)

def p_stmt_return_val(p):
    'stmt : RETURN expr SEMI'
    p[0] = SReturn(p[2])

def p_stmt_block(p):
    'stmt : block'
    p[0] = p[1]

# Expressions
def p_expr_num(p):
    'expr : NUMBER'
    p[0] = ENum(p[1])

def p_expr_true(p):
    'expr : TRUE'
    p[0] = EBool(True)

def p_expr_false(p):
    'expr : FALSE'
    p[0] = EBool(False)

def p_expr_ident(p):
    'expr : IDENT'
    p[0] = EVar(p[1])

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

def p_expr_call(p):
    'expr : IDENT LPAREN arglist RPAREN'
    p[0] = ECall(p[1], p[3])

def p_arglist_empty(p):
    'arglist : '
    p[0] = []

def p_arglist_list(p):
    'arglist : args'
    p[0] = p[1]

def p_args_one(p):
    'args : expr'
    p[0] = [p[1]]

def p_args_cons(p):
    'args : args COMMA expr'
    p[1].append(p[3]); p[0] = p[1]

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
        err("Parser", f"line {getattr(p,'lineno','?')}, pos {getattr(p,'lexpos','?')}: unexpected token {p.type} ({p.value!r})")
    raise SyntaxError

parser = yacc.yacc(start='program')

# =============================================================================
# TYPE CHECKING
# =============================================================================

class TypeErrorBX(Exception): pass
class SemErrorBX(Exception): pass

def check_program(prog: Program):
    # Procedure symbol table: name -> (param_types, ret_ty)
    procs: Dict[str, Tuple[List[Ty], Ty]] = {}

    # Predeclare print specializations (we will map 'print' to one of these)
    procs['__bx_print_int']  = (['int'],  'void')
    procs['__bx_print_bool'] = (['bool'], 'void')

    # 1) Collect procedure headers (no dups)
    for pd in prog.procs:
        if pd.name in procs:
            raise SemErrorBX(f"Procedure '{pd.name}' redeclared")
        param_tys = [p.ty for p in pd.params]
        for t in param_tys:
            if t not in ('int','bool'):
                raise TypeErrorBX(f"Invalid parameter type '{t}' in '{pd.name}'")
        if pd.ret_ty not in ('int','bool','void'):
            raise TypeErrorBX(f"Invalid return type '{pd.ret_ty}' in '{pd.name}'")
        procs[pd.name] = (param_tys, pd.ret_ty)

    if 'main' not in procs:
        raise SemErrorBX("Entry point 'main' missing")

    # 2) Check bodies
    def chk_proc(pd: ProcDecl):
        param_tys, ret_ty = procs[pd.name]

        # Block scopes for locals
        scope_stack: List[Dict[str,Ty]] = [ { p.name: p.ty for p in pd.params } ]

        def in_scope(name: str) -> Optional[Ty]:
            for d in reversed(scope_stack):
                if name in d: return d[name]
            return None

        def add_local(name: str, ty: Ty):
            if name in scope_stack[-1]:
                raise SemErrorBX(f"Variable '{name}' redeclared in the same block")
            scope_stack[-1][name] = ty

        def chk_e(e: Expr) -> Ty:
            if isinstance(e, ENum): e.ty = 'int'; return 'int'
            if isinstance(e, EBool): e.ty = 'bool'; return 'bool'
            if isinstance(e, EVar):
                ty = in_scope(e.name)
                if ty is None:
                    raise SemErrorBX(f"Use of undeclared variable '{e.name}'")
                e.ty = ty
                return ty
            if isinstance(e, EUn):
                t = chk_e(e.e)
                if e.op == '!':
                    if t != 'bool': raise TypeErrorBX("operator ! expects bool")
                    e.ty = 'bool'; return 'bool'
                if e.op in ('-','~'):
                    if t != 'int': raise TypeErrorBX(f"operator {e.op} expects int")
                    e.ty = 'int'; return 'int'
                raise TypeErrorBX(f"unknown unary {e.op}")
            if isinstance(e, EBin):
                if e.op in ('&&','||'):
                    tl = chk_e(e.l); tr = chk_e(e.r)
                    if tl!='bool' or tr!='bool': raise TypeErrorBX(f"{e.op} expects bool && bool")
                    e.ty = 'bool'; return 'bool'
                if e.op in ('==','!=','<','<=','>','>='):
                    tl = chk_e(e.l); tr = chk_e(e.r)
                    if tl!='int' or tr!='int': raise TypeErrorBX(f"{e.op} compares ints")
                    e.ty = 'bool'; return 'bool'
                # arithmetic/bitwise require ints
                tl = chk_e(e.l); tr = chk_e(e.r)
                if tl!='int' or tr!='int': raise TypeErrorBX(f"{e.op} expects ints")
                e.ty = 'int'; return 'int'
            if isinstance(e, ECall):
                # Specialize 'print' → __bx_print_int/bool
                if e.name == 'print':
                    if len(e.args) != 1:
                        raise TypeErrorBX("print expects exactly one argument")
                    aty = chk_e(e.args[0])
                    if aty == 'int':
                        e.name = '__bx_print_int'
                    elif aty == 'bool':
                        e.name = '__bx_print_bool'
                    else:
                        raise TypeErrorBX("print expects int or bool")
                # generic call
                if e.name not in procs:
                    raise SemErrorBX(f"Call to unknown procedure '{e.name}'")
                param_ts, rty = procs[e.name]
                if len(e.args) != len(param_ts):
                    raise TypeErrorBX(f"Procedure '{e.name}' expects {len(param_ts)} args, got {len(e.args)}")
                for i,(a,pt) in enumerate(zip(e.args, param_ts)):
                    at = chk_e(a)
                    if at != pt:
                        raise TypeErrorBX(f"Argument {i+1} type mismatch in call to '{e.name}': expected {pt}, got {at}")
                e.ty = rty
                return rty
            raise TypeErrorBX("unknown expr")

        # ensure all paths return for non-void procs
        def chk_s(s: Stmt) -> bool:
            """returns True if this statement definitely returns on all paths."""
            if isinstance(s, SVar):
                if s.ty_annot != 'int':
                    raise TypeErrorBX("local variables must be int in Lab 4")
                t = chk_e(s.init)
                if t != 'int':
                    raise TypeErrorBX("initializer must be int")
                add_local(s.name, 'int')
                return False
            if isinstance(s, SAssign):
                ty = in_scope(s.name)
                if ty is None: raise SemErrorBX(f"Assignment to undeclared variable '{s.name}'")
                if ty != 'int': raise TypeErrorBX("assignment target must be int")
                rt = chk_e(s.rhs)
                if rt != 'int': raise TypeErrorBX("assignment expects int")
                return False
            if isinstance(s, SExpr):
                t = chk_e(s.e)
                # disallow using non-void returning calls as statement? That's allowed; we just discard the value.
                return False
            if isinstance(s, SBlock):
                scope_stack.append({})
                must_return = False
                for st in s.ss:
                    if chk_s(st):
                        must_return = True
                        # but still check remaining for semantic errors
                scope_stack.pop()
                return must_return
            if isinstance(s, SIfElse):
                if chk_e(s.cond) != 'bool': raise TypeErrorBX("if expects bool condition")
                r1 = chk_s(s.thenb)
                r2 = False
                if s.elsep is not None:
                    r2 = chk_s(s.elsep)
                # returns on all paths only if both branches do
                return r1 and (s.elsep is not None) and r2
            if isinstance(s, SWhile):
                if chk_e(s.cond) != 'bool': raise TypeErrorBX("while expects bool condition")
                # Conservatively: while may not execute → does not guarantee return
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
                    if t != ret_ty:
                        raise TypeErrorBX(f"return type mismatch: expected {ret_ty}, got {t}")
                return True
            raise TypeErrorBX("unknown stmt")

        all_paths_return = chk_s(pd.body)
        if ret_ty != 'void' and not all_paths_return:
            raise TypeErrorBX(f"non-void procedure '{pd.name}' might not return a value on all paths")

    for pd in prog.procs:
        chk_proc(pd)

# =============================================================================
# DRIVER (frontend only)
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
    ap = argparse.ArgumentParser(description="BX Lab 4 – frontend only")
    ap.add_argument("source", help=".bx file")
    ap.add_argument("--debug", action="store_true", help="enable debug prints")
    args = ap.parse_args()
    DEBUG = args.debug or (os.environ.get("BX_DEBUG","") not in ("", "0", "false", "False"))

    try:
        src = open(args.source,"r",encoding="utf-8").read()
    except OSError as e:
        err("IO", f"cannot read {args.source}: {e}")
        sys.exit(1)

    ast = parse_text(src)
    if ast is None:
        err("Driver", "parse failed")
        sys.exit(1)

    try:
        check_program(ast)
        dbg("Type/Semantic OK")
    except (TypeErrorBX, SemErrorBX) as e:
        err("Type/Semantic", str(e)); sys.exit(1)
    except Exception as e:
        err("Type/Semantic", f"internal exception: {e}"); sys.exit(1)

    # Lab 4: success → no output, exit 0
    sys.exit(0)

if __name__=="__main__":
    main()


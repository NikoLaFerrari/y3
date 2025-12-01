#!/usr/bin/env python3
import sys, os, json, argparse, dataclasses as dc, abc
from typing import List, Tuple, Dict, Optional, Union
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
    'var': 'VAR',
    'def': 'DEF',
    'int': 'INT',
    'bool': 'BOOL',
    'void': 'VOID',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'return': 'RETURN',
    'true': 'TRUE',
    'false': 'FALSE',
    'print': 'PRINT',
}

tokens = (
    'IDENT', 'NUMBER',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'BAND', 'BOR', 'BXOR', 'BNOT',
    'LAND', 'LOR',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NEQ',
    'ASSIGN',
    'LPAREN', 'RPAREN',
    'LBRACE', 'RBRACE',
    'SEMI', 'COLON', 'COMMA',
) + tuple(reserved.values())

t_PLUS   = r'\+'
t_MINUS  = r'-'
t_TIMES  = r'\*'
t_DIVIDE = r'/'
t_MOD    = r'%'

t_LAND   = r'&&'
t_LOR    = r'\|\|'

t_BAND   = r'&'
t_BOR    = r'\|'
t_BXOR   = r'\^'
t_BNOT   = r'~'

t_LT     = r'<'
t_LE     = r'<='
t_GT     = r'>'
t_GE     = r'>='
t_EQ     = r'=='
t_NEQ    = r'!='

t_ASSIGN = r'='

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'

t_SEMI   = r';'
t_COLON  = r':'
t_COMMA  = r','

t_ignore = ' \t\r'

def t_COMMENT(t):
    r'//[^\n]*'
    pass

def t_BLOCKCOMMENT(t):
    r'/\*(.|\n)*?\*/'
    t.lexer.lineno += t.value.count('\n')

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_IDENT(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, 'IDENT')
    return t

def t_NUMBER(t):
    r'[0-9]+'
    try:
        t.value = int(t.value)
    except ValueError:
        err("Lexer", f"Bad integer literal {t.value!r}")
        t.lexer.skip(1)
        return None
    return t

def t_error(t):
    err("Lexer", f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)

lexer = lex.lex()

# =============================================================================
# AST
# =============================================================================

@dc.dataclass
class AST(abc.ABC):
    pass

# Expressions

class Expr(AST):
    ty: Optional[str] = None

@dc.dataclass
class ENum(Expr):
    n: int

@dc.dataclass
class EBool(Expr):
    b: bool

@dc.dataclass
class EVar(Expr):
    name: str

@dc.dataclass
class EUn(Expr):
    op: str
    e: Expr

@dc.dataclass
class EBin(Expr):
    op: str
    l: Expr
    r: Expr

@dc.dataclass
class ECall(Expr):
    target: str
    args: List[Expr]

# Statements

class Stmt(AST):
    pass

@dc.dataclass
class SVar(Stmt):
    name: str
    init: Expr
    ty: str

@dc.dataclass
class SAssign(Stmt):
    name: str
    rhs: Expr

@dc.dataclass
class SEval(Stmt):
    e: Expr

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
class SBreak(Stmt):
    pass

@dc.dataclass
class SContinue(Stmt):
    pass

@dc.dataclass
class SReturn(Stmt):
    value: Optional[Expr]

# Top-level decls

@dc.dataclass
class Decl(AST):
    pass

@dc.dataclass
class GlobalVar(Decl):
    name: str
    init: Union[int, bool]
    ty: str

@dc.dataclass
class Proc(Decl):
    name: str
    params: List[Tuple[str, str]]  # (name, type)
    ret_ty: str
    body: SBlock

@dc.dataclass
class Program(AST):
    decls: List[Decl]

# =============================================================================
# PARSER
# =============================================================================

precedence = (
    ('left', 'LOR'),
    ('left', 'LAND'),
    ('left', 'BOR'),
    ('left', 'BXOR'),
    ('left', 'BAND'),
    ('left', 'EQ', 'NEQ'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD'),
    ('right', 'BNOT', 'UMINUS'),
)

def p_program(p):
    'program : decls'
    p[0] = Program(p[1])

def p_decls_empty(p):
    'decls : '
    p[0] = []

def p_decls_more(p):
    'decls : decls decl'
    p[1].append(p[2])
    p[0] = p[1]

def p_decl_global(p):
    'decl : VAR IDENT ASSIGN literal COLON type_name SEMI'
    p[0] = GlobalVar(p[2], p[4], p[6])

def p_literal_num(p):
    'literal : NUMBER'
    p[0] = p[1]

def p_literal_true(p):
    'literal : TRUE'
    p[0] = True

def p_literal_false(p):
    'literal : FALSE'
    p[0] = False

def p_type_int(p):
    'type_name : INT'
    p[0] = "int"

def p_type_bool(p):
    'type_name : BOOL'
    p[0] = "bool"

def p_type_void(p):
    'type_name : VOID'
    p[0] = "void"

def p_decl_proc(p):
    'decl : DEF IDENT LPAREN params RPAREN ret_opt block'
    p[0] = Proc(p[2], p[4], p[6], p[7])

def p_params_empty(p):
    'params : '
    p[0] = []

def p_params_nonempty(p):
    'params : param_list'
    p[0] = p[1]

def p_param_list_one(p):
    'param_list : IDENT COLON type_name'
    p[0] = [(p[1], p[3])]

def p_param_list_more(p):
    'param_list : param_list COMMA IDENT COLON type_name'
    p[1].append((p[3], p[5]))
    p[0] = p[1]

def p_ret_opt_none(p):
    'ret_opt : '
    p[0] = "void"

def p_ret_opt_some(p):
    'ret_opt : COLON type_name'
    p[0] = p[2]

def p_block(p):
    'block : LBRACE stmts RBRACE'
    p[0] = SBlock(p[2])

def p_stmts_empty(p):
    'stmts : '
    p[0] = []

def p_stmts_more(p):
    'stmts : stmts stmt'
    p[1].append(p[2])
    p[0] = p[1]

# ----- var declarations -----

def p_vardecl(p):
    'vardecl : VAR varinits COLON type_name SEMI'
    ty = p[4]
    p[0] = [SVar(name, expr, ty) for (name, expr) in p[2]]

def p_varinits_one(p):
    'varinits : IDENT ASSIGN expr'
    p[0] = [(p[1], p[3])]

def p_varinits_more(p):
    'varinits : varinits COMMA IDENT ASSIGN expr'
    p[1].append((p[3], p[5]))
    p[0] = p[1]

def p_stmt_vardecl_block(p):
    'stmt : vardecl'
    # vardecl returns list[SVar]; wrap in a block so stmt is a single AST node
    p[0] = SBlock(p[1])

# ----- other stmts -----

def p_stmt_assign(p):
    'stmt : IDENT ASSIGN expr SEMI'
    p[0] = SAssign(p[1], p[3])

def p_stmt_eval(p):
    'stmt : expr SEMI'
    p[0] = SEval(p[1])

def p_stmt_if(p):
    'stmt : IF LPAREN expr RPAREN block ifrest'
    p[0] = SIfElse(p[3], p[5], p[6])

def p_ifrest_else(p):
    'ifrest : ELSE stmt'
    p[0] = p[2]

def p_ifrest_none(p):
    'ifrest : '
    p[0] = None

def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN block'
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

# ----- expressions -----

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

def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_unary(p):
    '''expr : MINUS expr %prec UMINUS
            | BNOT expr'''
    if p[1] == '-':
        p[0] = EUn('-', p[2])
    else:
        p[0] = EUn('~', p[2])

def p_expr_bin(p):
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr
            | expr DIVIDE expr
            | expr MOD expr
            | expr BAND expr
            | expr BOR expr
            | expr BXOR expr
            | expr LT expr
            | expr LE expr
            | expr GT expr
            | expr GE expr
            | expr EQ expr
            | expr NEQ expr
            | expr LAND expr
            | expr LOR expr'''
    op = p[2]
    p[0] = EBin(op, p[1], p[3])

def p_expr_call(p):
    'expr : IDENT LPAREN args RPAREN'
    p[0] = ECall(p[1], p[3])

def p_expr_print(p):
    'expr : PRINT LPAREN args RPAREN'
    # print(...) is parsed as a call to "print"
    p[0] = ECall("print", p[3])

def p_args_empty(p):
    'args : '
    p[0] = []

def p_args_some(p):
    'args : arg_list'
    p[0] = p[1]

def p_arg_list_one(p):
    'arg_list : expr'
    p[0] = [p[1]]

def p_arg_list_more(p):
    'arg_list : arg_list COMMA expr'
    p[1].append(p[3])
    p[0] = p[1]

def p_error(p):
    if p:
        err("Parser", f"Syntax error at token {p.type} (value={p.value})")
    else:
        err("Parser", "Syntax error at EOF")

parser = yacc.yacc(start='program')

# =============================================================================
# TYPE CHECKER
# =============================================================================

class SemanticError(Exception):
    pass

@dc.dataclass
class ProcType:
    param_tys: List[str]
    ret_ty: str

def check_program(prog: Program):
    global_scope: Dict[str, Union[str, ProcType]] = {}
    main_found = False

    # 1. gather globals & procs
    for d in prog.decls:
        if isinstance(d, GlobalVar):
            if d.name in global_scope:
                raise SemanticError(f"Duplicate global '{d.name}'")
            if isinstance(d.init, bool):
                if d.ty != "bool":
                    raise SemanticError(f"Global '{d.name}' declared {d.ty} but initialized with bool")
            else:
                if d.ty == "bool":
                    raise SemanticError(f"Global '{d.name}' declared bool but initialized with int")
            global_scope[d.name] = d.ty
        elif isinstance(d, Proc):
            if d.name in global_scope:
                raise SemanticError(f"Procedure '{d.name}' redeclared")
            pt = ProcType([ty for (_n, ty) in d.params], d.ret_ty)
            global_scope[d.name] = pt
            if d.name == "main":
                if d.params:
                    raise SemanticError("main() must take no arguments")
                if d.ret_ty != "void":
                    raise SemanticError("main() must have return type void")
                main_found = True

    if not main_found:
        raise SemanticError("Missing main() procedure")

    # built-in print
    global_scope["_bx_print_int"] = ProcType(["int"], "void")
    global_scope["_bx_print_bool"] = ProcType(["bool"], "void")

    # 2. check each proc
    for d in prog.decls:
        if isinstance(d, Proc):
            check_proc(d, global_scope)

def check_proc(proc: Proc, global_scope: Dict[str, Union[str, ProcType]]):

    env_stack: List[Dict[str, Union[str, ProcType]]] = []

    def push_env():
        env_stack.append({})

    def pop_env():
        env_stack.pop()

    def add_local(name: str, ty: Union[str, ProcType]):
        if name in env_stack[-1]:
            raise SemanticError(f"Duplicate local '{name}'")
        env_stack[-1][name] = ty

    def lookup(name: str) -> Union[str, ProcType]:
        for env in reversed(env_stack):
            if name in env:
                return env[name]
        if name in global_scope:
            return global_scope[name]
        raise SemanticError(f"Undeclared identifier '{name}'")

    def check_expr(e: Expr) -> str:
        if isinstance(e, ENum):
            e.ty = "int"; return "int"
        if isinstance(e, EBool):
            e.ty = "bool"; return "bool"
        if isinstance(e, EVar):
            t = lookup(e.name)
            if isinstance(t, ProcType):
                raise SemanticError(f"Cannot use procedure '{e.name}' as a value")
            e.ty = t
            return t
        if isinstance(e, EUn):
            t = check_expr(e.e)
            if e.op in ('-', '~'):
                if t != "int":
                    raise SemanticError(f"Unary {e.op} requires int")
                e.ty = "int"
                return "int"
        if isinstance(e, EBin):
            op = e.op
            if op in ['+', '-', '*', '/', '%', '&', '|', '^']:
                tl = check_expr(e.l); tr = check_expr(e.r)
                if tl != "int" or tr != "int":
                    raise SemanticError(f"Operator {op} requires int,int")
                e.ty = "int"; return "int"
            if op in ['<', '<=', '>', '>=']:
                tl = check_expr(e.l); tr = check_expr(e.r)
                if tl != "int" or tr != "int":
                    raise SemanticError(f"Operator {op} requires int,int")
                e.ty = "bool"; return "bool"
            if op in ['==', '!=']:
                tl = check_expr(e.l); tr = check_expr(e.r)
                if tl != tr:
                    raise SemanticError("==/!= operands must have same type")
                e.ty = "bool"; return "bool"
            if op in ['&&', '||']:
                tl = check_expr(e.l); tr = check_expr(e.r)
                if tl != "bool" or tr != "bool":
                    raise SemanticError(f"Operator {op} requires bool,bool")
                e.ty = "bool"; return "bool"
        if isinstance(e, ECall):
            # built-in print sugar
            if e.target == "print":
                if len(e.args) != 1:
                    raise SemanticError("print(...) takes exactly one argument")
                t0 = check_expr(e.args[0])
                if t0 == "int":
                    e.target = "_bx_print_int"
                elif t0 == "bool":
                    e.target = "_bx_print_bool"
                else:
                    raise SemanticError("print only supports int or bool")
                e.ty = "void"
                return "void"

            t = lookup(e.target)
            if not isinstance(t, ProcType):
                raise SemanticError(f"'{e.target}' is not a function")
            if len(e.args) != len(t.param_tys):
                raise SemanticError(f"Call to '{e.target}' has wrong argument count")
            for arg, pty in zip(e.args, t.param_tys):
                if check_expr(arg) != pty:
                    raise SemanticError(f"Call to '{e.target}' argument type mismatch")
            e.ty = t.ret_ty
            return t.ret_ty

        raise SemanticError("Unknown expression")

    def check_stmt(s: Stmt):
        nonlocal in_loop
        if isinstance(s, SBlock):
            push_env()
            for st in s.ss:
                check_stmt(st)
            pop_env()
        elif isinstance(s, SVar):
            t = check_expr(s.init)
            if t != s.ty:
                raise SemanticError(f"Initializer type does not match declared type {s.ty}")
            add_local(s.name, s.ty)
        elif isinstance(s, SAssign):
            tr = check_expr(s.rhs)
            tl = lookup(s.name)
            if isinstance(tl, ProcType):
                raise SemanticError("Cannot assign to a procedure")
            if tl != tr:
                raise SemanticError(f"Assignment type mismatch for '{s.name}'")
        elif isinstance(s, SEval):
            check_expr(s.e)
        elif isinstance(s, SIfElse):
            if check_expr(s.cond) != "bool":
                raise SemanticError("If condition must be bool")
            check_stmt(s.thenb)
            if s.elsep:
                check_stmt(s.elsep)
        elif isinstance(s, SWhile):
            if check_expr(s.cond) != "bool":
                raise SemanticError("While condition must be bool")
            saved = in_loop
            in_loop = True
            check_stmt(s.body)
            in_loop = saved
        elif isinstance(s, SBreak):
            if not in_loop:
                raise SemanticError("break not in loop")
        elif isinstance(s, SContinue):
            if not in_loop:
                raise SemanticError("continue not in loop")
        elif isinstance(s, SReturn):
            if s.value is None:
                if proc.ret_ty != "void":
                    raise SemanticError("return; in non-void function")
            else:
                vt = check_expr(s.value)
                if vt != proc.ret_ty:
                    raise SemanticError("return type mismatch")
        else:
            raise SemanticError("Unknown statement")

    # initialize
    push_env()
    for name, ty in proc.params:
        add_local(name, ty)
    in_loop = False
    for st in proc.body.ss:
        check_stmt(st)
    pop_env()

# =============================================================================
# TAC GENERATION
# =============================================================================

class TempGen:
    def __init__(self):
        self.i = 0
    def fresh(self) -> str:
        t = f"%t{self.i}"
        self.i += 1
        return t

class LabelGen:
    def __init__(self):
        self.i = 0
    def fresh(self) -> str:
        l = f"%.L{self.i}"
        self.i += 1
        return l

def gen_tac(prog: Program) -> List[Dict]:
    temps = TempGen()
    labels = LabelGen()
    units: List[Dict] = []

    # globals
    for d in prog.decls:
        if isinstance(d, GlobalVar):
            units.append({"global": d.name, "init": d.init, "ty": d.ty})

    # procs
    for d in prog.decls:
        if isinstance(d, Proc):
            units.append(gen_proc_tac(d, temps, labels))

    return units

def gen_proc_tac(proc: Proc, temps: TempGen, labels: LabelGen) -> Dict:
    instrs: List[Dict] = []

    def emit(op, args=None, result=None):
        inst = {"opcode": op}
        if args is not None:
            inst["args"] = args
        if result is not None:
            inst["result"] = result
        instrs.append(inst)
        return inst

    var2temp: Dict[str, str] = {}

    def bind(name: str, t: str):
        var2temp[name] = t

    def var_loc(name: str) -> str:
        if name not in var2temp:
            t = temps.fresh()
            bind(name, t)
        return var2temp[name]

    # parameters: param_get i → temp
    for i, (pname, _pty) in enumerate(proc.params):
        t = temps.fresh()
        emit("param_get", [i], t)
        bind(pname, t)

    loop_stack: List[Tuple[str, str]] = []  # (head, end)

    def emit_int(e: Expr) -> str:
        if isinstance(e, ENum):
            t = temps.fresh()
            emit("const", [e.n], t)
            return t
        if isinstance(e, EVar):
            return var_loc(e.name)
        if isinstance(e, EBool):
            t = temps.fresh()
            emit("const", [1 if e.b else 0], t)
            return t
        if isinstance(e, EUn):
            v = emit_int(e.e) if e.op in ('-', '~') else emit_bool_as_int(e.e)
            t = temps.fresh()
            if e.op == '-':
                emit("neg", [v], t)
            elif e.op == '~':
                emit("not", [v], t)
            else:
                # logical not
                emit("xor", [v, 1], t)
            return t
        if isinstance(e, EBin):
            if e.op in ['+', '-', '*', '/', '%', '&', '|', '^']:
                l = emit_int(e.l); r = emit_int(e.r)
                t = temps.fresh()
                opmap = {
                    '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod',
                    '&': 'and', '|': 'or', '^': 'xor'
                }
                emit(opmap[e.op], [l, r], t)
                return t
            # comparisons / logicals as bool-int
            return emit_bool_as_int(e)
        if isinstance(e, ECall):
            args = [emit_int(a) for a in e.args]
            for i, a in enumerate(args):
                emit("param", [i+1, a])
            t = temps.fresh() if e.ty != "void" else None
            emit("call", [e.target, len(args)], t)
            return t if t is not None else "%_"
        # fallback: treat bool as int
        return emit_bool_as_int(e)

    def emit_bool_as_int(e: Expr) -> str:
        if isinstance(e, EBool):
            t = temps.fresh()
            emit("const", [1 if e.b else 0], t)
            return t
        if isinstance(e, EBin) and e.op in ("==","!=","<","<=",">",">="):
            t = temps.fresh()
            l = emit_int(e.l); r = emit_int(e.r)
            emit("cmp_set", [e.op, l, r], t)
            return t
        if isinstance(e, EBin) and e.op in ("&&","||"):
            # no true short-circuit; compute both and do op on 0/1
            tl = emit_bool_as_int(e.l)
            tr = emit_bool_as_int(e.r)
            t = temps.fresh()
            if e.op == "&&":
                # t = (tl != 0) & (tr != 0)
                emit("and", [tl, tr], t)
            else:
                emit("or", [tl, tr], t)
            return t
        if isinstance(e, EUn) and e.op == '!':
            v = emit_bool_as_int(e.e)
            t = temps.fresh()
            emit("xor", [v, 1], t)
            return t
        # fallback: non-zero test
        v = emit_int(e)
        t = temps.fresh()
        emit("cmp_set", ["!=", v, 0], t)
        return t

    def emit_expr_cond(e: Expr, true_lbl: str, false_lbl: str):
        v = emit_bool_as_int(e)
        emit("br_cmp2", ["!=", v, 0, true_lbl, false_lbl])

    def emit_stmt(s: Stmt):
        if isinstance(s, SBlock):
            for st in s.ss:
                emit_stmt(st)
        elif isinstance(s, SVar):
            v = emit_int(s.init)
            bind(s.name, v)
        elif isinstance(s, SAssign):
            src = emit_int(s.rhs)
            dst = var_loc(s.name)
            emit("copy", [src], dst)
        elif isinstance(s, SEval):
            if isinstance(s.e, ECall):
                args = [emit_int(a) for a in s.e.args]
                for i, a in enumerate(args):
                    emit("param", [i+1, a])
                emit("call", [s.e.target, len(args)], None)
            else:
                emit_int(s.e)
        elif isinstance(s, SIfElse):
            l_then = labels.fresh()
            l_else = labels.fresh()
            l_end  = labels.fresh()
            emit_expr_cond(s.cond, l_then, l_else if s.elsep else l_end)
            emit("label", [l_then])
            emit_stmt(s.thenb)
            emit("jmp", [l_end])
            if s.elsep:
                emit("label", [l_else])
                emit_stmt(s.elsep)
                emit("jmp", [l_end])
            emit("label", [l_end])
        elif isinstance(s, SWhile):
            l_head = labels.fresh()
            l_body = labels.fresh()
            l_end  = labels.fresh()
            emit("label", [l_head])
            emit_expr_cond(s.cond, l_body, l_end)
            emit("label", [l_body])
            loop_stack.append((l_head, l_end))
            emit_stmt(s.body)
            loop_stack.pop()
            emit("jmp", [l_head])
            emit("label", [l_end])
        elif isinstance(s, SBreak):
            if not loop_stack:
                raise RuntimeError("break outside loop in TAC gen")
            _, l_end = loop_stack[-1]
            emit("jmp", [l_end])
        elif isinstance(s, SContinue):
            if not loop_stack:
                raise RuntimeError("continue outside loop in TAC gen")
            l_head, _ = loop_stack[-1]
            emit("jmp", [l_head])
        elif isinstance(s, SReturn):
            if s.value is None:
                emit("ret", [])
            else:
                if s.value.ty == "bool":
                    v = emit_bool_as_int(s.value)
                else:
                    v = emit_int(s.value)
                emit("ret", [v])
        else:
            raise RuntimeError("Unknown stmt in TAC gen")

    for st in proc.body.ss:
        emit_stmt(st)

    if not instrs or instrs[-1]["opcode"] != "ret":
        emit("ret", [])

    return {
        "proc": proc.name,
        "params": [n for (n, _t) in proc.params],
        "ret": proc.ret_ty,
        "body": instrs,
    }

# =============================================================================
# CFG (very lightweight) + placeholder opt
# =============================================================================

class Block:
    def __init__(self, label: str):
        self.label = label
        self.instrs: List[Dict] = []
        self.succs: List[str] = []

def optimize_cfg(proc: Dict) -> Dict:
    body = proc["body"]
    if not body:
        return proc

    blocks: List[Block] = []
    curr = Block("%.B0")
    blocks.append(curr)

    for inst in body:
        if inst["opcode"] == "label":
            if curr.instrs:
                nb = Block(inst["args"][0])
                blocks.append(nb)
                curr = nb
            else:
                curr.label = inst["args"][0]
        else:
            curr.instrs.append(inst)
            if inst["opcode"] in ("jmp", "ret", "br_cmp2", "br_if_true"):
                nb = Block(f"%.B{len(blocks)}")
                blocks.append(nb)
                curr = nb

    # succs
    for i, b in enumerate(blocks):
        if not b.instrs:
            continue
        last = b.instrs[-1]
        op = last["opcode"]
        if op == "jmp":
            b.succs.append(last["args"][0])
        elif op == "br_cmp2":
            # op, v, 0, ltrue, lfalse
            b.succs.append(last["args"][3])
            b.succs.append(last["args"][4])
        elif op == "br_if_true":
            b.succs.append(last["args"][1])
            b.succs.append(last["args"][2])
        elif op == "ret":
            pass
        else:
            if i+1 < len(blocks):
                b.succs.append(blocks[i+1].label)

    new_body: List[Dict] = []
    for b in blocks:
        if not b.instrs:
            continue
        new_body.append({"opcode": "label", "args": [b.label]})
        new_body.extend(b.instrs)

    proc["body"] = new_body
    return proc

# =============================================================================
# X86-64 CODEGEN
# =============================================================================

def tac_to_x64(units: List[Dict], out_path: str):

    out: List[str] = []

    def emit(line=""):
        out.append(line)

    def lbl(x: str) -> str:
        return x[2:] if x.startswith("%.") else x

    def loc_temp(tmap, x):
        if isinstance(x, int):
            return f"${x}"
        if isinstance(x, str) and x.startswith("@"):
            return f"{x[1:]}(%rip)"
        if isinstance(x, str) and x.startswith("%t"):
            return f"{tmap[x]}(%rbp)"
        return x

    emit("    .text")
    emit("    .globl main")

    has_globals = any("global" in u for u in units)
    if has_globals:
        emit("    .data")
        for u in units:
            if "global" in u:
                g = u["global"]
                init = int(u["init"])
                emit(f"{g}:")
                emit(f"    .quad {init}")
        emit("    .text")

    abi_int = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"]

    for u in units:
        if "proc" not in u:
            continue

        name = u["proc"]
        body = u["body"]

        emit(f"{name}:")
        emit("  pushq %rbp")
        emit("  movq %rsp, %rbp")

        temps = set()
        for inst in body:
            r = inst.get("result")
            if isinstance(r, str) and r.startswith("%t"):
                temps.add(r)
            for a in inst.get("args", []):
                if isinstance(a, str) and a.startswith("%t"):
                    temps.add(a)
        temps = sorted(temps)
        tmap = {t: -8*(i+1) for i, t in enumerate(temps)}
        stack_size = 8 * len(temps)
        if stack_size:
            emit(f"  subq ${stack_size}, %rsp")

        for inst in body:
            op = inst["opcode"]
            args = inst.get("args", [])
            res  = inst.get("result")

            if op == "label":
                emit(f".L{lbl(args[0])}:")
            elif op == "const":
                emit(f"  movq ${args[0]}, {loc_temp(tmap, res)}")
            elif op == "copy":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op in ("add","sub","and","or","xor"):
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                m = {"add": "addq", "sub": "subq", "and": "andq", "or": "orq", "xor": "xorq"}[op]
                emit(f"  {m} {loc_temp(tmap, args[1])}, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "mul":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit(f"  imulq {loc_temp(tmap, args[1])}, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op in ("div","mod"):
                emit(f"  movq {loc_temp(tmap, args[0])}, %rax")
                emit("  cqto")
                emit(f"  idivq {loc_temp(tmap, args[1])}")
                if op == "div":
                    emit(f"  movq %rax, {loc_temp(tmap, res)}")
                else:
                    emit(f"  movq %rdx, {loc_temp(tmap, res)}")
            elif op == "neg":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit("  negq %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "not":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit("  notq %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "shl":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit(f"  movb {loc_temp(tmap, args[1])}, %cl")
                emit("  shlq %cl, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "shr":
                emit(f"  movq {loc_temp(tmap, args[0])}, %r11")
                emit(f"  movb {loc_temp(tmap, args[1])}, %cl")
                emit("  sarq %cl, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "jmp":
                emit(f"  jmp .L{lbl(args[0])}")
            elif op == "br_cmp2":
                cmpop, a1, a2, lt, lf = args
                emit(f"  movq {loc_temp(tmap, a1)}, %r11")
                emit(f"  cmpq {loc_temp(tmap, a2)}, %r11")
                jmpmap = {
                    "==": "je", "!=": "jne",
                    "<": "jl", "<=": "jle",
                    ">": "jg", ">=": "jge",
                }
                j = jmpmap[cmpop]
                emit(f"  {j} .L{lbl(lt)}")
                emit(f"  jmp .L{lbl(lf)}")
            elif op == "br_if_true":
                cond, lt, lf = args
                emit(f"  movq {loc_temp(tmap, cond)}, %r11")
                emit("  cmpq $0, %r11")
                emit(f"  jne .L{lbl(lt)}")
                emit(f"  jmp .L{lbl(lf)}")
            elif op == "cmp_set":
                cmpop, a1, a2 = args
                emit(f"  movq {loc_temp(tmap, a1)}, %r11")
                emit(f"  cmpq {loc_temp(tmap, a2)}, %r11")
                setmap = {
                    "==": "sete", "!=": "setne",
                    "<": "setl", "<=": "setle",
                    ">": "setg", ">=": "setge",
                }
                s = setmap[cmpop]
                emit(f"  {s} %r10b")
                emit("  movzbq %r10b, %r11")
                emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "param_get":
                idx = args[0]
                if idx < len(abi_int):
                    reg = abi_int[idx]
                    emit(f"  movq {reg}, {loc_temp(tmap, res)}")
                else:
                    off = 16 + 8*(idx - len(abi_int))
                    emit(f"  movq {off}(%rbp), %r11")
                    emit(f"  movq %r11, {loc_temp(tmap, res)}")
            elif op == "param":
                idx, a = args
                idx -= 1
                if idx < len(abi_int):
                    emit(f"  movq {loc_temp(tmap, a)}, {abi_int[idx]}")
                else:
                    # ignore >6th arg in this toy compiler
                    emit(f"  # ignoring extra arg {idx+1}")
            elif op == "call":
                fname, argc = args
                emit(f"  call {fname}")
                if res:
                    emit(f"  movq %rax, {loc_temp(tmap, res)}")
            elif op == "ret":
                if args:
                    emit(f"  movq {loc_temp(tmap, args[0])}, %rax")
                else:
                    emit("  xorq %rax, %rax")
                emit("  movq %rbp, %rsp")
                emit("  popq %rbp")
                emit("  retq")

        # function epilogue in case of fall-through without ret
        if not body or body[-1]["opcode"] != "ret":
            emit("  movq %rbp, %rsp")
            emit("  popq %rbp")
            emit("  retq")

    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--keep-tac", action="store_true")
    args = ap.parse_args()

    with open(args.source) as f:
        src = f.read()

    try:
        ast = parser.parse(src, lexer=lexer)
    except Exception as e:
        err("Parser", str(e))
        sys.exit(1)

    try:
        check_program(ast)
    except SemanticError as e:
        err("Type/Semantic", str(e))
        sys.exit(1)

    tac = gen_tac(ast)
    if args.keep_tac:
        with open(os.path.splitext(args.source)[0] + ".tac.json", "w") as f:
            json.dump(tac, f, indent=2)

    opt_units: List[Dict] = []
    for u in tac:
        if "proc" in u:
            opt_units.append(optimize_cfg(u))
        else:
            opt_units.append(u)

    out_s = os.path.splitext(args.source)[0] + ".s"
    tac_to_x64(opt_units, out_s)

if __name__ == "__main__":
    main()


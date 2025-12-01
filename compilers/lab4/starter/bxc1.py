#!/usr/bin/env python3
import sys, os, json, argparse, dataclasses as dc, abc
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
    'main':  'MAIN',
    'var':   'VAR',
    'print': 'PRINT',   # still a statement in Lab 3
    'int':   'INT',
    # bool literals (no bool-typed vars yet in Lab 3)
    'true':  'TRUE',
    'false': 'FALSE',
    # control
    'if':    'IF',
    'else':  'ELSE',
    'while': 'WHILE',
    'break': 'BREAK',
    'continue': 'CONTINUE',
}

tokens = (
    # id/lits
    'IDENT', 'NUMBER',
    # punct
    'LPAREN','RPAREN','LBRACE','RBRACE',
    'COLON','SEMI','EQUAL',
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
    except ValueError as e:
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

# Expressions
class Expr(AST): pass

@dc.dataclass
class ENum(Expr):
    n: int
    ty: str = "int"

@dc.dataclass
class EBool(Expr):
    b: bool
    ty: str = "bool"

@dc.dataclass
class EVar(Expr):
    name: str
    ty: Optional[str] = None

@dc.dataclass
class EUn(Expr):
    op: str   # '-', '~', '!'
    e: Expr
    ty: Optional[str] = None

@dc.dataclass
class EBin(Expr):
    op: str   # + - * / % & | ^ << >>, == != < <= > >=, && ||
    l: Expr
    r: Expr
    ty: Optional[str] = None

# Statements
class Stmt(AST): pass

@dc.dataclass
class SVar(Stmt):
    name: str
    init: Expr         # ": int" always

@dc.dataclass
class SAssign(Stmt):
    name: str
    rhs: Expr

@dc.dataclass
class SPrint(Stmt):
    e: Expr

@dc.dataclass
class SBlock(Stmt):
    ss: List[Stmt]

@dc.dataclass
class SIfElse(Stmt):
    cond: Expr
    thenb: SBlock
    elsep: Optional[Stmt]  # either None, SBlock, or nested SIfElse for else-if chain

@dc.dataclass
class SWhile(Stmt):
    cond: Expr
    body: SBlock

@dc.dataclass
class SBreak(Stmt): pass

@dc.dataclass
class SContinue(Stmt): pass

@dc.dataclass
class Program(AST):
    body: SBlock

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
    'program : DEF MAIN LPAREN RPAREN block'
    p[0] = Program(body=p[5])

def p_block(p):
    'block : LBRACE stmt_list RBRACE'
    p[0] = SBlock(p[2])

def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = []

def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    p[1].append(p[2]); p[0] = p[1]

def p_stmt_vardecl(p):
    'stmt : VAR IDENT EQUAL expr COLON INT SEMI'
    p[0] = SVar(p[2], p[4])

def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = SAssign(p[1], p[3])

def p_stmt_print(p):
    'stmt : PRINT LPAREN expr RPAREN SEMI'
    p[0] = SPrint(p[3])

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

def p_ifrest_else_if(p):
    'ifrest : ELSE stmt'
    # stmt can be block or nested if
    p[0] = p[2]

def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN block'
    p[0] = SWhile(p[3], p[5])

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
# TYPE CHECKING (ints & bools; only int vars in Lab 3)
# =============================================================================

class TypeErrorBX(Exception): pass
class SemErrorBX(Exception): pass

def check_program(prog: Program):
    # Block scope stack (enables shadowing in inner blocks)
    scope_stack: List[Set[str]] = [set()]

    def in_scope(name: str) -> bool:
        return any(name in s for s in reversed(scope_stack))

    def add_local(name: str):
        if name in scope_stack[-1]:
            raise SemErrorBX(f"Variable '{name}' redeclared in the same block")
        scope_stack[-1].add(name)

    def chk_e(e: Expr) -> str:
        if isinstance(e, ENum): e.ty = "int"; return "int"
        if isinstance(e, EBool): e.ty = "bool"; return "bool"
        if isinstance(e, EVar):
            if not in_scope(e.name):
                raise SemErrorBX(f"Use of undeclared variable '{e.name}'")
            e.ty = "int"; return "int"
        if isinstance(e, EUn):
            t = chk_e(e.e)
            if e.op == '!':
                if t != "bool": raise TypeErrorBX("operator ! expects bool")
                e.ty = "bool"; return "bool"
            if e.op in ('-','~'):
                if t != "int": raise TypeErrorBX(f"operator {e.op} expects int")
                e.ty = "int"; return "int"
            raise TypeErrorBX(f"unknown unary {e.op}")
        if isinstance(e, EBin):
            if e.op in ('&&','||'):
                tl = chk_e(e.l); tr = chk_e(e.r)
                if tl != "bool" or tr != "bool": raise TypeErrorBX(f"{e.op} expects bool && bool")
                e.ty = "bool"; return "bool"
            if e.op in ('==','!=','<','<=','>','>='):
                tl = chk_e(e.l); tr = chk_e(e.r)
                if tl != "int" or tr != "int": raise TypeErrorBX(f"{e.op} compares ints")
                e.ty = "bool"; return "bool"
            # arithmetic/bitwise
            tl = chk_e(e.l); tr = chk_e(e.r)
            if tl != "int" or tr != "int": raise TypeErrorBX(f"{e.op} expects ints")
            e.ty = "int"; return "int"
        raise TypeErrorBX("unknown expr")

    def chk_block(b: SBlock):
        scope_stack.append(set())
        for st in b.ss: chk_s(st)
        scope_stack.pop()

    def chk_s(s: Stmt):
        if isinstance(s, SVar):
            if chk_e(s.init) != "int": raise TypeErrorBX("local vars must be int")
            add_local(s.name); return
        if isinstance(s, SAssign):
            if not in_scope(s.name): raise SemErrorBX(f"Assignment to undeclared variable '{s.name}'")
            if chk_e(s.rhs) != "int": raise TypeErrorBX("assignment expects int")
            return
        if isinstance(s, SPrint):
            if chk_e(s.e) != "int": raise TypeErrorBX("print expects int")
            return
        if isinstance(s, SBlock):
            chk_block(s); return
        if isinstance(s, SIfElse):
            if chk_e(s.cond) != "bool": raise TypeErrorBX("if expects bool condition")
            chk_block(s.thenb)
            if s.elsep: chk_s(s.elsep)
            return
        if isinstance(s, SWhile):
            if chk_e(s.cond) != "bool": raise TypeErrorBX("while expects bool condition")
            chk_block(s.body); return
        if isinstance(s, (SBreak,SContinue)): return
        raise TypeErrorBX("unknown stmt")

    chk_s(prog.body)

# =============================================================================
# TAC (with labels & branches)
# =============================================================================

def tconst(v:int, dst:str):   return {"opcode":"const", "args":[v], "result":dst}
def tcopy(a:str, dst:str):    return {"opcode":"copy",  "args":[a], "result":dst}
def tun(op:str, a:str, dst):  return {"opcode":op,      "args":[a], "result":dst}
def tbin(op:str, a:str,b:str,dst): return {"opcode":op,"args":[a,b], "result":dst}
def tprint(a:str):            return {"opcode":"print", "args":[a], "result":None}
def tlabel(lbl:str):          return {"opcode":"label", "args":[lbl], "result":None}
def tbr(lbl:str):             return {"opcode":"br",    "args":[lbl], "result":None}
def tbr_true(c:str,lbl:str):  return {"opcode":"br_if_true","args":[c,lbl], "result":None}
def tbr_false(c:str,lbl:str): return {"opcode":"br_if_false","args":[c,lbl], "result":None}

BINOPS = {
    '+':'add','-':'sub','*':'mul','/':'div','%':'mod',
    '&':'and','|':'or','^':'xor','<<':'shl','>>':'shr',
}
UNOPS = {'-':'neg','~':'not'}

class TempGen:
    def __init__(self): self.n = 0
    def fresh(self)->str: t=f"%{self.n}"; self.n+=1; return t

class LabelGen:
    def __init__(self): self.k=0
    def fresh(self,pfx="%.L")->str:
        s=f"{pfx}{self.k}"; self.k+=1; return s

def gen_tac(prog: Program) -> List[Dict]:
    temps, labels = TempGen(), LabelGen()
    code: List[Dict] = []

    # env stack for block scopes: name -> temp
    env_stack: List[Dict[str,str]] = [ {} ]

    def env_lookup(name: str) -> str:
        for m in reversed(env_stack):
            if name in m: return m[name]
        raise RuntimeError(f"undeclared var {name}")

    def env_bind(name: str, temp: str):
        env_stack[-1][name] = temp

    # -------------------------------------------------------------------------
    # expression helpers
    # -------------------------------------------------------------------------
    def emit_int(e: Expr) -> Tuple[str,List[Dict]]:
        if isinstance(e, ENum):
            t=temps.fresh(); return t, [tconst(e.n,t)]
        if isinstance(e, EVar):
            try:
                v = env_lookup(e.name)
            except RuntimeError as ex:
                err("Codegen", str(ex)); raise
            return v, []
        if isinstance(e, EUn) and e.op in ('-','~'):
            a, c = emit_int(e.e)
            t = temps.fresh()
            return t, c + [tun(UNOPS[e.op], a, t)]
        if isinstance(e, EBin) and e.op in BINOPS:
            l, cl = emit_int(e.l); r, cr = emit_int(e.r)
            t = temps.fresh()
            return t, cl + cr + [tbin(BINOPS[e.op], l, r, t)]
        # bool → int materialization (0/1)
        if isinstance(e, (EBool,EUn,EBin)) and e.ty=="bool":
            t = temps.fresh()
            L1, L2, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            c: List[Dict] = []
            c.append(tconst(0, t))
            emit_cond_into(e, L1, L2, out=c)
            c += [tlabel(L1), tconst(1, t), tbr(Lend),
                  tlabel(L2), tlabel(Lend)]
            return t, c
        raise RuntimeError("emit_int: unsupported expression")

    def emit_bool(e: Expr) -> Tuple[str,List[Dict]]:
        if isinstance(e, EBool):
            t = temps.fresh(); return t, [tconst(1 if e.b else 0, t)]
        if isinstance(e, EUn) and e.op=='!':
            a, ca = emit_bool(e.e)
            t = temps.fresh()
            Lz, Lnz, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            code_ = ca + [tconst(0,t), tbr_true(a, Lnz), tbr(Lz),
                          tlabel(Lnz), tconst(1,t), tbr(Lend),
                          tlabel(Lz), tlabel(Lend)]
            return t, code_
        if isinstance(e, EBin) and e.op in ('&&','||'):
            t = temps.fresh()
            Ltrue, Lfalse, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            c: List[Dict] = []
            c.append(tconst(0, t))
            emit_cond_into(e, Ltrue, Lfalse, out=c)
            c += [tlabel(Ltrue), tconst(1,t), tbr(Lend),
                  tlabel(Lfalse), tlabel(Lend)]
            return t, c
        if isinstance(e, EBin) and e.op in ('==','!=','<','<=','>','>='):
            l, cl = emit_int(e.l); r, cr = emit_int(e.r)
            t = temps.fresh()
            diff = temps.fresh()
            # Fallback materialisation: use the conditional branch to build 0/1
            Ltrue, Lfalse, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            c = cl + cr + [tconst(0,t), tbin('sub', l, r, diff),
                           {"opcode":"cmpflag", "args":[diff, e.op], "result":None},
                           {"opcode":"br_cmp2", "args":[diff, e.op, Ltrue, Lfalse], "result":None},
                           tlabel(Ltrue), tconst(1,t), tbr(Lend),
                           tlabel(Lfalse), tlabel(Lend)]
            return t, c
        if isinstance(e, (ENum,EVar,EUn,EBin)) and e.ty=="int":
            a, ca = emit_int(e)
            t = temps.fresh()
            Lnz, Lz, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            c = ca + [tconst(0,t), tbr_true(a, Lnz), tbr(Lz),
                      tlabel(Lnz), tconst(1,t), tbr(Lend),
                      tlabel(Lz), tlabel(Lend)]
            return t, c
        raise RuntimeError("emit_bool: unsupported")

    def emit_cond_into(e: Expr, Ltrue: str, Lfalse: str, out: List[Dict]):
        if isinstance(e, EBool):
            out.append(tbr(Ltrue if e.b else Lfalse)); return
        if isinstance(e, EUn) and e.op=='!':
            emit_cond_into(e.e, Lfalse, Ltrue, out); return
        if isinstance(e, EBin) and e.op=='&&':
            Lmid = labels.fresh()
            emit_cond_into(e.l, Lmid, Lfalse, out)
            out.append(tlabel(Lmid))
            emit_cond_into(e.r, Ltrue, Lfalse, out)
            return
        if isinstance(e, EBin) and e.op=='||':
            Lmid = labels.fresh()
            emit_cond_into(e.l, Ltrue, Lmid, out)
            out.append(tlabel(Lmid))
            emit_cond_into(e.r, Ltrue, Lfalse, out)
            return
        if isinstance(e, EBin) and e.op in ('==','!=','<','<=','>','>='):
            l, cl = emit_int(e.l); r, cr = emit_int(e.r)
            diff = temps.fresh()
            out += cl + cr + [tbin('sub', l, r, diff),
                              {"opcode":"cmpflag","args":[diff, e.op], "result":None},
                              {"opcode":"br_cmp2", "args":[diff, e.op, Ltrue, Lfalse], "result":None}]
            return
        a, ca = emit_int(e)
        out += ca + [tbr_true(a, Ltrue), tbr(Lfalse)]

    # -------------------------------------------------------------------------
    # statements
    # -------------------------------------------------------------------------
    loop_stack: List[Tuple[str,str]] = []  # (cont_label, break_label)

    def emit_block(b: SBlock):
        env_stack.append({})
        for st in b.ss: emit_stmt(st)
        env_stack.pop()

    def emit_stmt(s: Stmt):
        nonlocal code
        if isinstance(s, SVar):
            t = temps.fresh()
            env_bind(s.name, t)
            v, cv = emit_int(s.init)
            code += cv + [tcopy(v, t)]
            return
        if isinstance(s, SAssign):
            v, cv = emit_int(s.rhs)
            try:
                dst = env_lookup(s.name)
            except RuntimeError as ex:
                err("Codegen", str(ex)); raise
            code += cv + [tcopy(v, dst)]
            return
        if isinstance(s, SPrint):
            v, cv = emit_int(s.e)
            code += cv + [tprint(v)]
            return
        if isinstance(s, SBlock):
            emit_block(s); return
        if isinstance(s, SIfElse):
            Lthen, Lelse, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            emit_cond_into(s.cond, Lthen, Lelse, code)
            code.append(tlabel(Lthen)); emit_block(s.thenb); code.append(tbr(Lend))
            code.append(tlabel(Lelse))
            if s.elsep: emit_stmt(s.elsep)
            code.append(tlabel(Lend))
            return
        if isinstance(s, SWhile):
            Lhead, Lbody, Lend = labels.fresh(), labels.fresh(), labels.fresh()
            code.append(tlabel(Lhead))
            emit_cond_into(s.cond, Lbody, Lend, code)
            code.append(tlabel(Lbody))
            loop_stack.append((Lhead, Lend))
            emit_block(s.body)
            loop_stack.pop()
            code.append(tbr(Lhead))
            code.append(tlabel(Lend))
            return
        if isinstance(s, SBreak):
            if not loop_stack:
                err("Codegen", "break outside loop"); raise RuntimeError("break outside loop")
            code.append(tbr(loop_stack[-1][1])); return
        if isinstance(s, SContinue):
            if not loop_stack:
                err("Codegen", "continue outside loop"); raise RuntimeError("continue outside loop")
            code.append(tbr(loop_stack[-1][0])); return
        err("Codegen", f"unknown stmt {type(s).__name__}"); raise RuntimeError("unknown stmt")

    emit_stmt(prog.body)
    dbg(f"TAC instructions: {len(code)}")
    return [{"proc":"@main", "body": code}]

# =============================================================================
# CONTROL-FLOW GRAPH (Lab 4)
# =============================================================================

@dc.dataclass
class BasicBlock:
    name: str
    instrs: List[Dict]
    succs: Set[str]
    preds: Set[str]

def _is_temp(x: object) -> bool:
    return isinstance(x, str) and x.startswith('%') and not x.startswith('%.L')

def build_cfg_for_proc(proc: Dict) -> Tuple[str, Dict[str,BasicBlock], List[str]]:
    """
    Build a CFG for a single TAC procedure.
    Returns (entry_block_name, blocks_by_name, order_list).
    """
    body: List[Dict] = proc["body"]
    blocks: List[BasicBlock] = []

    i = 0
    n = len(body)
    bid = 0

    def new_block_name(prefix: str = "%.B") -> str:
        nonlocal bid
        name = f"{prefix}{bid}"
        bid += 1
        return name

    while i < n:
        # Determine block name.
        # If the block starts with a label, we can use a generated name 
        # and map the label later.
        name = new_block_name()

        instrs: List[Dict] = []
        
        # 1. Handle the block starting with a label
        if i < n and body[i]["opcode"] == "label":
            instrs.append(body[i])
            i += 1

        # 2. Collect instructions until branch, return, or the NEXT label
        while i < n:
            # FIX: Check if the *current* instruction is a label (start of NEW block)
            # If so, break immediately so it is processed in the next outer iteration
            if body[i]["opcode"] == "label":
                break

            op = body[i]["opcode"]
            instrs.append(body[i])
            i += 1
            
            # These opcodes end the current basic block
            if op in ("br", "br_if_true", "br_if_false", "br_cmp2", "br_cmp_true", "ret"):
                break

        # Only add the block if it's not empty (or it's just a label which is fine)
        if instrs:
            blocks.append(BasicBlock(name=name, instrs=instrs,
                                     succs=set(), preds=set()))

    # Map labels -> block names.
    label2block: Dict[str,str] = {}
    for b in blocks:
        # A block might contain multiple consecutive labels at the top, 
        # though current partitioning logic usually splits them. 
        # We check the first instruction.
        if b.instrs and b.instrs[0]["opcode"] == "label":
            label = b.instrs[0]["args"][0]
            label2block[label] = b.name

    # Compute succs (control-flow).
    for idx, b in enumerate(blocks):
        last_real = None
        # Find the last non-label instruction
        for ins in reversed(b.instrs):
            if ins["opcode"] != "label":
                last_real = ins
                break
        
        succs: Set[str] = set()
        
        # Logic to determine where execution goes next
        if last_real is not None:
            op = last_real["opcode"]
            args = last_real.get("args", [])
            
            if op == "br":
                target = args[0]
                if target in label2block:
                    succs.add(label2block[target])
            elif op in ("br_if_true", "br_if_false"):
                cond, target = args
                if target in label2block:
                    succs.add(label2block[target])
                # Conditional branches also fall through to the next block
                if idx + 1 < len(blocks):
                    succs.add(blocks[idx+1].name)
            elif op == "br_cmp2":
                diff, relop, Lt, Lf = args
                if Lt in label2block: succs.add(label2block[Lt])
                if Lf in label2block: succs.add(label2block[Lf])
            elif op == "br_cmp_true":
                diff, relop, Lt = args
                if Lt in label2block: succs.add(label2block[Lt])
                if idx + 1 < len(blocks):
                    succs.add(blocks[idx+1].name)
            elif op == "ret":
                # ret has NO successors
                pass
            else:
                # Arithmetic/Print/etc falls through
                if idx + 1 < len(blocks):
                    succs.add(blocks[idx+1].name)
        else:
            # Block contained only labels (or empty), falls through
            if idx + 1 < len(blocks):
                succs.add(blocks[idx+1].name)

        b.succs = succs

    # Fill preds.
    name2block: Dict[str,BasicBlock] = {b.name: b for b in blocks}
    for b in blocks:
        for s in b.succs:
            if s in name2block:
                name2block[s].preds.add(b.name)

    # Entry block = first block.
    entry = blocks[0].name if blocks else ""
    order = [b.name for b in blocks]
    return entry, name2block, order

def dump_cfg(entry: str, blocks: Dict[str,BasicBlock], order: List[str]) -> None:
    print(f"ENTRY {entry}")
    for name in order:
        b = blocks[name]
        succs = ", ".join(sorted(b.succs))
        preds = ", ".join(sorted(b.preds))
        print(f"BLOCK {name}:")
        print(f"  succs: {succs}")
        print(f"  preds: {preds}")

# =============================================================================
# LIVENESS ANALYSIS (Lab 5)
# =============================================================================

def _inst_uses_defs(ins: Dict) -> Tuple[Set[str], Set[str]]:
    op = ins["opcode"]
    args = ins.get("args", [])
    r = ins.get("result")

    uses: Set[str] = set()
    defs: Set[str] = set()

    if op == "label":
        return uses, defs

    if op == "br":
        # only label
        return uses, defs
    if op in ("br_if_true", "br_if_false"):
        cond = args[0]
        if _is_temp(cond):
            uses.add(cond)
        return uses, defs
    if op == "cmpflag":
        diff = args[0]
        if _is_temp(diff):
            uses.add(diff)
        return uses, defs
    if op == "br_cmp2":
        diff = args[0]
        if _is_temp(diff):
            uses.add(diff)
        return uses, defs
    if op == "br_cmp_true":
        diff = args[0]
        if _is_temp(diff):
            uses.add(diff)
        return uses, defs

    # generic arithmetic / copy / const / print etc.
    for a in args:
        if _is_temp(a):
            uses.add(a)
    if _is_temp(r):
        defs.add(r)
    return uses, defs

def liveness(blocks: Dict[str,BasicBlock],
             order: List[str]) -> Tuple[Dict[str,Set[str]], Dict[str,Set[str]]]:
    """
    Standard backward dataflow liveness on CFG blocks.
    Returns (live_in, live_out) maps keyed by block name.
    """
    use: Dict[str,Set[str]] = {}
    defs: Dict[str,Set[str]] = {}

    # Block-level use/def
    for name in order:
        b = blocks[name]
        u: Set[str] = set()
        d: Set[str] = set()
        for ins in b.instrs:
            uses_i, defs_i = _inst_uses_defs(ins)
            # use[b] ∪= uses_i \ d
            u |= (uses_i - d)
            # defs[b] ∪= defs_i
            d |= defs_i
        use[name] = u
        defs[name] = d

    live_in: Dict[str,Set[str]] = {name:set() for name in order}
    live_out: Dict[str,Set[str]] = {name:set() for name in order}

    changed = True
    while changed:
        changed = False
        # iterate in reverse order for faster convergence, but any order works
        for name in reversed(order):
            b = blocks[name]
            old_in = live_in[name].copy()
            old_out = live_out[name].copy()

            # out[b] = ⋃ in[s] over successors
            out_b: Set[str] = set()
            for s in b.succs:
                out_b |= live_in[s]
            live_out[name] = out_b

            # in[b] = use[b] ∪ (out[b] \ def[b])
            live_in[name] = use[name] | (out_b - defs[name])

            if live_in[name] != old_in or live_out[name] != old_out:
                changed = True

    return live_in, live_out

def dump_liveness(entry: str,
                  blocks: Dict[str,BasicBlock],
                  order: List[str],
                  live_in: Dict[str,Set[str]],
                  live_out: Dict[str,Set[str]]) -> None:
    print(f"ENTRY {entry}")
    for name in order:
        b = blocks[name]
        print(f"BLOCK {name}:")
        print(f"  succs: {', '.join(sorted(b.succs))}")
        print(f"  in:   {', '.join(sorted(live_in[name]))}")
        print(f"  out:  {', '.join(sorted(live_out[name]))}")

# =============================================================================
# TAC -> x64 (Linux SysV ABI)
# =============================================================================

def tac_to_x64(tac: List[Dict], out_s: str):
    body = tac[0]["body"]  # single @main
    # collect temps (exclude labels)
    temps: Set[str] = set()
    labels: Set[str] = set()
    for ins in body:
        op = ins["opcode"]
        if op=="label":
            labels.add(ins["args"][0])
        else:
            for a in ins.get("args", []):
                if isinstance(a, str) and a.startswith('%') and not a.startswith('%.L'):
                    temps.add(a)
            r = ins.get("result")
            if isinstance(r, str) and r.startswith('%') and not r.startswith('%.L'):
                temps.add(r)

    def tid(x:str)->int:
        try:
            return int(x[1:])  # %N -> N
        except Exception:
            err("Backend", f"non-temp passed to tid(): {x!r}")
            raise

    max_id = max([tid(t) for t in temps], default=-1)
    nslots = max_id + 1
    if nslots % 2 == 1: nslots += 1   # 16-byte alignment

    def slot(t:str)->int:
        i = tid(t) + 1
        return -8*i

    def mklbl(l:str)->str:
        if l.startswith('%.L'): return '.'+l[2:]
        return l.replace('%','')

    lines: List[str] = []
    emit = lines.append

    emit(".globl main")
    emit(".text")
    emit("main:")
    emit("  pushq %rbp")
    emit("  movq %rsp, %rbp")
    if nslots>0:
        emit(f"  subq ${8*nslots}, %rsp  # {nslots} slots")

    def load(src:str, reg:str):
        if isinstance(src, str) and src.startswith('%') and not src.startswith('%.L'):
            emit(f"  movq {slot(src)}(%rbp), {reg}")
        else:
            err("Backend", f"unexpected load operand {src!r}")
            raise RuntimeError("unexpected non-temp in load")

    def store(reg:str, dst:str):
        emit(f"  movq {reg}, {slot(dst)}(%rbp)")

    for ins in body:
        op = ins["opcode"]; args = ins.get("args", []); r = ins.get("result")
        if op=="const":
            v = args[0]; emit(f"  movq ${v}, %r11"); store("%r11", r); continue
        if op=="copy":
            a = args[0]; load(a, "%r11"); store("%r11", r); continue
        if op in ("neg","not"):
            a = args[0]; load(a, "%r11"); emit(f"  { 'negq' if op=='neg' else 'notq' } %r11"); store("%r11", r); continue
        if op in ("add","sub","and","or","xor"):
            a,b = args; load(a,"%r11"); emit(f"  {op}q {slot(b)}(%rbp), %r11"); store("%r11", r); continue
        if op == "mul":
            a, b = args
            load(a, "%r11")
            emit(f"  imulq {slot(b)}(%rbp), %r11")   # two-operand signed multiply
            store("%r11", r)
            continue
        if op in ("shl","shr"):
            a,b = args
            load(a,"%r11"); load(b,"%rcx")
            emit(f"  {'salq' if op=='shl' else 'sarq'} %cl, %r11")
            store("%r11", r); continue
        if op in ("div","mod"):
            a,b = args
            load(a, "%rax"); emit("  cqto")
            load(b, "%rbx"); emit("  idivq %rbx")
            store("%rax" if op=="div" else "%rdx", r); continue
        if op=="print":
            a = args[0]; load(a, "%rdi"); emit("  callq bx_print_int"); continue
        if op=="label":
            emit(f"{mklbl(args[0])}:"); continue
        if op=="br":
            emit(f"  jmp {mklbl(args[0])}"); continue
        if op=="br_if_true":
            a,lbl = args; load(a, "%r11"); emit("  cmpq $0, %r11"); emit(f"  jne {mklbl(lbl)}"); continue
        if op=="br_if_false":
            a,lbl = args; load(a, "%r11"); emit("  cmpq $0, %r11"); emit(f"  je {mklbl(lbl)}"); continue
        if op=="cmpflag":
            continue
        if op=="br_cmp2":
            diff, relop, Lt, Lf = args
            load(diff, "%r11"); emit("  cmpq $0, %r11")
            jcc = {'==':'je', '!=':'jne', '<':'jl', '<=':'jle', '>':'jg', '>=':'jge'}[relop]
            emit(f"  {jcc} {mklbl(Lt)}"); emit(f"  jmp {mklbl(Lf)}"); continue
        if op=="br_cmp_true":
            diff, relop, Lt = args
            load(diff, "%r11"); emit("  cmpq $0, %r11")
            jcc = {'==':'je', '!=':'jne', '<':'jl', '<=':'jle', '>':'jg', '>=':'jge'}[relop]
            emit(f"  {jcc} {mklbl(Lt)}"); continue
        err("Backend", f"unhandled opcode: {op} {args} -> {r}")
        emit(f"  # unhandled: {op} {args} -> {r}")

    emit("  movq %rbp, %rsp")
    emit("  popq %rbp")
    emit("  movq $0, %rax")
    emit("  retq")

    with open(out_s, "w", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")


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
    ap = argparse.ArgumentParser(description="BX Labs 3–5 driver")
    ap.add_argument("source", help=".bx file")
    ap.add_argument("--keep-tac", action="store_true",
                    help="also dump TAC JSON next to .s")
    ap.add_argument("--dump-cfg", action="store_true",
                    help="Lab4: build CFG from TAC and print it")
    ap.add_argument("--dump-liveness", action="store_true",
                    help="Lab5: run liveness analysis on CFG and print in/out sets")
    ap.add_argument("--debug", action="store_true",
                    help="enable debug prints")
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

    try:
        tac = gen_tac(ast)
        dbg("TAC OK")
    except Exception as e:
        err("Codegen", f"{e}")
        sys.exit(1)

    # If Lab4 / Lab5 modes requested, run analyses instead of x64 backend.
    if args.dump_cfg or args.dump_liveness:
        proc = tac[0]   # only @main for now
        entry, blocks, order = build_cfg_for_proc(proc)
        if args.dump_cfg:
            dump_cfg(entry, blocks, order)
        if args.dump_liveness:
            live_in, live_out = liveness(blocks, order)
            dump_liveness(entry, blocks, order, live_in, live_out)
        return

    base = os.path.splitext(args.source)[0]
    if args.keep_tac:
        try:
            with open(base + ".tac.json","w",encoding="utf-8") as f:
                json.dump(tac, f, indent=2)
        except OSError as e:
            err("IO", f"cannot write TAC JSON: {e}")

    out_s = base + ".s"
    try:
        tac_to_x64(tac, out_s)
    except Exception as e:
        err("Backend", f"{e}")
        sys.exit(1)

    print(out_s)

if __name__=="__main__":
    main()


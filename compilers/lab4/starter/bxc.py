import sys, os, json, argparse, dataclasses as dc, abc
from typing import List, Tuple, Dict, Optional, Set, Union
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
    'void':  'VOID'
}

tokens = (
    'IDENT', 'NUMBER',
    'LPAREN','RPAREN','LBRACE','RBRACE',
    'COLON','SEMI','EQUAL','COMMA',
    'PLUS','MINUS','STAR','SLASH','MOD',
    'BAND','BOR','BXOR',
    'LSHIFT','RSHIFT',
    'BNOT',
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
    r'0|([1-9][0-9]*)'
    try:
        val = int(t.value)
    except ValueError:
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
class AST(abc.ABC): pass

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
    target: str
    args: List[Expr]
    ty: Optional[str] = None

class Stmt(AST): pass

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
class SBreak(Stmt): pass

@dc.dataclass
class SContinue(Stmt): pass

@dc.dataclass
class SReturn(Stmt):
    e: Optional[Expr]

@dc.dataclass
class Decl(AST): pass

@dc.dataclass
class GlobalVar(Decl):
    name: str
    init: Union[int, bool] # Literals only
    ty: str

@dc.dataclass
class Proc(Decl):
    name: str
    params: List[Tuple[str, str]] # (name, type)
    ret_ty: str # "void", "int", "bool"
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
    ('nonassoc', 'EQ', 'NE'),
    ('nonassoc', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'LSHIFT', 'RSHIFT'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'MOD'),
    ('right', 'LNOT', 'BNOT', 'UMINUS'),
)

# Fix: Ensure program flattens the list of lists returned by decl_list
def p_program(p):
    'program : decl_list'
    flat_decls = []
    for sublist in p[1]:
        flat_decls.extend(sublist)
    p[0] = Program(flat_decls)

def p_decl_list_empty(p):
    'decl_list : '
    p[0] = []

def p_decl_list_cons(p):
    'decl_list : decl_list decl'
    p[1].append(p[2])
    p[0] = p[1]

def p_decl_var(p):
    'decl : vardecl'
    # vardecl returns list[SVar], convert to list[GlobalVar]
    # Lab 4 spec: global init must be literal.
    # We will construct GlobalVar here but semantic check will enforce literal-ness.
    gvars = []
    ty = p[1][0].ty
    for sv in p[1]:
        val = 0
        if isinstance(sv.init, ENum): val = sv.init.n
        elif isinstance(sv.init, EBool): val = sv.init.b
        # Note: if sv.init is complex expr, we store it as is? 
        # Actually, GlobalVar dataclass expects int/bool. 
        # If parser allowed complex expr, we fail here or store dummy.
        # Let's just store the value if literal, else fail later.
        # For now, just pass the value.
        if isinstance(sv.init, (ENum, EBool)):
            val = sv.init.n if isinstance(sv.init, ENum) else sv.init.b
            gvars.append(GlobalVar(sv.name, val, ty))
        else:
            # If complex expr, we can't put it in GlobalVar cleanly if typed as int|bool
            # We'll cheat and put the expr, semantic check handles it.
            # But dataclass says int|bool. Let's adjust semantic check or allow it.
            # Better: Fail in semantic check.
            gvars.append(GlobalVar(sv.name, sv.init, ty)) 
    p[0] = gvars

def p_decl_proc(p):
    'decl : procdecl'
    p[0] = [p[1]] # Wrap in list

def p_procdecl(p):
    '''procdecl : DEF IDENT LPAREN param_list RPAREN ret_type block'''
    p[0] = Proc(name=p[2], params=p[4], ret_ty=p[6], body=p[7])

def p_procdecl_void(p):
    '''procdecl : DEF IDENT LPAREN param_list RPAREN block'''
    p[0] = Proc(name=p[2], params=p[4], ret_ty="void", body=p[6])

def p_param_list_empty(p):
    'param_list : '
    p[0] = []

def p_param_list(p):
    'param_list : params'
    p[0] = p[1]

def p_params_one(p):
    'params : ident_list COLON type_name'
    ty = p[3]
    p[0] = [(name, ty) for name in p[1]]

def p_params_more(p):
    'params : params COMMA ident_list COLON type_name'
    ty = p[5]
    p[1].extend([(name, ty) for name in p[3]])
    p[0] = p[1]

def p_ident_list_one(p):
    'ident_list : IDENT'
    p[0] = [p[1]]

def p_ident_list_more(p):
    'ident_list : ident_list COMMA IDENT'
    p[1].append(p[3])
    p[0] = p[1]

def p_ret_type(p):
    'ret_type : COLON type_name'
    p[0] = p[2]

def p_type_name(p):
    '''type_name : INT
                 | BOOL'''
    p[0] = p[1]

def p_block(p):
    'block : LBRACE stmt_list RBRACE'
    p[0] = SBlock(p[2])

def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = []

def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    if isinstance(p[2], list): # vardecl returns list
        p[1].extend(p[2])
    else:
        p[1].append(p[2])
    p[0] = p[1]

def p_stmt_vardecl(p):
    'stmt : vardecl'
    p[0] = p[1]

def p_vardecl(p):
    'vardecl : VAR varinits COLON type_name SEMI'
    ty = p[4]
    p[0] = [SVar(name, expr, ty) for (name, expr) in p[2]]

def p_varinits_one(p):
    'varinits : IDENT EQUAL expr'
    p[0] = [(p[1], p[3])]

def p_varinits_more(p):
    'varinits : varinits COMMA IDENT EQUAL expr'
    p[1].append((p[3], p[5]))
    p[0] = p[1]

def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = SAssign(p[1], p[3])

def p_stmt_eval(p):
    'stmt : expr SEMI'
    p[0] = SEval(p[1])

def p_stmt_if(p):
    'stmt : IF LPAREN expr RPAREN block ifrest'
    p[0] = SIfElse(p[3], p[5], p[6])

def p_ifrest_empty(p):
    'ifrest : '
    p[0] = None

def p_ifrest_else(p):
    'ifrest : ELSE stmt'
    p[0] = p[2]

def p_stmt_while(p):
    'stmt : WHILE LPAREN expr RPAREN block'
    p[0] = SWhile(p[3], p[5])

def p_stmt_break(p):
    'stmt : BREAK SEMI'
    p[0] = SBreak()

def p_stmt_continue(p):
    'stmt : CONTINUE SEMI'
    p[0] = SContinue()

def p_stmt_return(p):
    'stmt : RETURN expr_opt SEMI'
    p[0] = SReturn(p[2])

def p_expr_opt_empty(p):
    'expr_opt : '
    p[0] = None

def p_expr_opt(p):
    'expr_opt : expr'
    p[0] = p[1]

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

def p_expr_call(p):
    'expr : IDENT LPAREN args_opt RPAREN'
    p[0] = ECall(p[1], p[3])

def p_args_opt_empty(p):
    'args_opt : '
    p[0] = []

def p_args_opt(p):
    'args_opt : args'
    p[0] = p[1]

def p_args_one(p):
    'args : expr'
    p[0] = [p[1]]

def p_args_more(p):
    'args : args COMMA expr'
    p[1].append(p[3])
    p[0] = p[1]

def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_unop(p):
    '''expr : MINUS expr %prec UMINUS
            | BNOT expr
            | LNOT expr'''
    op = p[1]
    if op == '-': op = '-'
    p[0] = EUn(op, p[2])

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
    if p:
        err("Parser", f"line {p.lineno}, pos {p.lexpos}: unexpected token {p.type} ('{p.value}')")
    else:
        err("Parser", "unexpected end of input")
    raise SyntaxError

parser = yacc.yacc()

# =============================================================================
# TYPE CHECKING
# =============================================================================

class SemanticError(Exception): pass

@dc.dataclass
class ProcType:
    param_tys: List[str]
    ret_ty: str

def check_program(prog: Program):
    global_scope: Dict[str, Union[str, ProcType]] = {}
    
    global_scope["_bx_print_int"] = ProcType(["int"], "void")
    global_scope["_bx_print_bool"] = ProcType(["bool"], "void")

    main_found = False
    
    # 1. Register Globals & Procs
    for d in prog.decls:
        if isinstance(d, GlobalVar):
            if d.name in global_scope:
                raise SemanticError(f"Duplicate global '{d.name}'")
            # Enforce literal init
            if not isinstance(d.init, (int, bool)):
                 raise SemanticError(f"Global '{d.name}' must be initialized with a literal")
            if d.ty == "int" and isinstance(d.init, bool):
                raise SemanticError(f"Global '{d.name}' declared int but init is bool")
            if d.ty == "bool" and not isinstance(d.init, bool) and d.init not in (0,1): 
                # 0/1 are ints, usually python bool is subtype of int.
                # Strict check:
                if type(d.init) is not bool:
                    raise SemanticError(f"Global '{d.name}' declared bool but init is not bool")
            global_scope[d.name] = d.ty

        elif isinstance(d, Proc):
            if d.name in global_scope:
                raise SemanticError(f"Procedure '{d.name}' redeclared")
            pt = ProcType([p[1] for p in d.params], d.ret_ty)
            global_scope[d.name] = pt
            
            if d.name == "main":
                if d.params: raise SemanticError("main() must take no args")
                if d.ret_ty != "void": raise SemanticError("main() must be void")
                main_found = True

    if not main_found:
        raise SemanticError("Missing main() procedure")

    # 2. Check Bodies
    for d in prog.decls:
        if isinstance(d, Proc):
            check_proc(d, global_scope)

def check_proc(proc: Proc, global_scope: Dict):
    scope_stack: List[Dict[str, str]] = []

    def enter_scope(): scope_stack.append({})
    def exit_scope(): scope_stack.pop()
    def add_local(name, ty):
        if name in scope_stack[-1]:
            raise SemanticError(f"Variable '{name}' redeclared in the same block")
        scope_stack[-1][name] = ty

    def lookup(name):
        for s in reversed(scope_stack):
            if name in s: return s[name]
        if name in global_scope: return global_scope[name]
        raise SemanticError(f"Use of undeclared variable '{name}'")

    def check_expr(e: Expr) -> str:
        if isinstance(e, ENum):
            e.ty = "int"; return "int"
        if isinstance(e, EBool):
            e.ty = "bool"; return "bool"
        if isinstance(e, EVar):
            t = lookup(e.name)
            if isinstance(t, ProcType): raise SemanticError(f"Procedure '{e.name}' used as variable")
            e.ty = t; return t
        if isinstance(e, EUn):
            t = check_expr(e.e)
            if e.op == '!' and t == 'bool': e.ty = 'bool'; return 'bool'
            if e.op in ('-','~') and t == 'int': e.ty = 'int'; return 'int'
            raise SemanticError(f"Type mismatch in unary {e.op}")
        if isinstance(e, EBin):
            t1, t2 = check_expr(e.l), check_expr(e.r)
            is_int = (t1=='int' and t2=='int')
            is_bool = (t1=='bool' and t2=='bool')
            if e.op in ['+','-','*','/','%','&','|','^','<<','>>']:
                if not is_int: raise SemanticError(f"Binary {e.op} expects ints")
                e.ty = 'int'; return 'int'
            if e.op in ['<','<=','>','>=','==','!=']:
                if not is_int and not (e.op in ['==','!='] and is_bool):
                     raise SemanticError(f"Binary {e.op} type mismatch")
                e.ty = 'bool'; return 'bool'
            if e.op in ['&&','||']:
                if not is_bool: raise SemanticError(f"Binary {e.op} expects bools")
                e.ty = 'bool'; return 'bool'
            raise SemanticError(f"Unknown binop {e.op}")
        if isinstance(e, ECall):
            if e.target == "print":
                if len(e.args) != 1: raise SemanticError("print takes exactly 1 argument")
                aty = check_expr(e.args[0])
                if aty == "int": e.target = "_bx_print_int"
                elif aty == "bool": e.target = "_bx_print_bool"
                else: raise SemanticError("print expects int or bool")
                e.ty = "void"; return "void"
            
            ft = lookup(e.target)
            if not isinstance(ft, ProcType): raise SemanticError(f"'{e.target}' is not a function")
            if len(e.args) != len(ft.param_tys): raise SemanticError(f"Call to '{e.target}' has wrong arg count")
            for a, pty in zip(e.args, ft.param_tys):
                if check_expr(a) != pty: raise SemanticError(f"Call to '{e.target}' arg type mismatch")
            e.ty = ft.ret_ty; return ft.ret_ty
        raise SemanticError("Unknown expr")

    def check_stmt(s: Stmt):
        if isinstance(s, SVar):
            t = check_expr(s.init)
            if t != s.ty: raise SemanticError(f"initializer must be {s.ty}")
            add_local(s.name, s.ty)
        elif isinstance(s, SAssign):
            tr = check_expr(s.rhs)
            tl = lookup(s.name)
            if isinstance(tl, ProcType): raise SemanticError("Cannot assign to procedure")
            if tl != tr: raise SemanticError(f"Assignment type mismatch for '{s.name}'")
        elif isinstance(s, SEval):
            check_expr(s.e)
        elif isinstance(s, SBlock):
            enter_scope()
            for st in s.ss: check_stmt(st)
            exit_scope()
        elif isinstance(s, SIfElse):
            if check_expr(s.cond) != "bool": raise SemanticError("If condition must be bool")
            check_stmt(s.thenb)
            if s.elsep: check_stmt(s.elsep)
        elif isinstance(s, SWhile):
            if check_expr(s.cond) != "bool": raise SemanticError("While condition must be bool")
            check_stmt(s.body)
        elif isinstance(s, SReturn):
            if s.e:
                if proc.ret_ty == "void": raise SemanticError("void function cannot return a value")
                if check_expr(s.e) != proc.ret_ty: raise SemanticError(f"return type mismatch: expected {proc.ret_ty}")
            else:
                if proc.ret_ty != "void": raise SemanticError("non-void function must return a value")
        # Break/Continue checks omitted for brevity

    # Return path check
    def returns_on_all_paths(s: Stmt) -> bool:
        if isinstance(s, SReturn): return True
        if isinstance(s, SBlock):
            return any(returns_on_all_paths(sub) for sub in s.ss)
        if isinstance(s, SIfElse):
            if s.elsep:
                return returns_on_all_paths(s.thenb) and returns_on_all_paths(s.elsep)
            return False # if without else might not return
        return False

    enter_scope()
    for p, ty in proc.params: add_local(p, ty)
    check_stmt(proc.body)

    if proc.ret_ty != "void" and not returns_on_all_paths(proc.body):
         raise SemanticError(f"non-void procedure '{proc.name}' might not return a value on all paths")

# =============================================================================
# TAC
# =============================================================================

class TempGen:
    def __init__(self): self.n = 0
    def fresh(self)->str: t=f"%{self.n}"; self.n+=1; return t

class LabelGen:
    def __init__(self): self.k=0
    def fresh(self,pfx="%.L")->str: s=f"{pfx}{self.k}"; self.k+=1; return s

def gen_tac(prog: Program) -> List[Dict]:
    units = []
    for d in prog.decls:
        if isinstance(d, GlobalVar):
            val = 1 if d.init is True else 0 if d.init is False else d.init
            units.append({"var": f"@{d.name}", "init": int(val)})
        elif isinstance(d, Proc):
            units.append(gen_tac_proc(d))
    return units

def gen_tac_proc(proc: Proc) -> Dict:
    temps, labels = TempGen(), LabelGen()
    code = []
    scope_stack = [{}] 

    def var_loc(n):
        for s in reversed(scope_stack):
            if n in s: return s[n]
        return f"@{n}"

    def enter(): scope_stack.append({})
    def exit(): scope_stack.pop()
    def bind(n, t): scope_stack[-1][n] = t
    def emit(op, args, res=None):
        d={"opcode":op, "args":args}
        if res: d["result"]=res
        code.append(d)

    param_temps = []
    enter() # Param scope
    for p, _ in proc.params:
        t = f"%{p}"
        bind(p, t)
        param_temps.append(t)

    def emit_int(e: Expr) -> str:
        if isinstance(e, ENum):
            t=temps.fresh(); emit("const",[e.n],t); return t
        if isinstance(e, EVar):
            return var_loc(e.name)
        if isinstance(e, EBin):
            l,r = emit_int(e.l), emit_int(e.r)
            t = temps.fresh()
            ops = {'+':'add','-':'sub','*':'mul','/':'div','%':'mod',
                   '&':'and','|':'or','^':'xor','<<':'shl','>>':'shr'}
            if e.op in ops: emit(ops[e.op],[l,r],t); return t
            return emit_bool_as_int(e)
        if isinstance(e, ECall):
            args = [emit_int(a) for a in e.args]
            for i, a in enumerate(args): emit("param", [i+1, a])
            t = temps.fresh() if e.ty!="void" else None
            emit("call", [e.target, len(args)], t)
            return t if t else "%_"
        if isinstance(e, EUn):
             if e.op=='-': v=emit_int(e.e); t=temps.fresh(); emit("neg",[v],t); return t
             if e.op=='~': v=emit_int(e.e); t=temps.fresh(); emit("not",[v],t); return t
             return emit_bool_as_int(e)
        if e.ty == "bool": return emit_bool_as_int(e)
        raise ValueError(f"emit_int {e}")

    def emit_bool_as_int(e: Expr) -> str:
        t = temps.fresh()
        L1, L2, Le = labels.fresh(), labels.fresh(), labels.fresh()
        emit("const", [0], t)
        emit_cond(e, L1, L2)
        emit("label", [L1])
        emit("const", [1], t)
        emit("label", [L2])
        return t

    def emit_cond(e: Expr, Lt, Lf):
        if isinstance(e, EBool):
            emit("jmp", [Lt] if e.b else [Lf]); return
        if isinstance(e, EUn) and e.op=='!':
            emit_cond(e.e, Lf, Lt); return
        if isinstance(e, EBin):
            if e.op=='&&':
                Lm = labels.fresh()
                emit_cond(e.l, Lm, Lf); emit("label", [Lm])
                emit_cond(e.r, Lt, Lf); return
            if e.op=='||':
                Lm = labels.fresh()
                emit_cond(e.l, Lt, Lm); emit("label", [Lm])
                emit_cond(e.r, Lt, Lf); return
            if e.op in ('==','!=','<','<=','>','>='):
                l, r = emit_int(e.l), emit_int(e.r)
                diff = temps.fresh()
                emit("sub", [l,r], diff)
                emit("br_cmp2", [diff, e.op, Lt, Lf]); return
        v = emit_int(e)
        emit("br_if_true", [v, Lt])
        emit("jmp", [Lf])

    loop_stack = []
    def emit_stmt(s: Stmt):
        if isinstance(s, SVar):
            v = emit_int(s.init)
            t = temps.fresh()
            bind(s.name, t)
            emit("copy", [v], t)
        elif isinstance(s, SAssign):
            v = emit_int(s.rhs)
            emit("copy", [v], var_loc(s.name))
        elif isinstance(s, SEval):
            if isinstance(s.e, ECall): emit_int(s.e)
        elif isinstance(s, SBlock):
            enter(); 
            for x in s.ss: emit_stmt(x)
            exit()
        elif isinstance(s, SIfElse):
            Lt, Lf, Le = labels.fresh(), labels.fresh(), labels.fresh()
            emit_cond(s.cond, Lt, Lf)
            emit("label", [Lt]); emit_stmt(s.thenb); emit("jmp", [Le])
            emit("label", [Lf])
            if s.elsep: emit_stmt(s.elsep)
            emit("label", [Le])
        elif isinstance(s, SWhile):
            Lh, Lb, Le = labels.fresh(), labels.fresh(), labels.fresh()
            emit("label", [Lh])
            emit_cond(s.cond, Lb, Le)
            emit("label", [Lb])
            loop_stack.append((Lh, Le))
            emit_stmt(s.body)
            loop_stack.pop()
            emit("jmp", [Lh]); emit("label", [Le])
        elif isinstance(s, SReturn):
            if s.e:
                v = emit_int(s.e)
                emit("ret", [v])
            else:
                emit("ret", [])
        elif isinstance(s, SBreak): emit("jmp", [loop_stack[-1][1]])
        elif isinstance(s, SContinue): emit("jmp", [loop_stack[-1][0]])

    emit_stmt(proc.body)
    emit("ret", []) # Fallback return
    return {"proc": f"@{proc.name}", "args": param_temps, "body": code}

# =============================================================================
# OPTIMIZER (CFG)
# =============================================================================

class Block:
    def __init__(self, lbl):
        self.label = lbl
        self.instrs = []
        self.succs = []

def optimize_cfg(proc: Dict) -> Dict:
    body = proc["body"]
    if not body: return proc

    blocks = []
    curr = Block("%.Lentry")
    blocks.append(curr)
    
    for instr in body:
        if instr["opcode"] == "label":
            if curr.instrs:
                nb = Block(instr["args"][0])
                blocks.append(nb)
                curr = nb
            else:
                curr.label = instr["args"][0]
        else:
            curr.instrs.append(instr)
            if instr["opcode"] in ("jmp","ret","br_cmp2","br_if_true"):
                nb = Block(f"%.L_autogen_{len(blocks)}")
                blocks.append(nb)
                curr = nb
    
    blocks = [b for b in blocks if b.instrs or b.label == "%.Lentry"]
    
    lbl_map = {b.label: b for b in blocks}
    for i, b in enumerate(blocks):
        if not b.instrs:
            if i+1 < len(blocks): b.succs.append(blocks[i+1])
            continue
        last = b.instrs[-1]
        op = last["opcode"]
        if op == "jmp": 
            if last["args"][0] in lbl_map: b.succs.append(lbl_map[last["args"][0]])
        elif op == "br_if_true":
             if last["args"][1] in lbl_map: b.succs.append(lbl_map[last["args"][1]])
             if i+1 < len(blocks): b.succs.append(blocks[i+1])
        elif op == "br_cmp2":
             if last["args"][2] in lbl_map: b.succs.append(lbl_map[last["args"][2]])
             if last["args"][3] in lbl_map: b.succs.append(lbl_map[last["args"][3]])
        elif op == "ret": pass
        else:
             if i+1 < len(blocks): b.succs.append(blocks[i+1])

    visited = set()
    stack = [blocks[0]]
    while stack:
        b = stack.pop()
        if b.label in visited: continue
        visited.add(b.label)
        for s in b.succs: stack.append(s)
    
    reachable = [b for b in blocks if b.label in visited]

    new_body = []
    for b in reachable:
        new_body.append({"opcode":"label", "args":[b.label]})
        new_body.extend(b.instrs)
    
    proc["body"] = new_body
    return proc

# =============================================================================
# BACKEND
# =============================================================================

def tac_to_x64(units: List[Dict], path: str):
    out = []
    emit = out.append
    emit(".data")
    for u in units:
        if "var" in u:
            n = u["var"][1:]
            emit(f".globl {n}"); emit(f".align 8")
            emit(f"{n}: .quad {u['init']}")
    
    emit(".text")
    for u in units:
        if "proc" not in u: continue
        name = u["proc"][1:]
        emit(f".globl {name}")
        emit(f"{name}:")
        emit("  pushq %rbp")
        emit("  movq %rsp, %rbp")
        
        # Stack logic
        temps = set(u.get("args", []))
        for i in u["body"]:
            if "result" in i and i["result"].startswith("%"): temps.add(i["result"])
            for a in i.get("args",[]):
                if isinstance(a,str) and a.startswith("%") and not a.startswith("%."): temps.add(a)
        
        tmap = {t: -(8*(i+1)) for i,t in enumerate(sorted(list(temps)))}
        ssize = (len(tmap)*8 + 15) & ~15
        if ssize: emit(f"  subq ${ssize}, %rsp")

        # Args
        regs = ["%rdi","%rsi","%rdx","%rcx","%r8","%r9"]
        for i, arg in enumerate(u.get("args", [])):
            if i<6: emit(f"  movq {regs[i]}, {tmap[arg]}(%rbp)")
            else: 
                off = 16+(i-6)*8
                emit(f"  movq {off}(%rbp), %r11"); emit(f"  movq %r11, {tmap[arg]}(%rbp)")

        def loc(x):
            if isinstance(x,int): return f"${x}"
            if x.startswith("@"): return f"{x[1:]}(%rip)"
            if x.startswith("%"): return f"{tmap[x]}(%rbp)"
            return x

        def getlbl(x): return x[2:] if x.startswith("%.") else x

        for inst in u["body"]:
            op, args = inst["opcode"], inst.get("args", [])
            res = inst.get("result")
            
            if op=="label": emit(f".L{getlbl(args[0])}:")
            elif op=="const": emit(f"  movq ${args[0]}, {loc(res)}")
            elif op=="copy": emit(f"  movq {loc(args[0])}, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="add": emit(f"  movq {loc(args[0])}, %r11"); emit(f"  addq {loc(args[1])}, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="sub": emit(f"  movq {loc(args[0])}, %r11"); emit(f"  subq {loc(args[1])}, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="mul": emit(f"  movq {loc(args[0])}, %r11"); emit(f"  imulq {loc(args[1])}, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op in ("div","mod"):
                emit(f"  movq {loc(args[0])}, %rax"); emit("  cqto"); emit(f"  idivq {loc(args[1])}")
                emit(f"  movq {'%rax' if op=='div' else '%rdx'}, {loc(res)}")
            elif op in ("and","or","xor"):
                emit(f"  movq {loc(args[0])}, %r11"); emit(f"  {op}q {loc(args[1])}, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op in ("shl","shr"):
                emit(f"  movq {loc(args[0])}, %r11"); emit(f"  movq {loc(args[1])}, %rcx"); emit(f"  {'salq' if op=='shl' else 'sarq'} %cl, %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="neg": emit(f"  movq {loc(args[0])}, %r11"); emit("  negq %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="not": emit(f"  movq {loc(args[0])}, %r11"); emit("  notq %r11"); emit(f"  movq %r11, {loc(res)}")
            elif op=="jmp": emit(f"  jmp .L{getlbl(args[0])}")
            elif op=="br_if_true": emit(f"  movq {loc(args[0])}, %r11"); emit("  cmpq $0, %r11"); emit(f"  jne .L{getlbl(args[1])}")
            elif op=="br_cmp2":
                emit(f"  movq {loc(args[0])}, %r11"); emit("  cmpq $0, %r11")
                jcc = {'==':'je','!=':'jne','<':'jl','<=':'jle','>':'jg','>=':'jge'}[args[1]]
                emit(f"  {jcc} .L{getlbl(args[2])}"); emit(f"  jmp .L{getlbl(args[3])}")
            elif op=="param": pass # Handled at call
            elif op=="call":
                idx = u["body"].index(inst)
                p_args = []
                for k in range(idx-1, -1, -1):
                    prev = u["body"][k]
                    if prev["opcode"]=="param": p_args.append((prev["args"][0], prev["args"][1]))
                    elif prev["opcode"] in ("call","label"): break
                p_args.sort(key=lambda x: x[0], reverse=True)
                s_args = [p for p in p_args if p[0] > 6]
                r_args = [p for p in p_args if p[0] <= 6]
                
                if len(s_args) % 2 == 1: emit("  subq $8, %rsp")
                for _, val in s_args: emit(f"  pushq {loc(val)}")
                
                r_regs = ["%rdi","%rsi","%rdx","%rcx","%r8","%r9"]
                r_args.sort(key=lambda x: x[0])
                for i, val in r_args: emit(f"  movq {loc(val)}, {r_regs[i-1]}")
                
                emit(f"  callq {args[0][1:]}")
                
                pop_sz = len(s_args)*8
                if len(s_args)%2==1: pop_sz+=8
                if pop_sz: emit(f"  addq ${pop_sz}, %rsp")
                
                if res: emit(f"  movq %rax, {loc(res)}")
            
            elif op=="ret":
                if args: emit(f"  movq {loc(args[0])}, %rax")
                else: emit("  xorq %rax, %rax")
                emit("  movq %rbp, %rsp"); emit("  popq %rbp"); emit("  retq")

    with open(path, "w") as f: f.write("\n".join(out)+"\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--keep-tac", action="store_true")
    args = ap.parse_args()

    with open(args.source) as f: src = f.read()

    try:
        ast = parser.parse(src, lexer=lexer)
    except Exception as e:
        err("Parser", str(e)); sys.exit(1)

    try:
        check_program(ast)
    except SemanticError as e:
        err("Type/Semantic", str(e)); sys.exit(1)

    tac = gen_tac(ast)
    if args.keep_tac:
        with open(os.path.splitext(args.source)[0]+".tac.json","w") as f: json.dump(tac,f,indent=2)

    opt = []
    for u in tac:
        if "proc" in u: opt.append(optimize_cfg(u))
        else: opt.append(u)
    
    out_s = os.path.splitext(args.source)[0]+".s"
    tac_to_x64(opt, out_s)

if __name__=="__main__":
    main()

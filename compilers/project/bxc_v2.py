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
class BXError(Exception): pass
class TypeErrorBX(Exception): pass
class SemErrorBX(Exception): pass

# =============================================================================
# LEXER
# =============================================================================
reserved = {
    'def': 'DEF', 'var': 'VAR', 'int': 'INT', 'bool': 'BOOL', 'void': 'VOID',
    'function': 'FUNCTION', 'true': 'TRUE', 'false': 'FALSE', 'if': 'IF',
    'else': 'ELSE', 'while': 'WHILE', 'break': 'BREAK', 'continue': 'CONTINUE',
    'return': 'RETURN', 'ret': 'RET',
}
tokens = ('IDENT', 'NUM', 'PLUS', 'MINUS', 'TIMES', 'DIV', 'MOD', 'BAND', 'BOR', 'BXOR',
    'RSHIFT', 'LSHIFT', 'BNOT', 'EQUAL', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'COMMA', 'ARROW', 'LNOT', 'LAND', 'LOR', 'EQ', 'NEQ', 'LT', 'LE', 'GT', 'GE'
) + tuple(reserved.values())

t_PLUS = r'\+'; t_MINUS = r'-'; t_TIMES = r'\*'; t_DIV = r'/'; t_MOD = r'%'
t_BAND = r'&'; t_BOR = r'\|'; t_BXOR = r'\^'; t_RSHIFT = r'>>'; t_LSHIFT = r'<<'; t_BNOT = r'~'
t_EQUAL = r'='; t_LPAREN = r'\('; t_RPAREN = r'\)'; t_LBRACE = r'\{'; t_RBRACE = r'\}'
t_COLON = r':'; t_SEMI = r';'; t_COMMA = r','; t_ARROW = r'->'
t_LNOT = r'!'; t_LAND = r'&&'; t_LOR = r'\|\|'; t_EQ = r'=='; t_NEQ = r'!='; t_LT = r'<'; t_LE = r'<='; t_GT = r'>'; t_GE = r'>='

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
def t_error(t): raise BXError(f"Illegal character {t.value[0]!r} at line {t.lineno}")
lexer = lex.lex()

# =============================================================================
# AST
# =============================================================================
class AST: pass
Ty = Union[str, 'FunTy']   
@dc(frozen=True)
class FunTy(AST):
    param_tys: Tuple[Ty, ...]
    ret_ty: str  
class Expr(AST): ty: Ty  
@dc
class ENum(Expr): n: int
@dc
class EBool(Expr): b: bool
@dc
class EVar(Expr): name: str
@dc
class EUn(Expr): op: str; e: Expr
@dc
class EBin(Expr): op: str; l: Expr; r: Expr
@dc
class ECall(Expr): name: str; args: List[Expr]
class Stmt(AST): pass
@dc
class SBlock(Stmt): ss: List[Stmt]
@dc
class SIfElse(Stmt): cond: Expr; thenb: Stmt; elsep: Optional[Stmt]
@dc
class SWhile(Stmt): cond: Expr; body: Stmt
@dc
class SBreak(Stmt): pass
@dc
class SContinue(Stmt): pass
@dc
class SVar(Stmt): name: str; init: Expr; ty_annot: Ty; vid: int = -1         
@dc
class SAssign(Stmt): name: str; e: Expr
@dc
class SExpr(Stmt): e: Expr
@dc
class SReturn(Stmt): e: Optional[Expr]  
@dc
class SProcDef(Stmt): proc: 'ProcDecl'
@dc
class Param(AST): name: str; ty: Ty; vid: int = -1  
@dc
class ProcDecl(AST): name: str; params: List[Param]; ret_ty: Ty; body: SBlock; captures: Set[int] = field(default_factory=set)
@dc
class Program(AST): procs: List[ProcDecl]

# =============================================================================
# PARSER
# =============================================================================
precedence = (('left', 'LOR'), ('left', 'LAND'), ('left', 'EQ', 'NEQ'), ('left', 'LT', 'LE', 'GT', 'GE'), ('left', 'PLUS', 'MINUS'), ('left', 'TIMES', 'DIV', 'MOD'), ('right', 'LNOT', 'BNOT'), ('right', 'ELSE'))
def p_program(p): 'program : procs'; p[0] = Program(p[1])
def p_procs_single(p): 'procs : proc'; p[0] = [p[1]]
def p_procs_many(p): 'procs : procs proc'; p[1].append(p[2]); p[0] = p[1]
def p_proc(p): 'proc : DEF IDENT LPAREN params RPAREN ret_annot block'; p[0] = ProcDecl(p[2], p[4], p[6], p[7])
def p_params_empty(p): 'params : '; p[0] = []
def p_params_nonempty(p): 'params : param_list'; p[0] = p[1]
def p_param_list_one(p): 'param_list : IDENT COLON type'; p[0] = [Param(p[1], p[3])]
def p_param_list_cons(p): 'param_list : param_list COMMA IDENT COLON type'; p[1].append(Param(p[3], p[5])); p[0] = p[1]
def p_type_int(p): 'type : INT'; p[0] = 'int'
def p_type_bool(p): 'type : BOOL'; p[0] = 'bool'
def p_type_fun(p): 'type : FUNCTION LPAREN type_list_opt RPAREN ARROW funrettype'; p[0] = FunTy(tuple(p[3]), p[6])
def p_type_list_opt_empty(p): 'type_list_opt : '; p[0] = []
def p_type_list_opt_list(p): 'type_list_opt : type_list'; p[0] = p[1]
def p_type_list_one(p): 'type_list : type'; p[0] = [p[1]]
def p_type_list_cons(p): 'type_list : type_list COMMA type'; p[1].append(p[3]); p[0] = p[1]
def p_funrettype_int(p): 'funrettype : INT'; p[0] = 'int'
def p_funrettype_bool(p): 'funrettype : BOOL'; p[0] = 'bool'
def p_funrettype_void(p): 'funrettype : VOID'; p[0] = 'void'
def p_ret_annot_void(p): 'ret_annot : '; p[0] = 'void'
def p_ret_annot_ty(p): 'ret_annot : COLON funrettype'; p[0] = p[2]
def p_block(p): 'block : LBRACE stmt_list RBRACE'; p[0] = SBlock(p[2])
def p_stmt_list_empty(p): 'stmt_list : '; p[0] = []
def p_stmt_list_cons(p): 'stmt_list : stmt_list stmt'; p[1].append(p[2]); p[0] = p[1]
def p_stmt_procdef(p): 'stmt : DEF IDENT LPAREN params RPAREN ret_annot block'; p[0] = SProcDef(ProcDecl(p[2], p[4], p[6], p[7]))
def p_stmt_vardecl(p): 'stmt : VAR IDENT EQUAL expr COLON type SEMI'; p[0] = SVar(p[2], p[4], p[6])
def p_stmt_assign(p): 'stmt : IDENT EQUAL expr SEMI'; p[0] = SAssign(p[1], p[3])
def p_stmt_expr(p): 'stmt : expr SEMI'; p[0] = SExpr(p[1])
def p_stmt_block(p): 'stmt : block'; p[0] = p[1]
def p_stmt_if(p): 'stmt : IF LPAREN expr RPAREN stmt %prec ELSE'; p[0] = SIfElse(p[3], p[5], None)
def p_stmt_if_else(p): 'stmt : IF LPAREN expr RPAREN stmt ELSE stmt'; p[0] = SIfElse(p[3], p[5], p[7])
def p_stmt_while(p): 'stmt : WHILE LPAREN expr RPAREN stmt'; p[0] = SWhile(p[3], p[5])
def p_stmt_break(p): 'stmt : BREAK SEMI'; p[0] = SBreak()
def p_stmt_continue(p): 'stmt : CONTINUE SEMI'; p[0] = SContinue()
def p_stmt_return_void(p): 'stmt : RETURN SEMI'; p[0] = SReturn(None)
def p_stmt_return_val(p): 'stmt : RETURN expr SEMI'; p[0] = SReturn(p[2])
def p_stmt_ret_short(p): 'stmt : RET SEMI'; p[0] = SReturn(None)
def p_expr_num(p): 'expr : NUM'; p[0] = ENum(p[1])
def p_expr_true(p): 'expr : TRUE'; p[0] = EBool(True)
def p_expr_false(p): 'expr : FALSE'; p[0] = EBool(False)
def p_expr_var(p): 'expr : IDENT'; p[0] = EVar(p[1])
def p_expr_parens(p): 'expr : LPAREN expr RPAREN'; p[0] = p[2]
def p_expr_unary(p): 
    '''expr : LNOT expr 
            | MINUS expr %prec LNOT 
            | BNOT expr %prec LNOT'''
    p[0] = EUn(p[1], p[2])
def p_expr_binary(p):
    '''expr : expr PLUS expr | expr MINUS expr | expr TIMES expr | expr DIV expr | expr MOD expr
            | expr BAND expr | expr BOR expr | expr BXOR expr | expr RSHIFT expr | expr LSHIFT expr
            | expr EQ expr | expr NEQ expr | expr LT expr | expr LE expr | expr GT expr | expr GE expr
            | expr LAND expr | expr LOR expr'''
    p[0] = EBin(p[2], p[1], p[3])
def p_expr_call(p): 'expr : IDENT LPAREN arglist RPAREN'; p[0] = ECall(p[1], p[3])
def p_arglist_empty(p): 'arglist : '; p[0] = []
def p_arglist_nonempty(p): 'arglist : expr_list'; p[0] = p[1]
def p_expr_list_one(p): 'expr_list : expr'; p[0] = [p[1]]
def p_expr_list_many(p): 'expr_list : expr_list COMMA expr'; p[1].append(p[3]); p[0] = p[1]
def p_error(p): raise BXError(f"[Parser] Syntax error at token {p.type}" if p else "EOF")
parser = yacc.yacc(start='program')

# =============================================================================
# TYPE CHECKER
# =============================================================================
VarInfo = Tuple[Ty, int]
def check_program(prog: Program) -> None:
    fun_env_global: Dict[str, FunTy] = {}
    for pd in prog.procs:
        param_tys = [p.ty for p in pd.params]
        fun_env_global[pd.name] = FunTy(tuple(param_tys), pd.ret_ty)
    next_var_id = 0
    def fresh_var_id() -> int:
        nonlocal next_var_id
        vid = next_var_id
        next_var_id += 1
        return vid
    def typecheck_proc(pd: ProcDecl, fun_ty: FunTy, outer_var_env: List[Dict[str, VarInfo]], fun_env_stack: List[Dict[str, FunTy]]) -> bool:
        pd.captures.clear()
        var_env_stack = [dict(f) for f in outer_var_env]
        param_env = {}
        local_ids = set()
        for param in pd.params:
            vid = fresh_var_id()
            param.vid = vid
            param_env[param.name] = (param.ty, vid)
            local_ids.add(vid)
        var_env_stack.append(param_env)
        ret_ty = fun_ty.ret_ty
        def lookup_var(name):
            for f in reversed(var_env_stack):
                if name in f: return f[name]
            return None
        def lookup_fun(name):
            for f in reversed(fun_env_stack):
                if name in f: return f[name]
            return None
        def add_local(name, ty):
            vid = fresh_var_id()
            var_env_stack[-1][name] = (ty, vid)
            local_ids.add(vid)
            return var_env_stack[-1][name]
        def chk_e(e):
            if isinstance(e, ENum): return 'int'
            if isinstance(e, EBool): return 'bool'
            if isinstance(e, EVar):
                vi = lookup_var(e.name)
                if vi:
                    ty, vid = vi
                    if vid not in local_ids: pd.captures.add(vid)
                    return ty
                return lookup_fun(e.name)
            if isinstance(e, EUn):
                chk_e(e.e)
                return 'bool' if e.op == '!' else 'int'
            if isinstance(e, EBin):
                chk_e(e.l); chk_e(e.r)
                return 'bool' if e.op in ('&&', '||', '==', '!=', '<', '<=', '>', '>=') else 'int'
            if isinstance(e, ECall):
                if e.name == 'print': return 'void'
                vi = lookup_var(e.name)
                fty = vi[0] if vi else lookup_fun(e.name)
                if vi and vi[1] not in local_ids: pd.captures.add(vi[1])
                return fty.ret_ty
            return 'void'
        def chk_s(s):
            if isinstance(s, SVar): add_local(s.name, s.ty_annot); s.vid = var_env_stack[-1][s.name][1]; return False
            if isinstance(s, SAssign):
                vi = lookup_var(s.name)
                if vi[1] not in local_ids: pd.captures.add(vi[1])
                return False
            if isinstance(s, SBlock):
                var_env_stack.append({})
                r = any(chk_s(st) for st in s.ss)
                var_env_stack.pop()
                return r
            if isinstance(s, SIfElse): return chk_s(s.thenb) and (s.elsep and chk_s(s.elsep))
            if isinstance(s, SWhile): chk_s(s.body); return False
            if isinstance(s, SProcDef):
                inner_fty = FunTy(tuple(p.ty for p in s.proc.params), s.proc.ret_ty)
                typecheck_proc(s.proc, inner_fty, var_env_stack, fun_env_stack + [dict(fun_env_stack[-1])])
                fun_env_stack[-1][s.proc.name] = inner_fty
                return False
            if isinstance(s, SReturn): return True
            return False
        return chk_s(pd.body)
    for pd in prog.procs:
        typecheck_proc(pd, fun_env_global[pd.name], [], [fun_env_global])

# =============================================================================
# TAC IR
# =============================================================================
class TacInstr: pass
@dc
class TacLabel(TacInstr): label: str
@dc
class TacBinOp(TacInstr): dst: str; op: str; lhs: str; rhs: str
@dc
class TacUnOp(TacInstr): dst: str; op: str; src: str
@dc
class TacCopy(TacInstr): dst: str; src: str
@dc
class TacJmp(TacInstr): target: str
@dc
class TacCJump(TacInstr): cond: str; target_true: str; target_false: str
@dc
class TacGetVar(TacInstr): dst: str; vid: int; hops: int
@dc
class TacSetVar(TacInstr): vid: int; hops: int; src: str
@dc
class TacMakeClosure(TacInstr): dst: str; proc_label: str; hops: int
@dc
class TacCall(TacInstr): dst: Optional[str]; func: str; static_link: str; args: List[str]; is_indirect: bool
@dc
class TacRet(TacInstr): val: Optional[str]
@dc
class TacProc:
    name: str; params: List[str]; body: List[TacInstr]; is_main: bool = False

# =============================================================================
# TAC GENERATOR
# =============================================================================
class TacGenerator:
    def __init__(self, prog: Program):
        self.prog = prog; self.procs = []; self.temp_counter = 0; self.label_counter = 0
        self.proc_depth = {}; self.proc_mangled = {}; self.vid_depth = {}
        self.current_depth = 0; self.env_stack = []

    def fresh_temp(self) -> str: self.temp_counter += 1; return f"%t{self.temp_counter}"
    def fresh_label(self) -> str: self.label_counter += 1; return f".L{self.label_counter}"

    def run_analysis(self):
        def walk(pd, depth, prefix):
            mangled = "main" if pd.name == "main" else (prefix + pd.name)
            self.proc_mangled[pd.name] = mangled; self.proc_depth[pd.name] = depth
            for p in pd.params: self.vid_depth[p.vid] = depth
            def scan(s):
                if isinstance(s, SBlock): [scan(x) for x in s.ss]
                elif isinstance(s, SVar): self.vid_depth[s.vid] = depth
                elif isinstance(s, SIfElse): scan(s.thenb); s.elsep and scan(s.elsep)
                elif isinstance(s, SWhile): scan(s.body)
                elif isinstance(s, SProcDef): walk(s.proc, depth + 1, mangled + "$")
            scan(pd.body)
        [walk(pd, 0, "") for pd in self.prog.procs]

    def gen_program(self):
        self.run_analysis()
        for pd in self.prog.procs: self.gen_proc(pd)
        return self.procs

    def gen_proc(self, pd: ProcDecl):
        prev_depth = self.current_depth
        self.current_depth = self.proc_depth[pd.name]
        body = []
        emit = body.append
        
        self.env_stack.append({})
        for p in pd.params: self.env_stack[-1][p.name] = p.vid
        proc_params = [f"%v_{p.name}_{p.vid}" for p in pd.params]
        loop_stack = []

        def lookup(n): 
            for f in reversed(self.env_stack): 
                if n in f: return f[n]
            return None

        def load_var(name):
            vid = lookup(name)
            if vid is not None:
                t = self.fresh_temp()
                emit(TacGetVar(t, vid, self.current_depth - self.vid_depth[vid]))
                return t
            if name in self.proc_mangled:
                t = self.fresh_temp()
                emit(TacMakeClosure(t, self.proc_mangled[name], -1))
                return t
            raise ValueError(f"Unknown var {name}")

        def compile_e(e):
            if isinstance(e, ENum): t=self.fresh_temp(); emit(TacCopy(t, str(e.n))); return t
            if isinstance(e, EBool): t=self.fresh_temp(); emit(TacCopy(t, "1" if e.b else "0")); return t
            if isinstance(e, EVar): return load_var(e.name)
            if isinstance(e, EBin): 
                l, r = compile_e(e.l), compile_e(e.r); t=self.fresh_temp(); emit(TacBinOp(t, e.op, l, r)); return t
            if isinstance(e, EUn):
                src = compile_e(e.e); t=self.fresh_temp(); emit(TacUnOp(t, e.op, src)); return t
            if isinstance(e, ECall):
                args = [compile_e(a) for a in e.args]
                vid = lookup(e.name)
                dst = self.fresh_temp() if e.ty != 'void' else None
                if vid is not None:
                    func = load_var(e.name)
                    emit(TacCall(dst, func, "0", args, True))
                else:
                    target = self.proc_mangled.get(e.name, e.name)
                    # Correct static link logic:
                    # If target is nested (depth > 0), we must pass the static link.
                    # Standard rule:
                    # - If calling sibling/self: SL = my SL
                    # - If calling inner: SL = my frame pointer
                    # General Hops: MyDepth - TargetDepth + 1
                    target_depth = self.proc_depth.get(e.name, 0)
                    if target_depth > 0:
                        hops = self.current_depth - target_depth + 1
                        sl = self.fresh_temp()
                        # hops=0 means "current frame pointer". 
                        # hops=1 means "parent frame pointer" (my SL)
                        # We use special vid -2 to represent "Get Frame Pointer"
                        # TacGetVar with vid=-2, hops=N will load the frame ptr N levels up.
                        # Actually, our TacGetVar hops logic is: 
                        # 0 -> RBP, 1 -> RBP->SL.
                        # So hops=0 gets RBP. hops=1 gets SL.
                        # This matches exactly what we need.
                        emit(TacGetVar(sl, -2, hops))
                    else:
                        sl = "0"
                    emit(TacCall(dst, target, sl, args, False))
                return dst

        def compile_s(s):
            if isinstance(s, SBlock):
                self.env_stack.append({}); [compile_s(x) for x in s.ss]; self.env_stack.pop()
            elif isinstance(s, SVar):
                v = compile_e(s.init); self.env_stack[-1][s.name] = s.vid; emit(TacSetVar(s.vid, 0, v))
            elif isinstance(s, SAssign):
                v = compile_e(s.e); vid = lookup(s.name); emit(TacSetVar(vid, self.current_depth - self.vid_depth[vid], v))
            elif isinstance(s, SExpr): compile_e(s.e)
            elif isinstance(s, SIfElse):
                l_t, l_e, l_end = self.fresh_label(), self.fresh_label(), self.fresh_label()
                emit(TacCJump(compile_e(s.cond), l_t, l_e))
                emit(TacLabel(l_t)); compile_s(s.thenb); emit(TacJmp(l_end))
                emit(TacLabel(l_e)); s.elsep and compile_s(s.elsep); emit(TacLabel(l_end))
            elif isinstance(s, SWhile):
                l_s, l_b, l_e = self.fresh_label(), self.fresh_label(), self.fresh_label()
                emit(TacLabel(l_s)); emit(TacCJump(compile_e(s.cond), l_b, l_e))
                emit(TacLabel(l_b)); loop_stack.append((l_s, l_e)); compile_s(s.body); loop_stack.pop(); emit(TacJmp(l_s)); emit(TacLabel(l_e))
            elif isinstance(s, SBreak): emit(TacJmp(loop_stack[-1][1]))
            elif isinstance(s, SContinue): emit(TacJmp(loop_stack[-1][0]))
            elif isinstance(s, SReturn): v = compile_e(s.e) if s.e else None; emit(TacRet(v))
            elif isinstance(s, SProcDef):
                self.gen_proc(s.proc)
                vid = hash(s.proc.name) % 100000 + 100000
                self.env_stack[-1][s.proc.name] = vid
                self.vid_depth[vid] = self.current_depth
                t = self.fresh_temp()
                emit(TacMakeClosure(t, self.proc_mangled[s.proc.name], 0))
                emit(TacSetVar(vid, 0, t))

        compile_s(pd.body)
        if not body or not isinstance(body[-1], TacRet): emit(TacRet(None))
        
        tp = TacProc(self.proc_mangled[pd.name], proc_params, body)
        if pd.name == 'main': tp.is_main = True
        self.procs.append(tp)
        self.env_stack.pop()
        self.current_depth = prev_depth

# =============================================================================
# ASSEMBLY GENERATION
# =============================================================================
class AsmGen:
    def __init__(self, procs: List[TacProc]):
        self.procs = procs
        self.output = []
        self.slots = {}
        # Precomputed map of {vid: offset_from_rbp}
        self.vid_offsets = {}
        # Precomputed map of {proc_name: stack_size_for_vars}
        self.proc_stack_base = {} 

    def emit(self, s): self.output.append(s)

    def precompute_offsets(self):
        # Global pass to assign stack slots for VIDs
        for proc in self.procs:
            off = 8 # Start after Static Link
            
            # 1. Map Params (Negative offsets)
            for param in proc.params:
                off += 8
                # Parse %v_name_vid
                try:
                    vid = int(param.split('_')[-1])
                    self.vid_offsets[vid] = -off
                except: pass
            
            # 2. Map Locals (Negative offsets)
            seen = set()
            for instr in proc.body:
                if isinstance(instr, (TacSetVar, TacGetVar)) and instr.hops == 0:
                    if instr.vid not in self.vid_offsets and instr.vid not in seen:
                        off += 8
                        self.vid_offsets[instr.vid] = -off
                        seen.add(instr.vid)
            
            self.proc_stack_base[proc.name] = off

    def gen_program(self):
        self.emit(".text")
        for p in self.procs:
            if p.is_main: self.emit(".globl main")
        for p in self.procs: self.gen_proc(p)
        return "\n".join(self.output) + "\n"

    def gen_proc(self, proc):
        self.emit(f"\n{proc.name}:")
        self.emit("    pushq %rbp")
        self.emit("    movq %rsp, %rbp")
        
        # Start offset where precompute left off
        offset = self.proc_stack_base.get(proc.name, 8)
        self.slots = {}

        # Important: Map params to self.slots so load_operand can find them
        for param in proc.params:
            try:
                vid = int(param.split('_')[-1])
                if vid in self.vid_offsets:
                    self.slots[param] = self.vid_offsets[vid]
            except: pass

        # Map Temps
        closure_cnt = 0
        for instr in proc.body:
            if hasattr(instr, 'dst') and instr.dst and instr.dst.startswith('%t'):
                if instr.dst not in self.slots:
                    offset += 8
                    self.slots[instr.dst] = -offset
            if isinstance(instr, TacMakeClosure): closure_cnt += 1
            
        # Alloc Closures
        closure_offs = []
        for _ in range(closure_cnt):
            offset += 16
            closure_offs.append(-offset)
            
        if offset % 16 != 0: offset += 16 - (offset % 16)
        self.emit(f"    subq ${offset}, %rsp")
        
        # Save Static Link
        self.emit("    movq %r10, -8(%rbp)")
        
        # Move Args
        regs = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
        for i, param in enumerate(proc.params):
            vid = int(param.split('_')[-1])
            slot = self.vid_offsets[vid]
            if i < 6:
                self.emit(f"    movq {regs[i]}, {slot}(%rbp)")
            else:
                self.emit(f"    movq {24 + (i-6)*8}(%rbp), %rax")
                self.emit(f"    movq %rax, {slot}(%rbp)")

        # Code Gen
        c_idx = 0
        for instr in proc.body:
            if isinstance(instr, TacLabel): self.emit(f"{instr.label}:")
            elif isinstance(instr, TacCopy):
                self.load_operand(instr.src, '%rax'); self.store_operand(instr.dst, '%rax')
            elif isinstance(instr, TacBinOp):
                self.load_operand(instr.lhs, '%rax'); self.load_operand(instr.rhs, '%rcx')
                if instr.op == '+': self.emit("    addq %rcx, %rax")
                elif instr.op == '-': self.emit("    subq %rcx, %rax")
                elif instr.op == '*': self.emit("    imulq %rcx, %rax")
                elif instr.op in ('/', '%'): self.emit("    cqto\n    idivq %rcx")
                elif instr.op in ('<','>','<=','>=','==','!='):
                    self.emit("    cmpq %rcx, %rax")
                    cc = {'<':'l','>':'g','<=':'le','>=':'ge','==':'e','!=':'ne'}[instr.op]
                    self.emit(f"    set{cc} %al\n    movzbq %al, %rax")
                if instr.op == '%': self.emit("    movq %rdx, %rax")
                self.store_operand(instr.dst, '%rax')
            elif isinstance(instr, TacUnOp):
                self.load_operand(instr.src, '%rax')
                if instr.op == '-': self.emit("    negq %rax")
                elif instr.op == '!': self.emit("    xorq $1, %rax")
                elif instr.op == '~': self.emit("    notq %rax")
                self.store_operand(instr.dst, '%rax')
            elif isinstance(instr, TacJmp): self.emit(f"    jmp {instr.target}")
            elif isinstance(instr, TacCJump):
                self.load_operand(instr.cond, '%rax'); self.emit("    testq %rax, %rax")
                self.emit(f"    jnz {instr.target_true}\n    jmp {instr.target_false}")
            elif isinstance(instr, (TacGetVar, TacSetVar)):
                reg = '%rbp'
                # Check if this is a 'Get Frame Pointer' request (vid=-2)
                if instr.vid == -2:
                    # hops=0 -> RBP, hops=1 -> SL, hops=2 -> SL->SL
                    if instr.hops == 0: self.emit("    movq %rbp, %rcx")
                    else:
                        self.emit("    movq -8(%rbp), %rcx") # Load SL
                        for _ in range(instr.hops - 1): self.emit("    movq -8(%rcx), %rcx")
                    self.store_operand(instr.dst, '%rcx')
                    continue

                if instr.hops > 0:
                    self.emit("    movq -8(%rbp), %rax")
                    for _ in range(instr.hops - 1): self.emit("    movq -8(%rax), %rax")
                    reg = '%rax'
                off = self.vid_offsets[instr.vid]
                if isinstance(instr, TacGetVar):
                    self.emit(f"    movq {off}({reg}), %rcx"); self.store_operand(instr.dst, '%rcx')
                else:
                    self.load_operand(instr.src, '%rcx'); self.emit(f"    movq %rcx, {off}({reg})")
            elif isinstance(instr, TacMakeClosure):
                bo = closure_offs[c_idx]; c_idx+=1
                self.emit(f"    leaq {instr.proc_label}(%rip), %rax"); self.emit(f"    movq %rax, {bo}(%rbp)")
                if instr.hops == -1: self.emit(f"    movq $0, {bo+8}(%rbp)")
                else: self.emit(f"    movq %rbp, {bo+8}(%rbp)")
                self.emit(f"    leaq {bo}(%rbp), %rax"); self.store_operand(instr.dst, '%rax')
            elif isinstance(instr, TacCall):
                if instr.is_indirect:
                    self.load_operand(instr.func, '%r11')
                    self.emit("    movq 0(%r11), %rax"); self.emit("    movq 8(%r11), %r10")
                else:
                    self.emit(f"    leaq {instr.func}(%rip), %rax")
                    if instr.static_link == '0': self.emit("    movq $0, %r10")
                    else: self.load_operand(instr.static_link, '%r10')
                regs = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
                for i, a in enumerate(instr.args):
                    if i < 6: self.load_operand(a, regs[i])
                    else: self.load_operand(a, '%r11'); self.emit("    pushq %r11")
                self.emit("    call *%rax")
                if len(instr.args) > 6: self.emit(f"    addq ${(len(instr.args)-6)*8}, %rsp")
                if instr.dst: self.store_operand(instr.dst, '%rax')
            elif isinstance(instr, TacRet):
                if instr.val: self.load_operand(instr.val, '%rax')
                elif proc.is_main: self.emit("    movq $0, %rax")
                self.emit("    leave\n    ret")

    def load_operand(self, op, reg):
        if op[0] in '-0123456789': self.emit(f"    movq ${op}, {reg}")
        elif op in self.slots: self.emit(f"    movq {self.slots[op]}(%rbp), {reg}")
        else: raise ValueError(f"Missing slot for {op}")

    def store_operand(self, op, reg):
        if op in self.slots: self.emit(f"    movq {reg}, {self.slots[op]}(%rbp)")
        else: raise ValueError(f"Missing slot for {op}")

def _build_vid_name_map(prog): return {}

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('file'); ap.add_argument('--dump-ast', action='store_true'); ap.add_argument('--dump-captures', action='store_true'); ap.add_argument('--dump-tac', action='store_true')
    args = ap.parse_args(argv)
    with open(args.file) as f: src = f.read()
    prog = parser.parse(src, lexer=lexer)
    check_program(prog)
    tac_gen = TacGenerator(prog); tac_procs = tac_gen.gen_program()
    if args.dump_tac:
        for p in tac_procs: 
            print(f"PROC {p.name}:"); [print(f"  {i}") for i in p.body]; print()
        return
    asm = AsmGen(tac_procs); asm.precompute_offsets(); code = asm.gen_program()
    with open(os.path.splitext(args.file)[0] + ".s", "w") as f: f.write(code)

if __name__ == '__main__': main(sys.argv[1:])

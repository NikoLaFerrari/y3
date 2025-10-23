import sys
import os
import json
import argparse
import dataclasses as dc
import abc
from ply import lex, yacc  

# =============================================================================

reserved = {
    'def':   'DEF',
    'main':  'MAIN',
    'var':   'VAR',
    'print': 'PRINT',
    'int':   'INT',
    'bool':  'BOOL'
}

tokens = (
    'IDENT', 'NUMBER',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'EQUAL',
    'PLUS', 'MINUS', 'STAR', 'SLASH', 'MOD',
    'BAND', 'BOR', 'BXOR',
    'LSHIFT', 'RSHIFT',
    'BNOT',
    'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
    'LAND', 'LOR',
    'LNOT',             
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

t_EQ     = r'=='
t_NE     = r'!='
t_LT     = r'<'
t_LE     = r'<='
t_GT     = r'>'
t_GE     = r'>='
t_LAND   = r'&&'
t_LOR    = r'||'
t_LNOT   = r'!'

t_ignore = '\t'

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
    val = int(t.value)
    if not (0 <= val < (1 << 63)):
        print(f"[Lexer] ERROR at pos {t.lexpos}: integer {val} out of range [0, 2^63)")
        t.lexer.error = True
        return None
    t.value = val
    return t

def t_error(t):
    print(f"[Lexer] ERROR at line {t.lineno}, pos {t.lexpos}: illegal char {t.value[0]!r}")
    t.lexer.skip(1)

lexer = lex.lex()

# =============================================================================
# AST NODES
# =============================================================================

@dc.dataclass
class AST(abc.ABC):
    pass

# --- Expressions ---
class Expression(AST):
    pass

@dc.dataclass
class VarExpression(Expression):
    name: str

@dc.dataclass
class NumberExpression(Expression):
    value: int

@dc.dataclass
class UnaryExpression(Expression):
    op: str  # '-', '~'
    expr: Expression

@dc.dataclass
class BinaryExpression(Expression):
    op: str  
    left: Expression
    right: Expression

# --- Statements ---
class Statement(AST):
    pass

@dc.dataclass
class VarDecl(Statement):
    name: str
    expr: Expression

@dc.dataclass
class Assign(Statement):
    name: str
    expr: Expression

@dc.dataclass
class Print(Statement):
    expr: Expression

@dc.dataclass
class Program(AST):
    body: list  

# =============================================================================
# PARSER -> AST
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
    ('right', 'BNOT', 'UMINUS'),
)

def p_program(p):
    'program : DEF MAIN LPAREN RPAREN LBRACE stmt_list RBRACE'
    p[0] = Program(body=p[6])

def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = []

def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    p[1].append(p[2])
    p[0] = p[1]

def p_stmt_vardecl(p):
    'stmt : VAR IDENT EQUAL expr COLON INT SEMI'
    p[0] = VarDecl(name=p[2], expr=p[4])

def p_stmt_assign(p):
    'stmt : IDENT EQUAL expr SEMI'
    p[0] = Assign(name=p[1], expr=p[3])

def p_stmt_print(p):
    'stmt : PRINT LPAREN expr RPAREN SEMI'
    p[0] = Print(expr=p[3])

def p_expr_number(p):
    'expr : NUMBER'
    p[0] = NumberExpression(value=p[1])

def p_expr_ident(p):
    'expr : IDENT'
    p[0] = VarExpression(name=p[1])

def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_uminus(p):
    'expr : MINUS expr %prec UMINUS'
    p[0] = UnaryExpression(op='-', expr=p[2])

def p_expr_bnot(p):
    'expr : BNOT expr'
    p[0] = UnaryExpression(op='~', expr=p[2])

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
            | expr RSHIFT expr'''
    p[0] = BinaryExpression(op=p[2], left=p[1], right=p[3])

def p_error(p):
    if p is None:
        print("[Parser] ERROR: unexpected end of input")
    else:
        print(f"[Parser] ERROR at line {getattr(p,'lineno','?')}, pos {getattr(p,'lexpos','?')}: unexpected token {p.type} ({p.value!r})")
    raise SyntaxError

parser = yacc.yacc(start='program')

# =============================================================================
# SEMANTIC CHECKS
# - variables must be declared before use
# - no redeclaration of same name
# =============================================================================

class SemanticError(Exception):
    pass

def check_program(prog: Program):
    declared = set()
    for stmt in prog.body:
        if isinstance(stmt, VarDecl):
            if stmt.name in declared:
                raise SemanticError(f"Variable '{stmt.name}' redeclared")
            _check_expr(stmt.expr, declared)
            declared.add(stmt.name)
        elif isinstance(stmt, Assign):
            if stmt.name not in declared:
                raise SemanticError(f"Assignment to undeclared variable '{stmt.name}'")
            _check_expr(stmt.expr, declared)
        elif isinstance(stmt, Print):
            _check_expr(stmt.expr, declared)
        else:
            raise SemanticError(f"Unknown statement: {stmt}")

def _check_expr(e: Expression, declared: set):
    if isinstance(e, NumberExpression):
        return
    if isinstance(e, VarExpression):
        if e.name not in declared:
            raise SemanticError(f"Use of undeclared variable '{e.name}'")
        return
    if isinstance(e, UnaryExpression):
        _check_expr(e.expr, declared)
        return
    if isinstance(e, BinaryExpression):
        _check_expr(e.left, declared)
        _check_expr(e.right, declared)
        return
    raise SemanticError(f"Unknown expression: {e}")

# =============================================================================

BINOPS = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
    '%': 'mod',
    '&': 'and',
    '|': 'or',
    '^': 'xor',
    '<<': 'shl',
    '>>': 'shr',
}
UNOPS = {
    '-': 'neg',
    '~': 'not',
}

class TempGen:
    def __init__(self):
        self.n = 0
    def fresh(self, hint: str | None = None) -> str:
        t = f"%{self.n}"
        self.n += 1
        return t

@dc.dataclass
class Instr:
    opcode: str
    args: tuple
    result: str | None

def instr(opcode, args, result):
    return {"opcode": opcode, "args": list(args), "result": result}

# Top-down maximal munch (preorder) 
def gen_expr_tmm(e: Expression, temps: TempGen):
    code = []
    if isinstance(e, NumberExpression):
        t = temps.fresh()
        code.append(instr("const", (e.value,), t))
        return t, code
    if isinstance(e, VarExpression):
        return None, code
    if isinstance(e, UnaryExpression):
        rt, rc = gen_expr_tmm(e.expr, temps)
        code += rc
        t = temps.fresh()
        code.append(instr(UNOPS[e.op], (rt,), t))
        return t, code
    if isinstance(e, BinaryExpression):
        lt, lc = gen_expr_tmm(e.left, temps)
        rt, rc = gen_expr_tmm(e.right, temps)
        code += lc + rc
        t = temps.fresh()
        code.append(instr(BINOPS[e.op], (lt, rt), t))
        return t, code
    raise RuntimeError(f"Unknown expr {e}")

# Bottom-up maximal munch (postorder, right-biased visit) 
def gen_expr_bmm(e: Expression, temps: TempGen):
    code = []
    if isinstance(e, NumberExpression):
        t = temps.fresh()
        code.append(instr("const", (e.value,), t))
        return t, code
    if isinstance(e, VarExpression):
        return None, code
    if isinstance(e, UnaryExpression):
        rt, rc = gen_expr_bmm(e.expr, temps)
        code += rc
        t = temps.fresh()
        code.append(instr(UNOPS[e.op], (rt,), t))
        return t, code
    if isinstance(e, BinaryExpression):
        rt, rc = gen_expr_bmm(e.right, temps)
        lt, lc = gen_expr_bmm(e.left, temps)
        code += rc + lc
        t = temps.fresh()
        code.append(instr(BINOPS[e.op], (lt, rt), t))
        return t, code
    raise RuntimeError(f"Unknown expr {e}")

def generate_tac(prog: Program, strategy: str = "tmm"):
    temps = TempGen()
    body = []
    env: dict[str, str] = {}  # var name -> temp holding its current value

    gen_expr = gen_expr_tmm if strategy == "tmm" else gen_expr_bmm

    def materialize_expr(expr: Expression) -> tuple[str, list]:
        """
        Returns (temp_name, code) for expr.
        Replaces VarExpression by their env temp.
        """
        if isinstance(expr, VarExpression):
            return env[expr.name], []
        t, code = gen_expr(expr, temps)
        return t, code

    for stmt in prog.body:
        if isinstance(stmt, VarDecl):
            if stmt.name in env:
                raise RuntimeError(f"Internal: redecl {stmt.name}")
            tvar = temps.fresh()  # dedicate a temp to this variable
            env[stmt.name] = tvar
            tval, code = materialize_expr(stmt.expr)
            body += code
            body.append(instr("copy", (tval,), tvar))
        elif isinstance(stmt, Assign):
            tval, code = materialize_expr(stmt.expr)
            body += code
            body.append(instr("copy", (tval,), env[stmt.name]))
        elif isinstance(stmt, Print):
            tval, code = materialize_expr(stmt.expr)
            body += code
            body.append(instr("print", (tval,), None))
        else:
            raise RuntimeError(f"Unknown stmt {stmt}")

    return [ { "proc": "@main", "body": body } ]

# DRIVER

def parse_text(src: str) -> Program | None:
    lexer.lineno = 1
    try:
        ast = parser.parse(src, lexer=lexer)
        return ast
    except SyntaxError:
        return None

def main():
    ap = argparse.ArgumentParser(description="BX -> TAC (JSON)")
    ap.add_argument("--tmm", action="store_true", help="top-down maximal munch (default)")
    ap.add_argument("--bmm", action="store_true", help="bottom-up maximal munch")
    ap.add_argument("source", help="BX source file")
    args = ap.parse_args()

    strategy = "tmm"
    if args.bmm:
        strategy = "bmm"
    if args.tmm:
        strategy = "tmm"

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"[IO] ERROR: cannot read {args.source}: {e}")
        sys.exit(1)

    ast = parse_text(src)
    if ast is None:
        sys.exit(1)

    try:
        check_program(ast)
    except SemanticError as e:
        print(f"[Semantic] ERROR: {e}")
        sys.exit(1)

    tac = generate_tac(ast, strategy=strategy)

    out_path = os.path.splitext(args.source)[0] + ".tac.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tac, f, indent=2)
    except OSError as e:
        print(f"[IO] ERROR: cannot write {out_path}: {e}")
        sys.exit(1)

    print(out_path)

if __name__ == "__main__":
    main()


import sys, pprint
from ply import lex, yacc

reserved = {
    'def':   'DEF',
    'main':  'MAIN',
    'var':   'VAR',
    'print': 'PRINT',
    'int':   'INT',
}

tokens = (
    'IDENT', 'NUMBER',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'EQUAL',
    'PLUS', 'MINUS', 'STAR', 'SLASH', 'MOD',
    'BAND', 'BOR', 'BXOR',
    'LSHIFT', 'RSHIFT',
    'BNOT',
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

# -----------------------------
precedence = (
    ('left', 'BOR'),
    ('left', 'BXOR'),
    ('left', 'BAND'),
    ('left', 'LSHIFT', 'RSHIFT'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'MOD'),
    ('right', 'BNOT', 'UMINUS'),
)

def Num(n): return ('num', n)
def Id(x):  return ('id', x)
def Un(op, e): return ('unop', op, e)
def Bin(op, l, r): return ('binop', op, l, r)
def StmtList(ss): return ('stmts', ss)
def VarDecl(name, expr): return ('vardecl', name, expr, 'int')
def Assign(name, expr): return ('assign', name, expr)
def Print(expr): return ('print', expr)
def Program(stmts): return ('program', stmts)

def p_program(p):
    'program : DEF MAIN LPAREN RPAREN LBRACE stmt_list RBRACE'
    p[0] = Program(p[6])

def p_stmt_list_empty(p):
    'stmt_list : '
    p[0] = StmtList([])

def p_stmt_list_cons(p):
    'stmt_list : stmt_list stmt'
    lst = list(p[1][1])  # ('stmts', [...])
    lst.append(p[2])
    p[0] = StmtList(lst)

def p_stmt_vardecl(p):
    'stmt : vardecl'
    p[0] = p[1]

def p_stmt_assign(p):
    'stmt : assign'
    p[0] = p[1]

def p_stmt_print(p):
    'stmt : print'
    p[0] = p[1]

def p_vardecl(p):
    'vardecl : VAR IDENT EQUAL expr COLON INT SEMI'
    p[0] = VarDecl(p[2], p[4])

def p_assign(p):
    'assign : IDENT EQUAL expr SEMI'
    p[0] = Assign(p[1], p[3])

def p_print(p):
    'print : PRINT LPAREN expr RPAREN SEMI'
    p[0] = Print(p[3])

def p_expr_number(p):
    'expr : NUMBER'
    p[0] = Num(p[1])

def p_expr_ident(p):
    'expr : IDENT'
    p[0] = Id(p[1])

def p_expr_group(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_uminus(p):
    'expr : MINUS expr %prec UMINUS'
    p[0] = Un('-', p[2])

def p_expr_bnot(p):
    'expr : BNOT expr'
    p[0] = Un('~', p[2])

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
    p[0] = Bin(p[2], p[1], p[3])

def p_error(p):
    if p is None:
        print("[Parser] ERROR: unexpected end of input")
    else:
        print(f"[Parser] ERROR at line {getattr(p,'lineno','?')}, pos {getattr(p,'lexpos','?')}: unexpected token {p.type} ({p.value!r})")
    raise SyntaxError

parser = yacc.yacc(start='program')

# -----------------------------
def parse_text(text):
    lexer.lineno = 1
    try:
        return parser.parse(text, lexer=lexer)
    except SyntaxError:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python bxparser.py <file.bx>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
    except OSError as e:
        print(f"[IO] ERROR: cannot read {path}: {e}")
        sys.exit(1)
    ast = parse_text(src)
    if ast is None:
        sys.exit(1)
    pprint.pprint(ast)
    sys.exit(0)

if __name__ == '__main__':
    main()


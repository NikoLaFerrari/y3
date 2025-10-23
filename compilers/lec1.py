import re
ident_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
wsp_re = re.compile(r'[\t\f\v\r\n]')

def lex(source, pos=0):
    while pos < len(source):
        if (match := wsp_re.match(source, pos)):
            pos += len(match.group(0))
        elif (match := ident_re.match(source, pos)):
            ident = match.group(0)
            yield('IDENT', ident, pos)
            pos += len(ident)
        else:
            print(f"unknown character at {pos}: {source[pos]}")
            pos += 1

print(*lex("mary had a little lambda"), sep="\n")

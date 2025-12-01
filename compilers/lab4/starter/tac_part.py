def tac_to_x64(tac: List[Dict], out_s: str):
    body = tac[0]["body"]
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
            return int(x[1:])  
        except Exception:
            err("Backend", f"non-temp passed to tid(): {x!r}")
            raise

    max_id = max([tid(t) for t in temps], default=-1)
    nslots = max_id + 1
    if nslots % 2 == 1: nslots += 1   

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
            emit(f"  imulq {slot(b)}(%rbp), %r11")   
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


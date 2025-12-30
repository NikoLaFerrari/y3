    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $1232, %rsp
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $2, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $3, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $4, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $5, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq $0, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -48(%rax)
.L0:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq $20, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq $5, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -144(%rbp)
    movq $0, %rax
    movq %rax, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -160(%rbp)
    movq -160(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -184(%rbp)
    movq -184(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq -192(%rbp), %rax
    movq -200(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -208(%rbp)
    movq -208(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -232(%rbp)
    movq %rbp, %rax
    movq -232(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -248(%rbp)
    movq -240(%rbp), %rax
    movq -248(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -256(%rbp)
    movq %rbp, %rax
    movq -256(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq $1, %rax
    movq %rax, -272(%rbp)
    movq -264(%rbp), %rax
    movq -272(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -280(%rbp)
    movq %rbp, %rax
    movq -280(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq $1, %rax
    movq %rax, -296(%rbp)
    movq -288(%rbp), %rax
    movq -296(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -304(%rbp)
    movq %rbp, %rax
    movq -304(%rbp), %rcx
    movq %rcx, -24(%rax)
.L11:
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -320(%rbp)
    movq -312(%rbp), %rax
    movq -320(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -328(%rbp)
    movq -328(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -336(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -344(%rbp)
    movq -336(%rbp), %rax
    movq -344(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -352(%rbp)
    movq %rbp, %rax
    movq -352(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -360(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -368(%rbp)
    movq -360(%rbp), %rax
    movq -368(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -376(%rbp)
    movq %rbp, %rax
    movq -376(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L14
.L13:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -384(%rbp)
    movq $1, %rax
    movq %rax, -392(%rbp)
    movq -384(%rbp), %rax
    movq -392(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -400(%rbp)
    movq %rbp, %rax
    movq -400(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -408(%rbp)
    movq $1, %rax
    movq %rax, -416(%rbp)
    movq -408(%rbp), %rax
    movq -416(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -424(%rbp)
    movq %rbp, %rax
    movq -424(%rbp), %rcx
    movq %rcx, -32(%rax)
.L14:
.L8:
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq $5, %rax
    movq %rax, -440(%rbp)
    movq -432(%rbp), %rax
    movq -440(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -448(%rbp)
    movq $1, %rax
    movq %rax, -456(%rbp)
    movq -448(%rbp), %rax
    movq -456(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -464(%rbp)
    movq -464(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -472(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -480(%rbp)
    movq -472(%rbp), %rax
    movq -480(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -488(%rbp)
    movq -488(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -496(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -504(%rbp)
    movq -496(%rbp), %rax
    movq -504(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -512(%rbp)
    movq -512(%rbp), %rax
    testq %rax, %rax
    jnz .L21
    jmp .L22
.L21:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -520(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -528(%rbp)
    movq -520(%rbp), %rax
    movq -528(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -536(%rbp)
    movq %rbp, %rax
    movq -536(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -544(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -552(%rbp)
    movq -544(%rbp), %rax
    movq -552(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -560(%rbp)
    movq %rbp, %rax
    movq -560(%rbp), %rcx
    movq %rcx, -40(%rax)
    jmp .L23
.L22:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -568(%rbp)
    movq $1, %rax
    movq %rax, -576(%rbp)
    movq -568(%rbp), %rax
    movq -576(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -584(%rbp)
    movq %rbp, %rax
    movq -584(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -592(%rbp)
    movq $2, %rax
    movq %rax, -600(%rbp)
    movq -592(%rbp), %rax
    movq -600(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -608(%rbp)
    movq %rbp, %rax
    movq -608(%rbp), %rcx
    movq %rcx, -8(%rax)
.L23:
    jmp .L20
.L19:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -616(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -624(%rbp)
    movq -616(%rbp), %rax
    movq -624(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -632(%rbp)
    movq -632(%rbp), %rax
    testq %rax, %rax
    jnz .L24
    jmp .L25
.L24:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -640(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -648(%rbp)
    movq -640(%rbp), %rax
    movq -648(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -656(%rbp)
    movq %rbp, %rax
    movq -656(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -664(%rbp)
    movq $3, %rax
    movq %rax, -672(%rbp)
    movq -664(%rbp), %rax
    movq -672(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -680(%rbp)
    movq %rbp, %rax
    movq -680(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L26
.L25:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -688(%rbp)
    movq $2, %rax
    movq %rax, -696(%rbp)
    movq -688(%rbp), %rax
    movq -696(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -704(%rbp)
    movq %rbp, %rax
    movq -704(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -712(%rbp)
    movq $1, %rax
    movq %rax, -720(%rbp)
    movq -712(%rbp), %rax
    movq -720(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -728(%rbp)
    movq %rbp, %rax
    movq -728(%rbp), %rcx
    movq %rcx, -16(%rax)
.L26:
.L20:
    jmp .L17
.L16:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -736(%rbp)
    movq $5, %rax
    movq %rax, -744(%rbp)
    movq -736(%rbp), %rax
    movq -744(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -752(%rbp)
    movq $2, %rax
    movq %rax, -760(%rbp)
    movq -752(%rbp), %rax
    movq -760(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -768(%rbp)
    movq -768(%rbp), %rax
    testq %rax, %rax
    jnz .L27
    jmp .L28
.L27:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -776(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -784(%rbp)
    movq -776(%rbp), %rax
    movq -784(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -792(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -800(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -808(%rbp)
    movq -800(%rbp), %rax
    movq -808(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -816(%rbp)
    movq -792(%rbp), %rax
    movq -816(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -824(%rbp)
    movq -824(%rbp), %rax
    testq %rax, %rax
    jnz .L30
    jmp .L31
.L30:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -832(%rbp)
    movq $1, %rax
    movq %rax, -840(%rbp)
    movq -832(%rbp), %rax
    movq -840(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -848(%rbp)
    movq %rbp, %rax
    movq -848(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -856(%rbp)
    movq $2, %rax
    movq %rax, -864(%rbp)
    movq -856(%rbp), %rax
    movq -864(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -872(%rbp)
    movq %rbp, %rax
    movq -872(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L32
.L31:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -880(%rbp)
    movq $1, %rax
    movq %rax, -888(%rbp)
    movq -880(%rbp), %rax
    movq -888(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -896(%rbp)
    movq %rbp, %rax
    movq -896(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -904(%rbp)
    movq $2, %rax
    movq %rax, -912(%rbp)
    movq -904(%rbp), %rax
    movq -912(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -920(%rbp)
    movq %rbp, %rax
    movq -920(%rbp), %rcx
    movq %rcx, -32(%rax)
.L32:
    jmp .L29
.L28:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -928(%rbp)
    movq $5, %rax
    movq %rax, -936(%rbp)
    movq -928(%rbp), %rax
    movq -936(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -944(%rbp)
    movq $3, %rax
    movq %rax, -952(%rbp)
    movq -944(%rbp), %rax
    movq -952(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -960(%rbp)
    movq -960(%rbp), %rax
    testq %rax, %rax
    jnz .L33
    jmp .L34
.L33:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -968(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -976(%rbp)
    movq -968(%rbp), %rax
    movq -976(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -984(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -992(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1000(%rbp)
    movq -992(%rbp), %rax
    movq -1000(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1008(%rbp)
    movq -984(%rbp), %rax
    movq -1008(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -1016(%rbp)
    movq -1016(%rbp), %rax
    testq %rax, %rax
    jnz .L36
    jmp .L37
.L36:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -1024(%rbp)
    movq $3, %rax
    movq %rax, -1032(%rbp)
    movq -1024(%rbp), %rax
    movq -1032(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1040(%rbp)
    movq %rbp, %rax
    movq -1040(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -1048(%rbp)
    movq $4, %rax
    movq %rax, -1056(%rbp)
    movq -1048(%rbp), %rax
    movq -1056(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1064(%rbp)
    movq %rbp, %rax
    movq -1064(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L38
.L37:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -1072(%rbp)
    movq $3, %rax
    movq %rax, -1080(%rbp)
    movq -1072(%rbp), %rax
    movq -1080(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1088(%rbp)
    movq %rbp, %rax
    movq -1088(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1096(%rbp)
    movq $4, %rax
    movq %rax, -1104(%rbp)
    movq -1096(%rbp), %rax
    movq -1104(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1112(%rbp)
    movq %rbp, %rax
    movq -1112(%rbp), %rcx
    movq %rcx, -40(%rax)
.L38:
    jmp .L35
.L34:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -1120(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1128(%rbp)
    movq -1120(%rbp), %rax
    movq -1128(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1136(%rbp)
    movq %rbp, %rax
    movq -1136(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1144(%rbp)
    movq $1, %rax
    movq %rax, -1152(%rbp)
    movq -1144(%rbp), %rax
    movq -1152(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1160(%rbp)
    movq %rbp, %rax
    movq -1160(%rbp), %rcx
    movq %rcx, -40(%rax)
.L35:
.L29:
.L17:
.L5:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -1168(%rbp)
    movq $1, %rax
    movq %rax, -1176(%rbp)
    movq -1168(%rbp), %rax
    movq -1176(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1184(%rbp)
    movq %rbp, %rax
    movq -1184(%rbp), %rcx
    movq %rcx, -48(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -1192(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1192(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -1200(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1200(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -1208(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1208(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -1216(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1216(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1224(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1224(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

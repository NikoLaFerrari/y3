    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $1120, %rsp
    movq $1, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $1, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $1, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $1, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $1, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq $1, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq $0, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -56(%rax)
.L0:
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq $12, %rax
    movq %rax, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -136(%rbp)
    movq -136(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq $4, %rax
    movq %rax, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -160(%rbp)
    movq $0, %rax
    movq %rax, -168(%rbp)
    movq -160(%rbp), %rax
    movq -168(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -176(%rbp)
    movq -176(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -184(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -200(%rbp)
    movq %rbp, %rax
    movq -200(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -208(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq -208(%rbp), %rax
    movq -216(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -224(%rbp)
    movq %rbp, %rax
    movq -224(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -248(%rbp)
    movq %rbp, %rax
    movq -248(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -272(%rbp)
    movq %rbp, %rax
    movq -272(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -280(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq -280(%rbp), %rax
    movq -288(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -296(%rbp)
    movq %rbp, %rax
    movq -296(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -304(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq -304(%rbp), %rax
    movq -312(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -320(%rbp)
    movq %rbp, %rax
    movq -320(%rbp), %rcx
    movq %rcx, -48(%rax)
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -328(%rbp)
    movq $4, %rax
    movq %rax, -336(%rbp)
    movq -328(%rbp), %rax
    movq -336(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -344(%rbp)
    movq $1, %rax
    movq %rax, -352(%rbp)
    movq -344(%rbp), %rax
    movq -352(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -360(%rbp)
    movq -360(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -368(%rbp)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -376(%rbp)
    movq -368(%rbp), %rax
    movq -376(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -384(%rbp)
    movq %rbp, %rax
    movq -384(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -392(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -400(%rbp)
    movq -392(%rbp), %rax
    movq -400(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -408(%rbp)
    movq %rbp, %rax
    movq -408(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -416(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -424(%rbp)
    movq -416(%rbp), %rax
    movq -424(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -432(%rbp)
    movq %rbp, %rax
    movq -432(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -440(%rbp)
    movq $50, %rax
    movq %rax, -448(%rbp)
    movq -440(%rbp), %rax
    movq -448(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -456(%rbp)
    movq -456(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -464(%rbp)
    movq $30, %rax
    movq %rax, -472(%rbp)
    movq -464(%rbp), %rax
    movq -472(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -480(%rbp)
    movq %rbp, %rax
    movq -480(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -488(%rbp)
    movq $5, %rax
    movq %rax, -496(%rbp)
    movq -488(%rbp), %rax
    movq -496(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -504(%rbp)
    movq %rbp, %rax
    movq -504(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -512(%rbp)
    movq $50, %rax
    movq %rax, -520(%rbp)
    movq -512(%rbp), %rax
    movq -520(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -528(%rbp)
    movq -528(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -536(%rbp)
    movq $30, %rax
    movq %rax, -544(%rbp)
    movq -536(%rbp), %rax
    movq -544(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -552(%rbp)
    movq %rbp, %rax
    movq -552(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -560(%rbp)
    movq $5, %rax
    movq %rax, -568(%rbp)
    movq -560(%rbp), %rax
    movq -568(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -576(%rbp)
    movq %rbp, %rax
    movq -576(%rbp), %rcx
    movq %rcx, -32(%rax)
    jmp .L14
.L13:
.L14:
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -584(%rbp)
    movq $4, %rax
    movq %rax, -592(%rbp)
    movq -584(%rbp), %rax
    movq -592(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -600(%rbp)
    movq $2, %rax
    movq %rax, -608(%rbp)
    movq -600(%rbp), %rax
    movq -608(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -616(%rbp)
    movq -616(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -624(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -632(%rbp)
    movq -624(%rbp), %rax
    movq -632(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -640(%rbp)
    movq %rbp, %rax
    movq -640(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -648(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -656(%rbp)
    movq -648(%rbp), %rax
    movq -656(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -664(%rbp)
    movq %rbp, %rax
    movq -664(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -672(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -680(%rbp)
    movq -672(%rbp), %rax
    movq -680(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -688(%rbp)
    movq %rbp, %rax
    movq -688(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -696(%rbp)
    movq $100, %rax
    movq %rax, -704(%rbp)
    movq -696(%rbp), %rax
    movq -704(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -712(%rbp)
    movq -712(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -720(%rbp)
    movq $2, %rax
    movq %rax, -728(%rbp)
    movq -720(%rbp), %rax
    movq -728(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -736(%rbp)
    movq %rbp, %rax
    movq -736(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -744(%rbp)
    movq $10, %rax
    movq %rax, -752(%rbp)
    movq -744(%rbp), %rax
    movq -752(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -760(%rbp)
    movq %rbp, %rax
    movq -760(%rbp), %rcx
    movq %rcx, -48(%rax)
    jmp .L20
.L19:
.L20:
    jmp .L17
.L16:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -768(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -776(%rbp)
    movq -768(%rbp), %rax
    movq -776(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -784(%rbp)
    movq %rbp, %rax
    movq -784(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -792(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -800(%rbp)
    movq -792(%rbp), %rax
    movq -800(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -808(%rbp)
    movq %rbp, %rax
    movq -808(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -816(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -824(%rbp)
    movq -816(%rbp), %rax
    movq -824(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -832(%rbp)
    movq %rbp, %rax
    movq -832(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -840(%rbp)
    movq $1, %rax
    movq %rax, -848(%rbp)
    movq -840(%rbp), %rax
    movq -848(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -856(%rbp)
    movq %rbp, %rax
    movq -856(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -864(%rbp)
    movq $2, %rax
    movq %rax, -872(%rbp)
    movq -864(%rbp), %rax
    movq -872(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -880(%rbp)
    movq %rbp, %rax
    movq -880(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -888(%rbp)
    movq $3, %rax
    movq %rax, -896(%rbp)
    movq -888(%rbp), %rax
    movq -896(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -904(%rbp)
    movq %rbp, %rax
    movq -904(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -912(%rbp)
    movq $0, %rax
    movq %rax, -920(%rbp)
    movq -912(%rbp), %rax
    movq -920(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -928(%rbp)
    movq -928(%rbp), %rax
    testq %rax, %rax
    jnz .L21
    jmp .L22
.L21:
    movq $0, %rax
    movq %rax, -936(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -944(%rbp)
    movq -936(%rbp), %rax
    movq -944(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -952(%rbp)
    movq %rbp, %rax
    movq -952(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L23
.L22:
.L23:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -960(%rbp)
    movq $0, %rax
    movq %rax, -968(%rbp)
    movq -960(%rbp), %rax
    movq -968(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -976(%rbp)
    movq -976(%rbp), %rax
    testq %rax, %rax
    jnz .L24
    jmp .L25
.L24:
    movq $0, %rax
    movq %rax, -984(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -992(%rbp)
    movq -984(%rbp), %rax
    movq -992(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -1000(%rbp)
    movq %rbp, %rax
    movq -1000(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L26
.L25:
.L26:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -1008(%rbp)
    movq $0, %rax
    movq %rax, -1016(%rbp)
    movq -1008(%rbp), %rax
    movq -1016(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -1024(%rbp)
    movq -1024(%rbp), %rax
    testq %rax, %rax
    jnz .L27
    jmp .L28
.L27:
    movq $0, %rax
    movq %rax, -1032(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -1040(%rbp)
    movq -1032(%rbp), %rax
    movq -1040(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -1048(%rbp)
    movq %rbp, %rax
    movq -1048(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L29
.L28:
.L29:
.L17:
.L8:
.L5:
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -1056(%rbp)
    movq $1, %rax
    movq %rax, -1064(%rbp)
    movq -1056(%rbp), %rax
    movq -1064(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -1072(%rbp)
    movq %rbp, %rax
    movq -1072(%rbp), %rcx
    movq %rcx, -56(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -1080(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1080(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -1088(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1088(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -1096(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1096(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -1104(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -1112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -1120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -1120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $640, %rsp
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $1, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -40(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $8, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -104(%rbp)
    movq -104(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq $0, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -120(%rbp), %rcx
    movq %rcx, -40(%rax)
.L3:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq $8, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -144(%rbp)
    movq -144(%rbp), %rax
    testq %rax, %rax
    jnz .L4
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq $2, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq $2, %rax
    movq %rax, -184(%rbp)
    movq -176(%rbp), %rax
    movq -184(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -192(%rbp)
    movq $0, %rax
    movq %rax, -200(%rbp)
    movq -192(%rbp), %rax
    movq -200(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -208(%rbp)
    movq -208(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq $10, %rax
    movq %rax, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -232(%rbp)
    movq %rbp, %rax
    movq -232(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq $1, %rax
    movq %rax, -248(%rbp)
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
    movq -32(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq $2, %rax
    movq %rax, -272(%rbp)
    movq -264(%rbp), %rax
    movq -272(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -280(%rbp)
    movq %rbp, %rax
    movq -280(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq $1, %rax
    movq %rax, -296(%rbp)
    movq -288(%rbp), %rax
    movq -296(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -304(%rbp)
    movq %rbp, %rax
    movq -304(%rbp), %rcx
    movq %rcx, -16(%rax)
.L11:
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq $5, %rax
    movq %rax, -320(%rbp)
    movq -312(%rbp), %rax
    movq -320(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -328(%rbp)
    movq -328(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -336(%rbp)
    movq $50, %rax
    movq %rax, -344(%rbp)
    movq -336(%rbp), %rax
    movq -344(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -352(%rbp)
    movq -352(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq $1, %rax
    movq %rax, -360(%rbp)
    movq %rbp, %rax
    movq -360(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq $100, %rax
    movq %rax, -368(%rbp)
    movq %rbp, %rax
    movq -368(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L17
.L16:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -376(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -384(%rbp)
    movq -376(%rbp), %rax
    movq -384(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -392(%rbp)
    movq %rbp, %rax
    movq -392(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -400(%rbp)
    movq $1, %rax
    movq %rax, -408(%rbp)
    movq -400(%rbp), %rax
    movq -408(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -416(%rbp)
    movq %rbp, %rax
    movq -416(%rbp), %rcx
    movq %rcx, -16(%rax)
.L17:
    jmp .L14
.L13:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -424(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq -424(%rbp), %rax
    movq -432(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -440(%rbp)
    movq $10, %rax
    movq %rax, -448(%rbp)
    movq -440(%rbp), %rax
    movq -448(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -456(%rbp)
    movq -456(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -464(%rbp)
    movq $3, %rax
    movq %rax, -472(%rbp)
    movq -464(%rbp), %rax
    movq -472(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -480(%rbp)
    movq %rbp, %rax
    movq -480(%rbp), %rcx
    movq %rcx, -32(%rax)
    jmp .L20
.L19:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -488(%rbp)
    movq $1, %rax
    movq %rax, -496(%rbp)
    movq -488(%rbp), %rax
    movq -496(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -504(%rbp)
    movq %rbp, %rax
    movq -504(%rbp), %rcx
    movq %rcx, -24(%rax)
.L20:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -512(%rbp)
    movq $1, %rax
    movq %rax, -520(%rbp)
    movq -512(%rbp), %rax
    movq -520(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -528(%rbp)
    movq %rbp, %rax
    movq -528(%rbp), %rcx
    movq %rcx, -16(%rax)
.L14:
.L8:
    jmp .L3
.L5:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -536(%rbp)
    movq $1, %rax
    movq %rax, -544(%rbp)
    movq -536(%rbp), %rax
    movq -544(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -552(%rbp)
    movq -552(%rbp), %rax
    testq %rax, %rax
    jnz .L21
    jmp .L22
.L21:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -560(%rbp)
    movq $5, %rax
    movq %rax, -568(%rbp)
    movq -560(%rbp), %rax
    movq -568(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -576(%rbp)
    movq %rbp, %rax
    movq -576(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L23
.L22:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -584(%rbp)
    movq $1, %rax
    movq %rax, -592(%rbp)
    movq -584(%rbp), %rax
    movq -592(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -600(%rbp)
    movq %rbp, %rax
    movq -600(%rbp), %rcx
    movq %rcx, -32(%rax)
.L23:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -608(%rbp)
    movq $1, %rax
    movq %rax, -616(%rbp)
    movq -608(%rbp), %rax
    movq -616(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -624(%rbp)
    movq %rbp, %rax
    movq -624(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -632(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -632(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -640(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -640(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

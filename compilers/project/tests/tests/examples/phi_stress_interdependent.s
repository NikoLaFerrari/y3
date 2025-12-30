    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $480, %rsp
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
    movq $0, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq $1, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $2, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $3, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $4, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $0, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -120(%rbp), %rcx
    movq %rcx, -40(%rax)
.L0:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq $5, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -144(%rbp)
    movq -144(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq $2, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -168(%rbp)
    movq $0, %rax
    movq %rax, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -184(%rbp)
    movq -184(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq -192(%rbp), %rax
    movq -200(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -208(%rbp)
    movq %rbp, %rax
    movq -208(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -232(%rbp)
    movq %rbp, %rax
    movq -232(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L5
.L4:
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
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -272(%rbp)
    movq -264(%rbp), %rax
    movq -272(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -280(%rbp)
    movq %rbp, %rax
    movq -280(%rbp), %rcx
    movq %rcx, -32(%rax)
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq $10, %rax
    movq %rax, -296(%rbp)
    movq -288(%rbp), %rax
    movq -296(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -304(%rbp)
    movq -304(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq $5, %rax
    movq %rax, -320(%rbp)
    movq -312(%rbp), %rax
    movq -320(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -328(%rbp)
    movq %rbp, %rax
    movq -328(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -336(%rbp)
    movq $1, %rax
    movq %rax, -344(%rbp)
    movq -336(%rbp), %rax
    movq -344(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -352(%rbp)
    movq %rbp, %rax
    movq -352(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -360(%rbp)
    movq $20, %rax
    movq %rax, -368(%rbp)
    movq -360(%rbp), %rax
    movq -368(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -376(%rbp)
    movq -376(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -384(%rbp)
    movq $8, %rax
    movq %rax, -392(%rbp)
    movq -384(%rbp), %rax
    movq -392(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -400(%rbp)
    movq %rbp, %rax
    movq -400(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -408(%rbp)
    movq $2, %rax
    movq %rax, -416(%rbp)
    movq -408(%rbp), %rax
    movq -416(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -424(%rbp)
    movq %rbp, %rax
    movq -424(%rbp), %rcx
    movq %rcx, -32(%rax)
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq $1, %rax
    movq %rax, -440(%rbp)
    movq -432(%rbp), %rax
    movq -440(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -448(%rbp)
    movq %rbp, %rax
    movq -448(%rbp), %rcx
    movq %rcx, -40(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -456(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -456(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -464(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -464(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -472(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -472(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -480(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -480(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

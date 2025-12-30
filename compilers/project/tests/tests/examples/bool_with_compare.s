    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $272, %rsp
    movq $5, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $10, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $20, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -80(%rbp)
    movq -56(%rbp), %rax
    movq -80(%rbp), %rcx
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $1, %rax
    movq %rax, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L2
.L1:
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq $5, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -144(%rbp)
    movq -120(%rbp), %rax
    movq -144(%rbp), %rcx
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $2, %rax
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -184(%rbp)
    movq -184(%rbp), %rax
    xorq $1, %rax
    movq %rax, -192(%rbp)
    movq -192(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $3, %rax
    movq %rax, -200(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -200(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -208(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq -208(%rbp), %rax
    movq -216(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -224(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq $100, %rax
    movq %rax, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -248(%rbp)
    movq -248(%rbp), %rax
    xorq $1, %rax
    movq %rax, -256(%rbp)
    movq -224(%rbp), %rax
    movq -256(%rbp), %rcx
    movq %rax, -264(%rbp)
    movq -264(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $4, %rax
    movq %rax, -272(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -272(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
.L11:
    movq $0, %rax
    leave
    ret

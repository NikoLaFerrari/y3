    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $288, %rsp
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $0, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $10, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -24(%rax)
.L0:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $10, %rax
    movq %rax, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -96(%rbp)
    movq -96(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq $3, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $100, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -128(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq $5, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $200, %rax
    movq %rax, -160(%rbp)
    movq %rbp, %rax
    movq -160(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq $7, %rax
    movq %rax, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -184(%rbp)
    movq -184(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $300, %rax
    movq %rax, -192(%rbp)
    movq %rbp, %rax
    movq -192(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq $250, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -216(%rbp)
    movq -216(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq -224(%rbp), %rax
    movq -232(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -240(%rbp)
    movq %rbp, %rax
    movq -240(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -248(%rbp)
    movq %rbp, %rax
    movq -248(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L14
.L13:
.L14:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq $1, %rax
    movq %rax, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -272(%rbp)
    movq %rbp, %rax
    movq -272(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -280(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -280(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -288(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -288(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

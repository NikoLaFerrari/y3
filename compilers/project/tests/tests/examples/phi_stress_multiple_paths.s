    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $336, %rsp
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
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $5, %rax
    movq %rax, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -96(%rbp)
    movq -96(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq $1, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -120(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
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
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq $1, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -168(%rbp)
    movq %rbp, %rax
    movq -168(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq $5, %rax
    movq %rax, -184(%rbp)
    movq -176(%rbp), %rax
    movq -184(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -192(%rbp)
    movq -192(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq $1, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -216(%rbp)
    movq %rbp, %rax
    movq -216(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L8
.L7:
.L8:
    jmp .L5
.L4:
.L5:
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq $0, %rax
    movq %rax, -232(%rbp)
    movq -224(%rbp), %rax
    movq -232(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -240(%rbp)
    movq -240(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -248(%rbp)
    movq $1, %rax
    movq %rax, -256(%rbp)
    movq -248(%rbp), %rax
    movq -256(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -264(%rbp)
    movq %rbp, %rax
    movq -264(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -272(%rbp)
    movq $0, %rax
    movq %rax, -280(%rbp)
    movq -272(%rbp), %rax
    movq -280(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -288(%rbp)
    movq -288(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -296(%rbp)
    movq $1, %rax
    movq %rax, -304(%rbp)
    movq -296(%rbp), %rax
    movq -304(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -312(%rbp)
    movq %rbp, %rax
    movq -312(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L14
.L13:
.L14:
    jmp .L11
.L10:
.L11:
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -320(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -320(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -328(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -328(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -336(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -336(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

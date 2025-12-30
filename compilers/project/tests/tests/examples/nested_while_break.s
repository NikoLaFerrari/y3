    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $224, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $5, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq $0, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -16(%rax)
.L3:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $5, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rax
    testq %rax, %rax
    jnz .L4
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $2, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -104(%rbp)
    movq -104(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq $3, %rax
    movq %rax, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -128(%rbp)
    movq -128(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    jmp .L5
    jmp .L11
.L10:
.L11:
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq $10, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -152(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -168(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -168(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq $1, %rax
    movq %rax, -184(%rbp)
    movq -176(%rbp), %rax
    movq -184(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -192(%rbp)
    movq %rbp, %rax
    movq -192(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L3
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq $1, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -216(%rbp)
    movq %rbp, %rax
    movq -216(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret

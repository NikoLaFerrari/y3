    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %r10, -8(%rbp)
    movq $3, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -32(%rbp)
    movq $15, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $1, %rax
    movq %rax, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -56(%rbp), %rdi
    call *%rax
    jmp .L2
.L1:
    movq $0, %rax
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -64(%rbp), %rdi
    call *%rax
.L2:
    movq $0, %rax
    leave
    ret

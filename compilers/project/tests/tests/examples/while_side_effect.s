    .data
    .globl glob_counter
glob_counter:
    .zero 8
    .text
    .globl main

increment:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq glob_counter(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_counter(%rip)
    movq glob_counter(%rip), %rcx
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
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
.L0:
    leaq increment(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rax, -8(%rbp)
    movq -8(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    jmp .L0
.L2:
    movq glob_counter(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

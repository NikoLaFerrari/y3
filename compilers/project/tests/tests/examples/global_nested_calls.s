    .data
    .globl glob_total
glob_total:
    .zero 8
    .text
    .globl main

add_to_total:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_total(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_total(%rip)
    leave
    ret

process:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    leaq add_to_total(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    leaq add_to_total(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq glob_total(%rip), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $5, %rax
    movq %rax, -16(%rbp)
    movq $10, %rax
    movq %rax, -24(%rbp)
    leaq process(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    movq -24(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq glob_total(%rip), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $3, %rax
    movq %rax, -40(%rbp)
    movq $7, %rax
    movq %rax, -48(%rbp)
    leaq process(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -48(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq glob_total(%rip), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

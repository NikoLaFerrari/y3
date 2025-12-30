    .text
    .globl main

apply:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %r10, -8(%rbp)
    movq %rdi, -16(%rbp)
    movq %rsi, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, -32(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    movq -32(%rbp), %rdi
    call *%rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret

double:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %r10, -8(%rbp)
    movq %rdi, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq $2, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %r10, -8(%rbp)
    leaq double(%rip), %rax
    movq %rax, -48(%rbp)
    movq $0, -40(%rbp)
    leaq -48(%rbp), %rax
    movq %rax, -16(%rbp)
    movq $1, %rax
    movq %rax, -24(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    movq -16(%rbp), %rdi
    movq -24(%rbp), %rsi
    call *%rax
    movq %rax, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -32(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret

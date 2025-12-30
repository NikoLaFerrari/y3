    .text
    .globl main

square:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %r10, -8(%rbp)
    movq %rdi, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -32(%rbp)
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
    subq $32, %rsp
    movq %r10, -8(%rbp)
    movq $9, %rax
    movq %rax, -16(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    movq -16(%rbp), %rdi
    call *%rax
    movq %rax, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -24(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret

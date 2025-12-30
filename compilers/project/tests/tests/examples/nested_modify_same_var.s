    .data
    .text
    .globl main

main$nested$deeply_nested:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $2, %rax
    movq %rax, -8(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq -8(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main$nested:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq 24(%rbp), %rax
    movq -16(%rbp), %rcx
    movq %rcx, 0(%rax)
    leaq main$nested$deeply_nested(%rip), %rax
    movq %rax, -48(%rbp)
    movq %rbp, -40(%rbp)
    leaq -48(%rbp), %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$nested(%rip), %rax
    movq %rax, -64(%rbp)
    movq %rbp, -56(%rbp)
    leaq -64(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

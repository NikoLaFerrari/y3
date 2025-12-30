    .data
    .text
    .globl main

identity:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main$add_base:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $112, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$add_base(%rip), %rax
    movq %rax, -112(%rbp)
    movq %rbp, -104(%rbp)
    leaq -112(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $5, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $7, %rax
    movq %rax, -56(%rbp)
    leaq identity(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    movq -56(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $20, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $7, %rax
    movq %rax, -88(%rbp)
    leaq identity(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    movq -88(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

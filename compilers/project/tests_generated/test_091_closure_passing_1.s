    .text
    .globl main

apply_func:
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

main$mult:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %r10, -8(%rbp)
    movq %rdi, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rax), %rcx
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
    subq $80, %rsp
    movq %r10, -8(%rbp)
    movq $4, %rax
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rbp)
    leaq main$mult(%rip), %rax
    movq %rax, -80(%rbp)
    movq %rbp, -72(%rbp)
    leaq -80(%rbp), %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, -48(%rbp)
    movq $4, %rax
    movq %rax, -56(%rbp)
    leaq apply_func(%rip), %rax
    movq $0, %r10
    movq -48(%rbp), %rdi
    movq -56(%rbp), %rsi
    call *%rax
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -64(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret

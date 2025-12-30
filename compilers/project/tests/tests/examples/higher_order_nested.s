    .data
    .text
    .globl main

higher$id:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rax
    leave
    ret

higher:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    leaq higher$id(%rip), %rax
    movq %rax, -56(%rbp)
    movq %rbp, -48(%rbp)
    leaq -56(%rbp), %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main$caller:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq $42, %rax
    movq %rax, -8(%rbp)
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

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    leaq main$caller(%rip), %rax
    movq %rax, -48(%rbp)
    movq %rbp, -40(%rbp)
    leaq -48(%rbp), %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    leaq higher(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

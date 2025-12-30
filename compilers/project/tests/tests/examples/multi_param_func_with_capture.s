    .data
    .text
    .globl main

apply:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -24(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret

main$complex_func:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
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
    subq $128, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$complex_func(%rip), %rax
    movq %rax, -120(%rbp)
    movq %rbp, -112(%rbp)
    leaq -120(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq $3, %rax
    movq %rax, -48(%rbp)
    movq $7, %rax
    movq %rax, -56(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -48(%rbp), %rsi
    movq -56(%rbp), %rdx
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
    movq $5, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $3, %rax
    movq %rax, -88(%rbp)
    movq $7, %rax
    movq %rax, -96(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    movq -88(%rbp), %rsi
    movq -96(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -104(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

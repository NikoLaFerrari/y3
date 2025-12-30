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

main$func_with_side_effect:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq 24(%rbp), %rax
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
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $2, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$func_with_side_effect(%rip), %rax
    movq %rax, -136(%rbp)
    movq %rbp, -128(%rbp)
    leaq -136(%rbp), %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $5, %rax
    movq %rax, -56(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    movq -56(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $10, %rax
    movq %rax, -96(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    movq -96(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

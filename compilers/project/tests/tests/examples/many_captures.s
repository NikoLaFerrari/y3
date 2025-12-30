    .data
    .text
    .globl main

main$nested:
    pushq %rbp
    movq %rsp, %rbp
    subq $112, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq 24(%rbp), %rax
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
    addq %rcx, %rax
    movq %rax, -40(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -56(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -80(%rbp)
    movq 24(%rbp), %rax
    movq -80(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $20, %rax
    movq %rax, -88(%rbp)
    movq 24(%rbp), %rax
    movq -88(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $30, %rax
    movq %rax, -96(%rbp)
    movq 24(%rbp), %rax
    movq -96(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $40, %rax
    movq %rax, -104(%rbp)
    movq 24(%rbp), %rax
    movq -104(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $50, %rax
    movq %rax, -112(%rbp)
    movq 24(%rbp), %rax
    movq -112(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $192, %rsp
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $2, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $3, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $4, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $5, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -40(%rax)
    leaq main$nested(%rip), %rax
    movq %rax, -192(%rbp)
    movq %rbp, -184(%rbp)
    leaq -192(%rbp), %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -104(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -144(%rbp)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -160(%rbp)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq -160(%rbp), %rax
    movq -168(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -176(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -176(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

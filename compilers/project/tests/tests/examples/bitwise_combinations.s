    .data
    .text
    .globl main

bitwise_magic:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -16(%rbp), %rax
    movq -24(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -56(%rbp)
    movq -32(%rbp), %rax
    movq -56(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -72(%rbp), %rax
    leave
    ret

shift_and_combine:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq $2, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    shlq %cl, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    sarq %cl, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $192, %rsp
    movq $12, %rax
    movq %rax, -24(%rbp)
    movq $10, %rax
    movq %rax, -32(%rbp)
    leaq bitwise_magic(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $15, %rax
    movq %rax, -48(%rbp)
    movq $7, %rax
    movq %rax, -56(%rbp)
    leaq bitwise_magic(%rip), %rax
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
    movq $8, %rax
    movq %rax, -72(%rbp)
    leaq shift_and_combine(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $16, %rax
    movq %rax, -88(%rbp)
    leaq shift_and_combine(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
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
    movq $15, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $255, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -136(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -136(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -184(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -184(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

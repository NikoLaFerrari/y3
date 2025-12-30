    .data
    .text
    .globl main

main$show_local:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $100, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main$show_flag:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $1, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $4, %rax
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -64(%rbp)
    movq -64(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -16(%rax)
    leaq main$show_local(%rip), %rax
    movq %rax, -144(%rbp)
    movq %rbp, -136(%rbp)
    leaq -144(%rbp), %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -24(%rax)
    leaq main$show_flag(%rip), %rax
    movq %rax, -160(%rbp)
    movq %rbp, -152(%rbp)
    leaq -160(%rbp), %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq -96(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
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
    movq $1, %rax
    movq %rax, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -128(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret

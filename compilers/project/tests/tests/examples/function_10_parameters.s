    .data
    .text
    .globl main

add:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
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
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -136(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq $1, %rax
    movq %rax, -8(%rbp)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq $3, %rax
    movq %rax, -24(%rbp)
    movq $4, %rax
    movq %rax, -32(%rbp)
    movq $5, %rax
    movq %rax, -40(%rbp)
    movq $6, %rax
    movq %rax, -48(%rbp)
    movq $7, %rax
    movq %rax, -56(%rbp)
    movq $8, %rax
    movq %rax, -64(%rbp)
    movq $9, %rax
    movq %rax, -72(%rbp)
    movq $10, %rax
    movq %rax, -80(%rbp)
    leaq add(%rip), %rax
    movq $0, %r10
    movq -80(%rbp), %r11
    pushq %r11
    movq -72(%rbp), %r11
    pushq %r11
    movq -64(%rbp), %r11
    pushq %r11
    movq -56(%rbp), %r11
    pushq %r11
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    movq -24(%rbp), %rdx
    movq -32(%rbp), %rcx
    movq -40(%rbp), %r8
    movq -48(%rbp), %r9
    call *%rax
    addq $48, %rsp
    movq %rax, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

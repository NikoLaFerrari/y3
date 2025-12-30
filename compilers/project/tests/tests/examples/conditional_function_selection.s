    .data
    .text
    .globl main

apply:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -64(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    leave
    ret
.L2:
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rax
    leave
    ret

main$neg:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq $0, %rax
    movq %rax, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main$id:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    leaq main$neg(%rip), %rax
    movq %rax, -112(%rbp)
    movq %rbp, -104(%rbp)
    leaq -112(%rbp), %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$id(%rip), %rax
    movq %rax, -128(%rbp)
    movq %rbp, -120(%rbp)
    leaq -128(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $5, %rax
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
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $-3, %rax
    movq %rax, -88(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    movq -80(%rbp), %rsi
    movq -88(%rbp), %rdx
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

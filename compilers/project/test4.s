    .text
    .globl main

apply_twice:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    movq -8(%rbp), %rdi
    call *%rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    movq -24(%rbp), %rdi
    call *%rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main$increment:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main$add_y:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq 16(%rbp), %rax
    movq -8(%rax), %rcx
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
    movq $5, %rax
    movq %rax, -8(%rbp)
    movq %rbp, %rax
    movq -8(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$increment(%rip), %rax
    movq %rax, -88(%rbp)
    movq %rbp, -80(%rbp)
    leaq -88(%rbp), %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -16(%rax)
    leaq main$add_y(%rip), %rax
    movq %rax, -104(%rbp)
    movq %rbp, -96(%rbp)
    leaq -104(%rbp), %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $10, %rax
    movq %rax, -40(%rbp)
    leaq apply_twice(%rip), %rax
    movq $0, %r10
    movq -32(%rbp), %rdi
    movq -40(%rbp), %rsi
    call *%rax
    movq %rax, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -48(%rbp), %rdi
    call *%rax
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $16, %rax
    movq %rax, -64(%rbp)
    leaq apply_twice(%rip), %rax
    movq $0, %r10
    movq -56(%rbp), %rdi
    movq -64(%rbp), %rsi
    call *%rax
    movq %rax, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -72(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret

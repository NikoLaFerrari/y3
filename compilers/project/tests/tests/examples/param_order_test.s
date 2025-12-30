    .data
    .text
    .globl main

check_param_order:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -88(%rbp)
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
    movq -136(%rbp), %rax
    leave
    ret

test_stack_args:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -64(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $192, %rsp
    movq $100, %rax
    movq %rax, -24(%rbp)
    movq $200, %rax
    movq %rax, -32(%rbp)
    movq $300, %rax
    movq %rax, -40(%rbp)
    movq $400, %rax
    movq %rax, -48(%rbp)
    movq $500, %rax
    movq %rax, -56(%rbp)
    movq $600, %rax
    movq %rax, -64(%rbp)
    movq $700, %rax
    movq %rax, -72(%rbp)
    movq $800, %rax
    movq %rax, -80(%rbp)
    movq $900, %rax
    movq %rax, -88(%rbp)
    movq $1000, %rax
    movq %rax, -96(%rbp)
    leaq check_param_order(%rip), %rax
    movq $0, %r10
    movq -96(%rbp), %r11
    pushq %r11
    movq -88(%rbp), %r11
    pushq %r11
    movq -80(%rbp), %r11
    pushq %r11
    movq -72(%rbp), %r11
    pushq %r11
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    movq -40(%rbp), %rdx
    movq -48(%rbp), %rcx
    movq -56(%rbp), %r8
    movq -64(%rbp), %r9
    call *%rax
    addq $48, %rsp
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -120(%rbp)
    movq $20, %rax
    movq %rax, -128(%rbp)
    movq $30, %rax
    movq %rax, -136(%rbp)
    movq $40, %rax
    movq %rax, -144(%rbp)
    movq $50, %rax
    movq %rax, -152(%rbp)
    movq $60, %rax
    movq %rax, -160(%rbp)
    movq $70, %rax
    movq %rax, -168(%rbp)
    leaq test_stack_args(%rip), %rax
    movq $0, %r10
    movq -168(%rbp), %r11
    pushq %r11
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    movq -128(%rbp), %rsi
    movq -136(%rbp), %rdx
    movq -144(%rbp), %rcx
    movq -152(%rbp), %r8
    movq -160(%rbp), %r9
    call *%rax
    addq $24, %rsp
    movq %rax, -176(%rbp)
    movq %rbp, %rax
    movq -176(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -184(%rbp)
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

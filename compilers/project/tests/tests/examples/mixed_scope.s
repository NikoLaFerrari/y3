    .data
    .globl glob_global_offset
glob_global_offset:
    .zero 8
    .text
    .globl main

add_with_offset:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq glob_global_offset(%rip), %rcx
    movq %rcx, -24(%rbp)
    movq -16(%rbp), %rax
    movq -24(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

multiply_and_add:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq glob_global_offset(%rip), %rcx
    movq %rcx, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -72(%rbp), %rax
    leave
    ret

update_offset:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_global_offset(%rip)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $50, %rax
    movq %rax, -40(%rbp)
    leaq add_with_offset(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $3, %rax
    movq %rax, -64(%rbp)
    movq $4, %rax
    movq %rax, -72(%rbp)
    leaq multiply_and_add(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    movq -72(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $200, %rax
    movq %rax, -96(%rbp)
    leaq update_offset(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $50, %rax
    movq %rax, -104(%rbp)
    leaq add_with_offset(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $3, %rax
    movq %rax, -128(%rbp)
    movq $4, %rax
    movq %rax, -136(%rbp)
    leaq multiply_and_add(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -128(%rbp), %rdi
    movq -136(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -144(%rbp)
    movq %rbp, %rax
    movq -144(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -152(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -152(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

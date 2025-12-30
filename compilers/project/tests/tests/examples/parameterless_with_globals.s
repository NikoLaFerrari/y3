    .data
    .globl glob_counter
glob_counter:
    .zero 8
    .globl glob_step_size
glob_step_size:
    .zero 8
    .text
    .globl main

get_counter:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq glob_counter(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rax
    leave
    ret

increment_by_step:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_counter(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_step_size(%rip), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_counter(%rip)
    leave
    ret

reset_counter:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $0, %rax
    movq %rax, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_counter(%rip)
    leave
    ret

double_step:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_step_size(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_step_size(%rip)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    leaq get_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rax, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq increment_by_step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq get_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rax, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq increment_by_step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq get_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rax, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq double_step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq increment_by_step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq get_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rax, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq reset_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq get_counter(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
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
    movq $0, %rax
    leave
    ret

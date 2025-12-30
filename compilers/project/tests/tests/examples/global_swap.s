    .data
    .globl glob_a
glob_a:
    .zero 8
    .globl glob_b
glob_b:
    .zero 8
    .globl glob_temp
glob_temp:
    .zero 8
    .text
    .globl main

save_a:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_temp(%rip)
    leave
    ret

copy_b_to_a:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_a(%rip)
    leave
    ret

restore_to_b:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq glob_temp(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_b(%rip)
    leave
    ret

swap:
    pushq %rbp
    movq %rsp, %rbp
    subq $0, %rsp
    leaq save_a(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq copy_b_to_a(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq restore_to_b(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq swap(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq swap(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret

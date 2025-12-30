    .data
    .globl glob_result
glob_result:
    .zero 8
    .globl glob_operand1
glob_operand1:
    .zero 8
    .globl glob_operand2
glob_operand2:
    .zero 8
    .text
    .globl main

load_operands:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_operand1(%rip)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, glob_operand2(%rip)
    leave
    ret

add:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_operand1(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_operand2(%rip), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_result(%rip)
    leave
    ret

subtract:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_operand1(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_operand2(%rip), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_result(%rip)
    leave
    ret

multiply:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_operand1(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_operand2(%rip), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_result(%rip)
    leave
    ret

use_result_as_operand1:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq glob_result(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_operand1(%rip)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq $15, %rax
    movq %rax, -8(%rbp)
    movq $5, %rax
    movq %rax, -16(%rbp)
    leaq load_operands(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq glob_operand1(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_operand2(%rip), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq add(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_result(%rip), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $20, %rax
    movq %rax, -48(%rbp)
    movq $3, %rax
    movq %rax, -56(%rbp)
    leaq load_operands(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    movq -56(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    leaq multiply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_result(%rip), %rcx
    movq %rcx, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq use_result_as_operand1(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_result(%rip), %rcx
    movq %rcx, -72(%rbp)
    movq $10, %rax
    movq %rax, -80(%rbp)
    leaq load_operands(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    movq -80(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    leaq subtract(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_result(%rip), %rcx
    movq %rcx, -88(%rbp)
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

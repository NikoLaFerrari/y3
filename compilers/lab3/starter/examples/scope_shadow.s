.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $48, %rsp  # 6 slots
  movq $10, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $20, %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -24(%rbp), %rdi
  callq bx_print_int
  movq $1, %r11
  movq %r11, -40(%rbp)
  movq -24(%rbp), %r11
  addq -40(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -8(%rbp), %rdi
  callq bx_print_int
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq

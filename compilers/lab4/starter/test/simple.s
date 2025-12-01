.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $32, %rsp  # 4 slots
  movq $5, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $1, %r11
  movq %r11, -24(%rbp)
  movq -8(%rbp), %r11
  addq -24(%rbp), %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -8(%rbp)
  movq -8(%rbp), %rdi
  callq bx_print_int
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq

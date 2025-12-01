.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $96, %rsp  # 12 slots
  movq $10, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq -8(%rbp), %rdi
  callq bx_print_int
  movq $20, %r11
  movq %r11, -32(%rbp)
  movq -8(%rbp), %r11
  addq -32(%rbp), %r11
  movq %r11, -40(%rbp)
  movq -40(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -24(%rbp), %rdi
  callq bx_print_int
  movq $42, %r11
  movq %r11, -56(%rbp)
  movq -24(%rbp), %r11
  imulq -56(%rbp), %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %rdi
  callq bx_print_int
  movq -8(%rbp), %r11
  imulq -24(%rbp), %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  subq -48(%rbp), %r11
  movq %r11, -88(%rbp)
  movq -88(%rbp), %r11
  movq %r11, -72(%rbp)
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq

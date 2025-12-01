.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $304, %rsp  # 38 slots
  movq $10, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $20, %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -40(%rbp)
  movq -40(%rbp), %r11
  cmpq $0, %r11
  je .L0
  jmp .L1
.L0:
  movq $0, %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %rdi
  callq bx_print_int
  jmp .L2
.L1:
.L2:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -56(%rbp)
  movq -56(%rbp), %r11
  cmpq $0, %r11
  jne .L3
  jmp .L4
.L3:
  movq $1, %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %rdi
  callq bx_print_int
  jmp .L5
.L4:
.L5:
  movq $2, %r11
  movq %r11, -72(%rbp)
  movq -8(%rbp), %r11
  imulq -72(%rbp), %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -88(%rbp)
  movq -88(%rbp), %r11
  cmpq $0, %r11
  je .L6
  jmp .L7
.L6:
  movq $2, %r11
  movq %r11, -96(%rbp)
  movq -96(%rbp), %rdi
  callq bx_print_int
  jmp .L8
.L7:
.L8:
  jmp .L9
.L12:
  jmp .L9
.L9:
  movq $3, %r11
  movq %r11, -104(%rbp)
  movq -104(%rbp), %rdi
  callq bx_print_int
  jmp .L11
.L10:
.L11:
  movq $1, %r11
  movq %r11, -112(%rbp)
  movq $1, %r11
  movq %r11, -120(%rbp)
  movq -112(%rbp), %r11
  subq -120(%rbp), %r11
  movq %r11, -128(%rbp)
  movq -128(%rbp), %r11
  cmpq $0, %r11
  je .L13
  jmp .L16
.L16:
  jmp .L14
.L13:
  movq $4, %r11
  movq %r11, -136(%rbp)
  movq -136(%rbp), %rdi
  callq bx_print_int
  jmp .L15
.L14:
.L15:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -144(%rbp)
  movq -144(%rbp), %r11
  cmpq $0, %r11
  jl .L17
  jmp .L18
.L17:
  movq $5, %r11
  movq %r11, -152(%rbp)
  movq -152(%rbp), %rdi
  callq bx_print_int
  jmp .L19
.L18:
.L19:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -160(%rbp)
  movq -160(%rbp), %r11
  cmpq $0, %r11
  jg .L20
  jmp .L21
.L20:
  movq $6, %r11
  movq %r11, -168(%rbp)
  movq -168(%rbp), %rdi
  callq bx_print_int
  jmp .L22
.L21:
.L22:
  movq -8(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -176(%rbp)
  movq -176(%rbp), %r11
  cmpq $0, %r11
  jl .L23
  jmp .L24
.L23:
  movq $7, %r11
  movq %r11, -184(%rbp)
  movq -184(%rbp), %rdi
  callq bx_print_int
  jmp .L25
.L24:
.L25:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -192(%rbp)
  movq -192(%rbp), %r11
  cmpq $0, %r11
  jle .L26
  jmp .L27
.L26:
  movq $8, %r11
  movq %r11, -200(%rbp)
  movq -200(%rbp), %rdi
  callq bx_print_int
  jmp .L28
.L27:
.L28:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -208(%rbp)
  movq -208(%rbp), %r11
  cmpq $0, %r11
  jge .L29
  jmp .L30
.L29:
  movq $9, %r11
  movq %r11, -216(%rbp)
  movq -216(%rbp), %rdi
  callq bx_print_int
  jmp .L31
.L30:
.L31:
  movq -8(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -224(%rbp)
  movq -224(%rbp), %r11
  cmpq $0, %r11
  jle .L32
  jmp .L33
.L32:
  movq $10, %r11
  movq %r11, -232(%rbp)
  movq -232(%rbp), %rdi
  callq bx_print_int
  jmp .L34
.L33:
.L34:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -240(%rbp)
  movq -240(%rbp), %r11
  cmpq $0, %r11
  jle .L35
  jmp .L38
.L38:
  movq -24(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -248(%rbp)
  movq -248(%rbp), %r11
  cmpq $0, %r11
  jle .L35
  jmp .L36
.L35:
  movq $11, %r11
  movq %r11, -256(%rbp)
  movq -256(%rbp), %rdi
  callq bx_print_int
  jmp .L37
.L36:
.L37:
  movq -8(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -264(%rbp)
  movq -264(%rbp), %r11
  cmpq $0, %r11
  je .L39
  jmp .L40
.L39:
  movq $12, %r11
  movq %r11, -272(%rbp)
  movq -272(%rbp), %rdi
  callq bx_print_int
  jmp .L41
.L40:
.L41:
  movq -8(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -280(%rbp)
  movq -280(%rbp), %r11
  cmpq $0, %r11
  je .L43
  jmp .L42
.L42:
  movq $13, %r11
  movq %r11, -288(%rbp)
  movq -288(%rbp), %rdi
  callq bx_print_int
  jmp .L44
.L43:
.L44:
  movq -8(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -296(%rbp)
  movq -296(%rbp), %r11
  cmpq $0, %r11
  jne .L46
  jmp .L45
.L45:
  movq $14, %r11
  movq %r11, -304(%rbp)
  movq -304(%rbp), %rdi
  callq bx_print_int
  jmp .L47
.L46:
.L47:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq

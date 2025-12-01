.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $864, %rsp  # 108 slots
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
  je .L12
  jmp .L11
.L12:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L11
.L11:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -56(%rbp)
  movq -56(%rbp), %r11
  cmpq $0, %r11
  jne .L10
  jmp .L3
.L10:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  cmpq $0, %r11
  jne .L9
  jmp .L3
.L9:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -72(%rbp)
  movq -72(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L8
.L8:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  cmpq $0, %r11
  je .L14
  jmp .L7
.L14:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -88(%rbp)
  movq -88(%rbp), %r11
  cmpq $0, %r11
  jne .L13
  jmp .L7
.L13:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -96(%rbp)
  movq -96(%rbp), %r11
  cmpq $0, %r11
  jne .L15
  jmp .L16
.L16:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -104(%rbp)
  movq -104(%rbp), %r11
  cmpq $0, %r11
  je .L15
  jmp .L3
.L15:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -112(%rbp)
  movq -112(%rbp), %r11
  cmpq $0, %r11
  jne .L7
  jmp .L17
.L17:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -120(%rbp)
  movq -120(%rbp), %r11
  cmpq $0, %r11
  je .L7
  jmp .L3
.L7:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -128(%rbp)
  movq -128(%rbp), %r11
  cmpq $0, %r11
  jne .L23
  jmp .L6
.L23:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -136(%rbp)
  movq -136(%rbp), %r11
  cmpq $0, %r11
  jne .L24
  jmp .L22
.L24:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -144(%rbp)
  movq -144(%rbp), %r11
  cmpq $0, %r11
  jne .L6
  jmp .L22
.L22:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -152(%rbp)
  movq -152(%rbp), %r11
  cmpq $0, %r11
  jne .L25
  jmp .L21
.L25:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -160(%rbp)
  movq -160(%rbp), %r11
  cmpq $0, %r11
  je .L6
  jmp .L21
.L21:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -168(%rbp)
  movq -168(%rbp), %r11
  cmpq $0, %r11
  jne .L6
  jmp .L20
.L20:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -176(%rbp)
  movq -176(%rbp), %r11
  cmpq $0, %r11
  je .L6
  jmp .L19
.L19:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -184(%rbp)
  movq -184(%rbp), %r11
  cmpq $0, %r11
  je .L26
  jmp .L27
.L27:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -192(%rbp)
  movq -192(%rbp), %r11
  cmpq $0, %r11
  jne .L26
  jmp .L18
.L26:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -200(%rbp)
  movq -200(%rbp), %r11
  cmpq $0, %r11
  je .L6
  jmp .L28
.L28:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -208(%rbp)
  movq -208(%rbp), %r11
  cmpq $0, %r11
  jne .L6
  jmp .L18
.L18:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -216(%rbp)
  movq -216(%rbp), %r11
  cmpq $0, %r11
  jne .L6
  jmp .L3
.L6:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -224(%rbp)
  movq -224(%rbp), %r11
  cmpq $0, %r11
  jne .L30
  jmp .L3
.L30:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -232(%rbp)
  movq -232(%rbp), %r11
  cmpq $0, %r11
  jne .L29
  jmp .L3
.L29:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -240(%rbp)
  movq -240(%rbp), %r11
  cmpq $0, %r11
  jne .L32
  jmp .L5
.L32:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -248(%rbp)
  movq -248(%rbp), %r11
  cmpq $0, %r11
  jne .L31
  jmp .L5
.L31:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -256(%rbp)
  movq -256(%rbp), %r11
  cmpq $0, %r11
  jne .L5
  jmp .L3
.L5:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -264(%rbp)
  movq -264(%rbp), %r11
  cmpq $0, %r11
  jne .L36
  jmp .L4
.L36:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -272(%rbp)
  movq -272(%rbp), %r11
  cmpq $0, %r11
  je .L35
  jmp .L4
.L35:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -280(%rbp)
  movq -280(%rbp), %r11
  cmpq $0, %r11
  jne .L34
  jmp .L4
.L34:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -288(%rbp)
  movq -288(%rbp), %r11
  cmpq $0, %r11
  je .L33
  jmp .L4
.L33:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -296(%rbp)
  movq -296(%rbp), %r11
  cmpq $0, %r11
  jne .L37
  jmp .L3
.L37:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -304(%rbp)
  movq -304(%rbp), %r11
  cmpq $0, %r11
  jne .L4
  jmp .L3
.L4:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -312(%rbp)
  movq -312(%rbp), %r11
  cmpq $0, %r11
  jne .L43
  jmp .L45
.L45:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -320(%rbp)
  movq -320(%rbp), %r11
  cmpq $0, %r11
  je .L43
  jmp .L44
.L44:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -328(%rbp)
  movq -328(%rbp), %r11
  cmpq $0, %r11
  jne .L38
  jmp .L47
.L47:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -336(%rbp)
  movq -336(%rbp), %r11
  cmpq $0, %r11
  jne .L38
  jmp .L46
.L46:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -344(%rbp)
  movq -344(%rbp), %r11
  cmpq $0, %r11
  jne .L43
  jmp .L38
.L43:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -352(%rbp)
  movq -352(%rbp), %r11
  cmpq $0, %r11
  je .L50
  jmp .L49
.L50:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -360(%rbp)
  movq -360(%rbp), %r11
  cmpq $0, %r11
  je .L48
  jmp .L49
.L49:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -368(%rbp)
  movq -368(%rbp), %r11
  cmpq $0, %r11
  jne .L51
  jmp .L42
.L51:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -376(%rbp)
  movq -376(%rbp), %r11
  cmpq $0, %r11
  je .L48
  jmp .L42
.L48:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -384(%rbp)
  movq -384(%rbp), %r11
  cmpq $0, %r11
  jne .L52
  jmp .L38
.L52:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -392(%rbp)
  movq -392(%rbp), %r11
  cmpq $0, %r11
  jne .L42
  jmp .L38
.L42:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -400(%rbp)
  movq -400(%rbp), %r11
  cmpq $0, %r11
  jne .L38
  jmp .L53
.L53:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -408(%rbp)
  movq -408(%rbp), %r11
  cmpq $0, %r11
  jne .L41
  jmp .L54
.L54:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -416(%rbp)
  movq -416(%rbp), %r11
  cmpq $0, %r11
  je .L41
  jmp .L38
.L41:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -424(%rbp)
  movq -424(%rbp), %r11
  cmpq $0, %r11
  je .L56
  jmp .L57
.L57:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -432(%rbp)
  movq -432(%rbp), %r11
  cmpq $0, %r11
  jne .L56
  jmp .L38
.L56:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -440(%rbp)
  movq -440(%rbp), %r11
  cmpq $0, %r11
  je .L55
  jmp .L38
.L55:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -448(%rbp)
  movq -448(%rbp), %r11
  cmpq $0, %r11
  jne .L40
  jmp .L38
.L40:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -456(%rbp)
  movq -456(%rbp), %r11
  cmpq $0, %r11
  je .L61
  jmp .L39
.L61:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -464(%rbp)
  movq -464(%rbp), %r11
  cmpq $0, %r11
  jne .L60
  jmp .L39
.L60:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -472(%rbp)
  movq -472(%rbp), %r11
  cmpq $0, %r11
  je .L39
  jmp .L59
.L59:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -480(%rbp)
  movq -480(%rbp), %r11
  cmpq $0, %r11
  jne .L39
  jmp .L58
.L58:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -488(%rbp)
  movq -488(%rbp), %r11
  cmpq $0, %r11
  jne .L38
  jmp .L39
.L39:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -496(%rbp)
  movq -496(%rbp), %r11
  cmpq $0, %r11
  jne .L0
  jmp .L62
.L62:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -504(%rbp)
  movq -504(%rbp), %r11
  cmpq $0, %r11
  je .L38
  jmp .L63
.L63:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -512(%rbp)
  movq -512(%rbp), %r11
  cmpq $0, %r11
  je .L38
  jmp .L0
.L38:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -520(%rbp)
  movq -520(%rbp), %r11
  cmpq $0, %r11
  je .L67
  jmp .L65
.L67:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -528(%rbp)
  movq -528(%rbp), %r11
  cmpq $0, %r11
  je .L65
  jmp .L66
.L66:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -536(%rbp)
  movq -536(%rbp), %r11
  cmpq $0, %r11
  je .L65
  jmp .L64
.L65:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -544(%rbp)
  movq -544(%rbp), %r11
  cmpq $0, %r11
  jne .L68
  jmp .L3
.L68:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -552(%rbp)
  movq -552(%rbp), %r11
  cmpq $0, %r11
  jne .L69
  jmp .L64
.L69:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -560(%rbp)
  movq -560(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L64
.L64:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -568(%rbp)
  movq -568(%rbp), %r11
  cmpq $0, %r11
  jne .L70
  jmp .L72
.L72:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -576(%rbp)
  movq -576(%rbp), %r11
  cmpq $0, %r11
  jne .L70
  jmp .L71
.L71:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -584(%rbp)
  movq -584(%rbp), %r11
  cmpq $0, %r11
  jne .L73
  jmp .L0
.L73:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -592(%rbp)
  movq -592(%rbp), %r11
  cmpq $0, %r11
  je .L70
  jmp .L0
.L70:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -600(%rbp)
  movq -600(%rbp), %r11
  cmpq $0, %r11
  jne .L76
  jmp .L75
.L76:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -608(%rbp)
  movq -608(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L75
.L75:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -616(%rbp)
  movq -616(%rbp), %r11
  cmpq $0, %r11
  jne .L3
  jmp .L74
.L74:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -624(%rbp)
  movq -624(%rbp), %r11
  cmpq $0, %r11
  jne .L3
  jmp .L0
.L3:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -632(%rbp)
  movq -632(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L85
.L85:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -640(%rbp)
  movq -640(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L84
.L84:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -648(%rbp)
  movq -648(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L83
.L83:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -656(%rbp)
  movq -656(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L82
.L82:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -664(%rbp)
  movq -664(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L81
.L81:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -672(%rbp)
  movq -672(%rbp), %r11
  cmpq $0, %r11
  je .L77
  jmp .L80
.L80:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -680(%rbp)
  movq -680(%rbp), %r11
  cmpq $0, %r11
  je .L79
  jmp .L77
.L79:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -688(%rbp)
  movq -688(%rbp), %r11
  cmpq $0, %r11
  jne .L86
  jmp .L87
.L87:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -696(%rbp)
  movq -696(%rbp), %r11
  cmpq $0, %r11
  je .L86
  jmp .L78
.L86:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -704(%rbp)
  movq -704(%rbp), %r11
  cmpq $0, %r11
  je .L78
  jmp .L77
.L78:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -712(%rbp)
  movq -712(%rbp), %r11
  cmpq $0, %r11
  jne .L88
  jmp .L89
.L89:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -720(%rbp)
  movq -720(%rbp), %r11
  cmpq $0, %r11
  jne .L88
  jmp .L1
.L88:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -728(%rbp)
  movq -728(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L90
.L90:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -736(%rbp)
  movq -736(%rbp), %r11
  cmpq $0, %r11
  jne .L77
  jmp .L1
.L77:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -744(%rbp)
  movq -744(%rbp), %r11
  cmpq $0, %r11
  je .L91
  jmp .L94
.L94:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -752(%rbp)
  movq -752(%rbp), %r11
  cmpq $0, %r11
  jne .L91
  jmp .L93
.L93:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -760(%rbp)
  movq -760(%rbp), %r11
  cmpq $0, %r11
  jne .L92
  jmp .L95
.L95:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -768(%rbp)
  movq -768(%rbp), %r11
  cmpq $0, %r11
  je .L92
  jmp .L91
.L92:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -776(%rbp)
  movq -776(%rbp), %r11
  cmpq $0, %r11
  jne .L91
  jmp .L0
.L91:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -784(%rbp)
  movq -784(%rbp), %r11
  cmpq $0, %r11
  jne .L96
  jmp .L100
.L100:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -792(%rbp)
  movq -792(%rbp), %r11
  cmpq $0, %r11
  jne .L96
  jmp .L99
.L99:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -800(%rbp)
  movq -800(%rbp), %r11
  cmpq $0, %r11
  jne .L98
  jmp .L101
.L101:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -808(%rbp)
  movq -808(%rbp), %r11
  cmpq $0, %r11
  jne .L98
  jmp .L96
.L98:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -816(%rbp)
  movq -816(%rbp), %r11
  cmpq $0, %r11
  jne .L97
  jmp .L96
.L97:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -824(%rbp)
  movq -824(%rbp), %r11
  cmpq $0, %r11
  jne .L1
  jmp .L96
.L96:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -832(%rbp)
  movq -832(%rbp), %r11
  cmpq $0, %r11
  jne .L102
  jmp .L0
.L102:
  movq -8(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -840(%rbp)
  movq -840(%rbp), %r11
  cmpq $0, %r11
  je .L1
  jmp .L0
.L0:
  movq $42, %r11
  movq %r11, -848(%rbp)
  movq -848(%rbp), %rdi
  callq bx_print_int
  jmp .L2
.L1:
  movq $42, %r11
  movq %r11, -856(%rbp)
  movq -856(%rbp), %r11
  negq %r11
  movq %r11, -864(%rbp)
  movq -864(%rbp), %rdi
  callq bx_print_int
.L2:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq

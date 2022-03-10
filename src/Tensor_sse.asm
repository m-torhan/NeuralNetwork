global _SSE_vector_inner_product

section .data

section .text

_SSE_vector_inner_product:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; N
	mov		eax, [ebp+16]		; *v1
	mov		esi, [ebp+20]		; *v2
	mov		edi, [ebp+24]		; *r
	
	sub		esp, 16
	xorps 	xmm0, xmm0

ups_mul_loop:
	sub		ecx, 4
	
	movups	xmm1, [eax + 4*ecx]
	movups	xmm2, [esi + 4*ecx]

	mulps	xmm1, xmm2
	addps	xmm0, xmm1

	cmp		ecx, 4
	jge		ups_mul_loop

	cmp		ecx, 0
	jle		end

mul_loop:
	sub		ecx, 1

	movss	xmm1, dword [eax + 4*ecx]
	movss	xmm2, dword [esi + 4*ecx]
	
	mulss	xmm1, xmm2
	addss	xmm0, xmm1

	cmp		ecx, 0
	jge		mul_loop

end:

	movhlps xmm1, xmm0
	addps   xmm0, xmm1
	movaps  xmm1, xmm0
	shufps  xmm1, xmm1, 0b01010101
	addss   xmm0, xmm1

	movss   [edi], xmm0

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

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

mul_loop:
	sub		ecx, 4
	
	movups	xmm5, [eax + 4*ecx]
	movups	xmm6, [esi + 4*ecx]

	mulps	xmm5, xmm6
 
	movups	[esp], xmm5

	fld		dword [edi]
	fadd	dword [esp]
	fadd	dword [esp+4]
	fadd	dword [esp+8]
	fadd	dword [esp+16]
	fstp	dword [edi]

	cmp		ecx, 0
	jl		mul_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

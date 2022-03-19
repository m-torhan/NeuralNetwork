global _SSE_vector_inner_product
global _SSE_vector_add
global _SSE_tensor_add
global _SSE_tensor_add_scalar

section .data

section .text

_SSE_vector_inner_product:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (scalar)
	
	xorps 	xmm0, xmm0

VIP_ups_mul_loop:
	cmp		ecx, 4
	jl		VIP_ss_mul_loop

	sub		ecx, 4
	
	movups	xmm1, [eax + 4*ecx]
	movups	xmm2, [esi + 4*ecx]

	mulps	xmm1, xmm2
	addps	xmm0, xmm1

	jmp		VIP_ups_mul_loop

VIP_ss_mul_loop:
	cmp		ecx, 1
	jl		VIP_end

	sub		ecx, 1

	movss	xmm1, dword [eax + 4*ecx]
	movss	xmm2, dword [esi + 4*ecx]
	
	mulss	xmm1, xmm2
	addss	xmm0, xmm1

	jmp		VIP_ss_mul_loop

VIP_end:

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

_SSE_vector_add:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)
	
	xorps 	xmm0, xmm0

VA_ups_add_loop:
	cmp		ecx, 4
	jl		VA_ss_add_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		VA_ups_add_loop

VA_ss_add_loop:
	cmp		ecx, 1
	jl		VA_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		VA_ss_add_loop

VA_end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

_SSE_tensor_add:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n1	uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		edx, [ebp+20]		; n2	uint32
	mov		esi, [ebp+24]		; *v2	float* (array)
	mov		edi, [ebp+28]		; *r	float* (array)
	
	mov		ebx, ecx

TA_row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

TA_ups_add_row_loop:
	cmp		ecx, 4
	jl		TA_ss_add_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		TA_ups_add_row_loop

TA_ss_add_loop:
	cmp		ecx, 1
	jl		TA_row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		TA_ss_add_loop

TA_row_end:
	cmp		ebx, 0
	jg		TA_row_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

_SSE_tensor_add_scalar:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v	float* (array)
	mov		esi, [ebp+20]		; *s	float* (scalar)
	mov		edi, [ebp+24]		; *r	float* (array)

	movss	xmm1, dword [esi]
	shufps	xmm1, xmm1, 0x0
	
	xorps 	xmm0, xmm0

TAS_ups_add_loop:
	cmp		ecx, 4
	jl		TAS_ss_add_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		TAS_ups_add_loop

TAS_ss_add_loop:
	cmp		ecx, 1
	jl		TAS_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		TAS_ss_add_loop

TAS_end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret
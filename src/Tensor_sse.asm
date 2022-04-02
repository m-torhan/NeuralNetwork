global _SSE_vector_inner_product

global _SSE_vector_add
global _SSE_tensor_add
global _SSE_tensor_add_scalar

global _SSE_vector_sub
global _SSE_tensor_sub
global _SSE_tensor_sub_scalar
global _SSE_scalar_sub_tensor

global _SSE_vector_mul
global _SSE_tensor_mul
global _SSE_tensor_mul_scalar

global _SSE_vector_div
global _SSE_tensor_div
global _SSE_tensor_div_scalar
global _SSE_scalar_div_tensor

section .data

section .text

; void SSE_vector_inner_product(const uint32_t n, const float* v1, const float* v2, float* r);
;	n - size of v1 and v2
;	v1 - first vector
;	v2 - second vector
;	r - return value
_SSE_vector_inner_product:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (scalar)
	
	xorps 	xmm0, xmm0

.ps_mul_loop:
	cmp		ecx, 4
	jl		.ss_mul_loop

	sub		ecx, 4
	
	movups	xmm1, [eax + 4*ecx]
	movups	xmm2, [esi + 4*ecx]

	mulps	xmm1, xmm2
	addps	xmm0, xmm1

	jmp		.ps_mul_loop

.ss_mul_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm1, dword [eax + 4*ecx]
	movss	xmm2, dword [esi + 4*ecx]
	
	mulss	xmm1, xmm2
	addss	xmm0, xmm1

	jmp		.ss_mul_loop

.end:

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

; void SSE_vector_add(const uint32_t n, const float* v1, const float* v2, float* r);
;	n - size of v1 and v2
;	v1 - first vector
;	v2 - second vector
;	r - return value
_SSE_vector_add:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)
	
	xorps 	xmm0, xmm0

.ps_add_loop:
	cmp		ecx, 4
	jl		.ss_add_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_add_loop

.ss_add_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_add_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_tensor_add(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
;	n1 - size of v1
;	v1 - first tensor
;	n2 - size of v1
;	v2 - second tensor
;	r - return value
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

.row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

.ps_add_row_loop:
	cmp		ecx, 4
	jl		.ss_add_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		.ps_add_row_loop

.ss_add_loop:
	cmp		ecx, 1
	jl		.row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		.ss_add_loop

.row_end:
	cmp		ebx, 0
	jg		.row_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret


; void SSE_tensor_add_scalar(const uint32_t n1, const float* v, const float* s, float* r);
;	n - size of v
;	v - tensor
;	s - scalar
;	r - return value
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

.ps_add_loop:
	cmp		ecx, 4
	jl		.ss_add_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	addps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_add_loop

.ss_add_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_add_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_vector_sub(const uint32_t n, const float* v1, const float* v2, float* r);
;	n - size of v1 and v2
;	v1 - first vector
;	v2 - second vector
;	r - return value
_SSE_vector_sub:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)
	
	xorps 	xmm0, xmm0

.ps_sub_loop:
	cmp		ecx, 4
	jl		.ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_sub_loop

.ss_sub_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	subss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_sub_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_tensor_sub(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
;	n1 - size of v1
;	v1 - first tensor
;	n2 - size of v1
;	v2 - second tensor
;	r - return value
_SSE_tensor_sub:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n1	uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		edx, [ebp+20]		; n2	uint32
	mov		esi, [ebp+24]		; *v2	float* (array)
	mov		edi, [ebp+28]		; *r	float* (array)
	
	mov		ebx, ecx

.row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

.ps_sub_row_loop:
	cmp		ecx, 4
	jl		.ss_sub_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		.ps_sub_row_loop

.ss_sub_loop:
	cmp		ecx, 1
	jl		.row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	subss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		.ss_sub_loop

.row_end:
	cmp		ebx, 0
	jg		.row_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_tensor_sub_scalar(const uint32_t n1, const float* v, const float* s, float* r);
;	n - size of v
;	v - tensor
;	s - scalar
;	r - return value
_SSE_tensor_sub_scalar:
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

.ps_sub_loop:
	cmp		ecx, 4
	jl		.ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_sub_loop

.ss_sub_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	subss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_sub_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_scalar_sub_tensor(const float* s, const uint32_t n, const float* v, float* r);
;	s - scalar
;	n - size of v
;	v - tensor
;	r - return value
_SSE_scalar_sub_tensor:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		esi, [ebp+12]		; *s	float* (scalar)
	mov		ecx, [ebp+16]		; n		uint32
	mov		eax, [ebp+20]		; *v	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)

	movss	xmm1, dword [esi]
	shufps	xmm1, xmm1, 0x0
	
	xorps 	xmm0, xmm0

.ps_sub_loop:
	cmp		ecx, 4
	jl		.ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	vsubps	xmm2, xmm1, xmm0

	movups	[edi + 4*ecx], xmm2

	jmp		.ps_sub_loop

.ss_sub_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	vsubss	xmm2, xmm1, xmm0

	movss	[edi + 4*ecx], xmm2

	jmp		.ss_sub_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_vector_mul(const uint32_t n, const float* v1, const float* v2, float* r);
;	n - size of v1 and v2
;	v1 - first vector
;	v2 - second vector
;	r - return value
_SSE_vector_mul:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)
	
	xorps 	xmm0, xmm0

.ps_mul_loop:
	cmp		ecx, 4
	jl		.ss_mul_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_mul_loop

.ss_mul_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_mul_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_tensor_mul(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
;	n1 - size of v1
;	v1 - first tensor
;	n2 - size of v1
;	v2 - second tensor
;	r - return value
_SSE_tensor_mul:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n1	uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		edx, [ebp+20]		; n2	uint32
	mov		esi, [ebp+24]		; *v2	float* (array)
	mov		edi, [ebp+28]		; *r	float* (array)
	
	mov		ebx, ecx

.row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

.ps_mul_row_loop:
	cmp		ecx, 4
	jl		.ss_mul_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		.ps_mul_row_loop

.ss_mul_loop:
	cmp		ecx, 1
	jl		.row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		.ss_mul_loop

.row_end:
	cmp		ebx, 0
	jg		.row_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret


; void SSE_tensor_mul_scalar(const uint32_t n1, const float* v, const float* s, float* r);
;	n - size of v
;	v - tensor
;	s - scalar
;	r - return value
_SSE_tensor_mul_scalar:
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

.ps_mul_loop:
	cmp		ecx, 4
	jl		.ss_mul_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_mul_loop

.ss_mul_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_mul_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret
	
; void SSE_vector_div(const uint32_t n, const float* v1, const float* v2, float* r);
;	n - size of v1 and v2
;	v1 - first vector
;	v2 - second vector
;	r - return value
_SSE_vector_div:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n		uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		esi, [ebp+20]		; *v2	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)
	
	xorps 	xmm0, xmm0

.ps_div_loop:
	cmp		ecx, 4
	jl		.ss_div_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_div_loop

.ss_div_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_div_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret

; void SSE_tensor_div(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
;	n1 - size of v1
;	v1 - first tensor
;	n2 - size of v1
;	v2 - second tensor
;	r - return value
_SSE_tensor_div:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		ecx, [ebp+12]		; n1	uint32
	mov		eax, [ebp+16]		; *v1	float* (array)
	mov		edx, [ebp+20]		; n2	uint32
	mov		esi, [ebp+24]		; *v2	float* (array)
	mov		edi, [ebp+28]		; *r	float* (array)
	
	mov		ebx, ecx

.row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

.ps_div_row_loop:
	cmp		ecx, 4
	jl		.ss_div_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		.ps_div_row_loop

.ss_div_loop:
	cmp		ecx, 1
	jl		.row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		.ss_div_loop

.row_end:
	cmp		ebx, 0
	jg		.row_loop

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret


; void SSE_tensor_div_scalar(const uint32_t n1, const float* v, const float* s, float* r);
;	n - size of v
;	v - tensor
;	s - scalar
;	r - return value
_SSE_tensor_div_scalar:
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

.ps_div_loop:
	cmp		ecx, 4
	jl		.ss_div_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		.ps_div_loop

.ss_div_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		.ss_div_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret
	
; void SSE_scalar_div_tensor(const float* s, const uint32_t n, const float* v, float* r);
;	s - scalar
;	v - tensor
;	n - size of v
;	r - return value
_SSE_scalar_div_tensor:
	push	ebp
	push	edi

	mov		ebp, esp
	
	mov		esi, [ebp+12]		; *s	float* (scalar)
	mov		ecx, [ebp+16]		; n		uint32
	mov		eax, [ebp+20]		; *v	float* (array)
	mov		edi, [ebp+24]		; *r	float* (array)

	movss	xmm1, dword [esi]
	shufps	xmm1, xmm1, 0x0
	
	xorps 	xmm0, xmm0

.ps_sub_loop:
	cmp		ecx, 4
	jl		.ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	vdivps	xmm2, xmm1, xmm0

	movups	[edi + 4*ecx], xmm2

	jmp		.ps_sub_loop

.ss_sub_loop:
	cmp		ecx, 1
	jl		.end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	vdivss	xmm2, xmm1, xmm0

	movss	[edi + 4*ecx], xmm2

	jmp		.ss_sub_loop

.end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret
global _SSE_vector_inner_product

global _SSE_vector_add
global _SSE_tensor_add
global _SSE_tensor_add_scalar

global _SSE_vector_sub
global _SSE_tensor_sub
global _SSE_tensor_sub_scalar

global _SSE_vector_mul
global _SSE_tensor_mul
global _SSE_tensor_mul_scalar

global _SSE_vector_div
global _SSE_tensor_div
global _SSE_tensor_div_scalar

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

VS_ups_sub_loop:
	cmp		ecx, 4
	jl		VS_ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		VS_ups_sub_loop

VS_ss_sub_loop:
	cmp		ecx, 1
	jl		VS_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	subss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		VS_ss_sub_loop

VS_end:

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

TS_row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

TS_ups_sub_row_loop:
	cmp		ecx, 4
	jl		TS_ss_sub_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		TS_ups_sub_row_loop

TS_ss_sub_loop:
	cmp		ecx, 1
	jl		TS_row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	subss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		TS_ss_sub_loop

TS_row_end:
	cmp		ebx, 0
	jg		TS_row_loop

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

TSS_ups_sub_loop:
	cmp		ecx, 4
	jl		TSS_ss_sub_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	subps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		TSS_ups_sub_loop

TSS_ss_sub_loop:
	cmp		ecx, 1
	jl		TSS_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	addss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		TSS_ss_sub_loop

TSS_end:

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

VM_ups_mul_loop:
	cmp		ecx, 4
	jl		VM_ss_mul_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		VM_ups_mul_loop

VM_ss_mul_loop:
	cmp		ecx, 1
	jl		VM_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		VM_ss_mul_loop

VM_end:

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

TM_row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

TM_ups_mul_row_loop:
	cmp		ecx, 4
	jl		TM_ss_mul_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		TM_ups_mul_row_loop

TM_ss_mul_loop:
	cmp		ecx, 1
	jl		TM_row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		TM_ss_mul_loop

TM_row_end:
	cmp		ebx, 0
	jg		TM_row_loop

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

TMS_ups_mul_loop:
	cmp		ecx, 4
	jl		TMS_ss_mul_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	mulps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		TMS_ups_mul_loop

TMS_ss_mul_loop:
	cmp		ecx, 1
	jl		TMS_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	mulss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		TMS_ss_mul_loop

TMS_end:

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

VD_ups_div_loop:
	cmp		ecx, 4
	jl		VD_ss_div_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]
	movups	xmm1, [esi + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		VD_ups_div_loop

VD_ss_div_loop:
	cmp		ecx, 1
	jl		VD_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	movss	xmm1, dword [esi + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		VD_ss_div_loop

VD_end:

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

TD_row_loop:
	mov		ecx, edx

	xorps 	xmm0, xmm0

TD_ups_div_row_loop:
	cmp		ecx, 4
	jl		TD_ss_div_loop

	sub		ecx, 4
	sub		ebx, 4
	
	movups	xmm0, [eax + 4*ebx]
	movups	xmm1, [esi + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ebx], xmm0

	jmp		TD_ups_div_row_loop

TD_ss_div_loop:
	cmp		ecx, 1
	jl		TD_row_end

	sub		ecx, 1
	sub		ebx, 1

	movss	xmm0, dword [eax + 4*ebx]
	movss	xmm1, dword [esi + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ebx], xmm0

	jmp		TD_ss_div_loop

TD_row_end:
	cmp		ebx, 0
	jg		TD_row_loop

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

TDS_ups_div_loop:
	cmp		ecx, 4
	jl		TDS_ss_div_loop

	sub		ecx, 4
	
	movups	xmm0, [eax + 4*ecx]

	divps	xmm0, xmm1

	movups	[edi + 4*ecx], xmm0

	jmp		TDS_ups_div_loop

TDS_ss_div_loop:
	cmp		ecx, 1
	jl		TDS_end

	sub		ecx, 1

	movss	xmm0, dword [eax + 4*ecx]
	
	divss	xmm0, xmm1

	movss	[edi + 4*ecx], xmm0

	jmp		TDS_ss_div_loop

TDS_end:

	mov     esp, ebp

	pop 	edi
	pop		ebp

	ret
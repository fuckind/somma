use crate::simd::{SimdArch, avx2::Avx2};
use std::arch::x86_64::*;

impl SimdArch<f32> for Avx2 {
    type Mem = __m256;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn setzero() -> Self::Mem {
        unsafe { _mm256_setzero_ps() }
    }

    #[inline(always)]
    unsafe fn set1(a: f32) -> Self::Mem {
        unsafe { _mm256_set1_ps(a) }
    }

    #[inline(always)]
    unsafe fn loadu(mem: *const f32) -> Self::Mem {
        unsafe { _mm256_loadu_ps(mem) }
    }

    #[inline(always)]
    unsafe fn storeu(mem: *mut f32, a: Self::Mem) {
        unsafe {
            _mm256_storeu_ps(mem, a);
        }
    }

    #[inline(always)]
    unsafe fn abs(a: Self::Mem) -> Self::Mem {
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
            _mm256_and_ps(a, mask)
        }
    }

    #[inline(always)]
    unsafe fn reduce_max(a: Self::Mem) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(a, 1);
            let lo = _mm256_castps256_ps128(a);
            let mut result = _mm_max_ps(lo, hi);
            result = _mm_max_ps(result, _mm_movehl_ps(result, result));
            result = _mm_max_ps(result, _mm_shuffle_ps::<0x01>(result, result));
            _mm_cvtss_f32(result)
        }
    }

    #[inline(always)]
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_and_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_max_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn div(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_div_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Mem, b: Self::Mem, c: Self::Mem) -> Self::Mem {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    unsafe fn add(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum(a: Self::Mem) -> f32 {
        unsafe {
            let mut sum = _mm256_add_ps(a, _mm256_permute2f128_ps::<0x01>(a, a));
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);
            _mm_cvtss_f32(_mm256_castps256_ps128(sum))
        }
    }
}

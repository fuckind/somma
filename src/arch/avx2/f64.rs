use crate::arch::{SimdArch, avx2::Avx2};
use std::arch::x86_64::*;

impl SimdArch<f64> for Avx2 {
    type Mem = __m256d;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn setzero() -> Self::Mem {
        unsafe { _mm256_setzero_pd() }
    }

    #[inline(always)]
    unsafe fn set1(a: f64) -> Self::Mem {
        unsafe { _mm256_set1_pd(a) }
    }

    #[inline(always)]
    unsafe fn loadu(mem: *const f64) -> Self::Mem {
        unsafe { _mm256_loadu_pd(mem) }
    }

    #[inline(always)]
    unsafe fn storeu(mem: *mut f64, a: Self::Mem) {
        unsafe {
            _mm256_storeu_pd(mem, a);
        }
    }

    #[inline(always)]
    unsafe fn abs(a: Self::Mem) -> Self::Mem {
        unsafe {
            let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff));
            _mm256_and_pd(a, mask)
        }
    }

    #[inline(always)]
    unsafe fn reduce_max(a: Self::Mem) -> f64 {
        unsafe {
            let mut result = _mm_max_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
            result = _mm_max_pd(result, _mm_shuffle_pd::<1>(result, result));
            _mm_cvtsd_f64(result)
        }
    }

    #[inline(always)]
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_and_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_max_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn div(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_div_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Mem, b: Self::Mem, c: Self::Mem) -> Self::Mem {
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    unsafe fn add(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        unsafe { _mm256_add_pd(a, b) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum(a: Self::Mem) -> f64 {
        unsafe {
            let sum = _mm256_add_pd(a, _mm256_permute2f128_pd::<0x01>(a, a));
            let sum = _mm256_hadd_pd(sum, sum);
            _mm_cvtsd_f64(_mm256_castpd256_pd128(sum))
        }
    }
}

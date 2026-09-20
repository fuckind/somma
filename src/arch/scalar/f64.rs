use crate::arch::{SimdArch, scalar::Scalar};

impl SimdArch<f64> for Scalar {
    type Mem = f64;
    const LANES: usize = 1;

    fn one() -> f64 {
        1.0
    }

    fn sqrt(a: f64) -> f64 {
        a.sqrt()
    }

    #[inline(always)]
    unsafe fn setzero() -> Self::Mem {
        0.0
    }

    #[inline(always)]
    unsafe fn set1(a: f64) -> Self::Mem {
        a
    }

    #[inline(always)]
    unsafe fn loadu(mem: *const f64) -> Self::Mem {
        unsafe { std::ptr::read_unaligned(mem) }
    }

    #[inline(always)]
    unsafe fn storeu(mem: *mut f64, a: Self::Mem) {
        unsafe { std::ptr::write_unaligned(mem, a) };
    }

    #[inline(always)]
    unsafe fn abs(a: Self::Mem) -> Self::Mem {
        a.abs()
    }

    #[inline(always)]
    unsafe fn reduce_max(a: Self::Mem) -> f64 {
        a
    }

    #[inline(always)]
    unsafe fn reduce_min(a: Self::Mem) -> f64 {
        a
    }

    #[inline(always)]
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        f64::from_bits(a.to_bits() & b.to_bits())
    }

    #[inline(always)]
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        a.max(b)
    }

    #[inline(always)]
    unsafe fn min(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        a.min(b)
    }

    #[inline(always)]
    unsafe fn div(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        a / b
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        a * b
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Mem, b: Self::Mem, c: Self::Mem) -> Self::Mem {
        a.mul_add(b, c)
    }

    #[inline(always)]
    unsafe fn add(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        a + b
    }

    #[inline(always)]
    unsafe fn horizontal_sum(a: Self::Mem) -> f64 {
        a
    }

    #[inline(always)]
    fn scalar_abs(a: f64) -> f64 {
        a.abs()
    }

    #[inline(always)]
    fn scalar_max(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    #[inline(always)]
    fn scalar_min(a: f64, b: f64) -> f64 {
        a.min(b)
    }
}

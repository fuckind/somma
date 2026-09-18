use crate::arch::{SimdArch, scalar::Scalar};

impl SimdArch<f64> for Scalar {
    type Mem = [f64; 4];
    const LANES: usize = 4;

    fn one() -> f64 {
        1.0
    }

    fn sqrt(a: f64) -> f64 {
        a.sqrt()
    }

    #[inline(always)]
    unsafe fn setzero() -> Self::Mem {
        [0.0; 4]
    }

    #[inline(always)]
    unsafe fn set1(a: f64) -> Self::Mem {
        [a; 4]
    }

    #[inline(always)]
    unsafe fn loadu(mem: *const f64) -> Self::Mem {
        unsafe { std::ptr::read_unaligned(mem as *const Self::Mem) }
    }

    #[inline(always)]
    unsafe fn storeu(mem: *mut f64, a: Self::Mem) {
        unsafe { std::ptr::write_unaligned(mem as *mut Self::Mem, a) };
    }

    #[inline(always)]
    unsafe fn abs(a: Self::Mem) -> Self::Mem {
        a.map(|value| value.abs())
    }

    #[inline(always)]
    unsafe fn reduce_max(a: Self::Mem) -> f64 {
        a.into_iter().reduce(|left, right| left.max(right)).unwrap()
    }

    #[inline(always)]
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| f64::from_bits(a[index].to_bits() & b[index].to_bits()))
    }

    #[inline(always)]
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| a[index].max(b[index]))
    }

    #[inline(always)]
    unsafe fn div(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| a[index] / b[index])
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| a[index] * b[index])
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Mem, b: Self::Mem, c: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| a[index].mul_add(b[index], c[index]))
    }

    #[inline(always)]
    unsafe fn add(a: Self::Mem, b: Self::Mem) -> Self::Mem {
        std::array::from_fn(|index| a[index] + b[index])
    }

    #[inline(always)]
    unsafe fn horizontal_sum(a: Self::Mem) -> f64 {
        a.into_iter().fold(0.0, |sum, value| sum + value)
    }

    #[inline(always)]
    fn scalar_abs(a: f64) -> f64 {
        a.abs()
    }

    #[inline(always)]
    fn scalar_max(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}

use crate::arch::{SimdArch, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn asum_avx2<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { asum::<T, Avx2>(x) }
}

#[inline(always)]
pub fn asum_scalar<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { asum::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_asum_avx2<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_asum::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub fn par_asum_scalar<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_asum::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_asum<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size)
        .map(|chunk| unsafe { asum::<T, ARCH>(chunk) })
        .reduce(|| T::default(), |a, b| a + b)
}

/// # Safety
#[inline(always)]
pub unsafe fn asum<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        let mut acc0 = ARCH::setzero();
        let mut acc1 = ARCH::setzero();
        let mut acc2 = ARCH::setzero();
        let mut acc3 = ARCH::setzero();

        let mut i = 0;
        while i + ARCH::LANES * 4 <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let x1 = ARCH::loadu(ptr_x.add(i + ARCH::LANES));
            let x2 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 2));
            let x3 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 3));

            let v0 = ARCH::abs(x0);
            let v1 = ARCH::abs(x1);
            let v2 = ARCH::abs(x2);
            let v3 = ARCH::abs(x3);

            acc0 = ARCH::add(acc0, v0);
            acc1 = ARCH::add(acc1, v1);
            acc2 = ARCH::add(acc2, v2);
            acc3 = ARCH::add(acc3, v3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let v0 = ARCH::abs(x0);
            acc0 = ARCH::add(acc0, v0);

            i += ARCH::LANES;
        }

        let acc0 = ARCH::add(acc0, acc1);
        let acc1 = ARCH::add(acc2, acc3);
        let acc = ARCH::add(acc0, acc1);
        let mut sum = ARCH::horizontal_sum(acc);

        while i < len {
            sum = sum + ARCH::scalar_abs(*ptr_x.add(i));
            i += 1;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::{asum_scalar, par_asum_scalar};

    #[test]
    fn sequential_and_parallel_asum_match() {
        let values = [-1.0_f32, 2.0, -3.0, 4.0, -5.0];

        assert_eq!(asum_scalar(&values), 15.0);
        assert_eq!(par_asum_scalar(&values, 2), 15.0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_asum_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let values = [-1.0_f32, 2.0, -3.0, 4.0, -5.0];

        assert_eq!(unsafe { super::asum_avx2(&values) }, 15.0);
        assert_eq!(unsafe { super::par_asum_avx2(&values, 2) }, 15.0);
    }
}

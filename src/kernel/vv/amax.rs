use crate::arch::{SimdArch, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn amax_avx2<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { amax::<T, Avx2>(x) }
}

#[inline(always)]
pub fn amax_scalar<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { amax::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_amax_avx2<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_amax::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub fn par_amax_scalar<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_amax::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_amax<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size)
        .map(|x| unsafe { amax::<T, ARCH>(x) })
        .reduce(|| T::default(), |a, b| ARCH::scalar_max(a, b))
}

/// # Safety
#[inline(always)]
pub unsafe fn amax<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
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

            acc0 = ARCH::max(acc0, v0);
            acc1 = ARCH::max(acc1, v1);
            acc2 = ARCH::max(acc2, v2);
            acc3 = ARCH::max(acc3, v3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let v0 = ARCH::abs(x0);
            acc0 = ARCH::max(acc0, v0);

            i += ARCH::LANES;
        }

        let acc0 = ARCH::max(acc0, acc1);
        let acc1 = ARCH::max(acc2, acc3);
        let acc = ARCH::max(acc0, acc1);

        let mut max = ARCH::reduce_max(acc);

        while i < len {
            max = ARCH::scalar_max(max, ARCH::scalar_abs(*ptr_x.add(i)));
            i += 1;
        }

        max
    }
}

#[cfg(test)]
mod tests {
    use super::{amax_scalar, par_amax_scalar};

    #[test]
    fn sequential_and_parallel_amax_match() {
        let values = [-8.0_f32, 3.0, -11.0, 5.0, -2.0];

        assert_eq!(amax_scalar(&values), 11.0);
        assert_eq!(par_amax_scalar(&values, 2), 11.0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_amax_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let values = [-8.0_f32, 3.0, -11.0, 5.0, -2.0];

        assert_eq!(unsafe { super::amax_avx2(&values) }, 11.0);
        assert_eq!(unsafe { super::par_amax_avx2(&values, 2) }, 11.0);
    }
}

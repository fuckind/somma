use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn sum_avx2<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { sum::<T, Avx2>(x) }
}

#[inline(always)]
pub fn sum_scalar<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { sum::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_sum_avx2<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_sum::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub fn par_sum_scalar<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_sum::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_sum<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size).map(|x| unsafe { sum::<T, ARCH>(x) }).reduce(|| T::default(), |a, b| a + b)
}

/// # Safety
#[inline(always)]
pub unsafe fn sum<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
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

            acc0 = ARCH::add(acc0, x0);
            acc1 = ARCH::add(acc1, x1);
            acc2 = ARCH::add(acc2, x2);
            acc3 = ARCH::add(acc3, x3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            acc0 = ARCH::add(acc0, x0);
            i += ARCH::LANES;
        }

        let acc0 = ARCH::add(acc0, acc1);
        let acc1 = ARCH::add(acc2, acc3);
        let acc = ARCH::add(acc0, acc1);
        let mut sum = ARCH::horizontal_sum(acc);

        while i < len {
            sum = sum + *ptr_x.add(i);
            i += 1;
        }

        sum
    }
}

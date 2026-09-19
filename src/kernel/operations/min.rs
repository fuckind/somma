use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn min_avx2<T: SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { min::<T, Avx2>(x) }
}

/// # Safety
#[inline(always)]
pub unsafe fn min_scalar<T: SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { min::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_min_avx2<T: SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_min::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_min_scalar<T: SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_min::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_min<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size)
        .map(|x| unsafe { min::<T, ARCH>(x) })
        .reduce(|| T::default(), |a, b| ARCH::scalar_min(a, b))
}

/// # Safety
#[inline(always)]
pub unsafe fn min<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        if len < ARCH::LANES {
            let mut min_val = *ptr_x;
            let mut i = 1;
            while i < len {
                min_val = ARCH::scalar_min(min_val, *ptr_x.add(i));
                i += 1;
            }
            return min_val;
        }

        let first = ARCH::loadu(ptr_x);
        let mut acc0 = first;
        let mut acc1 = first;
        let mut acc2 = first;
        let mut acc3 = first;

        let mut i = ARCH::LANES;
        while i + ARCH::LANES * 4 <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let x1 = ARCH::loadu(ptr_x.add(i + ARCH::LANES));
            let x2 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 2));
            let x3 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 3));

            acc0 = ARCH::min(acc0, x0);
            acc1 = ARCH::min(acc1, x1);
            acc2 = ARCH::min(acc2, x2);
            acc3 = ARCH::min(acc3, x3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            acc0 = ARCH::min(acc0, x0);
            i += ARCH::LANES;
        }

        let acc0 = ARCH::min(acc0, acc1);
        let acc1 = ARCH::min(acc2, acc3);
        let acc = ARCH::min(acc0, acc1);
        let mut min = ARCH::reduce_min(acc);

        while i < len {
            min = ARCH::scalar_min(min, *ptr_x.add(i));
            i += 1;
        }

        min
    }
}

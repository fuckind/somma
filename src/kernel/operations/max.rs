use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn max_avx2<T: SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { max::<T, Avx2>(x) }
}

/// # Safety
#[inline(always)]
pub unsafe fn max_scalar<T: SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { max::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_max_avx2<T: SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_max::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_max_scalar<T: SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_max::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_max<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size)
        .map(|x| unsafe { max::<T, ARCH>(x) })
        .reduce(|| T::default(), |a, b| ARCH::scalar_max(a, b))
}

/// # Safety
#[inline(always)]
pub unsafe fn max<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        if len < ARCH::LANES {
            let mut max_val = *ptr_x;
            let mut i = 1;
            while i < len {
                max_val = ARCH::scalar_max(max_val, *ptr_x.add(i));
                i += 1;
            }
            return max_val;
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

            acc0 = ARCH::max(acc0, x0);
            acc1 = ARCH::max(acc1, x1);
            acc2 = ARCH::max(acc2, x2);
            acc3 = ARCH::max(acc3, x3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            acc0 = ARCH::max(acc0, x0);
            i += ARCH::LANES;
        }

        let acc0 = ARCH::max(acc0, acc1);
        let acc1 = ARCH::max(acc2, acc3);
        let acc = ARCH::max(acc0, acc1);
        let mut max = ARCH::reduce_max(acc);

        while i < len {
            max = ARCH::scalar_max(max, *ptr_x.add(i));
            i += 1;
        }

        max
    }
}

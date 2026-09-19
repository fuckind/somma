use crate::{
    arch::{SimdArch, avx2::Avx2, scalar::Scalar},
    kernel::operations::amax::{amax, amax_avx2},
};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn nrm2_avx2<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { nrm2::<T, Avx2>(x) }
}

#[inline(always)]
pub fn nrm2_scalar<T: crate::arch::SimdScalar>(x: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { nrm2::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_nrm2_avx2<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = amax_avx2(chunk);
            if s == T::default() {
                (T::default(), T::default())
            } else {
                (s, raw_nrm2::<T, Avx2>(chunk, s))
            }
        })
        .reduce(
            || (T::default(), T::default()),
            |(s1, sum1), (s2, sum2)| {
                if s1 == T::default() {
                    return (s2, sum2);
                }
                if s2 == T::default() {
                    return (s1, sum1);
                }

                if s1 >= s2 {
                    let r = s2 / s1;
                    (s1, sum1 + sum2 * (r * r))
                } else {
                    let r = s1 / s2;
                    (s2, sum2 + sum1 * (r * r))
                }
            },
        );

    scale * Avx2::sqrt(sq_sum)
}

#[inline(always)]
pub fn par_nrm2_scalar<T: crate::arch::SimdScalar>(x: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_nrm2::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_nrm2<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> T {
    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = amax::<T, ARCH>(chunk);
            if s == T::default() {
                (T::default(), T::default())
            } else {
                (s, raw_nrm2::<T, ARCH>(chunk, s))
            }
        })
        .reduce(
            || (T::default(), T::default()),
            |(s1, sum1), (s2, sum2)| {
                if s1 == T::default() {
                    return (s2, sum2);
                }
                if s2 == T::default() {
                    return (s1, sum1);
                }

                if s1 >= s2 {
                    let r = s2 / s1;
                    (s1, sum1 + sum2 * (r * r))
                } else {
                    let r = s1 / s2;
                    (s2, sum2 + sum1 * (r * r))
                }
            },
        );

    scale * ARCH::sqrt(sq_sum)
}

/// # Safety
#[inline(always)]
pub unsafe fn nrm2<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> T {
    let scale = unsafe { amax::<T, ARCH>(x) };
    if scale == T::default() {
        return T::default();
    }

    unsafe { scale * ARCH::sqrt(raw_nrm2::<T, ARCH>(x, scale)) }
}

/// # Safety
#[inline(always)]
unsafe fn raw_nrm2<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], scale: T) -> T {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();
        let inv_scale = ARCH::one() / scale;
        let mem_scale = ARCH::set1(inv_scale);

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

            let v0 = ARCH::mul(x0, mem_scale);
            let v1 = ARCH::mul(x1, mem_scale);
            let v2 = ARCH::mul(x2, mem_scale);
            let v3 = ARCH::mul(x3, mem_scale);

            acc0 = ARCH::fmadd(v0, v0, acc0);
            acc1 = ARCH::fmadd(v1, v1, acc1);
            acc2 = ARCH::fmadd(v2, v2, acc2);
            acc3 = ARCH::fmadd(v3, v3, acc3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let v0 = ARCH::mul(x0, mem_scale);
            acc0 = ARCH::fmadd(v0, v0, acc0);

            i += ARCH::LANES;
        }

        let acc0 = ARCH::add(acc0, acc1);
        let acc1 = ARCH::add(acc2, acc3);
        let acc = ARCH::add(acc0, acc1);
        let mut sum = ARCH::horizontal_sum(acc);

        while i < len {
            let v0 = *ptr_x.add(i) * inv_scale;
            sum = sum + v0 * v0;
            i += 1;
        }

        sum
    }
}

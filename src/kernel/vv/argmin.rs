use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn argmin_avx2<T: SimdScalar>(x: &[T]) -> usize
where
    Avx2: SimdArch<T>,
{
    unsafe { argmin::<T, Avx2>(x) }
}

/// # Safety
#[inline(always)]
pub unsafe fn argmin_scalar<T: SimdScalar>(x: &[T]) -> usize
where
    Scalar: SimdArch<T>,
{
    unsafe { argmin::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_argmin_avx2<T: SimdScalar>(x: &[T], chunk_size: usize) -> usize
where
    Avx2: SimdArch<T>,
{
    unsafe { par_argmin::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_argmin_scalar<T: SimdScalar>(x: &[T], chunk_size: usize) -> usize
where
    Scalar: SimdArch<T>,
{
    unsafe { par_argmin::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_argmin<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> usize {
    x.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_i, chunk)| {
            let i = unsafe { argmin::<T, ARCH>(chunk) };
            (chunk[i], chunk_i * chunk_size + i)
        })
        .reduce(|| (T::default(), usize::MAX), |left, right| select_min::<T, ARCH>(left, right))
        .1
}

/// # Safety
#[inline(always)]
pub unsafe fn argmin<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> usize {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        if len < ARCH::LANES {
            let mut best_i = 0;
            let mut best = *ptr_x;
            let mut i = 1;
            while i < len {
                let value = *ptr_x.add(i);
                if ARCH::scalar_min(best, value) != best {
                    best = value;
                    best_i = i;
                }
                i += 1;
            }
            return best_i;
        }

        let first = ARCH::loadu(ptr_x);
        let mut acc0 = first;
        let mut acc1 = first;
        let mut acc2 = first;
        let mut acc3 = first;

        let mut i = ARCH::LANES;
        while i + ARCH::LANES * 4 <= len {
            acc0 = ARCH::min(acc0, ARCH::loadu(ptr_x.add(i)));
            acc1 = ARCH::min(acc1, ARCH::loadu(ptr_x.add(i + ARCH::LANES)));
            acc2 = ARCH::min(acc2, ARCH::loadu(ptr_x.add(i + ARCH::LANES * 2)));
            acc3 = ARCH::min(acc3, ARCH::loadu(ptr_x.add(i + ARCH::LANES * 3)));
            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            acc0 = ARCH::min(acc0, ARCH::loadu(ptr_x.add(i)));
            i += ARCH::LANES;
        }

        let acc = ARCH::min(ARCH::min(acc0, acc1), ARCH::min(acc2, acc3));
        let mut best = ARCH::reduce_min(acc);

        while i < len {
            best = ARCH::scalar_min(best, *ptr_x.add(i));
            i += 1;
        }

        i = 0;
        while i < len {
            if *ptr_x.add(i) == best {
                return i;
            }
            i += 1;
        }

        0
    }
}

#[inline(always)]
fn select_min<T: SimdScalar, ARCH: SimdArch<T>>(left: (T, usize), right: (T, usize)) -> (T, usize) {
    if left.1 == usize::MAX {
        return right;
    }
    if right.1 == usize::MAX || ARCH::scalar_min(left.0, right.0) == left.0 {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use super::{argmin_scalar, par_argmin_scalar};

    #[test]
    fn sequential_and_parallel_argmin_match() {
        let values = [1.0_f32, -7.0, 3.0, -7.0, 2.0];

        assert_eq!(unsafe { argmin_scalar(&values) }, 1);
        assert_eq!(unsafe { par_argmin_scalar(&values, 2) }, 1);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_argmin_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let values = [1.0_f32, -7.0, 3.0, -7.0, 2.0];

        assert_eq!(unsafe { super::argmin_avx2(&values) }, 1);
        assert_eq!(unsafe { super::par_argmin_avx2(&values, 2) }, 1);
    }
}

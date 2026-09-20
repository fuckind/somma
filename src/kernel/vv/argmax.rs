use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn argmax_avx2<T: SimdScalar>(x: &[T]) -> usize
where
    Avx2: SimdArch<T>,
{
    unsafe { argmax::<T, Avx2>(x) }
}

/// # Safety
#[inline(always)]
pub unsafe fn argmax_scalar<T: SimdScalar>(x: &[T]) -> usize
where
    Scalar: SimdArch<T>,
{
    unsafe { argmax::<T, Scalar>(x) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_argmax_avx2<T: SimdScalar>(x: &[T], chunk_size: usize) -> usize
where
    Avx2: SimdArch<T>,
{
    unsafe { par_argmax::<T, Avx2>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_argmax_scalar<T: SimdScalar>(x: &[T], chunk_size: usize) -> usize
where
    Scalar: SimdArch<T>,
{
    unsafe { par_argmax::<T, Scalar>(x, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_argmax<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T], chunk_size: usize) -> usize {
    x.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_i, chunk)| {
            let i = unsafe { argmax::<T, ARCH>(chunk) };
            (chunk[i], chunk_i * chunk_size + i)
        })
        .reduce(|| (T::default(), usize::MAX), |left, right| select_max::<T, ARCH>(left, right))
        .1
}

/// # Safety
#[inline(always)]
pub unsafe fn argmax<T: SimdScalar, ARCH: SimdArch<T>>(x: &[T]) -> usize {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        if len < ARCH::LANES {
            let mut best_i = 0;
            let mut best = *ptr_x;
            let mut i = 1;
            while i < len {
                let value = *ptr_x.add(i);
                if ARCH::scalar_max(best, value) != best {
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
            acc0 = ARCH::max(acc0, ARCH::loadu(ptr_x.add(i)));
            acc1 = ARCH::max(acc1, ARCH::loadu(ptr_x.add(i + ARCH::LANES)));
            acc2 = ARCH::max(acc2, ARCH::loadu(ptr_x.add(i + ARCH::LANES * 2)));
            acc3 = ARCH::max(acc3, ARCH::loadu(ptr_x.add(i + ARCH::LANES * 3)));
            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            acc0 = ARCH::max(acc0, ARCH::loadu(ptr_x.add(i)));
            i += ARCH::LANES;
        }

        let acc = ARCH::max(ARCH::max(acc0, acc1), ARCH::max(acc2, acc3));
        let mut best = ARCH::reduce_max(acc);

        while i < len {
            best = ARCH::scalar_max(best, *ptr_x.add(i));
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
fn select_max<T: SimdScalar, ARCH: SimdArch<T>>(left: (T, usize), right: (T, usize)) -> (T, usize) {
    if left.1 == usize::MAX {
        return right;
    }
    if right.1 == usize::MAX || ARCH::scalar_max(left.0, right.0) == left.0 {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use super::{argmax_scalar, par_argmax_scalar};

    #[test]
    fn sequential_and_parallel_argmax_match() {
        let values = [1.0_f32, 7.0, 3.0, 7.0, -2.0];

        assert_eq!(unsafe { argmax_scalar(&values) }, 1);
        assert_eq!(unsafe { par_argmax_scalar(&values, 2) }, 1);

        let values = [1.0_f64, 7.0, 3.0, 7.0, -2.0];
        assert_eq!(unsafe { argmax_scalar(&values) }, 1);
        assert_eq!(unsafe { par_argmax_scalar(&values, 2) }, 1);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_argmax_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let values = [1.0_f32, 7.0, 3.0, 7.0, -2.0];

        assert_eq!(unsafe { super::argmax_avx2(&values) }, 1);
        assert_eq!(unsafe { super::par_argmax_avx2(&values, 2) }, 1);

        let values = [1.0_f64, 7.0, 3.0, 7.0, -2.0];
        assert_eq!(unsafe { super::argmax_avx2(&values) }, 1);
        assert_eq!(unsafe { super::par_argmax_avx2(&values, 2) }, 1);
    }
}

use crate::arch::{SimdArch, avx2::Avx2, scalar::Scalar};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2<T: crate::arch::SimdScalar>(x: &[T], y: &[T]) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { dot::<T, Avx2>(x, y) }
}

/// # Safety
#[inline(always)]
pub unsafe fn dot_scalar<T: crate::arch::SimdScalar>(x: &[T], y: &[T]) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { dot::<T, Scalar>(x, y) }
}

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_dot_avx2<T: crate::arch::SimdScalar>(x: &[T], y: &[T], chunk_size: usize) -> T
where
    Avx2: SimdArch<T>,
{
    unsafe { par_dot::<T, Avx2>(x, y, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_dot_scalar<T: crate::arch::SimdScalar>(x: &[T], y: &[T], chunk_size: usize) -> T
where
    Scalar: SimdArch<T>,
{
    unsafe { par_dot::<T, Scalar>(x, y, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_dot<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], y: &[T], chunk_size: usize) -> T {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(x, y)| unsafe { dot::<T, ARCH>(x, y) })
        .reduce(|| T::default(), |a, b| a + b)
}

/// # Safety
#[inline(always)]
pub unsafe fn dot<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());

    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();
        let ptr_y = y.as_ptr();

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

            let y0 = ARCH::loadu(ptr_y.add(i));
            let y1 = ARCH::loadu(ptr_y.add(i + ARCH::LANES));
            let y2 = ARCH::loadu(ptr_y.add(i + ARCH::LANES * 2));
            let y3 = ARCH::loadu(ptr_y.add(i + ARCH::LANES * 3));

            acc0 = ARCH::fmadd(x0, y0, acc0);
            acc1 = ARCH::fmadd(x1, y1, acc1);
            acc2 = ARCH::fmadd(x2, y2, acc2);
            acc3 = ARCH::fmadd(x3, y3, acc3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let y0 = ARCH::loadu(ptr_y.add(i));
            acc0 = ARCH::fmadd(x0, y0, acc0);
            i += ARCH::LANES;
        }

        let acc0 = ARCH::add(acc0, acc1);
        let acc1 = ARCH::add(acc2, acc3);
        let acc = ARCH::add(acc0, acc1);
        let mut sum = ARCH::horizontal_sum(acc);

        while i < len {
            sum = sum + *ptr_x.add(i) * *ptr_y.add(i);
            i += 1;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::{dot_scalar, par_dot_scalar};

    #[test]
    fn sequential_and_parallel_dot_match() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f32, 4.0, 3.0, 2.0, 1.0];

        assert_eq!(unsafe { dot_scalar(&x, &y) }, 35.0);
        assert_eq!(unsafe { par_dot_scalar(&x, &y, 2) }, 35.0);

        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f64, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(unsafe { dot_scalar(&x, &y) }, 35.0);
        assert_eq!(unsafe { par_dot_scalar(&x, &y, 2) }, 35.0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f32, 4.0, 3.0, 2.0, 1.0];

        assert_eq!(unsafe { super::dot_avx2(&x, &y) }, 35.0);
        assert_eq!(unsafe { super::par_dot_avx2(&x, &y, 2) }, 35.0);

        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f64, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(unsafe { super::dot_avx2(&x, &y) }, 35.0);
        assert_eq!(unsafe { super::par_dot_avx2(&x, &y, 2) }, 35.0);
    }
}

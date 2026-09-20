use crate::arch::{SimdArch, avx2::Avx2, scalar::Scalar};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn axpy_avx2<T: crate::arch::SimdScalar>(y: &mut [T], x: &[T], a: T)
where
    Avx2: SimdArch<T>,
{
    unsafe { axpy::<T, Avx2>(y, x, a) }
}

/// # Safety
#[inline(always)]
pub unsafe fn axpy_scalar<T: crate::arch::SimdScalar>(y: &mut [T], x: &[T], a: T)
where
    Scalar: SimdArch<T>,
{
    unsafe { axpy::<T, Scalar>(y, x, a) }
}

/// # Safety
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_axpy_avx2<T: crate::arch::SimdScalar>(y: &mut [T], x: &[T], a: T, chunk_size: usize)
where
    Avx2: SimdArch<T>,
{
    unsafe { par_axpy::<T, Avx2>(y, x, a, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_axpy_scalar<T: crate::arch::SimdScalar>(y: &mut [T], x: &[T], a: T, chunk_size: usize)
where
    Scalar: SimdArch<T>,
{
    unsafe { par_axpy::<T, Scalar>(y, x, a, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_axpy<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(y: &mut [T], x: &[T], a: T, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(y, x)| unsafe { axpy::<T, ARCH>(y, x, a) })
}

/// # Safety
#[inline(always)]
pub unsafe fn axpy<T: crate::arch::SimdScalar, ARCH: SimdArch<T>>(y: &mut [T], x: &[T], a: T) {
    assert_eq!(y.len(), x.len());

    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();
        let ptr_y = y.as_mut_ptr();
        let mem_a = ARCH::set1(a);

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

            let v0 = ARCH::fmadd(x0, mem_a, y0);
            let v1 = ARCH::fmadd(x1, mem_a, y1);
            let v2 = ARCH::fmadd(x2, mem_a, y2);
            let v3 = ARCH::fmadd(x3, mem_a, y3);

            ARCH::storeu(ptr_y.add(i), v0);
            ARCH::storeu(ptr_y.add(i + ARCH::LANES), v1);
            ARCH::storeu(ptr_y.add(i + ARCH::LANES * 2), v2);
            ARCH::storeu(ptr_y.add(i + ARCH::LANES * 3), v3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let y0 = ARCH::loadu(ptr_y.add(i));
            let v0 = ARCH::fmadd(x0, mem_a, y0);
            ARCH::storeu(ptr_y.add(i), v0);

            i += ARCH::LANES;
        }

        while i < len {
            *ptr_y.add(i) = *ptr_y.add(i) + *ptr_x.add(i) * a;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{axpy_scalar, par_axpy_scalar};

    #[test]
    fn sequential_and_parallel_axpy_match() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut sequential = [10.0, 10.0, 10.0, 10.0, 10.0];
        let mut parallel = sequential;

        unsafe { axpy_scalar(&mut sequential, &x, 2.0) };
        unsafe { par_axpy_scalar(&mut parallel, &x, 2.0, 2) };

        assert_eq!(sequential, [12.0, 14.0, 16.0, 18.0, 20.0]);
        assert_eq!(parallel, sequential);

        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut sequential = [10.0; 5];
        let mut parallel = sequential;
        unsafe { axpy_scalar(&mut sequential, &x, 2.0) };
        unsafe { par_axpy_scalar(&mut parallel, &x, 2.0, 2) };
        assert_eq!(sequential, [12.0, 14.0, 16.0, 18.0, 20.0]);
        assert_eq!(parallel, sequential);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_axpy_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut result = [10.0, 10.0, 10.0, 10.0, 10.0];
        let mut parallel_result = result;

        unsafe { super::axpy_avx2(&mut result, &x, 2.0) };
        unsafe { super::par_axpy_avx2(&mut parallel_result, &x, 2.0, 2) };

        assert_eq!(result, [12.0, 14.0, 16.0, 18.0, 20.0]);
        assert_eq!(parallel_result, result);

        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut result = [10.0; 5];
        let mut parallel_result = result;
        unsafe { super::axpy_avx2(&mut result, &x, 2.0) };
        unsafe { super::par_axpy_avx2(&mut parallel_result, &x, 2.0, 2) };
        assert_eq!(result, [12.0, 14.0, 16.0, 18.0, 20.0]);
        assert_eq!(parallel_result, result);
    }
}

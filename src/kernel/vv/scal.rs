use crate::arch::{SimdArch, SimdScalar, avx2::Avx2, scalar::Scalar};
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn scal_avx2<T: crate::arch::SimdScalar>(x: &mut [T], a: T)
where
    Avx2: SimdArch<T>,
{
    unsafe { scal::<T, Avx2>(x, a) }
}

/// # Safety
#[inline(always)]
pub unsafe fn scal_scalar<T: crate::arch::SimdScalar>(x: &mut [T], a: T)
where
    Scalar: SimdArch<T>,
{
    unsafe { scal::<T, Scalar>(x, a) }
}

/// # Safety
#[target_feature(enable = "avx2")]
pub unsafe fn par_scal_avx2<T: crate::arch::SimdScalar>(x: &mut [T], a: T, chunk_size: usize)
where
    Avx2: SimdArch<T>,
{
    unsafe { par_scal::<T, Avx2>(x, a, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_scal_scalar<T: crate::arch::SimdScalar>(x: &mut [T], a: T, chunk_size: usize)
where
    Scalar: SimdArch<T>,
{
    unsafe { par_scal::<T, Scalar>(x, a, chunk_size) }
}

/// # Safety
#[inline(always)]
pub unsafe fn par_scal<T: SimdScalar, ARCH: SimdArch<T>>(x: &mut [T], a: T, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| unsafe { scal::<T, ARCH>(x, a) });
}

/// # Safety
#[inline(always)]
pub unsafe fn scal<T: SimdScalar, ARCH: SimdArch<T>>(x: &mut [T], a: T) {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_mut_ptr();
        let mem_a = ARCH::set1(a);

        let mut i = 0;
        while i + ARCH::LANES * 4 <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let x1 = ARCH::loadu(ptr_x.add(i + ARCH::LANES));
            let x2 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 2));
            let x3 = ARCH::loadu(ptr_x.add(i + ARCH::LANES * 3));

            let v0 = ARCH::mul(x0, mem_a);
            let v1 = ARCH::mul(x1, mem_a);
            let v2 = ARCH::mul(x2, mem_a);
            let v3 = ARCH::mul(x3, mem_a);

            ARCH::storeu(ptr_x.add(i), v0);
            ARCH::storeu(ptr_x.add(i + ARCH::LANES), v1);
            ARCH::storeu(ptr_x.add(i + ARCH::LANES * 2), v2);
            ARCH::storeu(ptr_x.add(i + ARCH::LANES * 3), v3);

            i += ARCH::LANES * 4;
        }

        while i + ARCH::LANES <= len {
            let x0 = ARCH::loadu(ptr_x.add(i));
            let v0 = ARCH::mul(x0, mem_a);
            ARCH::storeu(ptr_x.add(i), v0);

            i += ARCH::LANES;
        }

        while i < len {
            x[i] = x[i] * a;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{par_scal_scalar, scal_scalar};

    #[test]
    fn sequential_and_parallel_scal_match() {
        let mut sequential = [1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let mut parallel = sequential;

        unsafe { scal_scalar(&mut sequential, -2.0) };
        unsafe { par_scal_scalar(&mut parallel, -2.0, 2) };

        assert_eq!(sequential, [-2.0, 4.0, -6.0, 8.0, -10.0]);
        assert_eq!(parallel, sequential);

        let mut sequential = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
        let mut parallel = sequential;
        unsafe { scal_scalar(&mut sequential, -2.0) };
        unsafe { par_scal_scalar(&mut parallel, -2.0, 2) };
        assert_eq!(sequential, [-2.0, 4.0, -6.0, 8.0, -10.0]);
        assert_eq!(parallel, sequential);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_scal_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut result = [1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let mut parallel_result = result;

        unsafe { super::scal_avx2(&mut result, -2.0) };
        unsafe { super::par_scal_avx2(&mut parallel_result, -2.0, 2) };

        assert_eq!(result, [-2.0, 4.0, -6.0, 8.0, -10.0]);
        assert_eq!(parallel_result, result);

        let mut result = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
        let mut parallel_result = result;
        unsafe { super::scal_avx2(&mut result, -2.0) };
        unsafe { super::par_scal_avx2(&mut parallel_result, -2.0, 2) };
        assert_eq!(result, [-2.0, 4.0, -6.0, 8.0, -10.0]);
        assert_eq!(parallel_result, result);
    }
}

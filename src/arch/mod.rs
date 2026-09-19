pub mod avx2;
pub mod scalar;

use std::ops::{Add, Div, Mul, Sub};

pub trait SimdScalar: Copy + Default + PartialEq + PartialOrd + Send + Sync + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {}

impl SimdScalar for f32 {}
impl SimdScalar for f64 {}

pub trait SimdArch<T: SimdScalar> {
    type Mem: Copy + Send + Sync;
    const LANES: usize;

    fn one() -> T;

    fn sqrt(a: T) -> T;

    /// # Safety
    unsafe fn setzero() -> Self::Mem;

    /// # Safety
    unsafe fn set1(a: T) -> Self::Mem;

    /// # Safety
    unsafe fn loadu(mem: *const T) -> Self::Mem;

    /// # Safety
    unsafe fn storeu(mem: *mut T, a: Self::Mem);

    /// # Safety
    unsafe fn abs(a: Self::Mem) -> Self::Mem;

    /// # Safety
    fn scalar_abs(a: T) -> T;

    /// # Safety
    unsafe fn reduce_max(a: Self::Mem) -> T;

    /// # Safety
    unsafe fn reduce_min(a: Self::Mem) -> T;

    /// # Safety
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn min(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    fn scalar_max(a: T, b: T) -> T;

    fn scalar_min(a: T, b: T) -> T;

    /// # Safety
    unsafe fn div(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn mul(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn fmadd(a: Self::Mem, b: Self::Mem, c: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn add(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn horizontal_sum(a: Self::Mem) -> T;
}

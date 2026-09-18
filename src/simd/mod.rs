pub mod avx2;
pub mod scalar;

pub trait SimdArch<T> {
    type Mem: Copy + Send + Sync;
    const LANES: usize;

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
    unsafe fn reduce_max(a: Self::Mem) -> T;

    /// # Safety
    unsafe fn and(a: Self::Mem, b: Self::Mem) -> Self::Mem;

    /// # Safety
    unsafe fn max(a: Self::Mem, b: Self::Mem) -> Self::Mem;

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

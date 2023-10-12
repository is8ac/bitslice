#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(inline_const)]
//#![feature(adt_const_params)]

use crate::transpose::BitSwap;
#[cfg(target_feature = "avx512f")]
use core::arch::x86_64::__m512i;
#[cfg(target_feature = "x86_64")]
use core::arch::x86_64::{__m128i, __m256i};
use core::simd::Simd;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use std::fmt;
use std::mem;
use std::ops::{BitAnd, BitOr, BitXor, Not};

pub mod arithmetic;
mod transpose;

/// A wrapper to force 64 byte alignment.
/// This does mean that some small matrices will be suboptimally layed out when multiple if them are stored together.
#[repr(align(64))]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct BitMatrix<V, const L: usize>(pub [V; L]);

/// Things we can do to an array of bits.
pub trait BitArray:
    Sized
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Copy
{
    /// Number of bits on the array.
    const LEN: usize;
    /// Splats the sign across the bit array.
    fn splat(s: bool) -> Self;
    /// Get a bit. Poor performance, used only for testing and debugging.
    fn get_bit(self, index: usize) -> bool;
    /// Set a bit. Poor performance, used only for testing and debugging.
    fn set_bit(&mut self, index: usize, sign: bool);
    fn count_ones(self) -> u32;
    /// Transpose a square matrix in place. This will likely be better performance than using `Transpose::transpose()`.
    fn transpose_in_place(a: &mut BitMatrix<Self, { Self::LEN }>);
}

fn generic_set_bit<const L: usize>(bytes: &mut [u8; L], index: usize, sign: bool) {
    bytes[index / 8] &= !(1 << (7 - (index % 8)));
    bytes[index / 8] |= (sign as u8) << (7 - (index % 8));
}

fn generic_get_bit<const L: usize>(bytes: [u8; L], index: usize) -> bool {
    ((bytes[index / 8] >> (7 - (index % 8))) & 1) == 1
}

macro_rules! impl_bit_vec {
    ($name:ident, $blen:expr, $align:expr) => {
        /// An array of bits
        #[derive(Eq, PartialEq, Copy, Clone)]
        #[repr(align($align))]
        pub struct $name(pub [u8; $blen]);

        impl BitXor for $name {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                let x: Simd<u8, $blen> = Simd::from_array(self.0) ^ Simd::from_array(rhs.0);
                Self(x.into())
            }
        }

        impl BitAnd for $name {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                let x: Simd<u8, $blen> = Simd::from_array(self.0) & Simd::from_array(rhs.0);
                Self(x.into())
            }
        }

        impl BitOr for $name {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                let x: Simd<u8, $blen> = Simd::from_array(self.0) | Simd::from_array(rhs.0);
                Self(x.into())
            }
        }

        impl Not for $name {
            type Output = Self;

            fn not(self) -> Self::Output {
                let x: Simd<u8, $blen> = Simd::from_array(self.0);
                Self((!x).into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for i in 0..$blen {
                    write!(f, "{:08b}", self.0[i])?
                }
                Ok(())
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for i in 0..$blen {
                    write!(f, "{:08b}", self.0[i])?
                }
                Ok(())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self([0; $blen])
            }
        }
        impl Distribution<$name> for Standard {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $name {
                $name([(); $blen].map(|_| rng.gen()))
            }
        }
    };
}

impl_bit_vec!(B8, 1, 1);
impl_bit_vec!(B16, 2, 2);
impl_bit_vec!(B32, 4, 4);
impl_bit_vec!(B64, 8, 8);
impl_bit_vec!(B128, 16, 16);
impl_bit_vec!(B256, 32, 32);
impl_bit_vec!(B512, 64, 64);

impl BitArray for B8 {
    const LEN: usize = 8;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 1])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        self.0[0].count_ones()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 8>) {
        B8::transpose_step(&mut a.0, 0, B8::swap_bits_0);
        B8::transpose_step(&mut a.0, 1, B8::swap_bits_1);
        B8::transpose_step(&mut a.0, 2, B8::swap_bits_2);
    }
}

impl BitArray for B16 {
    const LEN: usize = 16;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 2])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        unsafe { mem::transmute::<_, u16>(self.0) }.count_ones()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 16>) {
        B16::transpose_step(&mut a.0, 0, B16::swap_bits_0);
        B16::transpose_step(&mut a.0, 1, B16::swap_bits_1);
        B16::transpose_step(&mut a.0, 2, B16::swap_bits_2);
        B16::transpose_step(&mut a.0, 3, B16::swap_bits_3);
    }
}

impl BitArray for B32 {
    const LEN: usize = 32;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 4])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        unsafe { mem::transmute::<_, u32>(self.0) }.count_ones()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 32>) {
        B32::transpose_step(&mut a.0, 0, B32::swap_bits_0);
        B32::transpose_step(&mut a.0, 1, B32::swap_bits_1);
        B32::transpose_step(&mut a.0, 2, B32::swap_bits_2);
        B32::transpose_step(&mut a.0, 3, B32::swap_bits_3);
        B32::transpose_step(&mut a.0, 4, B32::swap_bits_4);
    }
}

impl BitArray for B64 {
    const LEN: usize = 64;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 8])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        unsafe { mem::transmute::<_, u64>(self.0) }.count_ones()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 64>) {
        B64::transpose_step(&mut a.0, 0, B64::swap_bits_0);
        B64::transpose_step(&mut a.0, 1, B64::swap_bits_1);
        B64::transpose_step(&mut a.0, 2, B64::swap_bits_2);
        B64::transpose_step(&mut a.0, 3, B64::swap_bits_3);
        B64::transpose_step(&mut a.0, 4, B64::swap_bits_4);
        B64::transpose_step(&mut a.0, 5, B64::swap_bits_5);
    }
}

impl BitArray for B128 {
    const LEN: usize = 128;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 16])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        unsafe { mem::transmute::<_, u128>(self.0) }.count_ones()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 128>) {
        B128::transpose_step(&mut a.0, 0, B128::swap_bits_0);
        B128::transpose_step(&mut a.0, 1, B128::swap_bits_1);
        B128::transpose_step(&mut a.0, 2, B128::swap_bits_2);
        B128::transpose_step(&mut a.0, 3, B128::swap_bits_3);
        B128::transpose_step(&mut a.0, 4, B128::swap_bits_4);
        B128::transpose_step(&mut a.0, 5, B128::swap_bits_5);
        B128::transpose_step(&mut a.0, 6, B128::swap_bits_6);
    }
}

impl BitArray for B256 {
    const LEN: usize = 256;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 32])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        let words = unsafe { mem::transmute::<_, [u64; 4]>(self.0) };
        words.iter().map(|x| x.count_ones()).sum()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 256>) {
        B256::transpose_step(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_step(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_step(&mut a.0, 2, B256::swap_bits_2);
        B256::transpose_step(&mut a.0, 3, B256::swap_bits_3);
        B256::transpose_step(&mut a.0, 4, B256::swap_bits_4);
        B256::transpose_step(&mut a.0, 5, B256::swap_bits_5);
        B256::transpose_step(&mut a.0, 6, B256::swap_bits_6);
        B256::transpose_step(&mut a.0, 7, B256::swap_bits_7);
    }
}

impl BitArray for B512 {
    const LEN: usize = 512;
    fn splat(s: bool) -> Self {
        Self([0u8.saturating_sub(s as u8); 64])
    }
    fn get_bit(self, index: usize) -> bool {
        generic_get_bit(self.0, index)
    }
    fn set_bit(&mut self, index: usize, sign: bool) {
        generic_set_bit(&mut self.0, index, sign)
    }
    fn count_ones(self) -> u32 {
        let words = unsafe { mem::transmute::<_, [u64; 8]>(self.0) };
        words.iter().map(|x| x.count_ones()).sum()
    }
    fn transpose_in_place(a: &mut BitMatrix<Self, 512>) {
        B512::transpose_step(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_step(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_step(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_step(&mut a.0, 3, B512::swap_bits_3);
        B512::transpose_step(&mut a.0, 4, B512::swap_bits_4);
        B512::transpose_step(&mut a.0, 5, B512::swap_bits_5);
        B512::transpose_step(&mut a.0, 6, B512::swap_bits_6);
        B512::transpose_step(&mut a.0, 7, B512::swap_bits_7);
        B512::transpose_step(&mut a.0, 8, B512::swap_bits_8);
    }
}

/// Transpose the dimensions of a matrix.
/// Input must be wrapped in a `BitMatrix` to ensure correct alignment.
pub trait Transpose<T: BitArray>: BitArray {
    fn transpose(a: BitMatrix<Self, { T::LEN }>) -> BitMatrix<T, { Self::LEN }>;
}

impl Transpose<B128> for B8 {
    fn transpose(a: BitMatrix<B8, 128>) -> BitMatrix<B128, 8> {
        let mut target = BitMatrix([B128::splat(false); 8]);
        for i in 0..8 {
            let mut sub_target = [B8::splat(false); 16];
            for x in 0..16 {
                sub_target[x] = a.0[i + 8 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B128::transpose_partial::<_, 3>(&mut target.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 3>(&mut target.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 3>(&mut target.0, 2, B128::swap_bits_2);
        target
    }
}

impl Transpose<B256> for B8 {
    fn transpose(a: BitMatrix<B8, 256>) -> BitMatrix<B256, 8> {
        let mut target = BitMatrix([B256::splat(false); 8]);
        for i in 0..8 {
            let mut sub_target = [B8::splat(false); 32];
            for x in 0..32 {
                sub_target[x] = a.0[i + 8 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B256::transpose_partial::<_, 3>(&mut target.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 3>(&mut target.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 3>(&mut target.0, 2, B256::swap_bits_2);
        target
    }
}

impl Transpose<B512> for B8 {
    fn transpose(a: BitMatrix<B8, 512>) -> BitMatrix<B512, 8> {
        let mut target = BitMatrix([B512::splat(false); 8]);
        for i in 0..8 {
            let mut sub_target = [B8::splat(false); 64];
            for x in 0..64 {
                sub_target[x] = a.0[i + 8 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B512::transpose_partial::<_, 3>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 3>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 3>(&mut target.0, 2, B512::swap_bits_2);
        target
    }
}

// 16

impl Transpose<B128> for B16 {
    fn transpose(a: BitMatrix<B16, 128>) -> BitMatrix<B128, 16> {
        let mut target = BitMatrix([B128::splat(false); 16]);
        for i in 0..16 {
            let mut sub_target = [B16::splat(false); 8];
            for x in 0..8 {
                sub_target[x] = a.0[i + 16 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B128::transpose_partial::<_, 4>(&mut target.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 4>(&mut target.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 4>(&mut target.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 4>(&mut target.0, 3, B128::swap_bits_3);
        target
    }
}

impl Transpose<B256> for B16 {
    fn transpose(a: BitMatrix<B16, 256>) -> BitMatrix<B256, 16> {
        let mut target = BitMatrix([B256::splat(false); 16]);
        for i in 0..16 {
            let mut sub_target = [B16::splat(false); 16];
            for x in 0..16 {
                sub_target[x] = a.0[i + 16 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B256::transpose_partial::<_, 4>(&mut target.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 4>(&mut target.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 4>(&mut target.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 4>(&mut target.0, 3, B256::swap_bits_3);
        target
    }
}

impl Transpose<B512> for B16 {
    fn transpose(a: BitMatrix<B16, 512>) -> BitMatrix<B512, 16> {
        let mut target = BitMatrix([B512::splat(false); 16]);
        for i in 0..16 {
            let mut sub_target = [B16::splat(false); 32];
            for x in 0..32 {
                sub_target[x] = a.0[i + 16 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B512::transpose_partial::<_, 4>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 4>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 4>(&mut target.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 4>(&mut target.0, 3, B512::swap_bits_3);
        target
    }
}

// 32

impl Transpose<B128> for B32 {
    fn transpose(a: BitMatrix<B32, 128>) -> BitMatrix<B128, 32> {
        let mut target = BitMatrix([B128::splat(false); 32]);
        for i in 0..32 {
            let mut sub_target = [B32::splat(false); 4];
            for x in 0..4 {
                sub_target[x] = a.0[i + 32 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B128::transpose_partial::<_, 5>(&mut target.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 5>(&mut target.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 5>(&mut target.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 5>(&mut target.0, 3, B128::swap_bits_3);
        B128::transpose_partial::<_, 5>(&mut target.0, 4, B128::swap_bits_4);
        target
    }
}

impl Transpose<B256> for B32 {
    fn transpose(a: BitMatrix<B32, 256>) -> BitMatrix<B256, 32> {
        let mut target = BitMatrix([B256::splat(false); 32]);
        for i in 0..32 {
            let mut sub_target = [B32::splat(false); 8];
            for x in 0..8 {
                sub_target[x] = a.0[i + 32 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B256::transpose_partial::<_, 5>(&mut target.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 5>(&mut target.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 5>(&mut target.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 5>(&mut target.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 5>(&mut target.0, 4, B256::swap_bits_4);
        target
    }
}

impl Transpose<B512> for B32 {
    fn transpose(a: BitMatrix<B32, 512>) -> BitMatrix<B512, 32> {
        let mut target = BitMatrix([B512::splat(false); 32]);
        for i in 0..32 {
            let mut sub_target = [B32::splat(false); 16];
            for x in 0..16 {
                sub_target[x] = a.0[i + 32 * x];
            }
            target.0[i] = unsafe { mem::transmute(sub_target) };
        }
        B512::transpose_partial::<_, 5>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 5>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 5>(&mut target.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 5>(&mut target.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 5>(&mut target.0, 4, B512::swap_bits_4);
        target
    }
}

// 64

impl Transpose<B128> for B64 {
    fn transpose(a: BitMatrix<B64, 128>) -> BitMatrix<B128, 64> {
        let mut target = BitMatrix([B128::splat(false); 64]);
        for i in 0..64 {
            target.0[i] = unsafe { mem::transmute((a.0[i], a.0[i + 64])) };
        }
        B128::transpose_partial::<_, 6>(&mut target.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 6>(&mut target.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 6>(&mut target.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 6>(&mut target.0, 3, B128::swap_bits_3);
        B128::transpose_partial::<_, 6>(&mut target.0, 4, B128::swap_bits_4);
        B128::transpose_partial::<_, 6>(&mut target.0, 5, B128::swap_bits_5);
        target
    }
}

impl Transpose<B256> for B64 {
    fn transpose(a: BitMatrix<B64, 256>) -> BitMatrix<B256, 64> {
        let mut target = BitMatrix([B256::splat(false); 64]);
        for i in 0..64 {
            target.0[i] = unsafe {
                mem::transmute((
                    a.0[i + 64 * 0],
                    a.0[i + 64 * 1],
                    a.0[i + 64 * 2],
                    a.0[i + 64 * 3],
                ))
            };
        }
        B256::transpose_partial::<_, 6>(&mut target.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 6>(&mut target.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 6>(&mut target.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 6>(&mut target.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 6>(&mut target.0, 4, B256::swap_bits_4);
        B256::transpose_partial::<_, 6>(&mut target.0, 5, B256::swap_bits_5);
        target
    }
}

impl Transpose<B512> for B64 {
    fn transpose(a: BitMatrix<B64, 512>) -> BitMatrix<B512, 64> {
        let mut target = BitMatrix([B512::splat(false); 64]);
        for i in 0..64 {
            target.0[i] = unsafe {
                mem::transmute((
                    a.0[i + 64 * 0],
                    a.0[i + 64 * 1],
                    a.0[i + 64 * 2],
                    a.0[i + 64 * 3],
                    a.0[i + 64 * 4],
                    a.0[i + 64 * 5],
                    a.0[i + 64 * 6],
                    a.0[i + 64 * 7],
                ))
            };
        }
        B512::transpose_partial::<_, 6>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 6>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 6>(&mut target.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 6>(&mut target.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 6>(&mut target.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 6>(&mut target.0, 5, B512::swap_bits_5);
        target
    }
}

impl Transpose<B8> for B128 {
    fn transpose(mut a: BitMatrix<B128, 8>) -> BitMatrix<B8, 128> {
        B128::transpose_partial::<_, 3>(&mut a.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 3>(&mut a.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 3>(&mut a.0, 2, B128::swap_bits_2);
        let mut target = BitMatrix([B8::splat(false); 128]);
        for i in 0..8 {
            let tmp: [B8; 16] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..16 {
                target.0[i + 8 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B16> for B128 {
    fn transpose(mut a: BitMatrix<B128, 16>) -> BitMatrix<B16, 128> {
        B128::transpose_partial::<_, 4>(&mut a.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 4>(&mut a.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 4>(&mut a.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 4>(&mut a.0, 3, B128::swap_bits_3);
        let mut target = BitMatrix([B16::splat(false); 128]);
        for i in 0..16 {
            let tmp: [B16; 8] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..8 {
                target.0[i + 16 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B32> for B128 {
    fn transpose(mut a: BitMatrix<B128, 32>) -> BitMatrix<B32, 128> {
        B128::transpose_partial::<_, 5>(&mut a.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 5>(&mut a.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 5>(&mut a.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 5>(&mut a.0, 3, B128::swap_bits_3);
        B128::transpose_partial::<_, 5>(&mut a.0, 4, B128::swap_bits_4);
        let mut target = BitMatrix([B32::splat(false); 128]);
        for i in 0..32 {
            let tmp: [B32; 4] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..4 {
                target.0[i + 32 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B64> for B128 {
    fn transpose(mut a: BitMatrix<B128, 64>) -> BitMatrix<B64, 128> {
        B128::transpose_partial::<_, 6>(&mut a.0, 0, B128::swap_bits_0);
        B128::transpose_partial::<_, 6>(&mut a.0, 1, B128::swap_bits_1);
        B128::transpose_partial::<_, 6>(&mut a.0, 2, B128::swap_bits_2);
        B128::transpose_partial::<_, 6>(&mut a.0, 3, B128::swap_bits_3);
        B128::transpose_partial::<_, 6>(&mut a.0, 4, B128::swap_bits_4);
        B128::transpose_partial::<_, 6>(&mut a.0, 5, B128::swap_bits_5);

        let mut target = BitMatrix([B64::splat(false); 128]);
        for i in 0..64 {
            [target.0[i], target.0[i + 64]] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B128> for B128 {
    fn transpose(mut a: BitMatrix<B128, 128>) -> BitMatrix<B128, 128> {
        B128::transpose_in_place(&mut a);
        a
    }
}

impl Transpose<B256> for B128 {
    fn transpose(a: BitMatrix<B128, 256>) -> BitMatrix<B256, 128> {
        let mut target = BitMatrix([B256::splat(false); 128]);
        for i in 0..128 {
            target.0[i] = unsafe { mem::transmute((a.0[i], a.0[i + 128])) };
        }
        B256::transpose_partial::<_, 7>(&mut target.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 7>(&mut target.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 7>(&mut target.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 7>(&mut target.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 7>(&mut target.0, 4, B256::swap_bits_4);
        B256::transpose_partial::<_, 7>(&mut target.0, 5, B256::swap_bits_5);
        B256::transpose_partial::<_, 7>(&mut target.0, 6, B256::swap_bits_6);
        target
    }
}

impl Transpose<B512> for B128 {
    fn transpose(a: BitMatrix<B128, 512>) -> BitMatrix<B512, 128> {
        let mut target = BitMatrix([B512::splat(false); 128]);
        for i in 0..128 {
            target.0[i] = unsafe {
                mem::transmute((
                    a.0[i + 128 * 0],
                    a.0[i + 128 * 1],
                    a.0[i + 128 * 2],
                    a.0[i + 128 * 3],
                ))
            };
        }
        B512::transpose_partial::<_, 7>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 7>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 7>(&mut target.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 7>(&mut target.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 7>(&mut target.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 7>(&mut target.0, 5, B512::swap_bits_5);
        B512::transpose_partial::<_, 7>(&mut target.0, 6, B512::swap_bits_6);
        target
    }
}

impl Transpose<B8> for B256 {
    fn transpose(mut a: BitMatrix<B256, 8>) -> BitMatrix<B8, 256> {
        B256::transpose_partial::<_, 3>(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 3>(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 3>(&mut a.0, 2, B256::swap_bits_2);
        let mut target = BitMatrix([B8::splat(false); 256]);
        for i in 0..8 {
            let tmp: [B8; 32] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..32 {
                target.0[i + 8 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B16> for B256 {
    fn transpose(mut a: BitMatrix<B256, 16>) -> BitMatrix<B16, 256> {
        B256::transpose_partial::<_, 4>(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 4>(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 4>(&mut a.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 4>(&mut a.0, 3, B256::swap_bits_3);
        let mut target = BitMatrix([B16::splat(false); 256]);
        for i in 0..16 {
            let tmp: [B16; 16] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..16 {
                target.0[i + 16 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B32> for B256 {
    fn transpose(mut a: BitMatrix<B256, 32>) -> BitMatrix<B32, 256> {
        B256::transpose_partial::<_, 5>(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 5>(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 5>(&mut a.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 5>(&mut a.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 5>(&mut a.0, 4, B256::swap_bits_4);
        let mut target = BitMatrix([B32::splat(false); 256]);
        for i in 0..32 {
            let tmp: [B32; 8] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..8 {
                target.0[i + 32 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B64> for B256 {
    fn transpose(mut a: BitMatrix<B256, 64>) -> BitMatrix<B64, 256> {
        B256::transpose_partial::<_, 6>(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 6>(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 6>(&mut a.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 6>(&mut a.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 6>(&mut a.0, 4, B256::swap_bits_4);
        B256::transpose_partial::<_, 6>(&mut a.0, 5, B256::swap_bits_5);

        let mut target = BitMatrix([B64::splat(false); 256]);
        for i in 0..64 {
            [
                target.0[i + 64 * 0],
                target.0[i + 64 * 1],
                target.0[i + 64 * 2],
                target.0[i + 64 * 3],
            ] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B128> for B256 {
    fn transpose(mut a: BitMatrix<B256, 128>) -> BitMatrix<B128, 256> {
        B256::transpose_partial::<_, 7>(&mut a.0, 0, B256::swap_bits_0);
        B256::transpose_partial::<_, 7>(&mut a.0, 1, B256::swap_bits_1);
        B256::transpose_partial::<_, 7>(&mut a.0, 2, B256::swap_bits_2);
        B256::transpose_partial::<_, 7>(&mut a.0, 3, B256::swap_bits_3);
        B256::transpose_partial::<_, 7>(&mut a.0, 4, B256::swap_bits_4);
        B256::transpose_partial::<_, 7>(&mut a.0, 5, B256::swap_bits_5);
        B256::transpose_partial::<_, 7>(&mut a.0, 6, B256::swap_bits_6);
        let mut target = BitMatrix([B128::splat(false); 256]);
        for i in 0..128 {
            [target.0[i], target.0[i + 128]] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B256> for B256 {
    fn transpose(mut a: BitMatrix<B256, 256>) -> BitMatrix<B256, 256> {
        B256::transpose_in_place(&mut a);
        a
    }
}

impl Transpose<B512> for B256 {
    fn transpose(a: BitMatrix<B256, 512>) -> BitMatrix<B512, 256> {
        let mut target = BitMatrix([B512::splat(false); 256]);
        for i in 0..256 {
            target.0[i] = unsafe { mem::transmute((a.0[i], a.0[i + 256])) };
        }
        B512::transpose_partial::<_, 8>(&mut target.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 8>(&mut target.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 8>(&mut target.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 8>(&mut target.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 8>(&mut target.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 8>(&mut target.0, 5, B512::swap_bits_5);
        B512::transpose_partial::<_, 8>(&mut target.0, 6, B512::swap_bits_6);
        B512::transpose_partial::<_, 8>(&mut target.0, 7, B512::swap_bits_7);
        target
    }
}

// B512

impl Transpose<B8> for B512 {
    fn transpose(mut a: BitMatrix<B512, 8>) -> BitMatrix<B8, 512> {
        B512::transpose_partial::<_, 3>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 3>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 3>(&mut a.0, 2, B512::swap_bits_2);

        let mut target = BitMatrix([B8::splat(false); 512]);
        for i in 0..8 {
            let tmp: [B8; 64] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..64 {
                target.0[i + 8 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B16> for B512 {
    fn transpose(mut a: BitMatrix<B512, 16>) -> BitMatrix<B16, 512> {
        B512::transpose_partial::<_, 4>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 4>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 4>(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 4>(&mut a.0, 3, B512::swap_bits_3);

        let mut target = BitMatrix([B16::splat(false); 512]);
        for i in 0..16 {
            let tmp: [B16; 32] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..32 {
                target.0[i + 16 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B32> for B512 {
    fn transpose(mut a: BitMatrix<B512, 32>) -> BitMatrix<B32, 512> {
        B512::transpose_partial::<_, 5>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 5>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 5>(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 5>(&mut a.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 5>(&mut a.0, 4, B512::swap_bits_4);
        let mut target = BitMatrix([B32::splat(false); 512]);
        for i in 0..32 {
            let tmp: [B32; 16] = unsafe { mem::transmute(a.0[i]) };
            for x in 0..16 {
                target.0[i + 32 * x] = tmp[x];
            }
        }
        target
    }
}

impl Transpose<B64> for B512 {
    fn transpose(mut a: BitMatrix<B512, 64>) -> BitMatrix<B64, 512> {
        B512::transpose_partial::<_, 6>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 6>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 6>(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 6>(&mut a.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 6>(&mut a.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 6>(&mut a.0, 5, B512::swap_bits_5);
        let mut target = BitMatrix([B64::splat(false); 512]);
        for i in 0..64 {
            [
                target.0[i + 64 * 0],
                target.0[i + 64 * 1],
                target.0[i + 64 * 2],
                target.0[i + 64 * 3],
                target.0[i + 64 * 4],
                target.0[i + 64 * 5],
                target.0[i + 64 * 6],
                target.0[i + 64 * 7],
            ] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B128> for B512 {
    fn transpose(mut a: BitMatrix<B512, 128>) -> BitMatrix<B128, 512> {
        B512::transpose_partial::<_, 7>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 7>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 7>(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 7>(&mut a.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 7>(&mut a.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 7>(&mut a.0, 5, B512::swap_bits_5);
        B512::transpose_partial::<_, 7>(&mut a.0, 6, B512::swap_bits_6);
        let mut target = BitMatrix([B128::splat(false); 512]);
        for i in 0..128 {
            [
                target.0[i + 128 * 0],
                target.0[i + 128 * 1],
                target.0[i + 128 * 2],
                target.0[i + 128 * 3],
            ] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B256> for B512 {
    fn transpose(mut a: BitMatrix<B512, 256>) -> BitMatrix<B256, 512> {
        B512::transpose_partial::<_, 8>(&mut a.0, 0, B512::swap_bits_0);
        B512::transpose_partial::<_, 8>(&mut a.0, 1, B512::swap_bits_1);
        B512::transpose_partial::<_, 8>(&mut a.0, 2, B512::swap_bits_2);
        B512::transpose_partial::<_, 8>(&mut a.0, 3, B512::swap_bits_3);
        B512::transpose_partial::<_, 8>(&mut a.0, 4, B512::swap_bits_4);
        B512::transpose_partial::<_, 8>(&mut a.0, 5, B512::swap_bits_5);
        B512::transpose_partial::<_, 8>(&mut a.0, 6, B512::swap_bits_6);
        B512::transpose_partial::<_, 8>(&mut a.0, 7, B512::swap_bits_7);
        let mut target = BitMatrix([B256::splat(false); 512]);
        for i in 0..256 {
            [target.0[i], target.0[i + 256]] = unsafe { mem::transmute(a.0[i]) };
        }
        target
    }
}

impl Transpose<B512> for B512 {
    fn transpose(mut a: BitMatrix<B512, 512>) -> BitMatrix<B512, 512> {
        B512::transpose_in_place(&mut a);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    macro_rules! test_transpose {
        ($test_name:ident, $a:ty, $b:ty) => {
            #[test]
            fn $test_name() {
                let mut rng = Pcg64::seed_from_u64(42);
                let input: BitMatrix<$a, { <$b>::LEN }> =
                    BitMatrix([(); <$b>::LEN].map(|_| rng.gen()));
                let output_1 = universal_transpose::<$a, $b>(input);
                let output_2 = <$a as Transpose<$b>>::transpose(input);
                assert_eq!(output_1, output_2);
            }
        };
    }

    fn universal_transpose<A: BitArray, B: BitArray>(
        a: BitMatrix<A, { B::LEN }>,
    ) -> BitMatrix<B, { A::LEN }> {
        let mut target = BitMatrix([B::splat(false); A::LEN]);
        for x in 0..A::LEN {
            for y in 0..B::LEN {
                target.0[x].set_bit(y, a.0[y].get_bit(x));
            }
        }
        target
    }
    test_transpose!(test_512x8, B512, B8);
    test_transpose!(test_512x16, B512, B16);
    test_transpose!(test_512x32, B512, B32);
    test_transpose!(test_512x64, B512, B64);
    test_transpose!(test_512x128, B512, B128);
    test_transpose!(test_512x256, B512, B256);
    test_transpose!(test_512x512, B512, B512);

    test_transpose!(test_256x8, B256, B8);
    test_transpose!(test_256x16, B256, B16);
    test_transpose!(test_256x32, B256, B32);
    test_transpose!(test_256x64, B256, B64);
    test_transpose!(test_256x128, B256, B128);
    test_transpose!(test_256x256, B256, B256);
    test_transpose!(test_256x512, B256, B512);

    test_transpose!(test_128x8, B128, B8);
    test_transpose!(test_128x16, B128, B16);
    test_transpose!(test_128x32, B128, B32);
    test_transpose!(test_128x64, B128, B64);
    test_transpose!(test_128x128, B128, B128);
    test_transpose!(test_128x256, B128, B256);
    test_transpose!(test_128x512, B128, B512);

    test_transpose!(test_64x128, B64, B128);
    test_transpose!(test_64x256, B64, B256);
    test_transpose!(test_64x512, B64, B512);

    test_transpose!(test_32x128, B32, B128);
    test_transpose!(test_32x256, B32, B256);
    test_transpose!(test_32x512, B32, B512);

    test_transpose!(test_16x128, B16, B128);
    test_transpose!(test_16x256, B16, B256);
    test_transpose!(test_16x512, B16, B512);

    test_transpose!(test_8x128, B8, B128);
    test_transpose!(test_8x256, B8, B256);
    test_transpose!(test_8x512, B8, B512);
}

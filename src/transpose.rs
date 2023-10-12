use crate::{B128, B16, B256, B32, B512, B64, B8};
#[cfg(target_feature = "neon")]
use std::arch::aarch64;
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{uint16x8_t, uint32x4_t, uint64x2_t, uint8x16_t};
#[cfg(target_feature = "avx512f")]
use std::arch::x86_64;
#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::{__m128i, __m256i, __m512i, __mmask64};
#[cfg(target_feature = "avx512f")]
use std::simd::Simd;
#[cfg(target_feature = "neon")]
use std::mem;
#[cfg(target_feature = "neon")]
use std::simd::Simd;

fn swap_bits_0<const L: usize>(a: [u8; L], b: [u8; L]) -> ([u8; L], [u8; L]) {
    let mut target_a = [0u8; L];
    let mut target_b = [0u8; L];
    for i in 0..L {
        let t = (a[i] ^ b[i] >> 1) & 0b_01010101_u8;
        target_a[i] = a[i] ^ t;
        target_b[i] = b[i] ^ t << 1;
    }
    (target_a, target_b)
}

fn swap_bits_1<const L: usize>(a: [u8; L], b: [u8; L]) -> ([u8; L], [u8; L]) {
    let mut target_a = [0u8; L];
    let mut target_b = [0u8; L];
    for i in 0..L {
        let t = (a[i] ^ b[i] >> 2) & 0b_00110011_u8;
        target_a[i] = a[i] ^ t;
        target_b[i] = b[i] ^ t << 2;
    }
    (target_a, target_b)
}

fn swap_bits_2<const L: usize>(a: [u8; L], b: [u8; L]) -> ([u8; L], [u8; L]) {
    let mut target_a = [0u8; L];
    let mut target_b = [0u8; L];
    for i in 0..L {
        let t = (a[i] ^ b[i] >> 4) & 0b_00001111_u8;
        target_a[i] = a[i] ^ t;
        target_b[i] = b[i] ^ t << 4;
    }
    (target_a, target_b)
}

fn swap_bytes_n<const L: usize, const K: u32>(a: [u8; L], b: [u8; L]) -> ([u8; L], [u8; L]) {
    let mut target_a = [0u8; L];
    let mut target_b = [0u8; L];
    for i in 0..L {
        if ((i >> K) & 1) == 0 {
            target_a[i] = a[i];
            target_b[i] = a[i + 2usize.pow(K)];
        } else {
            target_a[i] = b[i - 2usize.pow(K)];
            target_b[i] = b[i];
        }
    }
    (target_a, target_b)
}

#[cfg(target_feature = "avx512f")]
impl From<__m512i> for B512 {
    fn from(value: __m512i) -> Self {
        let tmp: Simd<u8, 64> = value.into();
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B512> for __m512i {
    fn from(value: B512) -> Self {
        let tmp: Simd<u8, 64> = Simd::from_array(value.0);
        tmp.into()
    }
}

#[cfg(target_feature = "avx512f")]
impl From<Simd<u8, 64>> for B512 {
    fn from(value: Simd<u8, 64>) -> Self {
        Self(value.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B512> for Simd<u8, 64> {
    fn from(value: B512) -> Self {
        Simd::from_array(value.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl From<__m256i> for B256 {
    fn from(value: __m256i) -> Self {
        let tmp: Simd<u8, 32> = value.into();
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B256> for __m256i {
    fn from(value: B256) -> Self {
        let tmp: Simd<u8, 32> = Simd::from_array(value.0);
        tmp.into()
    }
}

#[cfg(target_feature = "avx512f")]
impl From<Simd<u8, 32>> for B256 {
    fn from(value: Simd<u8, 32>) -> Self {
        Self(value.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B256> for Simd<u8, 32> {
    fn from(value: B256) -> Self {
        Simd::from_array(value.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl From<__m128i> for B128 {
    fn from(value: __m128i) -> Self {
        let tmp: Simd<u8, 16> = value.into();
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B128> for __m128i {
    fn from(value: B128) -> Self {
        let tmp: Simd<u8, 16> = Simd::from_array(value.0);
        tmp.into()
    }
}

#[cfg(target_feature = "avx512f")]
impl From<Simd<u8, 16>> for B128 {
    fn from(value: Simd<u8, 16>) -> Self {
        Self(value.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl From<B128> for Simd<u8, 16> {
    fn from(value: B128) -> Self {
        Simd::from_array(value.0)
    }
}

#[cfg(target_feature = "neon")]
impl From<uint8x16_t> for B128 {
    fn from(value: uint8x16_t) -> Self {
        let tmp: Simd<u8, 16> = value.into();
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "neon")]
impl From<B128> for uint8x16_t {
    fn from(value: B128) -> Self {
        let tmp: Simd<u8, 16> = Simd::from_array(value.0);
        tmp.into()
    }
}

#[cfg(target_feature = "neon")]
impl From<uint16x8_t> for B128 {
    fn from(value: uint16x8_t) -> Self {
        let tmp: Simd<u8, 16> = unsafe { mem::transmute(value) };
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "neon")]
impl From<B128> for uint16x8_t {
    fn from(value: B128) -> Self {
        unsafe { mem::transmute(value) }
    }
}

#[cfg(target_feature = "neon")]
impl From<uint32x4_t> for B128 {
    fn from(value: uint32x4_t) -> Self {
        let tmp: Simd<u8, 16> = unsafe { mem::transmute(value) };
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "neon")]
impl From<B128> for uint32x4_t {
    fn from(value: B128) -> Self {
        unsafe { mem::transmute(value) }
    }
}

#[cfg(target_feature = "neon")]
impl From<uint64x2_t> for B128 {
    fn from(value: uint64x2_t) -> Self {
        let tmp: Simd<u8, 16> = unsafe { mem::transmute(value) };
        Self(tmp.to_array())
    }
}

#[cfg(target_feature = "neon")]
impl From<B128> for uint64x2_t {
    fn from(value: B128) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl BitSwap for B512 {
    const E: u32 = 9;
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_0(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a: __m512i = a.into();
        let b: __m512i = b.into();
        let mask: __m512i = Simd::<u8, 64>::splat(0b_01010101_u8).into();
        unsafe {
            let t = x86_64::_mm512_and_si512(
                x86_64::_mm512_xor_si512(a, x86_64::_mm512_srli_epi64(b.into(), 1)),
                mask,
            );
            (
                x86_64::_mm512_xor_si512(a, t).into(),
                x86_64::_mm512_xor_si512(b, x86_64::_mm512_slli_epi64(t.into(), 1)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 4] = mem::transmute(a);
            let b: [uint8x16_t; 4] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_01010101_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 4];
            let mut target_b = [uint8x16_t::from(Simd::default()); 4];
            for i in 0..4 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 1)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 1)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_1(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a: __m512i = a.into();
        let b: __m512i = b.into();
        let mask: __m512i = Simd::<u8, 64>::splat(0b_00110011_u8).into();
        unsafe {
            let t = x86_64::_mm512_and_si512(
                x86_64::_mm512_xor_si512(a, x86_64::_mm512_srli_epi64(b.into(), 2)),
                mask,
            );
            (
                x86_64::_mm512_xor_si512(a, t).into(),
                x86_64::_mm512_xor_si512(b, x86_64::_mm512_slli_epi64(t.into(), 2)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 4] = mem::transmute(a);
            let b: [uint8x16_t; 4] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00110011_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 4];
            let mut target_b = [uint8x16_t::from(Simd::default()); 4];
            for i in 0..4 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 2)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 2)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_2(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a: __m512i = a.into();
        let b: __m512i = b.into();
        let mask: __m512i = Simd::<u8, 64>::splat(0b_00001111_u8).into();
        unsafe {
            let t = x86_64::_mm512_and_si512(
                x86_64::_mm512_xor_si512(a, x86_64::_mm512_srli_epi64(b.into(), 4)),
                mask,
            );
            (
                x86_64::_mm512_xor_si512(a, t).into(),
                x86_64::_mm512_xor_si512(b, x86_64::_mm512_slli_epi64(t.into(), 4)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 4] = mem::transmute(a);
            let b: [uint8x16_t; 4] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00001111_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 4];
            let mut target_b = [uint8x16_t::from(Simd::default()); 4];
            for i in 0..4 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 4)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 4)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 0>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u8; 64];
            let mut idx_lo = [0u8; 64];
            let mut i = 0;
            while i < 32 {
                idx_hi[i * 2 + 0] = (i as u8) * 2;
                idx_hi[i * 2 + 1] = 64 + (i as u8) * 2;
                idx_lo[i * 2 + 0] = (i as u8) * 2 + 1;
                idx_lo[i * 2 + 1] = 64 + (i as u8) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 64>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm512_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 64>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 4] = mem::transmute(a);
            let b: [uint8x16_t; 4] = mem::transmute(b);
            let mut target_a = [uint8x16_t::from(Simd::default()); 4];
            let mut target_b = [uint8x16_t::from(Simd::default()); 4];
            for i in 0..4 {
                target_a[i] = aarch64::vtrn1q_u8(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u8(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 1>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u16; 32];
            let mut idx_lo = [0u16; 32];
            let mut i = 0;
            while i < 16 {
                idx_hi[i * 2 + 0] = (i as u16) * 2;
                idx_hi[i * 2 + 1] = 32 + (i as u16) * 2;
                idx_lo[i * 2 + 0] = (i as u16) * 2 + 1;
                idx_lo[i * 2 + 1] = 32 + (i as u16) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 32>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm512_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 32>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint16x8_t; 4] = mem::transmute(a);
            let b: [uint16x8_t; 4] = mem::transmute(b);
            let mut target_a = [uint16x8_t::from(Simd::default()); 4];
            let mut target_b = [uint16x8_t::from(Simd::default()); 4];
            for i in 0..4 {
                target_a[i] = aarch64::vtrn1q_u16(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u16(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 2>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u32; 16];
            let mut idx_lo = [0u32; 16];
            let mut i = 0;
            while i < 8 {
                idx_hi[i * 2 + 0] = (i as u32) * 2;
                idx_hi[i * 2 + 1] = 16 + (i as u32) * 2;
                idx_lo[i * 2 + 0] = (i as u32) * 2 + 1;
                idx_lo[i * 2 + 1] = 16 + (i as u32) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 16>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm512_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 16>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint32x4_t; 4] = mem::transmute(a);
            let b: [uint32x4_t; 4] = mem::transmute(b);
            let mut target_a = [uint32x4_t::from(Simd::default()); 4];
            let mut target_b = [uint32x4_t::from(Simd::default()); 4];
            for i in 0..4 {
                target_a[i] = aarch64::vtrn1q_u32(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u32(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 3>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u64; 8];
            let mut idx_lo = [0u64; 8];
            let mut i = 0;
            while i < 4 {
                idx_hi[i * 2 + 0] = (i as u64) * 2;
                idx_hi[i * 2 + 1] = 8 + (i as u64) * 2;
                idx_lo[i * 2 + 0] = (i as u64) * 2 + 1;
                idx_lo[i * 2 + 1] = 8 + (i as u64) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 8>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm512_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 8>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint64x2_t; 4] = mem::transmute(a);
            let b: [uint64x2_t; 4] = mem::transmute(b);
            let mut target_a = [uint64x2_t::from(Simd::default()); 4];
            let mut target_b = [uint64x2_t::from(Simd::default()); 4];
            for i in 0..4 {
                target_a[i] = aarch64::vtrn1q_u64(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u64(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 4>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        let idx_hi = Simd::<u64, 8>::from_array([0, 1, 8, 9, 4, 5, 12, 13]);
        let idx_lo = Simd::<u64, 8>::from_array([2, 3, 10, 11, 6, 7, 14, 15]);
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi64(a.into(), idx_hi.into(), b.into()).into(),
                x86_64::_mm512_permutex2var_epi64(a.into(), idx_lo.into(), b.into()).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint32x4_t; 4] = mem::transmute(a);
            let b: [uint32x4_t; 4] = mem::transmute(b);
            let target_a = [a[0], b[0], a[2], b[2]];
            let target_b = [a[1], b[1], a[3], b[3]];
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_8(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<64, 5>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_8(a: Self, b: Self) -> (Self, Self) {
        let idx_hi = Simd::<u64, 8>::from_array([0, 1, 2, 3, 8, 9, 10, 11]);
        let idx_lo = Simd::<u64, 8>::from_array([4, 5, 6, 7, 12, 13, 14, 15]);
        unsafe {
            (
                x86_64::_mm512_permutex2var_epi64(a.into(), idx_hi.into(), b.into()).into(),
                x86_64::_mm512_permutex2var_epi64(a.into(), idx_lo.into(), b.into()).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_8(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint32x4_t; 4] = mem::transmute(a);
            let b: [uint32x4_t; 4] = mem::transmute(b);
            let target_a = [a[0], a[1], b[0], b[1]];
            let target_b = [a[2], a[3], b[2], b[3]];
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
}

impl BitSwap for B256 {
    const E: u32 = 8;
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_0(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a: __m256i = a.into();
        let b: __m256i = b.into();
        let mask: __m256i = Simd::<u8, 32>::splat(0b_01010101_u8).into();
        unsafe {
            let t = x86_64::_mm256_and_si256(
                x86_64::_mm256_xor_si256(a, x86_64::_mm256_srli_epi64(b.into(), 1)),
                mask,
            );
            (
                x86_64::_mm256_xor_si256(a, t).into(),
                x86_64::_mm256_xor_si256(b, x86_64::_mm256_slli_epi64(t.into(), 1)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 2] = mem::transmute(a);
            let b: [uint8x16_t; 2] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_01010101_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 2];
            let mut target_b = [uint8x16_t::from(Simd::default()); 2];
            for i in 0..2 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 1)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 1)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_1(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a: __m256i = a.into();
        let b: __m256i = b.into();
        let mask: __m256i = Simd::<u8, 32>::splat(0b_00110011_u8).into();
        unsafe {
            let t = x86_64::_mm256_and_si256(
                x86_64::_mm256_xor_si256(a, x86_64::_mm256_srli_epi64(b.into(), 2)),
                mask,
            );
            (
                x86_64::_mm256_xor_si256(a, t).into(),
                x86_64::_mm256_xor_si256(b, x86_64::_mm256_slli_epi64(t.into(), 2)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 2] = mem::transmute(a);
            let b: [uint8x16_t; 2] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00110011_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 2];
            let mut target_b = [uint8x16_t::from(Simd::default()); 2];
            for i in 0..2 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 2)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 2)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_2(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a: __m256i = a.into();
        let b: __m256i = b.into();
        let mask: __m256i = Simd::<u8, 32>::splat(0b_00001111_u8).into();
        unsafe {
            let t = x86_64::_mm256_and_si256(
                x86_64::_mm256_xor_si256(a, x86_64::_mm256_srli_epi64(b.into(), 4)),
                mask,
            );
            (
                x86_64::_mm256_xor_si256(a, t).into(),
                x86_64::_mm256_xor_si256(b, x86_64::_mm256_slli_epi64(t.into(), 4)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 2] = mem::transmute(a);
            let b: [uint8x16_t; 2] = mem::transmute(b);
            let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00001111_u8).into();
            let mut target_a = [uint8x16_t::from(Simd::default()); 2];
            let mut target_b = [uint8x16_t::from(Simd::default()); 2];
            for i in 0..2 {
                let t = aarch64::vandq_u8(
                    aarch64::veorq_u8(a[i], aarch64::vshrq_n_u8(b[i].into(), 4)),
                    mask,
                );
                target_a[i] = aarch64::veorq_u8(a[i], t).into();
                target_b[i] = aarch64::veorq_u8(b[i], aarch64::vshlq_n_u8(t.into(), 4)).into();
            }

            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<32, 0>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u8; 32];
            let mut idx_lo = [0u8; 32];
            let mut i = 0;
            while i < 16 {
                idx_hi[i * 2 + 0] = (i as u8) * 2;
                idx_hi[i * 2 + 1] = 32 + (i as u8) * 2;
                idx_lo[i * 2 + 0] = (i as u8) * 2 + 1;
                idx_lo[i * 2 + 1] = 32 + (i as u8) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm256_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 32>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm256_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 32>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint8x16_t; 2] = mem::transmute(a);
            let b: [uint8x16_t; 2] = mem::transmute(b);
            let mut target_a = [uint8x16_t::from(Simd::default()); 2];
            let mut target_b = [uint8x16_t::from(Simd::default()); 2];
            for i in 0..2 {
                target_a[i] = aarch64::vtrn1q_u8(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u8(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<32, 1>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u16; 16];
            let mut idx_lo = [0u16; 16];
            let mut i = 0;
            while i < 8 {
                idx_hi[i * 2 + 0] = (i as u16) * 2;
                idx_hi[i * 2 + 1] = 16 + (i as u16) * 2;
                idx_lo[i * 2 + 0] = (i as u16) * 2 + 1;
                idx_lo[i * 2 + 1] = 16 + (i as u16) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm256_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 16>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm256_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 16>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint16x8_t; 2] = mem::transmute(a);
            let b: [uint16x8_t; 2] = mem::transmute(b);
            let mut target_a = [uint16x8_t::from(Simd::default()); 2];
            let mut target_b = [uint16x8_t::from(Simd::default()); 2];
            for i in 0..2 {
                target_a[i] = aarch64::vtrn1q_u16(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u16(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<32, 2>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u32; 8];
            let mut idx_lo = [0u32; 8];
            let mut i = 0;
            while i < 4 {
                idx_hi[i * 2 + 0] = (i as u32) * 2;
                idx_hi[i * 2 + 1] = 8 + (i as u32) * 2;
                idx_lo[i * 2 + 0] = (i as u32) * 2 + 1;
                idx_lo[i * 2 + 1] = 8 + (i as u32) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm256_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 8>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm256_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 8>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint32x4_t; 2] = mem::transmute(a);
            let b: [uint32x4_t; 2] = mem::transmute(b);
            let mut target_a = [uint32x4_t::from(Simd::default()); 2];
            let mut target_b = [uint32x4_t::from(Simd::default()); 2];
            for i in 0..2 {
                target_a[i] = aarch64::vtrn1q_u32(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u32(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<32, 3>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u64; 4];
            let mut idx_lo = [0u64; 4];
            let mut i = 0;
            while i < 2 {
                idx_hi[i * 2 + 0] = (i as u64) * 2;
                idx_hi[i * 2 + 1] = 4 + (i as u64) * 2;
                idx_lo[i * 2 + 0] = (i as u64) * 2 + 1;
                idx_lo[i * 2 + 1] = 4 + (i as u64) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm256_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 4>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm256_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 4>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint64x2_t; 2] = mem::transmute(a);
            let b: [uint64x2_t; 2] = mem::transmute(b);
            let mut target_a = [uint64x2_t::from(Simd::default()); 2];
            let mut target_b = [uint64x2_t::from(Simd::default()); 2];
            for i in 0..2 {
                target_a[i] = aarch64::vtrn1q_u64(a[i].into(), b[i].into()).into();
                target_b[i] = aarch64::vtrn2q_u64(a[i].into(), b[i].into()).into();
            }
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<32, 4>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        let idx_hi = Simd::<u64, 4>::from_array([0, 1, 4, 5]);
        let idx_lo = Simd::<u64, 4>::from_array([2, 3, 6, 7]);
        unsafe {
            (
                x86_64::_mm256_permutex2var_epi64(a.into(), idx_hi.into(), b.into()).into(),
                x86_64::_mm256_permutex2var_epi64(a.into(), idx_lo.into(), b.into()).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_7(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            let a: [uint32x4_t; 2] = mem::transmute(a);
            let b: [uint32x4_t; 2] = mem::transmute(b);
            let target_a = [a[0], b[0]];
            let target_b = [a[1], b[1]];
            (mem::transmute(target_a), mem::transmute(target_b))
        }
    }
}

impl BitSwap for B128 {
    const E: u32 = 7;
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_0(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a: __m128i = a.into();
        let b: __m128i = b.into();
        let mask: __m128i = Simd::<u8, 16>::splat(0b_01010101_u8).into();
        unsafe {
            let t = x86_64::_mm_and_si128(
                x86_64::_mm_xor_si128(a, x86_64::_mm_srli_epi64(b.into(), 1)),
                mask,
            );
            (
                x86_64::_mm_xor_si128(a, t).into(),
                x86_64::_mm_xor_si128(b, x86_64::_mm_slli_epi64(t.into(), 1)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a: uint8x16_t = a.into();
        let b: uint8x16_t = b.into();
        let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_01010101_u8).into();
        unsafe {
            let t = aarch64::vandq_u8(aarch64::veorq_u8(a, aarch64::vshrq_n_u8(b.into(), 1)), mask);
            (
                aarch64::veorq_u8(a, t).into(),
                aarch64::veorq_u8(b, aarch64::vshlq_n_u8(t.into(), 1)).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_1(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a: __m128i = a.into();
        let b: __m128i = b.into();
        let mask: __m128i = Simd::<u8, 16>::splat(0b_00110011_u8).into();
        unsafe {
            let t = x86_64::_mm_and_si128(
                x86_64::_mm_xor_si128(a, x86_64::_mm_srli_epi64(b.into(), 2)),
                mask,
            );
            (
                x86_64::_mm_xor_si128(a, t).into(),
                x86_64::_mm_xor_si128(b, x86_64::_mm_slli_epi64(t.into(), 2)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a: uint8x16_t = a.into();
        let b: uint8x16_t = b.into();
        let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00110011_u8).into();
        unsafe {
            let t = aarch64::vandq_u8(aarch64::veorq_u8(a, aarch64::vshrq_n_u8(b.into(), 2)), mask);
            (
                aarch64::veorq_u8(a, t).into(),
                aarch64::veorq_u8(b, aarch64::vshlq_n_u8(t.into(), 2)).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bits_2(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a: __m128i = a.into();
        let b: __m128i = b.into();
        let mask: __m128i = Simd::<u8, 16>::splat(0b_00001111_u8).into();
        unsafe {
            let t = x86_64::_mm_and_si128(
                x86_64::_mm_xor_si128(a, x86_64::_mm_srli_epi64(b.into(), 4)),
                mask,
            );
            (
                x86_64::_mm_xor_si128(a, t).into(),
                x86_64::_mm_xor_si128(b, x86_64::_mm_slli_epi64(t.into(), 4)).into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a: uint8x16_t = a.into();
        let b: uint8x16_t = b.into();
        let mask: uint8x16_t = Simd::<u8, 16>::splat(0b_00001111_u8).into();
        unsafe {
            let t = aarch64::vandq_u8(aarch64::veorq_u8(a, aarch64::vshrq_n_u8(b.into(), 4)), mask);
            (
                aarch64::veorq_u8(a, t).into(),
                aarch64::veorq_u8(b, aarch64::vshlq_n_u8(t.into(), 4)).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<16, 0>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u8; 16];
            let mut idx_lo = [0u8; 16];
            let mut i = 0;
            while i < 8 {
                idx_hi[i * 2 + 0] = (i as u8) * 2;
                idx_hi[i * 2 + 1] = 16 + (i as u8) * 2;
                idx_lo[i * 2 + 0] = (i as u8) * 2 + 1;
                idx_lo[i * 2 + 1] = 16 + (i as u8) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 16>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm_permutex2var_epi8(
                    a.into(),
                    Simd::<u8, 16>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            (
                aarch64::vtrn1q_u8(a.into(), b.into()).into(),
                aarch64::vtrn2q_u8(a.into(), b.into()).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<16, 1>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u16; 8];
            let mut idx_lo = [0u16; 8];
            let mut i = 0;
            while i < 4 {
                idx_hi[i * 2 + 0] = (i as u16) * 2;
                idx_hi[i * 2 + 1] = 8 + (i as u16) * 2;
                idx_lo[i * 2 + 0] = (i as u16) * 2 + 1;
                idx_lo[i * 2 + 1] = 8 + (i as u16) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 8>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm_permutex2var_epi16(
                    a.into(),
                    Simd::<u16, 8>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            (
                aarch64::vtrn1q_u16(a.into(), b.into()).into(),
                aarch64::vtrn2q_u16(a.into(), b.into()).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<16, 2>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u32; 4];
            let mut idx_lo = [0u32; 4];
            let mut i = 0;
            while i < 2 {
                idx_hi[i * 2 + 0] = (i as u32) * 2;
                idx_hi[i * 2 + 1] = 4 + (i as u32) * 2;
                idx_lo[i * 2 + 0] = (i as u32) * 2 + 1;
                idx_lo[i * 2 + 1] = 4 + (i as u32) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 4>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm_permutex2var_epi32(
                    a.into(),
                    Simd::<u32, 4>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            (
                aarch64::vtrn1q_u32(a.into(), b.into()).into(),
                aarch64::vtrn2q_u32(a.into(), b.into()).into(),
            )
        }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<16, 3>(a.0, b.0);
        (Self(a), Self(b))
    }
    #[cfg(target_feature = "avx512f")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        let (idx_hi, idx_lo) = const {
            let mut idx_hi = [0u64; 2];
            let mut idx_lo = [0u64; 2];
            let mut i = 0;
            while i < 1 {
                idx_hi[i * 2 + 0] = (i as u64) * 2;
                idx_hi[i * 2 + 1] = 2 + (i as u64) * 2;
                idx_lo[i * 2 + 0] = (i as u64) * 2 + 1;
                idx_lo[i * 2 + 1] = 2 + (i as u64) * 2 + 1;
                i += 1;
            }
            (idx_hi, idx_lo)
        };
        unsafe {
            (
                x86_64::_mm_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 2>::from_array(idx_hi).into(),
                    b.into(),
                )
                .into(),
                x86_64::_mm_permutex2var_epi64(
                    a.into(),
                    Simd::<u64, 2>::from_array(idx_lo).into(),
                    b.into(),
                )
                .into(),
            )
        }
    }
    #[cfg(target_feature = "neon")]
    fn swap_bits_6(a: Self, b: Self) -> (Self, Self) {
        unsafe {
            (
                aarch64::vtrn1q_u64(a.into(), b.into()).into(),
                aarch64::vtrn2q_u64(a.into(), b.into()).into(),
            )
        }
    }
}

impl BitSwap for B64 {
    const E: u32 = 6;
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a = u64::from_le_bytes(a.0);
        let b = u64::from_le_bytes(b.0);
        let t = (a ^ (b >> 1)) & u64::from_le_bytes([0b_01010101_u8; 8]);
        (
            B64(u64::to_le_bytes(a ^ t)),
            B64(u64::to_le_bytes(b ^ t << 1)),
        )
    }
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a = u64::from_le_bytes(a.0);
        let b = u64::from_le_bytes(b.0);
        let t = (a ^ b >> 2) & u64::from_le_bytes([0b_00110011_u8; 8]);
        (
            B64(u64::to_le_bytes(a ^ t)),
            B64(u64::to_le_bytes(b ^ t << 2)),
        )
    }
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a = u64::from_le_bytes(a.0);
        let b = u64::from_le_bytes(b.0);
        let t = (a ^ b >> 4) & u64::from_le_bytes([0b_00001111_u8; 8]);
        (
            B64(u64::to_le_bytes(a ^ t)),
            B64(u64::to_le_bytes(b ^ t << 4)),
        )
    }
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<8, 0>(a.0, b.0);
        (B64(a), B64(b))
    }
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<8, 1>(a.0, b.0);
        (B64(a), B64(b))
    }
    fn swap_bits_5(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<8, 2>(a.0, b.0);
        (B64(a), B64(b))
    }
}

impl BitSwap for B32 {
    const E: u32 = 5;
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a = u32::from_le_bytes(a.0);
        let b = u32::from_le_bytes(b.0);
        let t = (a ^ (b >> 1)) & u32::from_le_bytes([0b_01010101_u8; 4]);
        (
            B32(u32::to_le_bytes(a ^ t)),
            B32(u32::to_le_bytes(b ^ t << 1)),
        )
    }
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a = u32::from_le_bytes(a.0);
        let b = u32::from_le_bytes(b.0);
        let t = (a ^ b >> 2) & u32::from_le_bytes([0b_00110011_u8; 4]);
        (
            B32(u32::to_le_bytes(a ^ t)),
            B32(u32::to_le_bytes(b ^ t << 2)),
        )
    }
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a = u32::from_le_bytes(a.0);
        let b = u32::from_le_bytes(b.0);
        let t = (a ^ b >> 4) & u32::from_le_bytes([0b_00001111_u8; 4]);
        (
            B32(u32::to_le_bytes(a ^ t)),
            B32(u32::to_le_bytes(b ^ t << 4)),
        )
    }
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<4, 0>(a.0, b.0);
        (B32(a), B32(b))
    }
    fn swap_bits_4(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<4, 1>(a.0, b.0);
        (B32(a), B32(b))
    }
}

impl BitSwap for B16 {
    const E: u32 = 4;
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a = u16::from_le_bytes(a.0);
        let b = u16::from_le_bytes(b.0);
        let t = (a ^ (b >> 1)) & u16::from_le_bytes([0b_01010101_u8; 2]);
        (
            B16(u16::to_le_bytes(a ^ t)),
            B16(u16::to_le_bytes(b ^ t << 1)),
        )
    }
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a = u16::from_le_bytes(a.0);
        let b = u16::from_le_bytes(b.0);
        let t = (a ^ b >> 2) & u16::from_le_bytes([0b_00110011_u8; 2]);
        (
            B16(u16::to_le_bytes(a ^ t)),
            B16(u16::to_le_bytes(b ^ t << 2)),
        )
    }
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a = u16::from_le_bytes(a.0);
        let b = u16::from_le_bytes(b.0);
        let t = (a ^ b >> 4) & u16::from_le_bytes([0b_00001111_u8; 2]);
        (
            B16(u16::to_le_bytes(a ^ t)),
            B16(u16::to_le_bytes(b ^ t << 4)),
        )
    }
    fn swap_bits_3(a: Self, b: Self) -> (Self, Self) {
        let (a, b) = swap_bytes_n::<2, 0>(a.0, b.0);
        (B16(a), B16(b))
    }
}

impl BitSwap for B8 {
    const E: u32 = 3;
    fn swap_bits_0(a: Self, b: Self) -> (Self, Self) {
        let a = u8::from_le_bytes(a.0);
        let b = u8::from_le_bytes(b.0);
        let t = (a ^ (b >> 1)) & u8::from_le_bytes([0b_01010101_u8; 1]);
        (B8(u8::to_le_bytes(a ^ t)), B8(u8::to_le_bytes(b ^ t << 1)))
    }
    fn swap_bits_1(a: Self, b: Self) -> (Self, Self) {
        let a = u8::from_le_bytes(a.0);
        let b = u8::from_le_bytes(b.0);
        let t = (a ^ b >> 2) & u8::from_le_bytes([0b_00110011_u8; 1]);
        (B8(u8::to_le_bytes(a ^ t)), B8(u8::to_le_bytes(b ^ t << 2)))
    }
    fn swap_bits_2(a: Self, b: Self) -> (Self, Self) {
        let a = u8::from_le_bytes(a.0);
        let b = u8::from_le_bytes(b.0);
        let t = (a ^ b >> 4) & u8::from_le_bytes([0b_00001111_u8; 1]);
        (B8(u8::to_le_bytes(a ^ t)), B8(u8::to_le_bytes(b ^ t << 4)))
    }
}

pub trait BitSwap: Sized + Copy + Default {
    const E: u32;
    fn transpose_step_loop_seg<F: Fn(Self, Self) -> (Self, Self)>(
        a: &mut [Self; 2usize.pow(Self::E)],
        je: u32,
        base: usize,
        level: usize,
        swap_fn: F,
    ) {
        let lo_mask = !(!0 << je);
        let hi_mask = (!0) << (je + 1);
        for i in base..base + (1 << level) {
            let lo = i & lo_mask;
            let hi = (i << 1) & hi_mask;
            let x = lo | hi;
            let y = x | (1 << je);
            //println!("{:08b} {:08b}", x, y);
            (a[x], a[y]) = swap_fn(a[x], a[y]);
        }
    }
    fn transpose_partial<F: Fn(Self, Self) -> (Self, Self), const S: u32>(
        a: &mut [Self; 2usize.pow(S)],
        je: u32,
        swap_fn: F,
    ) {
        let lo_mask = !(!0 << je);
        let hi_mask = (!0) << (je + 1);
        for i in 0..2usize.pow(S - 1) {
            let lo = i & lo_mask;
            let hi = (i << 1) & hi_mask;
            let x = lo | hi;
            let y = x | (1 << je);
            //println!("{:08b} {:08b}", x, y);
            (a[x], a[y]) = swap_fn(a[x], a[y]);
        }
    }
    fn transpose_step<F: Fn(Self, Self) -> (Self, Self)>(
        a: &mut [Self; 2usize.pow(Self::E)],
        je: u32,
        swap_fn: F,
    ) {
        Self::transpose_step_loop_seg(a, je, 0, Self::E as usize - 1, &swap_fn);
    }
    fn swap_bits_0(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 0");
    }
    fn swap_bits_1(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 1");
    }
    fn swap_bits_2(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 2");
    }
    fn swap_bits_3(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 3");
    }
    fn swap_bits_4(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 4");
    }
    fn swap_bits_5(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 5");
    }
    fn swap_bits_6(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 6");
    }
    fn swap_bits_7(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 7");
    }
    fn swap_bits_8(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 8");
    }
    fn swap_bits_9(_: Self, _: Self) -> (Self, Self) {
        panic!("Cannot transpose at level 9");
    }
}

/*
#[cfg(test)]
mod tests {
    use super::{BitArray, B128, B256, B512, B64, B8};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    #[test]
    fn test_b8() {
        let mut rng = Pcg64::seed_from_u64(42);
        B8::test(&mut rng);
    }
    #[test]
    fn test_b64() {
        let mut rng = Pcg64::seed_from_u64(42);
        B64::test(&mut rng);
    }
    #[test]
    fn test_b128() {
        let mut rng = Pcg64::seed_from_u64(42);
        B128::test(&mut rng);
    }
    #[test]
    fn test_b256() {
        let mut rng = Pcg64::seed_from_u64(42);
        B256::test(&mut rng);
    }
    #[test]
    fn test_b512() {
        let mut rng = Pcg64::seed_from_u64(42);
        B512::test(&mut rng);
    }
}
*/

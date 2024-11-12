use crate::BitArray;
use std::mem;

pub fn half_adder<V: BitArray>(a: V, b: V) -> (V, V) {
    (a ^ b, a & b)
}

pub fn full_adder<V: BitArray>(a: V, b: V, c: V) -> (V, V) {
    let u = a ^ b;
    (u ^ c, (a & b) | (u & c))
}

/// Add two `L` bit numbers together to produce an `L+1` bit number. Thus it will not overflow.
/// Note that this uses big endian byte order! If your host CPU is little endian and your numbers are >8 bits, you will need to call `fix_endianness()` on the bits.
pub fn bit_add<V: BitArray, const L: usize>(a: &[V; L], b: &[V; L]) -> [V; L + 1] {
    let mut acc = [V::splat(false); L + 1];
    let (zero, c) = half_adder(a[L - 1], b[L - 1]);
    acc[L] = zero;
    let mut carry = c;
    for i in 2..L + 1 {
        let i = L - i;
        (acc[i + 1], carry) = full_adder(a[i], b[i], carry);
    }
    acc[0] = carry;
    acc
}

pub trait BitAdd<const L: usize> {
    /// Add two `L` bit numbers together to produce an `L+1` bit number. Thus it will not overflow.
    /// Note that this uses big endian byte order! If your host CPU is little endian and your numbers are >8 bits, you will need to call `fix_endianness()` on the bits.
    fn add<V: BitArray>(a: &[V; L], b: &[V; L]) -> [V; L + 1];
}

macro_rules! impl_add {
    ($l:expr) => {
        impl BitAdd<$l> for () {
            fn add<V: BitArray>(a: &[V; $l], b: &[V; $l]) -> [V; $l + 1] {
                let mut acc = [V::splat(false); $l + 1];
                let (zero, c) = half_adder(a[$l - 1], b[$l - 1]);
                acc[$l] = zero;
                let mut carry = c;
                for i in 2..$l + 1 {
                    let i = $l - i;
                    (acc[i + 1], carry) = full_adder(a[i], b[i], carry);
                }
                acc[0] = carry;
                acc
            }
        }
    };
}

impl_add!(1);
impl_add!(2);
impl_add!(3);
impl_add!(4);
impl_add!(5);
impl_add!(6);
impl_add!(7);
impl_add!(8);
impl_add!(9);
impl_add!(10);
impl_add!(11);
impl_add!(12);

pub trait Popcnt<const E: u32> {
    fn popcnt<V: BitArray>(a: &[V; 2usize.pow(E)]) -> [V; E as usize + 1];
}

impl Popcnt<1> for () {
    fn popcnt<V: BitArray>(a: &[V; 2]) -> [V; 2] {
        popcnt2(a)
    }
}

/// Counts the number of bits on an array of bits.
#[inline(always)]
pub fn popcnt2<V: BitArray>(bits: &[V; 2]) -> [V; 2] {
    let (s, c) = half_adder(bits[0], bits[1]);
    [c, s]
}

macro_rules! impl_popcnt {
    ($fn_name:ident, $sub_fn_name:ident, $n:expr) => {
        /// Counts the number of bits on an array of bits.
        pub fn $fn_name<V: BitArray>(bits: &[V; 2usize.pow($n)]) -> [V; $n + 1] {
            let a: [V; $n] = $sub_fn_name(bits[0..2usize.pow($n - 1)].try_into().unwrap());
            let b: [V; $n] =
                $sub_fn_name(bits[2usize.pow($n - 1)..2usize.pow($n)].try_into().unwrap());
            bit_add::<V, $n>(&a, &b)
        }
        impl Popcnt<$n> for () {
            fn popcnt<V: BitArray>(a: &[V; 2usize.pow($n)]) -> [V; $n + 1] {
                $fn_name(a)
            }
        }
    };
}

impl_popcnt!(popcnt4, popcnt2, 2);
impl_popcnt!(popcnt8, popcnt4, 3);
impl_popcnt!(popcnt16, popcnt8, 4);
impl_popcnt!(popcnt32, popcnt16, 5);
impl_popcnt!(popcnt64, popcnt32, 6);
impl_popcnt!(popcnt128, popcnt64, 7);
impl_popcnt!(popcnt256, popcnt128, 8);
impl_popcnt!(popcnt512, popcnt256, 9);
impl_popcnt!(popcnt1024, popcnt512, 10);

pub fn half_comparator<V: BitArray>(a: V, b: V) -> (V, V, V) {
    let lt = !a & b;
    let gt = a & !b;
    let eq = !(lt | gt);
    (lt, eq, gt)
}

pub fn full_comparator<V: BitArray>(a: V, b: V, c: (V, V, V)) -> (V, V, V) {
    let x = half_comparator(a, b);
    let lt = x.0 | (!x.2 & c.0);
    let gt = x.2 | (!x.0 & c.2);
    (lt, !(lt | gt), gt)
}

/// (a > b, a == b, a < b)
pub fn comparator<V: BitArray, const L: usize>(a: &[V; L], b: &[V; L]) -> (V, V, V) {
    let mut acc = half_comparator(a[L - 1], b[L - 1]);
    for i in 1..L {
        let i = (L - i) - 1;
        acc = full_comparator(a[i], b[i], acc);
    }
    acc
}

/// Since bitslice logic is big endian, if you are running on a little endian CPU (such as x86 and most ARMs) you must swap the byte order before and after doing math.
/// This function will swap byte order only on little endian machines, on big endian machines it should compile to a null op.
pub fn fix_endianness<T, const L: usize>(bits: &mut [T; L * 8]) {
    if cfg!(target_endian = "little") {
        let tmp: &mut [[T; 8]; L] = unsafe { mem::transmute(bits) };
        tmp.reverse();
    }
}

/// Pads the integer with zeros on the most significant side, thus preserving the value while permitting it to be transposed and such like.
pub fn pad<T: Copy + BitArray, const S: usize, const L: usize>(input: &[T; S]) -> [T; L] {
    let mut target = [T::splat(false); L];
    for i in 0..S {
        target[i + (L - S)] = input[i];
    }
    target
}

pub trait Expand<const E: u32> {
    fn onehot_raw<V: BitArray>(input: &[V; E as usize], mask: V, target: &mut [V; 2usize.pow(E)]);
    fn onehot<V: BitArray>(input: &[V; E as usize]) -> [V; 2usize.pow(E)] {
        let mut target = [V::splat(false); 2usize.pow(E)];
        Self::onehot_raw(input, V::splat(true), &mut target);
        target
    }
    fn hadamard_raw<V: BitArray>(input: &[V; E as usize], target: &mut [V; 2usize.pow(E)]);
    fn hadamard<V: BitArray>(input: &[V; E as usize]) -> [V; 2usize.pow(E)] {
        let mut target = [V::splat(false); 2usize.pow(E)];
        Self::hadamard_raw(input, &mut target);
        target
    }
    fn unary_raw<V: BitArray>(
        input: &[V; E as usize],         // The bits of the input numbers
        head: V,                         // The head of the input numbers. Init to the 0 vec.
        lt: V,                           // less than carry. Init to the 0 vec
        gt: V,                           // greater than carry. Init to the 0 vec
        idx: bool,                       // the current index bit. Init to false
        target: &mut [V; 2usize.pow(E)], // the target array into which to write the result.
    );
    fn unary<V: BitArray>(input: &[V; E as usize]) -> [V; 2usize.pow(E)] {
        let mut target = [V::splat(false); 2usize.pow(E)];
        Self::unary_raw(
            input,
            V::splat(false),
            V::splat(false),
            V::splat(false),
            false,
            &mut target,
        );
        target
    }
}

impl Expand<1> for () {
    fn onehot_raw<V: BitArray>(input: &[V; 1], mask: V, target: &mut [V; 2]) {
        target[0] = mask & !input[0];
        target[1] = mask & input[0];
    }
    fn hadamard_raw<V: BitArray>(input: &[V; 1], target: &mut [V; 2]) {
        target[0] = V::splat(false);
        target[1] = input[0];
    }
    fn unary_raw<V: BitArray>(
        input: &[V; 1],      // The bits of the input numbers
        head: V,             // The head of the input numbers. Init to the 0 vec.
        lt: V,               // less than carry. Init to the 0 vec
        gt: V,               // greater than carry. Init to the 0 vec
        idx: bool,           // the current index bit. Init to false
        target: &mut [V; 2], // the target array into which to write the result.
    ) {
        let x = (!head & V::splat(idx), head & !V::splat(idx));

        let lt = lt | (!gt & x.0);
        let gt = gt | (!lt & x.1);

        target[0] = lt;
        target[1] = lt | (!gt & !input[0]);
    }
}

macro_rules! impl_expand {
    ($e:expr, $es1:expr) => {
        impl Expand<$e> for () {
            fn onehot_raw<V: BitArray>(input: &[V; $e], mask: V, target: &mut [V; 2usize.pow($e)]) {
                <() as Expand<$es1>>::onehot_raw(
                    input[1..$e].try_into().unwrap(),
                    mask & !input[0],
                    <&mut [V; 2usize.pow($e - 1)]>::try_from(&mut target[0..2usize.pow($e - 1)]).unwrap(),
                );
                <() as Expand<$es1>>::onehot_raw(
                    input[1..$e].try_into().unwrap(),
                    mask & input[0],
                    <&mut [V; 2usize.pow($e - 1)]>::try_from(&mut target[2usize.pow($e - 1)..2usize.pow($e)]).unwrap(),
                );
            }
            fn hadamard_raw<V: BitArray>(input: &[V; $e], target: &mut [V; 2usize.pow($e)]) {
                <() as Expand<$es1>>::hadamard_raw(
                    input[0..$e - 1].try_into().unwrap(),
                    <&mut [V; 2usize.pow($e - 1)]>::try_from(&mut target[0..2usize.pow($e-1)]).unwrap(),
                );
                for i in 0..2usize.pow($e - 1) {
                    target[2usize.pow($e - 1) + i] = target[i] ^ input[$e - 1];
                }
            }
            fn unary_raw<V: BitArray>(
                input: &[V; $e],                  // The bits of the input numbers
                head: V,                          // The head of the input numbers. Init to the 0 vec.
                lt: V,                            // less than carry. Init to the 0 vec
                gt: V,                            // greater than carry. Init to the 0 vec
                idx: bool,                        // the current index bit. Init to false
                target: &mut [V; 2usize.pow($e)], // the target array into which to write the result.
            ) {
                let x = ((!head) & V::splat(idx), head & (!V::splat(idx)));

                let lt = lt | (!gt & x.0);
                let gt = gt | (!lt & x.1);

                let tail = input[1..].try_into().unwrap();

                <() as Expand<$es1>>::unary_raw(
                    tail,
                    input[0],
                    lt,
                    gt,
                    false,
                    <&mut [V; 2usize.pow($e - 1)]>::try_from(&mut target[0..2usize.pow($e - 1)])
                        .unwrap(),
                );
                <() as Expand<$es1>>::unary_raw(
                    tail,
                    input[0],
                    lt,
                    gt,
                    true,
                    <&mut [V; 2usize.pow($e - 1)]>::try_from(&mut target[2usize.pow($e - 1)..])
                        .unwrap(),
                );
            }
        }
    }
}

impl_expand!(2, 1);
impl_expand!(3, 2);
impl_expand!(4, 3);
impl_expand!(5, 4);
impl_expand!(6, 5);
impl_expand!(7, 6);
impl_expand!(8, 7);
impl_expand!(9, 8);
impl_expand!(10, 9);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitMatrix, Transpose, B128, B16, B256, B32, B64, B8};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use std::mem;

    #[test]
    fn expand_unary_test() {
        let input_a: [B8; 128] = (0..=127)
            .map(|i| B8([i]))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let bitsliced_input = <B8 as Transpose<B128>>::transpose(BitMatrix(input_a));

        let target = <() as Expand<3>>::unary::<B128>(&bitsliced_input.0[5..].try_into().unwrap());
        let correct = [
            B128([0b_0000_0000_u8; 16]),
            B128([0b_1000_0000_u8; 16]),
            B128([0b_1100_0000_u8; 16]),
            B128([0b_1110_0000_u8; 16]),
            B128([0b_1111_0000_u8; 16]),
            B128([0b_1111_1000_u8; 16]),
            B128([0b_1111_1100_u8; 16]),
            B128([0b_1111_1110_u8; 16]),
        ];
        assert_eq!(target, correct);
    }
    #[test]
    fn expand_onehot_test() {
        let input_a: [B8; 128] = (0..=127)
            .map(|i| B8([i]))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let bitsliced_input = <B8 as Transpose<B128>>::transpose(BitMatrix(input_a));

        let target = <() as Expand<3>>::onehot::<B128>(&bitsliced_input.0[5..].try_into().unwrap());
        dbg!(target);
        let correct = [
            B128([0b_1000_0000_u8; 16]),
            B128([0b_0100_0000_u8; 16]),
            B128([0b_0010_0000_u8; 16]),
            B128([0b_0001_0000_u8; 16]),
            B128([0b_0000_1000_u8; 16]),
            B128([0b_0000_0100_u8; 16]),
            B128([0b_0000_0010_u8; 16]),
            B128([0b_0000_0001_u8; 16]),
        ];
        dbg!(correct);
        assert_eq!(target, correct);
    }
    #[test]
    fn expand_hadamard_test() {
        let input_a: [B8; 128] = (0..=127)
            .map(|i| B8([i]))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let bitsliced_input = <B8 as Transpose<B128>>::transpose(BitMatrix(input_a));

        let target =
            <() as Expand<3>>::hadamard::<B128>(&bitsliced_input.0[5..].try_into().unwrap());
        dbg!(target);
        let correct = [
            B128([0b_00000000_u8; 16]),
            B128([0b_00001111_u8; 16]),
            B128([0b_00110011_u8; 16]),
            B128([0b_00111100_u8; 16]),
            B128([0b_01010101_u8; 16]),
            B128([0b_01011010_u8; 16]),
            B128([0b_01100110_u8; 16]),
            B128([0b_01101001_u8; 16]),
        ];
        dbg!(correct);
        assert_eq!(target, correct);
    }

    #[test]
    fn add_test_8() {
        let mut rng = Pcg64::seed_from_u64(42);
        let input_a: [B8; 256] = [(); 256].map(|_| B8([rng.gen::<u8>() >> 1]));
        let bitsliced_input_a = <B8 as Transpose<B256>>::transpose(BitMatrix(input_a));

        let input_b: [B8; 256] = [(); 256].map(|_| B8([rng.gen::<u8>() >> 1]));
        let bitsliced_input_b = <B8 as Transpose<B256>>::transpose(BitMatrix(input_b));

        let sum: [B256; 8] = bit_add::<B256, 7>(
            &bitsliced_input_a.0[1..8].try_into().unwrap(),
            &bitsliced_input_b.0[1..8].try_into().unwrap(),
        );
        let int_sum = <B256 as Transpose<B8>>::transpose(BitMatrix(sum));

        input_a
            .iter()
            .zip(input_b.iter())
            .map(|(a, b)| a.0[0] + b.0[0])
            .zip(int_sum.0.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y.0[0]);
            });
    }
    #[test]
    fn add_test_16() {
        let mut rng = Pcg64::seed_from_u64(42);
        let input_a: [u16; 256] = [(); 256].map(|_| rng.gen::<u16>() >> 1);
        let mut bitsliced_input_a =
            <B16 as Transpose<B256>>::transpose(BitMatrix(unsafe { mem::transmute(input_a) }));
        fix_endianness::<_, 2>(&mut bitsliced_input_a.0);

        let input_b: [u16; 256] = [(); 256].map(|_| rng.gen::<u16>() >> 1);
        let mut bitsliced_input_b =
            <B16 as Transpose<B256>>::transpose(BitMatrix(unsafe { mem::transmute(input_b) }));
        fix_endianness::<_, 2>(&mut bitsliced_input_b.0);

        let mut sum: [B256; 16] = bit_add::<B256, 15>(
            &bitsliced_input_a.0[1..16].try_into().unwrap(),
            &bitsliced_input_b.0[1..16].try_into().unwrap(),
        );
        fix_endianness::<_, 2>(&mut sum);
        let int_sum = <B256 as Transpose<B16>>::transpose(BitMatrix(sum));
        let int_sum: [u16; 256] = unsafe { mem::transmute(int_sum) };

        input_a
            .iter()
            .zip(input_b.iter())
            .map(|(a, b)| a + b)
            .zip(int_sum.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, *y);
            });
    }
    #[test]
    fn add_test_32() {
        let mut rng = Pcg64::seed_from_u64(42);
        let input_a: [u32; 256] = [(); 256].map(|_| rng.gen::<u32>() >> 1);
        let mut bitsliced_input_a =
            <B32 as Transpose<B256>>::transpose(BitMatrix(unsafe { mem::transmute(input_a) }));
        fix_endianness::<_, 4>(&mut bitsliced_input_a.0);

        let input_b: [u32; 256] = [(); 256].map(|_| rng.gen::<u32>() >> 1);
        let mut bitsliced_input_b =
            <B32 as Transpose<B256>>::transpose(BitMatrix(unsafe { mem::transmute(input_b) }));
        fix_endianness::<_, 4>(&mut bitsliced_input_b.0);

        let mut sum: [B256; 32] = bit_add::<B256, 31>(
            &bitsliced_input_a.0[1..32].try_into().unwrap(),
            &bitsliced_input_b.0[1..32].try_into().unwrap(),
        );
        fix_endianness::<_, 4>(&mut sum);
        let int_sum = <B256 as Transpose<B32>>::transpose(BitMatrix(sum));
        let int_sum: [u32; 256] = unsafe { mem::transmute(int_sum) };

        input_a
            .iter()
            .zip(input_b.iter())
            .map(|(a, b)| a + b)
            .zip(int_sum.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, *y);
            });
    }

    #[test]
    fn popcnt64_test() {
        let mut rng = Pcg64::seed_from_u64(42);
        let input_a: [u64; 256] = [(); 256].map(|_| rng.gen::<u64>() >> 1);
        let bitsliced_input =
            <B64 as Transpose<B256>>::transpose(BitMatrix(unsafe { mem::transmute(input_a) }));

        let popcnts: [B256; 7] = popcnt64(&bitsliced_input.0);
        let padded = {
            let mut target = [B256::splat(false); 8];
            for i in 0..7 {
                target[i + 1] = popcnts[i];
            }
            target
        };
        let int_popcnts = <B256 as Transpose<B8>>::transpose(BitMatrix(padded));

        input_a
            .iter()
            .map(|a| a.count_ones() as u8)
            .zip(int_popcnts.0.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y.0[0]);
            });
    }
    #[test]
    fn cmp_8_test() {
        let mut rng = Pcg64::seed_from_u64(42);
        let input_a: [B8; 256] = [(); 256].map(|_| B8([rng.gen::<u8>()]));
        let bitsliced_input_a = <B8 as Transpose<B256>>::transpose(BitMatrix(input_a));

        let input_b: [B8; 256] = [(); 256].map(|_| B8([rng.gen::<u8>()]));
        let bitsliced_input_b = <B8 as Transpose<B256>>::transpose(BitMatrix(input_b));

        let cmp: (B256, B256, B256) =
            comparator::<B256, 8>(&bitsliced_input_a.0, &bitsliced_input_b.0);
        let padded = {
            let mut target = [B256::splat(false); 8];
            target[5] = cmp.0;
            target[6] = cmp.1;
            target[7] = cmp.2;
            target
        };
        let cmps = <B256 as Transpose<B8>>::transpose(BitMatrix(padded));

        input_a
            .iter()
            .zip(input_b.iter())
            .map(|(a, b)| {
                ((a.0[0] > b.0[0]) as u8)
                    | (((a.0[0] == b.0[0]) as u8) << 1)
                    | (((a.0[0] < b.0[0]) as u8) << 2)
            })
            .zip(cmps.0.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y.0[0]);
            });
    }
}

use bitslice::arithmetic::{bit_add, comparator, fix_endianness, pad, popcnt128};
use bitslice::{BitArray, BitMatrix, Transpose, B16, B32, B512};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::mem;

fn main() {
    let mut rng = Pcg64::seed_from_u64(42);
    // let us make some numbers! In production, this would be some meaningful data.
    let ints_a = [(); 512].map(|_| rng.gen::<u16>());
    let ints_b = [(); 512].map(|_| rng.gen::<u16>());

    let bits_a = {
        // This words are not adequately aligned so let's transmute the words and put them in a container.
        // the transmute is a value level, and should not incur alignment issues, so it should be safe.
        let aligned_words: BitMatrix<B16, 512> = BitMatrix(unsafe { mem::transmute(ints_a) });
        // Now let's transpose the words so that we have some nice bits to work with.
        let mut bits: BitMatrix<B512, 16> = <B16 as Transpose<B512>>::transpose(aligned_words);
        // if our CPU is little endian, we must swap endiness.
        fix_endianness::<B512, 2>(&mut bits.0);
        bits.0
    };
    // now we do it all over again with the other ints
    let bits_b = {
        let mut bits: BitMatrix<B512, 16> =
            <B16 as Transpose<B512>>::transpose(BitMatrix(unsafe { mem::transmute(ints_b) }));
        fix_endianness::<B512, 2>(&mut bits.0);
        bits.0
    };
    // Now we have our ints in bitslice mode, we can add them.
    let sum: [B512; 17] = bit_add::<B512, 16>(&bits_a, &bits_b);
    // Since the sum of two 16 bit ints needs 17 bits to not overflow, we must now pad the 17 bit int to a 32 bit int.
    let mut padded = pad::<_, 17, 32>(&sum);
    // Now we can switch the endianess back,
    fix_endianness::<B512, 4>(&mut padded);
    // transpose it back,
    let mut bits: BitMatrix<B32, 512> = <B512 as Transpose<B32>>::transpose(BitMatrix(padded));
    // and transmute back to u32s.
    let ints_sum: [u32; 512] = unsafe { mem::transmute(bits) };

    // Now let us check that we added the numbers correctly:
    ints_a
        .into_iter()
        .zip(ints_b.into_iter())
        .map(|(a, b)| a as u32 + b as u32)
        .zip(ints_sum)
        .for_each(|(x, y)| {
            assert_eq!(x, y);
        })
}

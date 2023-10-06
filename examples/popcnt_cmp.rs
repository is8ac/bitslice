use bitslice::{Transpose, B512, B128, B8, BitMatrix, BitArray};
use bitslice::arithmetic::{popcnt128, comparator};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::mem;

fn main(){
    // let us make some words. In production, this would be some meaningful data.
    let mut rng = Pcg64::seed_from_u64(42);
    let input_words: [u128; 512] = [(); 512].map(|_|rng.gen());
    // This words are not adequately aligned so let's transmute the words and put them in a container.
    // the transmute is a value level, and should not incur alignment issues, so it should be safe.
    let aligned_words: BitMatrix<B128, 512> = BitMatrix(unsafe{mem::transmute(input_words)});
    // Now let's transpose the words so that we have some nice bits to work with.
    let bits: BitMatrix<B512, 128> = <B128 as Transpose<B512>>::transpose(aligned_words);
    // Our data is now bitsliced. What shall we do with it? Let is count how many bits are set.
    // Since log2(128) is 7, we will need 8 bits to avoid overflow.
    let popcnts: [B512; 8] = popcnt128(&bits.0);    

    // We need to load some more data to play with. We downshift the u8s to give them a range more pleasing to our purpose.
    let input_ints: [u8; 512] = [(); 512].map(|_|rng.gen::<u8>()>>1);
    // as before we align it,
    let aligned_ints: BitMatrix<B8, 512> = BitMatrix(unsafe{mem::transmute(input_ints)});
    // and transpose
    let ints: BitMatrix<B512, 8> = <B8 as Transpose<B512>>::transpose(aligned_ints);

    // Let us see how many of the popcnts are greater than the corresponding int.
    let (_, _, gt) = comparator(&popcnts, &ints.0);
    // now we turn the comparisons back into normal mode. We must pad them first however.
    let padded: [B512; 8] = [B512::splat(false), B512::splat(false), B512::splat(false), B512::splat(false), B512::splat(false), B512::splat(false), B512::splat(false), gt];
    // Now we can transpose back.
    let bools: BitMatrix<B8, 512> = <B512 as Transpose<B8>>::transpose(BitMatrix(padded));
    // and we can safely transmute back.
    let bools: [bool; 512] = unsafe{mem::transmute(bools)};

    // did we do this correctly? Let's test it!
    let normal_bools: [bool; 512] = input_words.into_iter().zip(input_ints.into_iter()).map(|(w,i)|w.count_ones() as u8 > i).collect::<Vec<_>>().try_into().unwrap();
    assert_eq!(bools, normal_bools);
}

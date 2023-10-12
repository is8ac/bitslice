## A crate to transpose data into and out of bitslice format, and perform various operations on bitsliced data.

See `examples/` for basic usage examples.

This crate is in early stages. While I do not expect significant API changes, performance is currently likely significantly suboptimal, and it is missing many desirable features.

## Portability
This crate has only been tested on bigendian machines.
It has intrinsics for AVX512 and for ARM Neon, and a fallback implementation which should compile to most any architecture.
For good performance, be sure to set `rustflags = ["-C", "target-cpu=native"]`

## nice to have TODO
- [ ] benchmarks
- [ ] Perf improvements
- [ ] AVX2 implementation
- [ ] Flexible counter
- [ ] Flexible bit expand
- [ ] Runtime dynamic AIG logic
- [ ] LUT mapper
- [ ] reg mapper
- [ ] proc macro compiler

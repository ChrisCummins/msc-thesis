#ifndef EXPERIMENTS_20141105_SKEL_OPT_SPACE_NBITS_H_  // NOLINT(legal/copyright)
#define EXPERIMENTS_20141105_SKEL_OPT_SPACE_NBITS_H_

// Pre-processor trick for calculating no of bits required to store an
// unsigned integer at compile time. See: http://stackoverflow.com/a/6835421
#define NBITS2(n)  ((n&2)          ? 1 : 0)
#define NBITS4(n)  ((n&(0xC))      ? (2 + NBITS2(n >> 2)) : (NBITS2(n)))
#define NBITS8(n)  ((n&0xF0)       ? (4 + NBITS4(n >> 4)) : (NBITS4(n)))
#define NBITS16(n) ((n&0xFF00)     ? (8 + NBITS8(n >> 8)) : (NBITS8(n)))
#define NBITS32(n) ((n&0xFFFF0000) ? (16 + NBITS16(n >> 16)) : (NBITS16(n)))
#define NBITS(n)   (n == 0 ? 0 : NBITS32(n) + 1)

#endif  // EXPERIMENTS_20141105_SKEL_OPT_SPACE_NBITS_H_

//! rANS (range Asymmetric Numeral Systems) entropy coder.
//!
//! Implements the standard byte-IO tANS/rANS with:
//!   - State space  L ∈ [2^23, 2^31)   (L = 1 << LOG2_L)
//!   - Renorm base  b = 256 (one byte emitted/consumed per step)
//!   - Symbol frequency precision M = 2^FREQ_BITS (16 bits → 65536)
//!
//! # Encoding (streaming, produces output in *reverse* order)
//!
//! For a symbol with cumulative frequency [fs, fs + f) out of total M:
//!   1. Renorm: while x >= (f << (32 - FREQ_BITS)), emit x & 0xFF, x >>= 8
//!   2. Encode: x = (x / f) * M + fs + (x % f)
//!
//! # Decoding (consumes bytes in forward order)
//!
//! 1. slot = x & (M - 1)
//! 2. Find symbol s where CDF[s] <= slot < CDF[s+1]
//! 3. x = f[s] * (x >> FREQ_BITS) + slot - CDF[s]
//! 4. Renorm: while x < L, x = (x << 8) | read_byte()

pub const LOG2_L: u32 = 23;
pub const L: u32 = 1 << LOG2_L;
pub const FREQ_BITS: u32 = 16;
pub const M: u32 = 1 << FREQ_BITS; // total frequency = 65536

// Encoder

pub struct RansEncoder {
    state: u32,
    output: Vec<u8>, // bytes written in *reverse* order; caller must reverse
}

impl RansEncoder {
    pub fn new() -> Self {
        Self {
            state: L,
            output: Vec::new(),
        }
    }

    /// Encode one symbol.
    ///
    /// `freq`   : probability weight of the symbol (must be > 0, ≤ M).
    /// `cum_low`: cumulative frequency before this symbol.
    pub fn encode(&mut self, freq: u32, cum_low: u32) {
        debug_assert!(freq > 0 && freq <= M);
        // Pre-condition target range: x ∈ [freq·(L/M), freq·(b·L/M))
        //   = [freq·2^(LOG2_L-FREQ_BITS), freq·2^(LOG2_L-FREQ_BITS+8))
        //   = [freq·128,                   freq·32768)
        // upper_bound = freq * 2^(LOG2_L - FREQ_BITS + 8) = freq << 15
        let upper_bound = freq << (LOG2_L - FREQ_BITS + 8);
        let mut x = self.state;
        while x >= upper_bound {
            self.output.push((x & 0xFF) as u8);
            x >>= 8;
        }
        // Encode: x' = floor(x/freq)·M + cum_low + (x mod freq)
        self.state = (x / freq) * M + cum_low + (x % freq);
    }

    /// Flush remaining state bytes (always 4 bytes, MSB first in the *reversed* stream).
    pub fn flush(mut self) -> Vec<u8> {
        let x = self.state;
        // Emit state MSB-first (will be at the *end* of the reversed output).
        self.output.push(((x >> 24) & 0xFF) as u8);
        self.output.push(((x >> 16) & 0xFF) as u8);
        self.output.push(((x >> 8) & 0xFF) as u8);
        self.output.push((x & 0xFF) as u8);
        // Reverse: encoder output must be read forward by decoder.
        self.output.reverse();
        self.output
    }
}

// Decoder

pub struct RansDecoder<'a> {
    state: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        // Read initial state (4 bytes, little-endian — matches the encoder's
        // flush which reverses the output byte stream).
        let state = (data[0] as u32)
            | ((data[1] as u32) << 8)
            | ((data[2] as u32) << 16)
            | ((data[3] as u32) << 24);
        Self {
            state,
            data,
            pos: 4,
        }
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Decode one symbol given a CDF table of length `n+1` (n symbols).
    ///
    /// `cdf`: slice of length `vocab+1` where `cdf[s]` is the cumulative
    ///        frequency up to (but not including) symbol s, and `cdf[vocab]=M`.
    ///
    /// Returns the decoded symbol index.
    pub fn decode(&mut self, cdf: &[u32]) -> usize {
        let slot = self.state & (M - 1); // slot ∈ [0, M)

        // Binary search for symbol: find s where cdf[s] <= slot < cdf[s+1].
        let mut lo = 0usize;
        let mut hi = cdf.len() - 2; // last valid symbol index
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if cdf[mid] <= slot {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let s = lo;
        let freq = cdf[s + 1] - cdf[s];

        // Update state.
        self.state = freq * (self.state >> FREQ_BITS) + slot - cdf[s];

        // Renormalise.
        while self.state < L {
            self.state = (self.state << 8) | (self.read_byte() as u32);
        }

        s
    }
}

// Frequency table helpers

/// Convert a slice of probability weights (non-negative floats) to integer
/// cumulative frequencies that sum exactly to M=65536.
///
/// Every symbol is guaranteed a frequency of at least 1.
pub fn probs_to_cdf(probs: &[f32]) -> Vec<u32> {
    let n = probs.len();
    let mut freqs: Vec<u32> = probs
        .iter()
        .map(|&p| (p * M as f32).round().max(1.0) as u32)
        .collect();

    // Adjust sum to exactly M.
    let sum: u32 = freqs.iter().sum();
    if sum < M {
        let diff = M - sum;
        // Add the deficit to the most probable symbol.
        let max_idx = freqs
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        freqs[max_idx] += diff;
    } else if sum > M {
        let mut excess = sum - M;
        // Remove excess from largest symbols without going below 1.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| freqs[b].cmp(&freqs[a]));
        for idx in order {
            if excess == 0 {
                break;
            }
            let available = freqs[idx] - 1;
            let removed = available.min(excess);
            freqs[idx] -= removed;
            excess -= removed;
        }
    }

    // Build CDF.
    let mut cdf = Vec::with_capacity(n + 1);
    cdf.push(0u32);
    for &f in &freqs {
        cdf.push(cdf.last().unwrap() + f);
    }
    cdf
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn uniform_cdf(n: usize) -> Vec<u32> {
        let f = M / n as u32;
        let mut cdf = vec![0u32; n + 1];
        for i in 0..n {
            cdf[i + 1] = cdf[i] + f;
        }
        cdf[n] = M;
        cdf
    }

    #[test]
    fn round_trip_uniform() {
        let mut rng = SmallRng::seed_from_u64(42);
        let cdf = uniform_cdf(256);
        let symbols: Vec<usize> = (0..1000).map(|_| rng.gen_range(0..256)).collect();
        let freqs: Vec<u32> = (0..256).map(|i| cdf[i + 1] - cdf[i]).collect();

        let mut enc = RansEncoder::new();
        for &s in symbols.iter().rev() {
            enc.encode(freqs[s], cdf[s]);
        }
        let bytes = enc.flush();

        let mut dec = RansDecoder::new(&bytes);
        for &expected in &symbols {
            let got = dec.decode(&cdf);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn probs_to_cdf_sums_to_m() {
        let probs: Vec<f32> = (0..1024).map(|i| (i + 1) as f32).collect();
        let s: f32 = probs.iter().sum();
        let probs_norm: Vec<f32> = probs.iter().map(|&p| p / s).collect();
        let cdf = probs_to_cdf(&probs_norm);
        assert_eq!(*cdf.last().unwrap(), M);
    }
}

//! video_compressor — Rust CLI entry point.
//!
//! This binary provides a fast rANS encode / decode path that can be used as
//! a drop-in replacement for the Python/constriction range coder.
//!
//! # Binary I/O protocol (stdin → stdout)
//!
//! ## Encode mode  (`video_compressor encode`)
//!
//!   stdin:
//!     [4 bytes LE]  num_frames  (e.g. 1200)
//!     [4 bytes LE]  num_tokens  (e.g. 128)
//!     [4 bytes LE]  vocab_size  (e.g. 1024)
//!     for each frame:
//!       [vocab_size × 4 bytes LE f32]  probability table (sums ≈ 1 per token pos)
//!                                      shape: (num_tokens, vocab_size) row-major
//!       [num_tokens × 4 bytes LE i32]  symbol IDs
//!
//!   stdout:
//!     [4 bytes LE]  num_frames
//!     [4 bytes LE]  num_compressed_bytes
//!     [num_compressed_bytes bytes]  rANS bitstream
//!
//! ## Decode mode  (`video_compressor decode`)
//!
//!   stdin  = stdout format from encode, followed by the same probability tables
//!            (in the same frame order) as used during encoding.
//!   stdout = raw symbol IDs as (num_frames × num_tokens) × 4-byte LE i32 words.

mod rans_coder;

use std::io::{self, Read, Write};

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn write_u32_le(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}
fn write_i32_le(v: i32) -> [u8; 4] {
    v.to_le_bytes()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("encode");

    let mut stdin_buf = Vec::new();
    io::stdin().read_to_end(&mut stdin_buf).expect("stdin read");

    let num_frames = read_u32_le(&stdin_buf, 0) as usize;
    let num_tokens = read_u32_le(&stdin_buf, 4) as usize;
    let vocab_size = read_u32_le(&stdin_buf, 8) as usize;

    match mode {
        "encode" => encode_mode(&stdin_buf[12..], num_frames, num_tokens, vocab_size),
        "decode" => decode_mode(&stdin_buf[12..], num_frames, num_tokens, vocab_size),
        other => eprintln!("Unknown mode '{}'; use encode or decode.", other),
    }
}

fn encode_mode(data: &[u8], num_frames: usize, num_tokens: usize, vocab_size: usize) {
    let probs_per_frame = num_tokens * vocab_size;
    let bytes_per_prob_block = probs_per_frame * 4;
    let bytes_per_sym_block = num_tokens * 4;
    let _bytes_per_frame = bytes_per_prob_block + bytes_per_sym_block;

    // Collect symbols and CDFs.
    let mut all_cdfs: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_frames);
    let mut all_syms: Vec<Vec<usize>> = Vec::with_capacity(num_frames);

    let mut off = 0usize;
    for _ in 0..num_frames {
        let mut frame_cdfs = Vec::with_capacity(num_tokens);
        for j in 0..num_tokens {
            let probs: Vec<f32> = (0..vocab_size)
                .map(|v| read_f32_le(data, off + (j * vocab_size + v) * 4))
                .collect();
            frame_cdfs.push(rans_coder::probs_to_cdf(&probs));
        }
        off += bytes_per_prob_block;

        let syms: Vec<usize> = (0..num_tokens)
            .map(|j| read_u32_le(data, off + j * 4) as usize)
            .collect();
        off += bytes_per_sym_block;

        all_cdfs.push(frame_cdfs);
        all_syms.push(syms);
    }

    // Encode backwards (rANS is LIFO).
    let mut enc = rans_coder::RansEncoder::new();
    for f in (0..num_frames).rev() {
        for j in (0..num_tokens).rev() {
            let s = all_syms[f][j];
            let cdf = &all_cdfs[f][j];
            let freq = cdf[s + 1] - cdf[s];
            let cum_low = cdf[s];
            enc.encode(freq, cum_low);
        }
    }
    let compressed = enc.flush();

    let mut stdout = io::stdout();
    stdout.write_all(&write_u32_le(num_frames as u32)).unwrap();
    stdout
        .write_all(&write_u32_le(compressed.len() as u32))
        .unwrap();
    stdout.write_all(&compressed).unwrap();
}

fn decode_mode(data: &[u8], num_frames: usize, num_tokens: usize, vocab_size: usize) {
    // First: read the compressed bitstream header.
    let stored_frames = read_u32_le(data, 0) as usize;
    let comp_len = read_u32_le(data, 4) as usize;
    assert_eq!(stored_frames, num_frames);

    let bitstream = &data[8..8 + comp_len];
    let probs_off = 8 + comp_len;

    // Read probability tables (same order as during encoding).
    let mut all_cdfs: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_frames);
    let mut off = probs_off;
    for _ in 0..num_frames {
        let mut frame_cdfs = Vec::with_capacity(num_tokens);
        for j in 0..num_tokens {
            let probs: Vec<f32> = (0..vocab_size)
                .map(|v| read_f32_le(data, off + (j * vocab_size + v) * 4))
                .collect();
            frame_cdfs.push(rans_coder::probs_to_cdf(&probs));
        }
        off += num_tokens * vocab_size * 4;
        all_cdfs.push(frame_cdfs);
    }

    let mut dec = rans_coder::RansDecoder::new(bitstream);
    let mut stdout = io::stdout();
    for frame_cdfs in all_cdfs.iter() {
        for cdf in frame_cdfs.iter() {
            let s = dec.decode(cdf);
            stdout.write_all(&write_i32_le(s as i32)).unwrap();
        }
    }
}

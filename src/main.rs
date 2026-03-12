use std::env;
use std::io::{self, Read, Write};
use std::process;

fn usage() -> ! {
    eprintln!(
        "Usage:
  video_compressor pack10
  video_compressor unpack10 <count>

Commands read from stdin and write to stdout.
`pack10` expects little-endian u16 values in [0, 1023].
`unpack10` emits little-endian u16 values."
    );
    process::exit(1);
}

fn read_all_stdin() -> io::Result<Vec<u8>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    Ok(input)
}

fn pack10(input: &[u8]) -> io::Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length must be a multiple of 2 bytes",
        ));
    }

    let mut output = Vec::with_capacity((input.len() / 2 * 10).div_ceil(8));
    let mut bit_buffer: u32 = 0;
    let mut bits_in_buffer = 0usize;

    for chunk in input.chunks_exact(2) {
        let value = u16::from_le_bytes([chunk[0], chunk[1]]);
        if value >= 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("token value {value} is outside 10-bit range"),
            ));
        }

        bit_buffer |= u32::from(value) << bits_in_buffer;
        bits_in_buffer += 10;

        while bits_in_buffer >= 8 {
            output.push((bit_buffer & 0xFF) as u8);
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }

    if bits_in_buffer > 0 {
        output.push((bit_buffer & 0xFF) as u8);
    }

    Ok(output)
}

fn unpack10(input: &[u8], count: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(count * 2);
    let mut input_index = 0usize;
    let mut bit_buffer: u32 = 0;
    let mut bits_in_buffer = 0usize;

    for _ in 0..count {
        while bits_in_buffer < 10 {
            if input_index >= input.len() {
                panic!("not enough packed bytes to unpack {count} values");
            }
            bit_buffer |= u32::from(input[input_index]) << bits_in_buffer;
            bits_in_buffer += 8;
            input_index += 1;
        }

        let value = (bit_buffer & 0x3FF) as u16;
        output.extend_from_slice(&value.to_le_bytes());
        bit_buffer >>= 10;
        bits_in_buffer -= 10;
    }

    output
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }

    let result = match args[1].as_str() {
        "pack10" => read_all_stdin().and_then(|input| pack10(&input)),
        "unpack10" => {
            if args.len() != 3 {
                usage();
            }
            let count = args[2]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid count: {}", args[2]));
            read_all_stdin().map(|input| unpack10(&input, count))
        }
        _ => usage(),
    };

    match result {
        Ok(output) => {
            io::stdout().write_all(&output).unwrap();
        }
        Err(err) => {
            eprintln!("error: {err}");
            process::exit(1);
        }
    }
}

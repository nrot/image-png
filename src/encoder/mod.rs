mod sync_encoder;

pub use sync_encoder::{Encoder, Result, EncodingError, StreamWriter, Writer};
pub(crate) use sync_encoder::write_chunk;

#[cfg(feature="async")]
mod async_encoder;

use crate::common::Compression;

/// Mod to encapsulate the converters depending on the `deflate` crate.
///
/// Since this only contains trait impls, there is no need to make this public, they are simply
/// available when the mod is compiled as well.
impl Compression {
    fn to_options(self) -> deflate::CompressionOptions {
        match self {
            Compression::Default => deflate::CompressionOptions::default(),
            Compression::Fast => deflate::CompressionOptions::fast(),
            Compression::Best => deflate::CompressionOptions::high(),
            Compression::Huffman => deflate::CompressionOptions::huffman_only(),
            Compression::Rle => deflate::CompressionOptions::rle(),
        }
    }
}

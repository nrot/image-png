mod stream;
mod sync_decoder;
mod zlib;

use std::ops::Range;

use self::stream::FormatErrorInner;
pub use self::stream::{Decoded, DecodingError, StreamingDecoder};

pub use self::sync_decoder::{Decoder, Reader};

use crate::common::{BitDepth, ColorType, Info};

use crate::utils;

#[cfg(feature = "async")]
mod async_decoder;

/*
pub enum InterlaceHandling {
    /// Outputs the raw rows
    RawRows,
    /// Fill missing the pixels from the existing ones
    Rectangle,
    /// Only fill the needed pixels
    Sparkle
}
*/

/// Output info.
///
/// This describes one particular frame of the image that was written into the output buffer.
#[derive(Debug, PartialEq, Eq)]
pub struct OutputInfo {
    /// The pixel width of this frame.
    pub width: u32,
    /// The pixel height of this frame.
    pub height: u32,
    /// The chosen output color type.
    pub color_type: ColorType,
    /// The chosen output bit depth.
    pub bit_depth: BitDepth,
    /// The byte count of each scan line in the image.
    pub line_size: usize,
}

impl OutputInfo {
    /// Returns the size needed to hold a decoded frame
    /// If the output buffer was larger then bytes after this count should be ignored. They may
    /// still have been changed.
    pub fn buffer_size(&self) -> usize {
        self.line_size * self.height as usize
    }
}

#[derive(Clone, Copy, Debug)]
/// Limits on the resources the `Decoder` is allowed too use
pub struct Limits {
    /// maximum number of bytes the decoder is allowed to allocate, default is 64Mib
    pub bytes: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            bytes: 1024 * 1024 * 64,
        }
    }
}

/// A row of data with interlace information attached.
#[derive(Clone, Copy, Debug)]
pub struct InterlacedRow<'data> {
    data: &'data [u8],
    interlace: InterlaceInfo,
}

impl<'data> InterlacedRow<'data> {
    pub fn data(&self) -> &'data [u8] {
        self.data
    }

    pub fn interlace(&self) -> InterlaceInfo {
        self.interlace
    }
}

/// PNG (2003) specifies two interlace modes, but reserves future extensions.
#[derive(Clone, Copy, Debug)]
pub enum InterlaceInfo {
    /// the null method means no interlacing
    Null,
    /// Adam7 derives its name from doing 7 passes over the image, only decoding a subset of all pixels in each pass.
    /// The following table shows pictorially what parts of each 8x8 area of the image is found in each pass:
    ///
    /// 1 6 4 6 2 6 4 6
    /// 7 7 7 7 7 7 7 7
    /// 5 6 5 6 5 6 5 6
    /// 7 7 7 7 7 7 7 7
    /// 3 6 4 6 3 6 4 6
    /// 7 7 7 7 7 7 7 7
    /// 5 6 5 6 5 6 5 6
    /// 7 7 7 7 7 7 7 7
    Adam7 { pass: u8, line: u32, width: u32 },
}

/// A row of data without interlace information.
#[derive(Clone, Copy, Debug)]
pub struct Row<'data> {
    data: &'data [u8],
}

impl<'data> Row<'data> {
    pub fn data(&self) -> &'data [u8] {
        self.data
    }
}

#[derive(Clone)]
enum InterlaceIter {
    None(Range<u32>),
    Adam7(utils::Adam7Iterator),
}

/// Denote a frame as given by sequence numbers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SubframeIdx {
    /// The info has not yet been decoded.
    Uninit,
    /// The initial frame in an IDAT chunk without fcTL chunk applying to it.
    /// Note that this variant precedes `Some` as IDAT frames precede fdAT frames and all fdAT
    /// frames must have a fcTL applying to it.
    Initial,
    /// An IDAT frame with fcTL or an fdAT frame.
    Some(u32),
    /// The past-the-end index.
    End,
}

/// The subframe specific information.
///
/// In APNG the frames are constructed by combining previous frame and a new subframe (through a
/// combination of `dispose_op` and `overlay_op`). These sub frames specify individual dimension
/// information and reuse the global interlace options. This struct encapsulates the state of where
/// in a particular IDAT-frame or subframe we are.
struct SubframeInfo {
    width: u32,
    height: u32,
    rowlen: usize,
    interlace: InterlaceIter,
    consumed_and_flushed: bool,
}

impl SubframeInfo {
    fn not_yet_init() -> Self {
        SubframeInfo {
            width: 0,
            height: 0,
            rowlen: 0,
            interlace: InterlaceIter::None(0..0),
            consumed_and_flushed: false,
        }
    }

    fn new(info: &Info) -> Self {
        // The apng fctnl overrides width and height.
        // All other data is set by the main info struct.
        let (width, height) = if let Some(fc) = info.frame_control {
            (fc.width, fc.height)
        } else {
            (info.width, info.height)
        };

        let interlace = if info.interlaced {
            InterlaceIter::Adam7(utils::Adam7Iterator::new(width, height))
        } else {
            InterlaceIter::None(0..height)
        };

        SubframeInfo {
            width,
            height,
            rowlen: info.raw_row_length_from_width(width),
            interlace,
            consumed_and_flushed: false,
        }
    }
}

fn expand_gray_u8(buffer: &mut [u8], info: &Info) {
    let rescale = true;
    let scaling_factor = if rescale {
        (255) / ((1u16 << info.bit_depth as u8) - 1) as u8
    } else {
        1
    };
    if let Some(ref trns) = info.trns {
        utils::unpack_bits(buffer, 2, info.bit_depth as u8, |pixel, chunk| {
            if pixel == trns[0] {
                chunk[1] = 0
            } else {
                chunk[1] = 0xFF
            }
            chunk[0] = pixel * scaling_factor
        })
    } else {
        utils::unpack_bits(buffer, 1, info.bit_depth as u8, |val, chunk| {
            chunk[0] = val * scaling_factor
        })
    }
}

fn expand_paletted(buffer: &mut [u8], info: &Info) -> Result<(), DecodingError> {
    if let Some(palette) = info.palette.as_ref() {
        if let BitDepth::Sixteen = info.bit_depth {
            // This should have been caught earlier but let's check again. Can't hurt.
            Err(DecodingError::Format(
                FormatErrorInner::InvalidColorBitDepth {
                    color: ColorType::Indexed,
                    depth: BitDepth::Sixteen,
                }
                .into(),
            ))
        } else {
            let black = [0, 0, 0];
            if let Some(ref trns) = info.trns {
                utils::unpack_bits(buffer, 4, info.bit_depth as u8, |i, chunk| {
                    let (rgb, a) = (
                        palette
                            .get(3 * i as usize..3 * i as usize + 3)
                            .unwrap_or(&black),
                        *trns.get(i as usize).unwrap_or(&0xFF),
                    );
                    chunk[0] = rgb[0];
                    chunk[1] = rgb[1];
                    chunk[2] = rgb[2];
                    chunk[3] = a;
                });
            } else {
                utils::unpack_bits(buffer, 3, info.bit_depth as u8, |i, chunk| {
                    let rgb = palette
                        .get(3 * i as usize..3 * i as usize + 3)
                        .unwrap_or(&black);
                    chunk[0] = rgb[0];
                    chunk[1] = rgb[1];
                    chunk[2] = rgb[2];
                })
            }
            Ok(())
        }
    } else {
        Err(DecodingError::Format(
            FormatErrorInner::PaletteRequired.into(),
        ))
    }
}

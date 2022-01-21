use super::stream::{Decoded, DecodingError, StreamingDecoder};
use super::stream::{FormatErrorInner, CHUNCK_BUFFER_SIZE};

use std::io::Write;
use std::mem;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};

use crate::chunk;
use crate::common::{
    BitDepth, BytesPerPixel, ColorType, Info, ParameterErrorKind, Transformations,
};
use crate::filter::{unfilter, FilterType};
use crate::utils;

use super::{
    expand_gray_u8, expand_paletted, InterlaceInfo, InterlaceIter, InterlacedRow, Limits,
    OutputInfo, Row, SubframeIdx, SubframeInfo,
};

macro_rules! get_info(
    ($this:expr) => {
        $this.decoder.info().unwrap()
    }
);

/// PNG Decoder
#[derive(Debug)]
pub struct AsyncDecoder<R: Unpin> {
    /// Reader
    r: R,
    /// Output transformations
    transform: Transformations,
    /// Limits on resources the Decoder is allowed to use
    limits: Limits,
}

impl<R: AsyncRead + Unpin> AsyncDecoder<R> {
    /// Create a new decoder configuration with default limits.
    pub fn new(r: R) -> AsyncDecoder<R> {
        AsyncDecoder::new_with_limits(r, Limits::default())
    }
    /// Create a new decoder configuration with custom limits.
    pub fn new_with_limits(r: R, limits: Limits) -> AsyncDecoder<R> {
        AsyncDecoder {
            r,
            transform: Transformations::IDENTITY,
            limits,
        }
    }

    /// Reads all meta data until the first IDAT chunk
    pub async fn read_info(self) -> Result<AsyncReader<R>, DecodingError> {
        let mut reader =
            AsyncReader::new(self.r, StreamingDecoder::new(), self.transform, self.limits);
        reader.init().await?;

        let color_type = reader.info().color_type;
        let bit_depth = reader.info().bit_depth;
        if color_type.is_combination_invalid(bit_depth) {
            return Err(DecodingError::Format(
                FormatErrorInner::InvalidColorBitDepth {
                    color: color_type,
                    depth: bit_depth,
                }
                .into(),
            ));
        }

        // Check if the output buffer can be represented at all.
        if reader.checked_output_buffer_size().is_none() {
            return Err(DecodingError::LimitsExceeded);
        }

        Ok(reader)
    }

    /// Limit resource usage.
    ///
    /// Note that your allocations, e.g. when reading into a pre-allocated buffer, are __NOT__
    /// considered part of the limits. Nevertheless, required intermediate buffers such as for
    /// singular lines is checked against the limit.
    ///
    /// Note that this is a best-effort basis.
    ///
    /// ```
    /// use std::fs::File;
    /// use png::{Decoder, Limits};
    /// // This image is 32×32, 1bit per pixel. The reader buffers one row which requires 4 bytes.
    /// let mut limits = Limits::default();
    /// limits.bytes = 3;
    /// let mut decoder = Decoder::new_with_limits(File::open("tests/pngsuite/basi0g01.png").unwrap(), limits);
    /// assert!(decoder.read_info().is_err());
    ///
    /// // This image is 32x32 pixels, so the decoder will allocate less than 10Kib
    /// let mut limits = Limits::default();
    /// limits.bytes = 10*1024;
    /// let mut decoder = Decoder::new_with_limits(File::open("tests/pngsuite/basi0g01.png").unwrap(), limits);
    /// assert!(decoder.read_info().is_ok());
    /// ```
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Set the allowed and performed transformations.
    ///
    /// A transformation is a pre-processing on the raw image data modifying content or encoding.
    /// Many options have an impact on memory or CPU usage during decoding.
    pub fn set_transformations(&mut self, transform: Transformations) {
        self.transform = transform;
    }
}

struct AsyncReadDecoder<R: Unpin> {
    reader: BufReader<R>,
    decoder: StreamingDecoder,
    at_eof: bool,
}

impl<R: AsyncReadExt + Unpin> Unpin for AsyncReadDecoder<R> {}

impl<R: AsyncReadExt + Unpin> AsyncReadDecoder<R> {
    /// Returns the next decoded chunk. If the chunk is an ImageData chunk, its contents are written
    /// into image_data.
    async fn decode_next(
        &mut self,
        image_data: &mut Vec<u8>,
    ) -> Result<Option<Decoded>, DecodingError> {
        while !self.at_eof {
            let (consumed, result) = {
                let buf = self.reader.fill_buf().await?;
                if buf.is_empty() {
                    return Err(DecodingError::Format(
                        FormatErrorInner::UnexpectedEof.into(),
                    ));
                }
                self.decoder.update(buf, image_data)?
            };
            self.reader.consume(consumed);
            match result {
                Decoded::Nothing => (),
                Decoded::ImageEnd => self.at_eof = true,
                result => return Ok(Some(result)),
            }
        }
        Ok(None)
    }

    async fn finished_decoding(&mut self) -> Result<(), DecodingError> {
        while !self.at_eof {
            let buf = self.reader.fill_buf().await?;
            if buf.is_empty() {
                return Err(DecodingError::Format(
                    FormatErrorInner::UnexpectedEof.into(),
                ));
            }
            let (consumed, event) = self.decoder.update(buf, &mut vec![])?;
            self.reader.consume(consumed);
            match event {
                Decoded::Nothing => (),
                Decoded::ImageEnd => self.at_eof = true,
                // ignore more data
                Decoded::ChunkComplete(_, _) | Decoded::ChunkBegin(_, _) | Decoded::ImageData => {}
                Decoded::ImageDataFlushed => return Ok(()),
                Decoded::PartialChunk(_) => {}
                new => unreachable!("{:?}", new),
            }
        }

        Err(DecodingError::Format(
            FormatErrorInner::UnexpectedEof.into(),
        ))
    }

    fn info(&self) -> Option<&Info> {
        self.decoder.info.as_ref()
    }
}

/// PNG reader (mostly high-level interface)
///
/// Provides a high level that iterates over lines or whole images.
pub struct AsyncReader<R: Unpin> {
    decoder: AsyncReadDecoder<R>,
    bpp: BytesPerPixel,
    subframe: SubframeInfo,
    /// Number of frame control chunks read.
    /// By the APNG specification the total number must equal the count specified in the animation
    /// control chunk. The IDAT image _may_ have such a chunk applying to it.
    fctl_read: u32,
    next_frame: SubframeIdx,
    /// Previous raw line
    prev: Vec<u8>,
    /// Current raw line
    current: Vec<u8>,
    /// Start index of the current scan line.
    scan_start: usize,
    /// Output transformations
    transform: Transformations,
    /// Processed line
    processed: Vec<u8>,
    limits: Limits,
}

impl<'a, R: AsyncRead + Unpin> AsyncReader<R> {
    /// Creates a new PNG reader
    fn new(r: R, d: StreamingDecoder, t: Transformations, limits: Limits) -> AsyncReader<R> {
        AsyncReader {
            decoder: AsyncReadDecoder {
                reader: BufReader::with_capacity(CHUNCK_BUFFER_SIZE, r),
                decoder: d,
                at_eof: false,
            },
            bpp: BytesPerPixel::One,
            subframe: SubframeInfo::not_yet_init(),
            fctl_read: 0,
            next_frame: SubframeIdx::Initial,
            prev: Vec::new(),
            current: Vec::new(),
            scan_start: 0,
            transform: t,
            processed: Vec::new(),
            limits,
        }
    }

    /// Reads all meta data until the next frame data starts.
    /// Requires IHDR before the IDAT and fcTL before fdAT.
    async fn init(&mut self) -> Result<OutputInfo, DecodingError> {
        if self.next_frame == self.subframe_idx() {
            return Ok(self.output_info());
        } else if self.next_frame == SubframeIdx::End {
            return Err(DecodingError::Parameter(
                ParameterErrorKind::PolledAfterEndOfImage.into(),
            ));
        }

        loop {
            match self.decoder.decode_next(&mut Vec::new()).await? {
                Some(Decoded::ChunkBegin(_, chunk::IDAT))
                | Some(Decoded::ChunkBegin(_, chunk::fdAT)) => break,
                Some(Decoded::FrameControl(_)) => {
                    self.subframe = SubframeInfo::new(self.info());
                    // The next frame is the one to which this chunk applies.
                    self.next_frame = SubframeIdx::Some(self.fctl_read);
                    // TODO: what about overflow here? That would imply there are more fctl chunks
                    // than can be specified in the animation control but also that we have read
                    // several gigabytes of data.
                    self.fctl_read += 1;
                }
                None => {
                    return Err(DecodingError::Format(
                        FormatErrorInner::MissingImageData.into(),
                    ))
                }
                Some(Decoded::Header { .. }) => {
                    self.validate_buffer_sizes()?;
                }
                // Ignore all other chunk events. Any other chunk may be between IDAT chunks, fdAT
                // chunks and their control chunks.
                _ => {}
            }
        }
        {
            let info = match self.decoder.info() {
                Some(info) => info,
                None => return Err(DecodingError::Format(FormatErrorInner::MissingIhdr.into())),
            };
            self.bpp = info.bpp_in_prediction();
            // Check if the output buffer can be represented at all.
            // Now we can init the subframe info.
            // TODO: reuse the results obtained during the above check.
            self.subframe = SubframeInfo::new(info);
        }
        self.allocate_out_buf()?;
        self.prev = vec![0; self.subframe.rowlen];
        Ok(self.output_info())
    }

    fn output_info(&self) -> OutputInfo {
        let width = self.subframe.width;
        let height = self.subframe.height;

        let (color_type, bit_depth) = self.output_color_type();

        OutputInfo {
            width,
            height,
            color_type,
            bit_depth,
            line_size: self.output_line_size(width),
        }
    }

    fn reset_current(&mut self) {
        self.current.clear();
        self.scan_start = 0;
    }

    /// Get information on the image.
    ///
    /// The structure will change as new frames of an animated image are decoded.
    pub fn info(&self) -> &Info {
        self.decoder.info().unwrap()
    }

    /// Get the subframe index of the current info.
    fn subframe_idx(&self) -> SubframeIdx {
        let info = match self.decoder.info() {
            None => return SubframeIdx::Uninit,
            Some(info) => info,
        };

        match info.frame_control() {
            None => SubframeIdx::Initial,
            Some(_) => SubframeIdx::Some(self.fctl_read - 1),
        }
    }

    /// Call after decoding an image, to advance expected state to the next.
    fn finished_frame(&mut self) {
        // Should only be called after frame is done, so we have an info.
        let info = self.info();

        let past_end_subframe = match info.animation_control() {
            // a non-APNG has no subframes
            None => 0,
            // otherwise the count is the past-the-end index. It can not be 0 per spec.
            Some(ac) => ac.num_frames,
        };

        self.next_frame = match self.next_frame {
            SubframeIdx::Uninit => unreachable!("Next frame can never be initial"),
            SubframeIdx::End => unreachable!("Next frame called when already at image end"),
            // Reached the end of non-animated image.
            SubframeIdx::Initial if past_end_subframe == 0 => SubframeIdx::End,
            // An animated image, expecting first subframe.
            SubframeIdx::Initial => SubframeIdx::Some(0),
            // This was the last subframe, slightly fuzzy condition in case of programmer error.
            SubframeIdx::Some(idx) if past_end_subframe <= idx + 1 => SubframeIdx::End,
            // Expecting next subframe.
            SubframeIdx::Some(idx) => SubframeIdx::Some(idx + 1),
        }
    }

    /// Decodes the next frame into `buf`.
    ///
    /// Note that this decodes raw subframes that need to be mixed according to blend-op and
    /// dispose-op by the caller.
    ///
    /// The caller must always provide a buffer large enough to hold a complete frame (the APNG
    /// specification restricts subframes to the dimensions given in the image header). The region
    /// that has been written be checked afterwards by calling `info` after a successful call and
    /// inspecting the `frame_control` data. This requirement may be lifted in a later version of
    /// `png`.
    ///
    /// Output lines will be written in row-major, packed matrix with width and height of the read
    /// frame (or subframe), all samples are in big endian byte order where this matters.
    pub async fn next_frame(&mut self, buf: &mut [u8]) -> Result<OutputInfo, DecodingError> {
        // Advance until we've read the info / fcTL for this frame.
        let info = self.init().await?;
        // TODO 16 bit
        let (color_type, bit_depth) = self.output_color_type();
        if buf.len() < self.output_buffer_size() {
            return Err(DecodingError::Parameter(
                ParameterErrorKind::ImageBufferSize {
                    expected: buf.len(),
                    actual: self.output_buffer_size(),
                }
                .into(),
            ));
        }

        self.reset_current();
        let width = self.info().width;
        if self.info().interlaced {
            while let Some(InterlacedRow {
                data: row,
                interlace,
                ..
            }) = self.next_interlaced_row().await?
            {
                let (line, pass) = match interlace {
                    InterlaceInfo::Adam7 { line, pass, .. } => (line, pass),
                    InterlaceInfo::Null => unreachable!("expected interlace information"),
                };
                let samples = color_type.samples() as u8;
                utils::expand_pass(buf, width, row, pass, line, samples * (bit_depth as u8));
            }
        } else {
            let mut len = 0;
            while let Some(Row { data: row, .. }) = self.next_row().await? {
                len += (&mut buf[len..]).write(row)?;
            }
        }
        // Advance over the rest of data for this (sub-)frame.
        if !self.subframe.consumed_and_flushed {
            self.decoder.finished_decoding().await?;
        }
        // Advance our state to expect the next frame.
        self.finished_frame();

        Ok(info)
    }

    /// Returns the next processed row of the image
    pub async fn next_row(&'a mut self) -> Result<Option<Row<'a>>, DecodingError> {
        self.next_interlaced_row()
            .await
            .map(move |v| v.map(|v| Row { data: v.data }))
    }

    /// Returns the next processed row of the image
    pub async fn next_interlaced_row(
        &'a mut self,
    ) -> Result<Option<InterlacedRow<'a>>, DecodingError> {
        match self.next_interlaced_row_impl().await {
            Err(err) => Err(err),
            Ok(None) => Ok(None),
            Ok(s) => Ok(s),
        }
    }

    /// Fetch the next interlaced row and filter it according to our own transformations.
    async fn next_interlaced_row_impl(
        &'a mut self,
    ) -> Result<Option<InterlacedRow<'a>>, DecodingError> {
        use crate::common::ColorType::*;
        let transform = self.transform;

        if transform == Transformations::IDENTITY {
            return self.next_raw_interlaced_row().await;
        }

        // swap buffer to circumvent borrow issues
        let mut buffer = mem::replace(&mut self.processed, Vec::new());
        let (got_next, adam7) = if let Some(row) = self.next_raw_interlaced_row().await? {
            (&mut buffer[..]).write_all(row.data)?;
            (true, row.interlace)
        } else {
            (false, InterlaceInfo::Null)
        };
        // swap back
        let _ = mem::replace(&mut self.processed, buffer);

        if !got_next {
            return Ok(None);
        }

        let (color_type, bit_depth, trns) = {
            let info = self.info();
            (info.color_type, info.bit_depth as u8, info.trns.is_some())
        };
        let output_buffer = if let InterlaceInfo::Adam7 { width, .. } = adam7 {
            let width = self
                .line_size(width)
                .expect("Adam7 interlaced rows are shorter than the buffer.");
            &mut self.processed[..width]
        } else {
            &mut *self.processed
        };

        let mut len = output_buffer.len();
        if transform.contains(Transformations::EXPAND) {
            match color_type {
                Indexed => expand_paletted(output_buffer, get_info!(self))?,
                Grayscale | GrayscaleAlpha if bit_depth < 8 => {
                    expand_gray_u8(output_buffer, get_info!(self))
                }
                Grayscale | Rgb if trns => {
                    let channels = color_type.samples();
                    let trns = get_info!(self).trns.as_ref().unwrap();
                    if bit_depth == 8 {
                        utils::expand_trns_line(output_buffer, &*trns, channels);
                    } else {
                        utils::expand_trns_line16(output_buffer, &*trns, channels);
                    }
                }
                _ => (),
            }
        }

        if bit_depth == 16 && transform.intersects(Transformations::STRIP_16) {
            len /= 2;
            for i in 0..len {
                output_buffer[i] = output_buffer[2 * i];
            }
        }

        Ok(Some(InterlacedRow {
            data: &output_buffer[..len],
            interlace: adam7,
        }))
    }

    /// Returns the color type and the number of bits per sample
    /// of the data returned by `Reader::next_row` and Reader::frames`.
    pub fn output_color_type(&self) -> (ColorType, BitDepth) {
        use crate::common::ColorType::*;
        let t = self.transform;
        let info = self.info();
        if t == Transformations::IDENTITY {
            (info.color_type, info.bit_depth)
        } else {
            let bits = match info.bit_depth as u8 {
                16 if t.intersects(Transformations::STRIP_16) => 8,
                n if n < 8 && t.contains(Transformations::EXPAND) => 8,
                n => n,
            };
            let color_type = if t.contains(Transformations::EXPAND) {
                let has_trns = info.trns.is_some();
                match info.color_type {
                    Grayscale if has_trns => GrayscaleAlpha,
                    Rgb if has_trns => Rgba,
                    Indexed if has_trns => Rgba,
                    Indexed => Rgb,
                    ct => ct,
                }
            } else {
                info.color_type
            };
            (color_type, BitDepth::from_u8(bits).unwrap())
        }
    }

    /// Returns the number of bytes required to hold a deinterlaced image frame
    /// that is decoded using the given input transformations.
    pub fn output_buffer_size(&self) -> usize {
        let (width, height) = self.info().size();
        let size = self.output_line_size(width);
        size * height as usize
    }

    fn validate_buffer_sizes(&self) -> Result<(), DecodingError> {
        // Check if the decoding buffer of a single raw line has a valid size.
        if self.info().checked_raw_row_length().is_none() {
            return Err(DecodingError::LimitsExceeded);
        }

        // Check if the output buffer has a valid size.
        if self.checked_output_buffer_size().is_none() {
            return Err(DecodingError::LimitsExceeded);
        }

        Ok(())
    }

    fn checked_output_buffer_size(&self) -> Option<usize> {
        let (width, height) = self.info().size();
        let (color, depth) = self.output_color_type();
        let rowlen = color.checked_raw_row_length(depth, width)? - 1;
        let height: usize = std::convert::TryFrom::try_from(height).ok()?;
        rowlen.checked_mul(height)
    }

    /// Returns the number of bytes required to hold a deinterlaced row.
    pub fn output_line_size(&self, width: u32) -> usize {
        let (color, depth) = self.output_color_type();
        color.raw_row_length_from_width(depth, width) - 1
    }

    /// Returns the number of bytes required to decode a deinterlaced row.
    fn line_size(&self, width: u32) -> Option<usize> {
        use crate::common::ColorType::*;
        let t = self.transform;
        let info = self.info();
        let trns = info.trns.is_some();

        let expanded = if info.bit_depth == BitDepth::Sixteen {
            BitDepth::Sixteen
        } else {
            BitDepth::Eight
        };
        // The color type and depth representing the decoded line
        // TODO 16 bit
        let (color, depth) = match info.color_type {
            Indexed if trns && t.contains(Transformations::EXPAND) => (Rgba, expanded),
            Indexed if t.contains(Transformations::EXPAND) => (Rgb, expanded),
            Rgb if trns && t.contains(Transformations::EXPAND) => (Rgba, expanded),
            Grayscale if trns && t.contains(Transformations::EXPAND) => (GrayscaleAlpha, expanded),
            Grayscale if t.contains(Transformations::EXPAND) => (Grayscale, expanded),
            GrayscaleAlpha if t.contains(Transformations::EXPAND) => (GrayscaleAlpha, expanded),
            other => (other, info.bit_depth),
        };

        // Without the filter method byte
        color.checked_raw_row_length(depth, width).map(|n| n - 1)
    }

    fn allocate_out_buf(&mut self) -> Result<(), DecodingError> {
        let width = self.subframe.width;
        let bytes = self.limits.bytes;
        let buflen = match self.line_size(width) {
            Some(buflen) if buflen <= bytes => buflen,
            // Should we differentiate between platform limits and others?
            _ => return Err(DecodingError::LimitsExceeded),
        };
        self.processed.resize(buflen, 0u8);
        Ok(())
    }

    fn next_pass(&mut self) -> Option<(usize, InterlaceInfo)> {
        match self.subframe.interlace {
            InterlaceIter::Adam7(ref mut adam7) => {
                let last_pass = adam7.current_pass();
                let (pass, line, width) = adam7.next()?;
                let rowlen = self.info().raw_row_length_from_width(width);
                if last_pass != pass {
                    self.prev.clear();
                    self.prev.resize(rowlen, 0u8);
                }
                Some((rowlen, InterlaceInfo::Adam7 { pass, line, width }))
            }
            InterlaceIter::None(ref mut height) => {
                let _ = height.next()?;
                Some((self.subframe.rowlen, InterlaceInfo::Null))
            }
        }
    }

    /// Returns the next raw scanline of the image interlace pass.
    /// The scanline is filtered against the previous scanline according to the specification.
    async fn next_raw_interlaced_row(
        &mut self,
    ) -> Result<Option<InterlacedRow<'_>>, DecodingError> {
        let bpp = self.bpp;
        let (rowlen, passdata) = match self.next_pass() {
            Some((rowlen, passdata)) => (rowlen, passdata),
            None => return Ok(None),
        };
        loop {
            if self.current.len() - self.scan_start >= rowlen {
                let row = &mut self.current[self.scan_start..];
                let filter = match FilterType::from_u8(row[0]) {
                    None => {
                        self.scan_start += rowlen;
                        return Err(DecodingError::Format(
                            FormatErrorInner::UnknownFilterMethod(row[0]).into(),
                        ));
                    }
                    Some(filter) => filter,
                };

                if let Err(message) =
                    unfilter(filter, bpp, &self.prev[1..rowlen], &mut row[1..rowlen])
                {
                    return Err(DecodingError::Format(
                        FormatErrorInner::BadFilter(message).into(),
                    ));
                }

                self.prev[..rowlen].copy_from_slice(&row[..rowlen]);
                self.scan_start += rowlen;

                return Ok(Some(InterlacedRow {
                    data: &self.prev[1..rowlen],
                    interlace: passdata,
                }));
            } else {
                if self.subframe.consumed_and_flushed {
                    return Err(DecodingError::Format(
                        FormatErrorInner::NoMoreImageData.into(),
                    ));
                }

                // Clear the current buffer before appending more data.
                if self.scan_start > 0 {
                    self.current.drain(..self.scan_start).for_each(drop);
                    self.scan_start = 0;
                }

                let val = self.decoder.decode_next(&mut self.current).await?;
                match val {
                    Some(Decoded::ImageData) => {}
                    Some(Decoded::ImageDataFlushed) => {
                        self.subframe.consumed_and_flushed = true;
                    }
                    None => {
                        if !self.current.is_empty() {
                            return Err(DecodingError::Format(
                                FormatErrorInner::UnexpectedEndOfChunk.into(),
                            ));
                        } else {
                            return Ok(None);
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AsyncDecoder;
    use std::mem::discriminant;
    use tokio::io::{AsyncBufRead, AsyncBufReadExt};

    /// A reader that reads at most `n` bytes.
    // struct SmalBuf<R: AsyncBufReadExt + AsyncBufRead> {
    //     inner: R,
    //     cap: usize,
    // }

    // impl<R: AsyncBufReadExt> SmalBuf<R> {
    //     fn new(inner: R, cap: usize) -> Self {
    //         SmalBuf { inner, cap }
    //     }
    // }

    // impl<R: AsyncBufReadExt + AsyncRead> AsyncRead for SmalBuf<R>{
    //     fn poll_read(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>, buf: &mut tokio::io::ReadBuf<'_>) -> std::task::Poll<std::io::Result<()>> {
    //         self.inner.
    //     }
    // }

    // impl<R: AsyncBufReadExt + AsyncBufRead> AsyncBufRead for SmalBuf<R> {
    //     fn poll_fill_buf(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<std::io::Result<&[u8]>> {
    //         self.inner.fill_buf()
    //     }
    //     // fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
    //     //     let len = buf.len().min(self.cap);
    //     //     self.inner.read(&mut buf[..len]).await
    //     // }
    // }

    // impl<R: AsyncBufReadExt> AsyncBufReadExt for SmalBuf<R> {

    //     // fn fill_buf(&mut self) -> Result<&[u8]> {
    //     //     let buf = self.inner.fill_buf()?;
    //     //     let len = buf.len().min(self.cap);
    //     //     Ok(&buf[..len])
    //     // }

    //     // fn consume(&mut self, amt: usize) {
    //     //     assert!(amt <= self.cap);
    //     //     self.inner.consume(amt)
    //     // }
    // }

    #[tokio::test]
    async fn async_no_data_dup_on_finish() {
        const IMG: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/bugfixes/x_issue#214.png"
        ));

        let mut normal = AsyncDecoder::new(IMG).read_info().await.unwrap();

        let mut buffer = vec![0; normal.output_buffer_size()];
        let normal = normal.next_frame(&mut buffer).await.unwrap_err();

        // let smal = AsyncDecoder::new(SmalBuf::new(IMG, 1))
        //     .read_info()
        //     .unwrap()
        //     .next_frame(&mut buffer)
        //     .unwrap_err();
        let smal = AsyncDecoder::new_with_limits(IMG, crate::Limits { bytes: 1 })
            .read_info()
            .await
            .unwrap()
            .next_frame(&mut buffer)
            .await
            .unwrap_err();

        assert_eq!(discriminant(&normal), discriminant(&smal));
    }
}

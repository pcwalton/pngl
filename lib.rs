#![feature(vec_push_all)]

extern crate byteorder;
extern crate flate2;
extern crate opencl;
extern crate time;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::{Decompress, Flush, Status};
use opencl::cl::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_PROFILING_COMMAND_END};
use opencl::cl::{CL_PROFILING_COMMAND_START};
use opencl::mem::CLBuffer;
use opencl::util::{self, PreferedType};
use std::cmp;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::iter;

const BPP: u32 = 4;
const BUFFER_SIZE: usize = 4096;

static MAGIC: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
static IDAT: [u8; 4] = [b'I', b'D', b'A', b'T'];
static IEND: [u8; 4] = [b'I', b'E', b'N', b'D'];

static KERNEL_SOURCE: &'static str = "
uchar4 get_color(__global char *src, int stride, int bpp, int x, int y) {
    return uchar4(src[y * stride + x * bpp + 1],
                  src[y * stride + x * bpp + 2],
                  src[y * stride + x * bpp + 3],
                  src[y * stride + x * bpp + 4]);
}

void set_color(__global uchar4 *dest, int stride, int bpp, int x, int y, uchar4 color) {
    dest[y * stride + x] = color;
}

uchar4 png_none_defilter(__global uchar4 *dest,
                         int dest_stride,
                         __global char *src,
                         int src_stride,
                         int width,
                         int height,
                         int bpp,
                         int x,
                         int y,
                         uchar4 color) {
    return color;
}

uchar4 png_paeth_defilter(__global uchar4 *dest,
                          int dest_stride,
                          __global char *src,
                          int src_stride,
                          int width,
                          int height,
                          int bpp,
                          int y,
                          int x,
                          uchar4 color) {
    uchar4 a = x > 0 ? get_color(src, src_stride, bpp, x - 1, y) : uchar4(0);
    uchar4 b = get_color(src, src_stride, bpp, x, y - 1);
    uchar4 c = x > 0 ? get_color(src, src_stride, bpp, x - 1, y - 1) : uchar4(0);
    short4 sa = convert_short4(a);
    short4 sb = convert_short4(b);
    short4 sc = convert_short4(c);
    short4 sp = sa + sb - sc;
    ushort4 spa = abs(sp - sa);
    ushort4 spb = abs(sp - sb);
    ushort4 spc = abs(sp - sc);
    short4 spaeth = select(sa, select(sb, sc, spb <= spc), spa <= spb & spa <= spc);
    uchar4 paeth = convert_uchar4(spaeth);
    return color + paeth;
}

__kernel void png_defilter(__global char *dest,
                           __global char *src,
                           int width,
                           int height) {
    int bpp = 4;

    int src_stride = width * bpp + 1;
    int dest_stride = width;

    int y = get_global_id(0);
    uchar filter = src[y * src_stride];
    for (int i = 0; i < width + height * 2; i++) {
        int x = i - y;
        if (x >= 0 && x < width) {
            uchar4 color = get_color(src, src_stride, bpp, x, y);
            if (y == 0) {
                color = png_none_defilter(dest,
                                          dest_stride,
                                          src,
                                          src_stride,
                                          width,
                                          height,
                                          bpp,
                                          x,
                                          y,
                                          color);
            } else {
                color = png_paeth_defilter(dest,
                                           dest_stride,
                                           src,
                                           src_stride,
                                           width,
                                           height,
                                           bpp,
                                           x,
                                           y,
                                           color);
            }
            set_color(dest, dest_stride, bpp, x, y, color);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
";

pub trait ReadSeek : Read + Seek {}

impl<T> ReadSeek for T where T: Read + Seek {}

struct Chunk {
    chunk_type: [u8; 4],
    start: u64,
    length: u32,
}

impl Chunk {
    fn read(f: &mut ReadSeek) -> Result<Chunk,()> {
        let length = try!(f.read_u32::<BigEndian>().map_err(|_| ()));
        let mut chunk_type = [0; 4];
        try!(f.read_exact(&mut chunk_type).map_err(|_| ()));
        let start = try!(f.seek(SeekFrom::Current(0)).map_err(|_| ()));
        Ok(Chunk {
            chunk_type: chunk_type,
            start: start,
            length: length,
        })
    }

    fn next(self, f: &mut ReadSeek) -> Result<(),()> {
        match f.seek(SeekFrom::Start(self.start + (self.length as u64) + 4)) {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Image {
    pub fn decode(f: &mut ReadSeek) -> Result<Image,()> {
        let mut header = [0; 8];
        match f.read_exact(&mut header) {
            Ok(()) if header == MAGIC => {}
            _ => return Err(()),
        }

        let mut chunk = try!(Chunk::read(f));
        let width = try!(f.read_u32::<BigEndian>().map_err(|_| ()));
        let height = try!(f.read_u32::<BigEndian>().map_err(|_| ()));
        if try!(f.read_u8().map_err(|_| ())) != 8 {
            // Bit depth must be 8 for now.
            return Err(())
        }
        if try!(f.read_u8().map_err(|_| ())) != 6 {
            // Color type must be 1 for now.
            return Err(())
        }
        if try!(f.read_u16::<BigEndian>().map_err(|_| ())) != 0 {
            // Compression method and filter method must be 0.
            return Err(())
        }
        if try!(f.read_u8().map_err(|_| ())) != 0 {
            // Interlace method must be 0 for now.
            return Err(())
        }
        try!(chunk.next(f));

        // Entropy decode.
        //
        // TODO(pcwalton): Use a faster entropy decoder. This is leaving at least 3x perf on the
        // table.
        //println!("starting entropy decode, width={} height={}", width, height);
        let before = time::precise_time_s();
        let mut unfiltered_data = Vec::with_capacity((width * height * BPP + height) as usize);
        let mut buffer = vec![0; BUFFER_SIZE];
        let mut decoder = Decompress::new(true);
        {
            'outer: loop {
                loop {
                    chunk = try!(Chunk::read(f));
                    if chunk.chunk_type == IEND {
                        break 'outer
                    }
                    if chunk.chunk_type == IDAT {
                        break
                    }
                    try!(chunk.next(f));
                }

                //println!("got IDAT chunk, length={}", chunk.length);
                let mut remaining = chunk.length as usize;
                while remaining != 0 {
                    let nread = try!(f.read(&mut buffer[0..cmp::min(remaining, BUFFER_SIZE)])
                                      .map_err(|_| ()));
                    if nread == 0 {
                        return Err(())
                    }
                    remaining -= nread;

                    match decoder.decompress_vec(&buffer[0..nread],
                                                 &mut unfiltered_data,
                                                 Flush::None) {
                        Ok(Status::Ok) | Ok(Status::StreamEnd) => {}
                        _ => return Err(()),
                    }
                    //println!("new data size={}", data.len());
                }
                //println!("decompression of IDAT ok");
                try!(chunk.next(f));
            }
        }
        let elapsed = time::precise_time_s() - before;
        println!("entropy decode: {}ms", elapsed * 1000.0);

        try!(defilter(width,
                      height,
                      &unfiltered_data[..],
                      PreferedType::CPUPrefered,
                      "CPU"));
        let pixels = try!(defilter(width,
                                   height,
                                   &unfiltered_data[..],
                                   PreferedType::GPUPrefered,
                                   "GPU"));

        ::std::thread::sleep(::std::time::Duration::from_secs(60));

        Ok(Image {
            width: width,
            height: height,
            data: pixels,
        })
    }
}

fn defilter(width: u32,
            height: u32,
            unfiltered_data: &[u8],
            cl_type: PreferedType,
            cl_type_description: &str)
            -> Result<Vec<u8>, ()> {
    let (device, context, queue) =
        try!(util::create_compute_context_prefer(cl_type).map_err(|_| ()));
    let src_buffer: CLBuffer<u8> = context.create_buffer(unfiltered_data.len(),
                                                         CL_MEM_READ_ONLY);
    let dest_buffer: CLBuffer<u8> = context.create_buffer((width * height * BPP) as usize,
                                                          CL_MEM_WRITE_ONLY);

    queue.write(&src_buffer, &&unfiltered_data[..], ());

    let program = context.create_program_from_source(KERNEL_SOURCE);
    match program.build(&device) {
        Ok(_) => {}
        Err(log) => {
            println!("{}", log);
            return Err(())
        }
    }

    let kernel = program.create_kernel("png_defilter");
    kernel.set_arg(0, &dest_buffer);
    kernel.set_arg(1, &src_buffer);
    kernel.set_arg(2, &width);
    kernel.set_arg(3, &height);

    let event = queue.enqueue_async_kernel(&kernel, (height - 1) as isize, None, ());
    let pixels: Vec<u8> = queue.get(&dest_buffer, &event);

    let elapsed = event.end_time() - event.start_time();
    println!("defiltering ({}): {}ms", cl_type_description, (elapsed as f64) / 1_000_000.0);
    Ok(pixels)
}


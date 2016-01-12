#![feature(vec_push_all)]

extern crate byteorder;
extern crate flate2;
extern crate libc;
extern crate opencl;
extern crate time;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::{Decompress, Flush, Status};
use libc::size_t;
use opencl::cl::ll::{clCreateImage2D, clEnqueueReadImage, clSetKernelArg};
use opencl::cl::{CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl::cl::{CL_RGBA, CL_TRUE, CL_UNSIGNED_INT8, CLStatus};
use opencl::cl::{cl_channel_order, cl_channel_type, cl_kernel, cl_mem};
use opencl::hl::Kernel;
use opencl::mem::CLBuffer;
use opencl::util::{self, PreferedType};
use std::cmp;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::iter;
use std::mem;
use std::ptr;

const BPP: u32 = 4;
const BUFFER_SIZE: usize = 4096;

static MAGIC: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
static IDAT: [u8; 4] = [b'I', b'D', b'A', b'T'];
static IEND: [u8; 4] = [b'I', b'E', b'N', b'D'];

static KERNEL_SOURCE: &'static str = "
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

uchar4 get_prev(__global uchar4 *prevs, int x, int y, int i) {
    return prevs[y * 3 + (x + i) % 3];
}

uchar4 get_src_color(read_only image2d_t src, int bpp, int x, int y) {
    return convert_uchar4(read_imageui(src, sampler, int2(x, y)));
}

void set_prev(__global uchar4 *prevs, int x, int y, int i, uchar4 color) {
    prevs[y * 3 + (x + i) % 3] = color;
}

void set_color(read_write image2d_t dest, int bpp, int x, int y, uchar4 color) {
    write_imageui(dest, int2(x, y), convert_uint4(color));
}

uchar4 png_none_defilter(read_write image2d_t dest,
                         read_only image2d_t src,
                         __global uchar4 *prevs,
                         int width,
                         int height,
                         int bpp,
                         int x,
                         int y,
                         uchar4 color) {
    return color;
}

uchar4 png_paeth_defilter(read_write image2d_t dest,
                          read_only image2d_t src,
                          __global uchar4 *prevs,
                          int width,
                          int height,
                          int bpp,
                          int x,
                          int y,
                          int i,
                          uchar4 color) {
    uchar4 a = get_prev(prevs, 1, y, i);
    uchar4 b = y > 0 ? get_prev(prevs, 1, y - 1, i) : uchar4(0);
    uchar4 c = y > 0 ? get_prev(prevs, 2, y - 1, i) : uchar4(0);
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

__kernel void png_defilter(write_only image2d_t dest,
                           read_only image2d_t src,
                           __global char *src_buffer,
                           __global uchar4 *prevs,
                           int width,
                           int height) {
    int bpp = 4;
    int y = get_global_id(0);

    int src_stride = width * bpp + 1;
    uchar filter = src_buffer[y * src_stride];

    for (int i = 0; i < width + height * 2; i++) {
        int x = i - y;
        if (x >= 0 && x < width) {
            uchar4 color = get_src_color(src, bpp, x, y);
            color = png_paeth_defilter(dest, src, prevs, width, height, bpp, x, y, i, color);
            set_color(dest, bpp, x, y, color);
            set_prev(prevs, 0, y, i, color);
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

        /*try!(defilter(width,
                      height,
                      &unfiltered_data[..],
                      PreferedType::CPUPrefered,
                      "CPU"));*/
        let pixels = try!(defilter(width,
                                   height,
                                   &unfiltered_data[..],
                                   PreferedType::GPUPrefered,
                                   "GPU"));

        //::std::thread::sleep(::std::time::Duration::from_secs(60));

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
    queue.write(&src_buffer, &&unfiltered_data[..], ());

    let mut errcode = 0;
    let mut format = cl_image_format {
        image_channel_order: CL_RGBA,
        image_channel_data_type: CL_UNSIGNED_INT8,
    };
    let format_ref = &mut format as *mut cl_image_format;
    let src_image;
    let dest_image;
    unsafe {
        src_image = clCreateImage2D(context.ctx,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    format_ref as *mut _,
                                    width as u64,
                                    height as u64,
                                    (width * BPP) as u64,
                                    unfiltered_data.as_ptr() as *mut _,
                                    &mut errcode);
        println!("errcode={}", errcode);
        assert!(errcode == CLStatus::CL_SUCCESS as i32);
        dest_image = clCreateImage2D(context.ctx,
                                     CL_MEM_READ_WRITE,
                                     format_ref as *mut _,
                                     width as u64,
                                     height as u64,
                                     (width * BPP) as u64,
                                     ptr::null_mut(),
                                     &mut errcode);
        assert!(errcode == CLStatus::CL_SUCCESS as i32);
    }
    let prevs_buffer: CLBuffer<u8> = context.create_buffer((height * BPP * 3) as usize,
                                                           CL_MEM_READ_WRITE);

    let program = context.create_program_from_source(KERNEL_SOURCE);
    match program.build(&device) {
        Ok(_) => {}
        Err(log) => {
            println!("{}", log);
            return Err(())
        }
    }

    let event;
    let mut pixels: Vec<u8>;
    unsafe {
        let kernel = program.create_kernel("png_defilter");
        let dest_image_ref = &dest_image as *const _;
        let cl_kernel = *(&kernel as *const Kernel as *const cl_kernel);
        assert!(clSetKernelArg(cl_kernel,
                               0,
                               mem::size_of::<cl_mem>() as u64,
                               dest_image_ref as *const _) == CLStatus::CL_SUCCESS as i32);
        let src_image_ref = &src_image as *const _;
        assert!(clSetKernelArg(cl_kernel,
                               1,
                               mem::size_of::<cl_mem>() as u64,
                               src_image_ref as *const _) == CLStatus::CL_SUCCESS as i32);
        kernel.set_arg(2, &src_buffer);
        kernel.set_arg(3, &prevs_buffer);
        kernel.set_arg(4, &width);
        kernel.set_arg(5, &height);

        event = queue.enqueue_async_kernel(&kernel, (height - 1) as isize, None, ());
        //let pixels: Vec<u8> = queue.get(&dest_image, &event);
        pixels = iter::repeat(0).take((width * height * BPP) as usize).collect();
        let err = clEnqueueReadImage(queue.cqueue,
                                     dest_image,
                                     CL_TRUE,
                                     [0u64, 0u64, 0u64].as_mut_ptr(),
                                     [width as u64, height as u64, 1].as_mut_ptr(),
                                     0,
                                     0,
                                     pixels.as_mut_ptr() as *mut _,
                                     1,
                                     &event.event,
                                     ptr::null_mut());
        println!("err={}", err);
        assert!(err == CLStatus::CL_SUCCESS as i32);
    }

    let elapsed = event.end_time() - event.start_time();
    println!("defiltering ({}): {}ms", cl_type_description, (elapsed as f64) / 1_000_000.0);
    Ok(pixels)
}

#[repr(C)]
struct cl_image_format {
    image_channel_order: cl_channel_order,
    image_channel_data_type: cl_channel_type,
}


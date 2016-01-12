#![feature(vec_push_all)]

extern crate byteorder;
extern crate flate2;
extern crate libc;
extern crate opencl;
extern crate time;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::{Decompress, Flush, Status};
use libc::size_t;
use opencl::cl::ll::{clCreateImage2D, clEnqueueReadImage, clEnqueueWriteImage, clGetDeviceInfo};
use opencl::cl::ll::{clSetKernelArg};
use opencl::cl::{CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_ITEM_SIZES, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl::cl::{CL_RGBA, CL_TRUE, CL_UNSIGNED_INT8, CLStatus};
use opencl::cl::{cl_channel_order, cl_channel_type, cl_device_id, cl_kernel, cl_mem};
use opencl::hl::{Device, Kernel};
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

int imod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
    //return abs(r);
}

uchar4 get_prev(__local uchar4 *prevs, int x, int y, int i) {
    int index = y * 3 + imod(i - x, 3);
    return prevs[index];
}

uchar4 get_src_color(read_only image2d_t src, int bpp, int x, int y) {
    return convert_uchar4(read_imageui(src, sampler, (int2)(x, y)));
}

void set_prev(__local uchar4 *prevs, int x, int y, int i, uchar4 color) {
    //prevs[y * 3 + imod(i - x, 3)] = color;
    int index = y * 3 + imod(i - x, 3);
    //printf(\"%d writes to %d:%d\\n\", get_global_id(0), y, index);
    prevs[index] = color;
}

void set_color(write_only image2d_t dest, int bpp, int x, int y, uchar4 color) {
    write_imageui(dest, (int2)(x, y), convert_uint4(color));
}

uchar4 png_avg_defilter(write_only image2d_t dest,
                        read_only image2d_t src,
                        __local uchar4 *prevs,
                        int width,
                        int height,
                        int bpp,
                        int x,
                        int y,
                        int i,
                        uchar4 color,
                        uchar4 a,
                        uchar4 b,
                        uchar4 c) {
    return convert_uchar4((convert_short4(a) + convert_short4(b)) / (short4)(2, 2, 2, 2)) + color;
}

uchar4 png_paeth_defilter(write_only image2d_t dest,
                          read_only image2d_t src,
                          __local uchar4 *prevs,
                          int width,
                          int height,
                          int bpp,
                          int x,
                          int y,
                          int i,
                          uchar4 color,
                          uchar4 a,
                          uchar4 b,
                          uchar4 c) {
    short4 sa = convert_short4(a);
    short4 sb = convert_short4(b);
    short4 sc = convert_short4(c);
    short4 sp = sa + sb - sc;
    ushort4 spa = abs(sp - sa);
    ushort4 spb = abs(sp - sb);
    ushort4 spc = abs(sp - sc);
    short4 spaeth = select(select(sc, sb, spb <= spc), sa, (spa <= spb) & (spa <= spc));
    uchar4 paeth = convert_uchar4(spaeth);
    return color + paeth;
}

__kernel void png_defilter(write_only image2d_t dest,
                           read_only image2d_t src,
                           __global int2 *scanlines,
                           __global int *skews,
                           __global uchar *filters,
                           __local uchar4 *prevs,
                           int width,
                           int height) {
    int bpp = 4;
    int index = get_global_id(0);
    int x_offset = scanlines[index][0];
    int y = scanlines[index][1];
    int skew = skews[get_group_id(0)];
    uchar filter = filters[y];

    for (int i = 0; i < width + skew; i++) {
        int x = i + x_offset;
        uchar4 color;
        if (y < height && x >= 0 && x < width) {
            color = get_src_color(src, bpp, x, y);
            uchar4 a = get_prev(prevs, 1, y, i);
            uchar4 b = get_prev(prevs, 1, y - 1, i);
            uchar4 c = get_prev(prevs, 2, y - 1, i);
            if (filter == 1)
                color = a + color;
            else if (filter == 2)
                color = b + color;
            else if (filter == 3)
                color = png_avg_defilter(dest, src, prevs, width, height, bpp, x, y, i, color, a, b, c);
            else if (filter == 4)
                color = png_paeth_defilter(dest, src, prevs, width, height, bpp, x, y, i, color, a, b, c);
            set_color(dest, bpp, x, y, color);
        } else {
            color = (uchar4)(0, 0, 0, 0);
        }
        set_prev(prevs, 0, y, i, color);
        barrier(CLK_LOCAL_MEM_FENCE);
        //printf(\"--- next ---\\n\");
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

        /*let _pixels = try!(defilter(width,
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

struct Plan {
    scanlines: Vec<(i32, i32)>,
    skews: Vec<i32>,
}

impl Plan {
    fn create(max_workgroup_size: usize, filters: &[u8]) -> Plan {
        #[derive(Copy, Clone)]
        struct Run {
            start: u32,
            length: u32,
        }

        let mut runs = vec![];
        let mut run = Run {
            start: 0,
            length: 0,
        };
        for (i, filter) in filters.iter().enumerate() {
            if *filter < 2 {
                if run.length > 0 {
                    runs.push(run);
                }
                run = Run {
                    start: i as u32,
                    length: 1,
                };
                continue
            }

            // FIXME(pcwalton): Do something about runs too big to fit in a workgroup.
            if (run.length as usize) < max_workgroup_size {
                run.length += 1
            }
        }
        runs.push(run);

        runs.sort_by(|a, b| a.length.cmp(&b.length));
        let mut scanlines = Vec::with_capacity(filters.len());
        let mut skews = Vec::with_capacity(filters.len() / max_workgroup_size + 1);
        let mut workgroup_capacity_remaining = max_workgroup_size;
        let mut workgroup_skew = 0;
        while !runs.is_empty() {
            let last = *runs.last().unwrap();
            if (last.length as usize) <= workgroup_capacity_remaining {
                for i in 0..(last.length as i32) {
                    scanlines.push((-i, (last.start as i32) + i))
                }
                workgroup_capacity_remaining -= last.length as usize;
                workgroup_skew = cmp::max((last.length as i32) - 1, workgroup_skew);
                runs.pop();
                continue
            }

            scanlines.extend(iter::repeat((0, -1)).take(workgroup_capacity_remaining));
            skews.push(workgroup_skew);
            workgroup_capacity_remaining = max_workgroup_size;
            workgroup_skew = 0
        }
        scanlines.extend(iter::repeat((0, -1)).take(workgroup_capacity_remaining));
        skews.push(workgroup_skew);

        Plan {
            scanlines: scanlines,
            skews: skews,
        }
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
    let filter_buffer: CLBuffer<u8> = context.create_buffer(height as usize, CL_MEM_READ_ONLY);
    let mut filter_data: Vec<u8> =
        (0..height).map(|y| unfiltered_data[(y * ((width * BPP) + 1)) as usize]).collect();
    queue.write(&filter_buffer, &&filter_data[..], ());

    let mut max_workgroup_size: u64 = 0;
    unsafe {
        let cl_device = *(&device as *const Device as *const cl_device_id);
        clGetDeviceInfo(cl_device,
                        CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        8,
                        &mut max_workgroup_size as *mut _ as *mut _,
                        ptr::null_mut());
        println!("workgroup size={}", max_workgroup_size);
    }

    let plan = Plan::create(max_workgroup_size as usize, &filter_data[..]);
    println!("Plan:\n  Scanlines:");
    for (i, work_unit) in plan.scanlines.iter().enumerate() {
        println!("{},{}: {} {}",
                 i / (max_workgroup_size as usize),
                 i % (max_workgroup_size as usize),
                 work_unit.0,
                 work_unit.1);
    }
    println!("  Skews:");
    for (i, skew) in plan.skews.iter().enumerate() {
        println!("{}: {}", i, skew);
    }

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
                                    CL_MEM_READ_ONLY,
                                    format_ref as *mut _,
                                    width as u64,
                                    height as u64,
                                    0,
                                    ptr::null_mut(),
                                    &mut errcode);
        assert!(errcode == CLStatus::CL_SUCCESS as i32);
        dest_image = clCreateImage2D(context.ctx,
                                     CL_MEM_WRITE_ONLY,
                                     format_ref as *mut _,
                                     width as u64,
                                     height as u64,
                                     0,
                                     ptr::null_mut(),
                                     &mut errcode);
        assert!(errcode == CLStatus::CL_SUCCESS as i32);

        errcode = clEnqueueWriteImage(queue.cqueue,
                                      src_image,
                                      CL_TRUE,
                                      [0, 0, 0].as_mut_ptr(),
                                      [width as u64, height as u64, 1].as_mut_ptr(),
                                      (width * BPP + 1) as u64,
                                      0,
                                      unfiltered_data[1..].as_ptr() as *const _ as *mut _,
                                      0,
                                      ptr::null(),
                                      ptr::null_mut());
        println!("errcode={}", errcode);
        assert!(errcode == CLStatus::CL_SUCCESS as i32);
    }
    /*let prevs_buffer: CLBuffer<u8> = context.create_buffer((height * BPP * 3) as usize,
                                                           CL_MEM_READ_WRITE);*/

    let scanlines_buffer: CLBuffer<(i32, i32)> = context.create_buffer(plan.scanlines.len(),
                                                                       CL_MEM_READ_ONLY);
    queue.write(&scanlines_buffer, &&plan.scanlines[..], ());
    let skews_buffer: CLBuffer<i32> = context.create_buffer(plan.skews.len(), CL_MEM_READ_ONLY);
    queue.write(&skews_buffer, &&plan.skews[..], ());

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
        kernel.set_arg(2, &scanlines_buffer);
        kernel.set_arg(3, &skews_buffer);
        kernel.set_arg(4, &filter_buffer);
        assert!(clSetKernelArg(cl_kernel,
                               5,
                               max_workgroup_size * 4,
                               ptr::null()) == CLStatus::CL_SUCCESS as i32);
        kernel.set_arg(6, &width);
        kernel.set_arg(7, &height);

        let mut max_work_item_sizes: (u64, u64, u64) = (0, 0, 0);
        let cl_device = *(&device as *const Device as *const cl_device_id);
        clGetDeviceInfo(cl_device,
                        CL_DEVICE_MAX_WORK_ITEM_SIZES,
                        24,
                        &mut max_work_item_sizes as *mut _ as *mut _,
                        ptr::null_mut());
        println!("work item sizes={:?}", max_work_item_sizes);

        //event = queue.enqueue_kernel(&kernel, (height / 3 + 25) as isize, None, ());
        event = queue.enqueue_kernel(
            &kernel,
            plan.scanlines.len(),
            Some(max_workgroup_size as usize),
            ());
        let stuff: Vec<u8> = queue.get(&filter_buffer, &event);
        pixels = iter::repeat(0).take((width * height * BPP) as usize).collect();
        let err = clEnqueueReadImage(queue.cqueue,
                                     dest_image,
                                     CL_TRUE,
                                     [0u64, 0u64, 0u64].as_mut_ptr(),
                                     [width as u64, height as u64, 1].as_mut_ptr(),
                                     (width * BPP) as u64,
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


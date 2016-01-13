#![feature(asm, vec_push_all)]

extern crate byteorder;
extern crate flate2;
extern crate libc;
extern crate opencl;
extern crate simd;
extern crate time;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::{Decompress, Flush, Status};
use libc::size_t;
use opencl::cl::ll::{clCreateImage2D, clEnqueueReadImage, clEnqueueWriteImage, clGetDeviceInfo};
use opencl::cl::ll::{clSetKernelArg};
use opencl::cl::{CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_ITEM_SIZES, CL_FALSE, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl::cl::{CL_RGBA, CL_TRUE, CL_UNSIGNED_INT8, CLStatus};
use opencl::cl::{cl_channel_order, cl_channel_type, cl_device_id, cl_kernel, cl_mem};
use opencl::hl::{Device, Event, EventList, Kernel};
use opencl::mem::CLBuffer;
use opencl::util::{self, PreferedType};
use simd::{i16x8, u32x4, u8x16};
use std::cmp;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::iter;
use std::mem;
use std::ptr;

const BPP: u32 = 4;
const BUFFER_SIZE: usize = 4096;
const MIN_WORKGROUP_SIZE: usize = 128;

static MAGIC: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
static IDAT: [u8; 4] = [b'I', b'D', b'A', b'T'];
static IEND: [u8; 4] = [b'I', b'E', b'N', b'D'];

static KERNEL_SOURCE: &'static str = "
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

uchar4 get_prev(__local uchar4 *prevs, int x, int y, int i) {
    int index = y * 3 + (i - x) % 3;
    return prevs[index];
}

uchar4 get_src_color(read_only image2d_t src, int x, int y) {
    return convert_uchar4(read_imageui(src, sampler, (int2)(x, y)));
}

void set_prev(__local uchar4 *prevs, int x, int y, int i, uchar4 color) {
    int index = y * 3 + (i - x) % 3;
    prevs[index] = color;
}

void set_color(write_only image2d_t dest, int x, int y, uchar4 color) {
    write_imageui(dest, (int2)(x, y), convert_uint4(color));
}

uchar4 png_avg_defilter(write_only image2d_t dest,
                        read_only image2d_t src,
                        __local uchar4 *prevs,
                        int width,
                        int height,
                        int x,
                        int y,
                        int i,
                        uchar4 color,
                        uchar4 a,
                        uchar4 b) {
    return convert_uchar4((convert_short4(a) + convert_short4(b)) / (short4)(2, 2, 2, 2)) + color;
}

uchar4 png_paeth_defilter(write_only image2d_t dest,
                          read_only image2d_t src,
                          __local uchar4 *prevs,
                          int width,
                          int height,
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

uchar4 png_defilter_pixel(write_only image2d_t dest,
                          read_only image2d_t src,
                          __local uchar4 *prevs,
                          int width,
                          int height,
                          int x,
                          int y,
                          int i,
                          uchar filter) {
    if (y >= height || x < 0 || x >= width)
        return (uchar4)(0, 0, 0, 0);

    uchar4 color = get_src_color(src, x, y);
    uchar4 a = get_prev(prevs, 1, y, i);
    uchar4 b = get_prev(prevs, 1, y - 1, i);
    uchar4 c = get_prev(prevs, 2, y - 1, i);
    if (filter == 1)
        color = a + color;
    else if (filter == 2)
        color = b + color;
    else if (filter == 3)
        color = png_avg_defilter(dest, src, prevs, width, height, x, y, i, color, a, b);
    else if (filter == 4)
        color = png_paeth_defilter(dest, src, prevs, width, height, x, y, i, color, a, b, c);
    set_color(dest, x, y, color);
    return color;
}

__kernel void png_defilter(write_only image2d_t dest,
                           read_only image2d_t src,
                           __global int2 *scanlines,
                           __global int *skews,
                           __global uchar *filters,
                           __local uchar4 *prevs,
                           int width,
                           int height) {
    int index = get_global_id(0);
    int group_id = get_group_id(0);
    int x_offset = scanlines[index][0];
    int start_y = scanlines[index][1];
    int y = start_y;
    int skew = skews[group_id];
    uchar filter = filters[y];

    for (int i = 0; i < width + skew; i++) {
        int x = i + x_offset;
        uchar4 color = png_defilter_pixel(dest, src, prevs, width, height, x, y, i, filter);
        set_prev(prevs, 0, y, i, color);
        barrier(CLK_LOCAL_MEM_FENCE);
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

#[derive(Copy, Clone, Debug)]
struct Run {
    start: u32,
    length: u32,
}

impl Run {
    fn end(&self) -> u32 {
        self.start + self.length
    }
}

#[derive(Copy, Clone)]
struct Overflow {
    start: u32,
    length: u32,
}

struct Plan {
    scanlines: Vec<(i32, i32)>,
    skews: Vec<i32>,
    overflows: Vec<Overflow>,
}

impl Plan {
    fn create(max_workgroup_size: usize, filters: &[u8]) -> Plan {
        let (mut runs, mut overflows) = (vec![], vec![]);
        {
            let mut flush_run = |mut run: Run| {
                if run.length > 0 {
                    if (run.length as usize) > max_workgroup_size {
                        overflows.push(Overflow {
                            start: run.start + (max_workgroup_size as u32),
                            length: run.length - (max_workgroup_size as u32),
                        });
                        run.length = max_workgroup_size as u32
                    }
                    runs.push(run);
                }
            };

            let mut run = Run {
                start: 0,
                length: 0,
            };
            for (i, filter) in filters.iter().enumerate() {
                if *filter < 2 {
                    flush_run(run);
                    run = Run {
                        start: i as u32,
                        length: 1,
                    };
                    continue
                }

                run.length += 1
            }
            flush_run(run);
        }

        let mut scanlines = Vec::with_capacity(filters.len());
        let mut skews = Vec::with_capacity(filters.len() / max_workgroup_size + 1);
        let mut workgroup_capacity_remaining = max_workgroup_size;
        let mut workgroup_skew = 0;
        runs.sort_by(|a, b| a.length.cmp(&b.length));
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
            overflows: overflows,
        }
    }

    fn dump(&self, max_workgroup_size: usize) {
        println!("Plan:");
        println!("  Scanlines:");
        for (i, work_unit) in self.scanlines.iter().enumerate() {
            println!("    {},{}: {} {}",
                     i / (max_workgroup_size as usize),
                     i % (max_workgroup_size as usize),
                     work_unit.0,
                     work_unit.1);
        }
        println!("  Skews:");
        for (i, skew) in self.skews.iter().enumerate() {
            println!("    {}: {}", i, skew);
        }
        println!("  Overflows:");
        for (i, overflow) in self.overflows.iter().enumerate() {
            println!("    {}: {} {}", i, overflow.start, overflow.length);
        }
    }
}

#[inline(never)]
fn cpu_defilter(data: &mut [u8], width: u32, height: u32, filters: &[u8], skip_first: bool) {
    /*let mut first_row = Vec::new();
    for byte in data[0..((width * BPP) as usize)].iter() {
        first_row.push(*byte);
    }*/

    let first = if skip_first {
        1
    } else {
        0
    };
    for y in first..height {
        /*for (i, byte) in first_row.iter().enumerate() {
            data[((y * width * BPP) as usize) + i] = *byte;
        }*/
        let filter = filters[y as usize];
        for x in 0..width {
            let mut color = get_the_pixel(data, width, x, y);
            let a = if x == 0 {
                i16x8::splat(0)
            } else {
                get_the_pixel(data, width, x - 1, y)
            };
            let b = if y == 0 {
                i16x8::splat(0)
            } else {
                get_the_pixel(data, width, x, y - 1)
            };
            let c = if x == 0 || y == 0 {
                i16x8::splat(0)
            } else {
                get_the_pixel(data, width, x - 1, y - 1)
            };
            color = color + match filter {
                0 => i16x8::splat(0),
                1 => a,
                2 => b,
                3 => (a + b) >> 1u16,
                _ => {
                    let p = a + b - c;
                    let pa = pabsw(p - a);
                    let pb = pabsw(p - b);
                    let pc = pabsw(p - c);
                    (pa.le(pb) & pa.le(pc)).select(a, pb.le(pc).select(b, c))
                }
            };
            set_the_pixel(data, width, x, y, color)
        }
    }

    fn pabsw(mut x: i16x8) -> i16x8 {
        unsafe {
            asm!("pabsw $0,$0" : "+x"(x) : : : "intel");
        }
        x
    }

    fn get_the_pixel(data: &[u8], width: u32, x: u32, y: u32) -> i16x8 {
        unsafe {
            let color: u32 = *mem::transmute::<*const u8,*const u32>(
                &data[((y * width + x) * BPP) as usize] as *const u8);
            let mut color = u32x4::splat(color);
            let mask = u8x16::new(0, 0x80,
                                  1, 0x80,
                                  2, 0x80,
                                  3, 0x80,
                                  0x80, 0x80,
                                  0x80, 0x80,
                                  0x80, 0x80,
                                  0x80, 0x80);
            asm!("pshufb $0,$1" : "+x"(color) : "x"(mask) : : "intel");
            mem::transmute::<u32x4, i16x8>(color)
        }
    }

    fn set_the_pixel(data: &mut [u8], width: u32, x: u32, y: u32, color: i16x8) {
        unsafe {
            let dest: *mut u32 = mem::transmute::<*mut u8,*mut u32>(
                &mut data[((y * width + x) * BPP + 0) as usize] as *mut u8);
            let mut color = mem::transmute::<i16x8, u32x4>(color);
            let mask = u8x16::new(0, 2, 4, 6,
                                  0x80, 0x80, 0x80, 0x80,
                                  0x80, 0x80, 0x80, 0x80,
                                  0x80, 0x80, 0x80, 0x80);
            asm!("pshufb $0,$1" : "+x"(color) : "x"(mask) : : "intel");
            *dest = color.extract(0);
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
    let mut filters: Vec<u8> =
        (0..height).map(|y| unfiltered_data[(y * ((width * BPP) + 1)) as usize]).collect();
    queue.write(&filter_buffer, &&filters[..], ());

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

    let plan = Plan::create(max_workgroup_size as usize, &filters[..]);
    plan.dump(max_workgroup_size as usize);

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
                                      CL_FALSE,
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

    let mut cpu_data = vec![];
    {
        let mut src_data = &unfiltered_data[1..];
        for y in 0..height {
            cpu_data.push_all(&src_data[0..(width * BPP) as usize]);
            if y < height - 1 {
                src_data = &src_data[(width * BPP + 1) as usize..]
            }
        }
    }

    let kernel_event;
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

        let workgroup_size = cmp::min(cmp::max(plan.skews.iter().cloned().max().unwrap_or(1),
                                               MIN_WORKGROUP_SIZE as i32) as u64,
                                      max_workgroup_size).next_power_of_two();
        println!("selected workgroup size: {}", workgroup_size);

        kernel_event = queue.enqueue_async_kernel(
            &kernel,
            plan.scanlines.len() as isize,
            Some(workgroup_size as isize),
            ());
    }

    let mut overflow_readback_events = vec![];
    for overflow in &plan.overflows {
        assert!(overflow.start > 0);
        let byte_start = ((overflow.start - 1) * width * BPP) as usize;
        let byte_length = (width * BPP) as usize;
        let mut overflow_readback_event = ptr::null_mut();
        unsafe {
            errcode = clEnqueueReadImage(queue.cqueue,
                                         dest_image,
                                         CL_FALSE,
                                         [0, (overflow.start - 1) as u64, 0].as_mut_ptr(),
                                         [width as u64, 1, 1].as_mut_ptr(),
                                         (width * BPP) as u64,
                                         0,
                                         (&mut cpu_data[byte_start..]).as_mut_ptr() as *mut _,
                                         1,
                                         &kernel_event.event,
                                         &mut overflow_readback_event);
            assert!(errcode == CLStatus::CL_SUCCESS as i32);
        }
        overflow_readback_events.push(Event {
            event: overflow_readback_event,
        });
    }

    let mut accelerated_execute_time = 0.0;
    let mut overflow_upload_events = vec![];
    for (i, overflow) in plan.overflows.iter().enumerate() {
        overflow_readback_events[i].wait();

        let before = time::precise_time_s();
        let byte_start = ((overflow.start - 1) * width * BPP) as usize;
        let byte_length = ((overflow.length + 1) * width * BPP) as usize;
        cpu_defilter(&mut cpu_data[byte_start..(byte_start + byte_length)],
                     width,
                     overflow.length + 1,
                     &filters[((overflow.start - 1) as usize)..],
                     true);
        let elapsed = time::precise_time_s() - before;
        accelerated_execute_time += elapsed * 1000.0;
        println!("CPU overflow defiltering: {}ms", elapsed * 1000.0);

        let mut overflow_upload_event = ptr::null_mut();
        unsafe {
            errcode = clEnqueueWriteImage(
                queue.cqueue,
                dest_image,
                CL_FALSE,
                [0, overflow.start as u64, 0].as_mut_ptr(),
                [width as u64, overflow.length as u64, 1].as_mut_ptr(),
                (width * BPP) as u64,
                0,
                (&mut cpu_data[byte_start..]).as_mut_ptr() as *mut _,
                0,
                ptr::null(),
                &mut overflow_upload_event);
            assert!(errcode == CLStatus::CL_SUCCESS as i32);
            overflow_upload_events.push(Event {
                event: overflow_upload_event,
            });
        }
    }

    kernel_event.wait();
    if !overflow_upload_events.is_empty() {
        (&overflow_upload_events[..]).wait();
    }

    accelerated_execute_time += event_elapsed_time(&kernel_event);
    println!("GPU defiltering ({}): {}ms", cl_type_description, event_elapsed_time(&kernel_event));
    for event in &overflow_readback_events {
        accelerated_execute_time += event_elapsed_time(&event);
        println!("CPU overflow readback: {}ms", event_elapsed_time(&event))
    }
    for event in &overflow_upload_events {
        accelerated_execute_time += event_elapsed_time(&event);
        println!("CPU overflow upload: {}ms", event_elapsed_time(&event))
    }
    println!("GPU-accelerated execution time: {}ms", accelerated_execute_time);

    let elapsed_total_time = match overflow_upload_events.last() {
        Some(last_overflow_upload_event) => last_overflow_upload_event.end_time(),
        None => kernel_event.end_time(),
    } - kernel_event.start_time();
    println!("GPU-accelerated wallclock time: {}ms", ns_to_ms(elapsed_total_time));

    let cpu_before = time::precise_time_s();
    cpu_defilter(&mut cpu_data[..], width, height, &filters[..], false);
    let cpu_elapsed = (time::precise_time_s() - cpu_before) * 1000.0;
    println!("CPU decode: {}ms", cpu_elapsed);

    println!("GPU-accelerated speedup: {}x", cpu_elapsed / accelerated_execute_time);

    unsafe {
        let stuff: Vec<u8> = queue.get(&filter_buffer, &kernel_event);
        pixels = iter::repeat(0).take((width * height * BPP) as usize).collect();
        let err = clEnqueueReadImage(queue.cqueue,
                                     dest_image,
                                     CL_TRUE,
                                     [0u64, 0u64, 0u64].as_mut_ptr(),
                                     [width as u64, height as u64, 1].as_mut_ptr(),
                                     (width * BPP) as u64,
                                     0,
                                     pixels.as_mut_ptr() as *mut _,
                                     0,
                                     ptr::null(),
                                     ptr::null_mut());
        assert!(err == CLStatus::CL_SUCCESS as i32);
    }

    Ok(pixels)
}

#[repr(C)]
struct cl_image_format {
    image_channel_order: cl_channel_order,
    image_channel_data_type: cl_channel_type,
}

fn event_elapsed_time(event: &Event) -> f64 {
    ns_to_ms(event.end_time() - event.start_time())
}

fn ns_to_ms(ns: u64) -> f64 {
    (ns as f64) / 1_000_000.0
}


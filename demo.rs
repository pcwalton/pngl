extern crate byteorder;
extern crate pngl;

use byteorder::{LittleEndian, WriteBytesExt};
use pngl::Image;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

const BPP: u32 = 4;

pub fn main() {
    let image =
        Image::decode(&mut File::open(env::args().skip(1).next().unwrap()).unwrap()).unwrap();
    println!("w={} h={} d={}", image.width, image.height, image.data.len());

    // Write out the image into TGA format.
    let mut tga = BufWriter::new(File::create(env::args().skip(2).next().unwrap()).unwrap());
    tga.write_all(&[0; 2]).unwrap();
    tga.write_u16::<LittleEndian>(2).unwrap();
    tga.write_all(&[0; 8]).unwrap();
    tga.write_u16::<LittleEndian>(image.width as u16).unwrap();
    tga.write_u16::<LittleEndian>(image.height as u16).unwrap();
    tga.write_u16::<LittleEndian>(24).unwrap();

    let stride = image.width * BPP;
    for i in 0..image.height {
        let y = image.height - i - 1;
        for x in 0..image.width {
            let start = (y * stride + x * BPP) as usize;
            let mut data = [image.data[start+2], image.data[start+1], image.data[start+0]];
            tga.write_all(&data[..]).unwrap();
        }
    }
}


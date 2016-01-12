extern crate pngl;

use pngl::Image;
use std::env;
use std::fs::File;

pub fn main() {
    let image =
        Image::decode(&mut File::open(env::args().skip(1).next().unwrap()).unwrap()).unwrap();
    println!("w={} h={} d={}", image.width, image.height, image.data.len());
}


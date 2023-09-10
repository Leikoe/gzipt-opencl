use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING, cl_char, cl_int, cl_uchar, cl_half};
use opencl3::Result;
use std::{fs, ptr};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use kdtree::KdTree;
use kdtree::ErrorKind;
use kdtree::distance::squared_euclidean;
use opencl3::error_codes::ClError;

const KERNEL_NAME: &str = "ncd_kernel";

// return (chars, lens, offsets)
fn to_cl(v: &(&Vec<u8>, &Vec<(usize, usize)>)) -> (Vec<cl_char>, Vec<cl_int>, Vec<cl_int>) {
    let mut strings = v.0.iter().map(|x| *x as cl_char).collect();
    let mut lens = Vec::new();
    let mut offsets = Vec::new();

    for (len, offset) in v.1.iter() {
        lens.push(*len as cl_int);
        offsets.push(*offset as cl_int);
    }

    (strings, lens, offsets)
}

fn get_ncds(x: (&Vec<u8>, &Vec<(usize, usize)>), y: (&Vec<u8>, &Vec<(usize, usize)>)) -> Vec<Vec<f32>> {
    let same_vec = x == y;

    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("couldn't get all devices")
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program_source = fs::read_to_string("./src/kernels/shoco_compress.cl")
        .expect("Should have been able to read the file");
    let program = Program::create_and_build_from_source(&context, program_source.as_str(), "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    /////////////////////////////////////////////////////////////////////
    // Compute data

    let (strings_x, lens_x, offsets_x) = to_cl(&x);

    // Create OpenCL device buffers
    // buffers for x
    let mut strings_x_buffer = unsafe {
        Buffer::<cl_char>::create(&context, CL_MEM_READ_ONLY, strings_x.len(), ptr::null_mut()).expect("couldn't create strings_x_buffer")
    };
    let mut lens_x_buffer = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, x.1.len(), ptr::null_mut()).expect("couldn't create lens_x_buffer")
    };
    let mut offsets_x_buffer = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, x.1.len(), ptr::null_mut()).expect("couldn't create offsets_x_buffer")
    };

    // output buffer
    let ncds_buffer = unsafe {
        Buffer::<cl_half>::create(&context, CL_MEM_WRITE_ONLY, x.1.len() * y.1.len(), ptr::null_mut()).expect("couldn't create ncds_buffer")
    };

    // Blocking write
    unsafe { queue.enqueue_write_buffer(&mut strings_x_buffer, CL_BLOCKING, 0, &strings_x, &[]).expect("couldb't write_buffer strings_x_buffer") };
    unsafe { queue.enqueue_write_buffer(&mut lens_x_buffer, CL_BLOCKING, 0, &lens_x, &[]).expect("couldb't write_buffer lens_x_buffer") };
    unsafe { queue.enqueue_write_buffer(&mut offsets_x_buffer, CL_BLOCKING, 0, &offsets_x, &[]).expect("couldb't write_buffer offsets_x_buffer") };

    println!("[CL] Launching {}x{} kernel", x.1.len(), y.1.len());

    // optimization for the same_vec case
    let kernel_event = if !same_vec {
        let (strings_y, lens_y, offsets_y) = to_cl(&y);

        // buffers for y
        let mut strings_y_buffer = unsafe {
            Buffer::<cl_char>::create(&context, CL_MEM_READ_ONLY, strings_y.len(), ptr::null_mut()).expect("couldn't create strings_y_buffer")
        };
        let mut lens_y_buffer = unsafe {
            Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, y.1.len(), ptr::null_mut()).expect("couldn't create lens_y_buffer")
        };
        let mut offsets_y_buffer = unsafe {
            Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, y.1.len(), ptr::null_mut()).expect("couldn't create offsets_y_buffer")
        };

        unsafe { queue.enqueue_write_buffer(&mut strings_y_buffer, CL_BLOCKING, 0, &strings_y, &[]).expect("couldb't write_buffer strings_y_buffer") };
        unsafe { queue.enqueue_write_buffer(&mut lens_y_buffer, CL_BLOCKING, 0, &lens_y, &[]).expect("couldb't write_buffer lens_y_buffer") };
        unsafe { queue.enqueue_write_buffer(&mut offsets_y_buffer, CL_BLOCKING, 0, &offsets_y, &[]).expect("couldb't write_buffer offsets_y_buffer") };

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&strings_x_buffer)
                .set_arg(&lens_x_buffer)
                .set_arg(&offsets_x_buffer)
                .set_arg(&strings_y_buffer)
                .set_arg(&lens_y_buffer)
                .set_arg(&offsets_y_buffer)
                .set_arg(&ncds_buffer)
                .set_global_work_sizes(&[x.1.len(), y.1.len()])
                .enqueue_nd_range(&queue)
                .expect("couldn't enqueue nd range")
        }
    } else {
        println!("[CL] same_vec optimization used");
        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&strings_x_buffer)
                .set_arg(&lens_x_buffer)
                .set_arg(&offsets_x_buffer)
                .set_arg(&strings_x_buffer)
                .set_arg(&lens_x_buffer)
                .set_arg(&offsets_x_buffer)
                .set_arg(&ncds_buffer)
                .set_global_work_sizes(&[x.1.len(), y.1.len()])
                .enqueue_nd_range(&queue)
                .expect("couldn't enqueue nd range")
        }
    };


    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut results = vec![0.0 as cl_half; x.1.len() * y.1.len()];
    let _read_event =
        unsafe { queue.enqueue_read_buffer(&ncds_buffer, CL_BLOCKING, 0, &mut results, &events).expect("couldn't copy results") };

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start().expect("couldn't start profiling");
    let end_time = kernel_event.profiling_command_end().expect("couldn't end profiling");
    let duration = end_time - start_time;
    println!("[CL] kernel execution duration (ms): {}", Duration::from_nanos(duration).as_millis());

    (0..x.1.len())
        .map(|i| results[i*y.1.len()..(i+1)*y.1.len()].iter().map(|x| *x as f32).collect())
        .collect()
}



fn main() {
    let start = Instant::now();
    let text = fs::read_to_string("./input.txt").unwrap()[..50].to_string();

    // # here are all the unique characters that occur in this text
    let chars = {
        let chars_set: HashSet<char> = HashSet::from_iter(text.chars());
        let mut chars: Vec<char> = Vec::from_iter(chars_set);
        chars.sort();
        chars
    };

    // create a mapping from characters to integers
    let mut stoi : HashMap<char, u8> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (c.clone(), i as u8))
    );
    let mut itos : HashMap<u8, char> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (i as u8, c.clone()))
    );

    // encoder: take a string, output a list of integers
    let encode = |s: &str| s.chars().map(|c| *stoi.get(&c).unwrap()).collect::<Vec<u8>>();
    // decoder: take a list of integers, output a string
    let decode = |l: Vec<u8>| l.iter().map(|i| itos.get(i).unwrap()).collect::<String>();

    let data = encode(text.as_str());
    let n_vocab: i64 = chars.len() as i64;
    let n_ctx : i64 = 8;
    let n_train: i64 = data.len() as i64;

    println!("n_vocab = {n_vocab}");
    println!("n_ctx   = {n_ctx}");
    println!("n_train = {n_train}");


    let mut X = Vec::with_capacity((n_train*n_ctx) as usize);
    let mut Y= Vec::with_capacity((n_train*n_ctx) as usize);

    let before = Instant::now();
    let d = get_data(&data, n_ctx);
    println!("get_data | Elapsed time: {:.2?}", before.elapsed());

    let before = Instant::now();
    for (x, y) in d.0.iter().zip(d.1.iter()) {
        for token_idx in 0..n_ctx as usize {
            let context = (min(x.0, token_idx+1), x.1);
            let target = &y[token_idx];

            // println!("when context is {:?}, target is {}", decode(context.to_vec()), itos.get(target).unwrap());
            X.push(context);
            Y.push(target);
        }
    }
    println!("X, Y | took: {:.2?}", before.elapsed());

    let before = Instant::now();
    let ncds = get_ncds((&data, &X), (&data, &X));
    println!("ncd_scores | took: {:.2?}", before.elapsed());

    for line in &ncds {
        println!("{:?}", line);
    }

    let before = Instant::now();
    // use the kiddo::KdTree type to get up and running quickly with default settings
    // let mut kdtree = KdTree::new(ncds[0].len());
    // 'outer_loop: for i in 0..ncds.len() {
    //     for j in 0..ncds[i].len() {
    //         if !ncds[i][j].is_finite() {
    //             println!("[ERR] non finite f32 at ({i}, {j}), char: {:?}", itos.get(Y[i]));
    //             continue 'outer_loop;
    //         }
    //     }
    //     kdtree.add(&ncds[i], i).expect("TODO: panic message");
    // }
    // println!("build kdtree | took: {:.2?}", before.elapsed());
    //
    // let query = get_ncds(&vec![&data[0..n_ctx as usize]], &X)[0].clone();
    // let closest_points = kdtree.nearest(&query, 7, &squared_euclidean).unwrap();
    // let closest_points_v = closest_points.iter().map(|x| itos.get(Y[*x.1])).collect::<Vec<_>>();
    // println!("{:?}", closest_points_v);

    println!("TOOK: {}", start.elapsed().as_secs_f64());
}


fn get_data(data: &Vec<u8>, n_ctx: i64) -> (Vec<(usize, usize)>, Vec<&[u8]>) {
    let ix = Vec::from_iter(0..(data.len() - n_ctx as usize));
    let x = ix.iter().map(|&i| (n_ctx as usize, i)).collect();
    let y = ix.iter().map(|&i| &data[(i+1)..(i+n_ctx as usize+1)]).collect();

    (x, y)
}
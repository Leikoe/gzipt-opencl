use std::fs::File;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING, cl_char, cl_int, cl_uchar};
use opencl3::Result;
use std::{fs, ptr};
use std::ptr::null;

const KERNEL_NAME: &str = "nomnom";

fn main() -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program_source = fs::read_to_string("./src/gzip.cl")
        .expect("Should have been able to read the file");
    let program = Program::create_and_build_from_source(&context, program_source.as_str(), "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    /////////////////////////////////////////////////////////////////////
    // Compute data

    // The input data
    let inputs = vec![
        &program_source[..1000];1_000
    ];

    let mut strings = Vec::new();
    let mut lens = Vec::new();
    let mut offsets = Vec::new();

    let mut cursor = 0;
    for input in inputs.iter() {
        input.chars().for_each(|x| strings.push(x as cl_uchar));
        lens.push(input.len() as cl_int);
        offsets.push(cursor as cl_int);

        cursor += input.len();
    }


    // Create OpenCL device buffers
    let mut strings_buffer = unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, strings.len(), ptr::null_mut())?
    };
    let mut lens_buffer = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, inputs.len(), ptr::null_mut())?
    };
    let mut offsets_buffer = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, offsets.len(), ptr::null_mut())?
    };
    let compressed_lens_buffer = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_WRITE_ONLY, inputs.len(), ptr::null_mut())?
    };

    // Blocking write
    let _strings_buffer_write_event = unsafe { queue.enqueue_write_buffer(&mut strings_buffer, CL_BLOCKING, 0, &strings, &[])? };
    let _lens_buffer_write_event = unsafe { queue.enqueue_write_buffer(&mut lens_buffer, CL_BLOCKING, 0, &lens, &[])? };
    let _offsets_buffer_write_event = unsafe { queue.enqueue_write_buffer(&mut offsets_buffer, CL_BLOCKING, 0, &offsets, &[])? };

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&strings_buffer)
            .set_arg(&lens_buffer)
            .set_arg(&offsets_buffer)
            .set_arg(&compressed_lens_buffer)
            .set_global_work_size(inputs.len())
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut results = vec![0; inputs.len()];
    let _read_event =
        unsafe { queue.enqueue_read_buffer(&compressed_lens_buffer, CL_BLOCKING, 0, &mut results, &events)? };

    // Output the first and last results
    println!("Original lens: {:?}", lens[0]);
    dbg!(results[0]);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

    Ok(())
}
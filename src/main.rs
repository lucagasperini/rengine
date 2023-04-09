use core::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use vulkano::buffer::TypedBufferAccess;
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassContents};
use vulkano::device::Queue;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    swapchain::{acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, KeyboardInput};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod render;
mod types;

use render::{Mesh, RenderScene};
use types::{ColorRGBA, Point2D};

struct GameStatus {
    pub viewport: Viewport,
    pub sensitivity: f32,
    pub scene: RenderScene,
}

fn create_instance() -> Arc<Instance> {
    let required_extensions = vulkano_win::required_extensions();
    return Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
        enumerate_portability: true,
        ..Default::default()
    })
    .unwrap();
}

fn create_window(instance: &Arc<Instance>) -> (Arc<Surface<Window>>, EventLoop<()>) {
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    return (surface, event_loop);
}

fn list_devices<'a>(
    instance: &'a Arc<Instance>,
    extensions: DeviceExtensions,
    features: Features,
) -> Option<PhysicalDevice<'a>> {
    let physical_devices = PhysicalDevice::enumerate(instance);

    let physical_devices_filtred = physical_devices
        .filter(|&p| p.api_version() >= Version::V1_3)
        .filter(|&p| p.supported_extensions().is_superset_of(&extensions))
        .filter(|&p| p.supported_features().is_superset_of(&features));

    let mut selected_device: Option<PhysicalDevice> = None;
    for device in physical_devices_filtred {
        println!(
            "Found device: {} (type: {:?})",
            device.properties().device_name,
            device.properties().device_type,
        );

        if device.properties().device_type == PhysicalDeviceType::DiscreteGpu {
            selected_device = Some(device);
        }
    }

    return selected_device;
}

fn get_queue_family_vector<'a>(
    pdevice: &'a PhysicalDevice,
    surface: &'a Arc<Surface<Window>>,
) -> Vec<QueueCreateInfo<'a>> {
    let queue_family = pdevice
        .queue_families()
        .find(|&q| q.supports_graphics() && q.supports_surface(surface).unwrap_or(false));

    let mut queue_vector: Vec<QueueCreateInfo> = Vec::new();
    for queue in queue_family {
        queue_vector.push(QueueCreateInfo::family(queue));
    }
    return queue_vector;
}

fn create_vulkan_device<'a>(
    pdevice: &'a PhysicalDevice,
    surface: &'a Arc<Surface<Window>>,
    extensions: DeviceExtensions,
    features: Features,
) -> (Arc<Device>, Vec<Arc<Queue>>) {
    let queue_vector = get_queue_family_vector(pdevice, surface);

    let (device, queues) = Device::new(
        *pdevice,
        DeviceCreateInfo {
            enabled_extensions: extensions,
            queue_create_infos: queue_vector,
            enabled_features: features,
            ..Default::default()
        },
    )
    .unwrap();

    return (device, Vec::from_iter(queues));
}

fn create_swapchain(
    pdevice: &PhysicalDevice,
    surface: &Arc<Surface<Window>>,
    device: &Arc<Device>,
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let surface_capabilities = pdevice
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let image_format = Some(
        pdevice
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    return Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,

            image_format,

            image_extent: surface.window().inner_size().into(),

            image_usage: ImageUsage::color_attachment(),

            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .iter()
                .next()
                .unwrap(),

            ..Default::default()
        },
    )
    .unwrap();
}

fn create_viewport() -> Viewport {
    return Viewport {
        origin: [0.0, 0.0],
        dimensions: [1.0, 1.0],
        depth_range: 0.0..1.0,
    };
}

fn recreate_swapchain(
    old_swapchain: &Arc<Swapchain<Window>>,
    surface: &Arc<Surface<Window>>,
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let (new_swapchain, new_images) = match old_swapchain.recreate(SwapchainCreateInfo {
        image_extent: surface.window().inner_size().into(),
        ..old_swapchain.create_info()
    }) {
        Ok(r) => r,
        // This error tends to happen when the user is manually resizing the window.
        // Simply restarting the loop is the easiest way to fix this issue.
        //Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return, //TODO: Add it again?
        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
    };
    return (new_swapchain, new_images);
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn keypressed(input: &KeyboardInput, gs: &mut GameStatus) {
    if input.state == ElementState::Released {
        return;
    }

    static CURRENT_ID: AtomicUsize = AtomicUsize::new(1);
    let obj_id = CURRENT_ID.load(Ordering::Acquire);

    let current_obj = match gs.scene.get_mut(obj_id) {
        Some(obj) => obj,
        None => {
            println!("obj_id {} is not found!", obj_id);
            return;
        }
    };

    match input.virtual_keycode {
        /*
        Some(VirtualKeyCode::Up) => current_obj.move_up(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Down) => current_obj.move_down(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Left) => current_obj.move_left(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Right) => current_obj.move_right(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Q) => current_obj.move_absolute(-1.0, -1.0),
        Some(VirtualKeyCode::P) => current_obj.move_absolute(0.2, -1.0),
        Some(VirtualKeyCode::C) => current_obj.move_absolute(0.0, 0.0),
        Some(VirtualKeyCode::Z) => current_obj.move_absolute(-1.0, 0.2),
        Some(VirtualKeyCode::M) => current_obj.move_absolute(0.2, 0.2),
        Some(VirtualKeyCode::L) => current_obj.rotate_center(0.1),
        Some(VirtualKeyCode::A) => current_obj.rotate_center(-0.1),
        Some(VirtualKeyCode::D) => current_obj.scaling(0.9),
        Some(VirtualKeyCode::E) => current_obj.scaling(1.1),
        Some(VirtualKeyCode::K) => {
            let mut found_id = false;
            // TODO: add get_keys instead
            for (key, obj) in gs.scene.get_objs() {
                if found_id {
                    CURRENT_ID.store(*key, Ordering::Relaxed);
                    found_id = false;
                }
                if obj_id == *key {
                    found_id = true;
                }
            }
            if found_id {
                CURRENT_ID.store(
                    *gs.scene.get_objs().iter().next().unwrap().0,
                    Ordering::Relaxed,
                );
            }
        }
        Some(VirtualKeyCode::R) => current_obj.set_color(Color::new(1.0, 0.0, 0.0, 1.0)),
        Some(VirtualKeyCode::G) => current_obj.set_color(Color::new(0.0, 1.0, 0.0, 1.0)),
        Some(VirtualKeyCode::B) => current_obj.set_color(Color::new(0.0, 0.0, 1.0, 1.0)),
        Some(VirtualKeyCode::F) => {} /*current_obj.set_fill(!current_obj.get_fill())*/
        */
        Some(_) => {}
        None => {}
    }
}

fn main() {
    let instance = create_instance();

    let (surface, event_loop) = create_window(&instance);

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let features = Features {
        tessellation_shader: true,
        fill_mode_non_solid: true,
        ..Features::none()
    };

    let pdevice = list_devices(&instance, device_extensions.clone(), features.clone())
        .expect("No graphic devices found!");

    let (device, mut queues) =
        create_vulkan_device(&pdevice, &surface, device_extensions, features);
    // Take first queue
    let queue = queues.pop().unwrap();

    let (mut swapchain, images) = create_swapchain(&pdevice, &surface, &device);

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let mut render_scene = RenderScene::new();

    render_scene.add_triangle(
        Point2D::new(-0.5, -0.25),
        Point2D::new(0.0, 0.5),
        Point2D::new(0.25, -0.1),
        ColorRGBA::new(1.0, 1.0, 1.0, 1.0),
        true,
    );

    render_scene.add_triangle(
        Point2D::new(-0.9, 0.9),
        Point2D::new(-0.7, 0.6),
        Point2D::new(-0.5, 0.9),
        ColorRGBA::new(0.0, 0.0, 1.0, 1.0),
        false,
    );

    render_scene.add_rectangle(
        Point2D::new(-0.3, 0.4),
        0.3,
        0.7,
        ColorRGBA::new(0.0, 1.0, 0.0, 1.0),
        false,
    );

    render_scene.add_square(
        Point2D::new(0.3, -0.4),
        0.3,
        ColorRGBA::new(0.0, 1.0, 0.0, 1.0),
        true,
    );

    let mut gs = GameStatus {
        viewport: create_viewport(),
        sensitivity: 1.5,
        scene: render_scene,
    };

    for (id, obj) in gs.scene.get_objs_mut() {
        obj.init_pipelines(&device, &render_pass);
    }

    let mut b_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut gs.viewport);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                b_swapchain = true;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        device_id,
                        input,
                        is_synthetic,
                    },
                ..
            } => {
                println!("User event {:?}, {:?}", device_id, input);
                keypressed(&input, &mut gs);
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if b_swapchain {
                    let (new_swapchain, new_images) = recreate_swapchain(&swapchain, &surface);
                    // Get the new dimensions of the window.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut gs.viewport,
                    );
                    swapchain = new_swapchain;
                    b_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            b_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    b_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap();

                for (id, obj) in gs.scene.get_objs() {
                    let buffer = CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::all(),
                        false,
                        obj.get_vulkan_vertex(),
                    )
                    .unwrap();
                    builder
                        .set_viewport(0, [gs.viewport.clone()])
                        .bind_pipeline_graphics(obj.get_pipeline())
                        .bind_vertex_buffers(0, buffer.clone())
                        .draw(buffer.len() as u32, 1, 0, 0)
                        .unwrap();
                }

                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        b_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

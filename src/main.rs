use bytemuck::{Pod, Zeroable};
use std::f32::consts::SQRT_2;
use std::sync::Arc;
use vulkano::buffer::TypedBufferAccess;
use vulkano::command_buffer::{PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::device::Queue;
use vulkano::image::swapchain;
use vulkano::pipeline::compute;
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::GraphicsPipelineBuilder;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            render_pass::PipelineRenderingCreateInfo,
            tessellation::TessellationState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{LoadOp, StoreOp},
    swapchain::{
        acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};
use winit::event_loop;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct GameStatus {
    pub viewport: Viewport,
    pub sensitivity: f32,
    pub scene: RenderScene,
}

struct RenderObject {
    id: String,
    pipeline: Arc<GraphicsPipeline>,
    vertex: Vec<VulkanVertex2D>,
}

impl RenderObject {
    pub fn new(id: String, pipeline: Arc<GraphicsPipeline>, vertex: Vec<VulkanVertex2D>) -> Self {
        let (c_x, c_y) = Self::compute_center(&vertex);
        return Self {
            id: id,
            pipeline: pipeline,
            vertex: vertex,
        };
    }

    fn compute_center(vertex: &Vec<VulkanVertex2D>) -> (f32, f32) {
        let mut proj_sum_x = 0.0;
        let mut proj_sum_y = 0.0;
        let num_vertex: f32 = vertex.len() as f32;
        if (num_vertex == 0.0) {
            return (0.0, 0.0);
        }

        for v in vertex {
            proj_sum_x += v.position[0];
            proj_sum_y += v.position[1];
        }

        return (proj_sum_x / num_vertex, proj_sum_y / num_vertex);
    }

    pub fn move_absolute(&mut self, x: f32, y: f32) {
        let delta_x = x - self.vertex[0].position[0];
        let delta_y = y - self.vertex[0].position[1];
        self.move_relative(delta_x, delta_y);
    }
    pub fn move_relative(&mut self, delta_x: f32, delta_y: f32) {
        for v in &mut self.vertex {
            v.position[0] = v.position[0] + delta_x;
            v.position[1] = v.position[1] + delta_y;
        }
        println!(
            "Moving object id: {} at: (x={},y={})",
            self.id, self.vertex[0].position[0], self.vertex[0].position[1]
        );
    }
    pub fn move_right(&mut self, delta: f32) {
        self.move_relative(delta, 0.0);
    }
    pub fn move_left(&mut self, delta: f32) {
        self.move_relative(-delta, 0.0);
    }
    pub fn move_down(&mut self, delta: f32) {
        self.move_relative(0.0, delta);
    }
    pub fn move_up(&mut self, delta: f32) {
        self.move_relative(0.0, -delta);
    }
    pub fn rotate_point(&mut self, c_x: f32, c_y: f32, alpha: f32) {
        for v in &mut self.vertex {
            // taking relative center
            let delta_x = v.position[0] - c_x;
            let delta_y = v.position[1] - c_y;
            // rotating vertex
            v.position[0] = delta_x * alpha.cos() - delta_y * alpha.sin();
            v.position[1] = delta_x * alpha.sin() + delta_y * alpha.cos();
            // moving to center
            v.position[0] = v.position[0] + c_x;
            v.position[1] = v.position[1] + c_y;
        }
    }

    pub fn rotate_center(&mut self, alpha: f32) {
        let (c_x, c_y) = Self::compute_center(&self.vertex);
        self.rotate_point(c_x, c_y, alpha);
    }

    pub fn scaling(&mut self, x: f32) {
        assert!(x > 0.0);
        for v in &mut self.vertex {
            v.position[0] = v.position[0] * x;
            v.position[1] = v.position[1] * x;
        }
    }
}

struct RenderScene {
    pub objs: Vec<RenderObject>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct VulkanVertex2D {
    position: [f32; 2],
}
impl_vertex!(VulkanVertex2D, position);

mod tcs {
    vulkano_shaders::shader! {
        ty: "tess_ctrl",
        src: "
                    #version 450
                    layout (vertices = 3) out; // a value of 3 means a patch consists of a single triangle
                    void main(void)
                    {
                        // save the position of the patch, so the tes can access it
                        // We could define our own output variables for this,
                        // but gl_out is handily provided.
                        gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
                        gl_TessLevelInner[0] = 10; // many triangles are generated in the center
                        gl_TessLevelOuter[0] = 1;  // no triangles are generated for this edge
                        gl_TessLevelOuter[1] = 10; // many triangles are generated for this edge
                        gl_TessLevelOuter[2] = 10; // many triangles are generated for this edge
                        // gl_TessLevelInner[1] = only used when tes uses layout(quads)
                        // gl_TessLevelOuter[3] = only used when tes uses layout(quads)
                    }
                "
    }
}

mod tes {
    vulkano_shaders::shader! {
        ty: "tess_eval",
        src: "
        #version 450
        layout(triangles, equal_spacing, cw) in;
        void main(void)
        {
            // retrieve the vertex positions set by the tcs
            vec4 vert_x = gl_in[0].gl_Position;
            vec4 vert_y = gl_in[1].gl_Position;
            vec4 vert_z = gl_in[2].gl_Position;
            // convert gl_TessCoord from barycentric coordinates to cartesian coordinates
            gl_Position = vec4(
                gl_TessCoord.x * vert_x.x + gl_TessCoord.y * vert_y.x + gl_TessCoord.z * vert_z.x,
                gl_TessCoord.x * vert_x.y + gl_TessCoord.y * vert_y.y + gl_TessCoord.z * vert_z.y,
                gl_TessCoord.x * vert_x.z + gl_TessCoord.y * vert_y.z + gl_TessCoord.z * vert_z.z,
                1.0
            );
        }"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "#version 450
        layout(location = 0) out vec4 f_color;
        void main() {
            f_color = vec4(1.0, 1.0, 1.0, 1.0);
        }"
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
                    #version 450
                    layout(location = 0) in vec2 position;
                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                "
    }
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

fn create_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();
    let tcs = tcs::load(device.clone()).unwrap();
    let tes = tes::load(device.clone()).unwrap();

    return GraphicsPipeline::start()
        // We need to indicate the layout of the vertices.
        .vertex_input_state(BuffersDefinition::new().vertex::<VulkanVertex2D>())
        .tessellation_shaders(
            tcs.entry_point("main").unwrap(),
            (),
            tes.entry_point("main").unwrap(),
            (),
        )
        // The content of the vertex buffer describes a list of triangles.
        .input_assembly_state(
            /*InputAssemblyState::new()*/
            InputAssemblyState::new().topology(PrimitiveTopology::PatchList),
        )
        .rasterization_state(RasterizationState::new().polygon_mode(PolygonMode::Line))
        .tessellation_state(
            TessellationState::new()
                // Use a patch_control_points of 3, because we want to convert one triangle into
                // lots of little ones. A value of 4 would convert a rectangle into lots of little
                // triangles.
                .patch_control_points(3),
        )
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one.
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // Use a resizable viewport set to draw over the entire window
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        // See `vertex_shader`.
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
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
        //Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
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

    match input.virtual_keycode {
        Some(VirtualKeyCode::Up) => gs.scene.objs[0].move_up(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Down) => gs.scene.objs[0].move_down(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Left) => gs.scene.objs[0].move_left(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Right) => gs.scene.objs[0].move_right(0.1 * gs.sensitivity),
        Some(VirtualKeyCode::Q) => gs.scene.objs[0].move_absolute(-1.0, -1.0),
        Some(VirtualKeyCode::P) => gs.scene.objs[0].move_absolute(0.2, -1.0),
        Some(VirtualKeyCode::C) => gs.scene.objs[0].move_absolute(0.0, 0.0),
        Some(VirtualKeyCode::Z) => gs.scene.objs[0].move_absolute(-1.0, 0.2),
        Some(VirtualKeyCode::M) => gs.scene.objs[0].move_absolute(0.2, 0.2),
        Some(VirtualKeyCode::L) => gs.scene.objs[0].rotate_center(0.1),
        Some(VirtualKeyCode::A) => gs.scene.objs[0].rotate_center(-0.1),
        Some(VirtualKeyCode::D) => gs.scene.objs[0].scaling(0.9),
        Some(VirtualKeyCode::F) => gs.scene.objs[0].scaling(1.1),
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

    let mut render_scene = RenderScene { objs: Vec::new() };

    let render_obj1 = RenderObject::new(
        "render1".to_string(),
        create_pipeline(&device, &render_pass),
        vec![
            VulkanVertex2D {
                position: [-0.5, -0.25],
            },
            VulkanVertex2D {
                position: [0.0, 0.5],
            },
            VulkanVertex2D {
                position: [0.25, -0.1],
            },
        ],
    );

    let render_obj2 = RenderObject::new(
        "render2".to_string(),
        create_pipeline(&device, &render_pass),
        vec![
            VulkanVertex2D {
                position: [-0.9, 0.9],
            },
            VulkanVertex2D {
                position: [-0.7, 0.6],
            },
            VulkanVertex2D {
                position: [-0.5, 0.9],
            },
        ],
    );

    render_scene.objs.push(render_obj1);
    render_scene.objs.push(render_obj2);

    let mut gs = GameStatus {
        viewport: create_viewport(),
        sensitivity: 1.5,
        scene: render_scene,
    };

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
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap();

                for render_obj in &gs.scene.objs {
                    let buffer = CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::all(),
                        false,
                        render_obj.vertex.clone(),
                    )
                    .unwrap();
                    builder
                        .set_viewport(0, [gs.viewport.clone()])
                        .bind_pipeline_graphics(render_obj.pipeline.clone())
                        .bind_vertex_buffers(0, buffer.clone())
                        .draw(buffer.len() as u32, 1, 0, 0)
                        .unwrap();
                }

                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    //.join(acquire_future)
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

use std::{
    collections::HashMap,
    hash::Hash,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use vulkano::{
    device::Device,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline,
    },
    render_pass::{RenderPass, Subpass},
};

use crate::types::{ColorRGBA, Point2D, Vertex2D, VulkanVertex2D};

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
    
    
            layout(location = 0) in vec4 color;
    
            layout(location = 0) out vec4 f_color;
            
            void main() {
                f_color = color;
            }"
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
                    #version 450
    
                    layout(location = 0) in vec2 position;
                    layout(location = 1) in vec4 color;
    
                    layout(location = 0) out vec4 f_color;
                    
    
                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                        f_color = color;
                    }
                "
    }
}

pub struct Mesh {
    vec_vertex: Vec<Vertex2D>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    primitive: PrimitiveTopology,
    polygon_mode: PolygonMode,
}

impl Mesh {
    pub fn new(
        vec_vertex: Vec<Vertex2D>,
        primitive: PrimitiveTopology,
        polygon_mode: PolygonMode,
    ) -> Self {
        Self {
            vec_vertex: vec_vertex,
            pipeline: None,
            primitive: primitive,
            polygon_mode: polygon_mode,
        }
    }

    pub fn get_pipeline(&self) -> Option<Arc<GraphicsPipeline>> {
        self.pipeline.clone()
    }

    pub fn build_pipeline(&mut self, device: &Arc<Device>, render_pass: &Arc<RenderPass>) {
        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();
        //let tcs = tcs::load(device.clone()).unwrap();
        //let tes = tes::load(device.clone()).unwrap();

        /*.tessellation_shaders(
            tcs.entry_point("main").unwrap(),
            (),
            tes.entry_point("main").unwrap(),
            (),
        )*/

        let builder = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<VulkanVertex2D>())
            .input_assembly_state(InputAssemblyState::new().topology(self.primitive))
            .rasterization_state(RasterizationState::new().polygon_mode(self.polygon_mode))
            /*.tessellation_state(
                TessellationState::new()
                    // Use a patch_control_points of 3, because we want to convert one triangle into
                    // lots of little ones. A value of 4 would convert a rectangle into lots of little
                    // triangles.
                    .patch_control_points(3),
            )*/
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // Use a resizable viewport set to draw over the entire window
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            // See `vertex_shader`.
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap());
        self.pipeline = Some(builder.build(device.clone()).unwrap());
    }
    /*
        pub fn get_vertex_x(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.position[0]),
                None => None
            }
        }

        pub fn get_vertex_y(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.position[1]),
                None => None
            }
        }

        pub fn get_vertex_r(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.color[0]),
                None => None
            }
        }

        pub fn get_vertex_g(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.color[1]),
                None => None
            }
        }

        pub fn get_vertex_b(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.color[2]),
                None => None
            }
        }

        pub fn get_vertex_a(&self, index: usize) -> Option<f32> {
            match self.vec_vertex.get(index) {
                Some(value) => Some(value.color[3]),
                None => None
            }
        }

        pub fn set_vertex_x(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.position[0] = val; true},
                None => false
            }
        }

        pub fn set_vertex_y(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.position[1] = val; true},
                None => false
            }
        }

        pub fn set_vertex_r(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.color[0] = val; true},
                None => false
            }
        }

        pub fn set_vertex_g(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.color[1] = val; true},
                None => false
            }
        }

        pub fn set_vertex_b(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.color[2] = val; true},
                None => false
            }
        }

        pub fn set_vertex_a(&mut self, index: usize, val: f32) -> bool {
            match self.vec_vertex.get(index) {
                Some(value) => {value.color[3] = val; true},
                None => false
            }
        }

    */
    pub fn get_vertex_position(&mut self, index: usize) -> Option<Point2D> {
        match self.vec_vertex.get(index) {
            Some(v) => Some(v.get_position()),
            None => None,
        }
    }

    pub fn set_vertex_position(&mut self, index: usize, value: Point2D) -> bool {
        match self.vec_vertex.get_mut(index) {
            Some(v) => {
                v.set_position(value);
                true
            }
            None => false,
        }
    }

    pub fn get_vertex_color(&mut self, index: usize) -> Option<ColorRGBA> {
        match self.vec_vertex.get(index) {
            Some(v) => Some(v.get_color()),
            None => None,
        }
    }

    pub fn set_vertex_color(&mut self, index: usize, value: ColorRGBA) -> bool {
        match self.vec_vertex.get_mut(index) {
            Some(v) => {
                v.set_color(value);
                true
            }
            None => false,
        }
    }

    fn compute_center(vertex: &Vec<Vertex2D>) -> Option<Point2D> {
        let mut proj_sum_x = 0.0;
        let mut proj_sum_y = 0.0;
        let num_vertex: f32 = vertex.len() as f32;
        if num_vertex == 0.0 {
            return None;
        }

        for v in vertex {
            let pos = v.get_position();
            proj_sum_x += pos.get_x();
            proj_sum_y += pos.get_y();
        }

        return Some(Point2D::new(
            proj_sum_x / num_vertex,
            proj_sum_y / num_vertex,
        ));
    }

    pub fn move_absolute(&mut self, x: f32, y: f32) {
        let pos = self.get_vertex_position(0).unwrap();
        let delta_x = x - pos.get_x();
        let delta_y = y - pos.get_y();
        self.move_relative(delta_x, delta_y);
    }
    pub fn move_relative(&mut self, delta_x: f32, delta_y: f32) {
        for v in &mut self.vec_vertex {
            v.pmove(delta_x, delta_y);
        }
        println!("Moving object at: {:?}", self.vec_vertex[0].get_position());
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
    pub fn rotate_point(&mut self, cp: Point2D, alpha: f32) {
        for v in &mut self.vec_vertex {
            v.rotate(cp.clone(), alpha);
        }
    }

    pub fn rotate_center(&mut self, alpha: f32) {
        let cp = match Self::compute_center(&self.vec_vertex) {
            Some(point) => point,
            None => return, //TODO: add a message
        };

        self.rotate_point(cp, alpha);
    }

    pub fn scale(&mut self, x: f32) {
        assert!(x > 0.0);
        for v in &mut self.vec_vertex {
            v.scale(x);
        }
    }

    pub fn set_color(&mut self, color: ColorRGBA) {
        for v in &mut self.vec_vertex {
            v.set_color(color.clone());
        }
    }

    pub fn get_vulkan_mesh(&self) -> Vec<VulkanVertex2D> {
        let mut vulkan_mesh = Vec::with_capacity(self.vec_vertex.len());
        for v in &self.vec_vertex {
            vulkan_mesh.push(v.get_vulkan_vertex());
        }

        vulkan_mesh
    }
}

pub struct RenderObject {
    id: usize,
    mesh: Mesh,
}

impl RenderObject {
    pub fn new(mesh: Mesh) -> Self {
        static ID_COUNTER: AtomicUsize = AtomicUsize::new(1);
        let id = ID_COUNTER.fetch_add(1, Ordering::Acquire);
        println!("Added RenderObject id:{}", id);
        Self { id: id, mesh: mesh }
    }

    pub fn init_pipelines(&mut self, device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> bool {
        self.mesh.build_pipeline(device, render_pass);
        true
    }

    pub fn get_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.mesh
            .get_pipeline()
            .expect("Cannot get pipeline from RenderObject")
    }

    pub fn get_vulkan_vertex(&self) -> Vec<VulkanVertex2D> {
        self.mesh.get_vulkan_mesh()
    }
}

pub struct RenderScene {
    objs: HashMap<usize, RenderObject>,
}

impl RenderScene {
    pub fn new() -> Self {
        Self {
            objs: HashMap::new(),
        }
    }
    pub fn get_objs(&self) -> &HashMap<usize, RenderObject> {
        return &self.objs;
    }

    pub fn get_objs_mut(&mut self) -> &mut HashMap<usize, RenderObject> {
        return &mut self.objs;
    }

    pub fn get(&self, k: usize) -> Option<&RenderObject> {
        self.objs.get(&k)
    }

    pub fn get_mut(&mut self, k: usize) -> Option<&mut RenderObject> {
        self.objs.get_mut(&k)
    }

    pub fn add_triangle(
        &mut self,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        color: ColorRGBA,
        is_fill: bool,
    ) -> usize {
        let obj = RenderObject::new(Mesh::new(
            vec![
                Vertex2D::new(p1, color.clone()),
                Vertex2D::new(p2, color.clone()),
                Vertex2D::new(p3, color.clone()),
            ],
            PrimitiveTopology::TriangleList,
            match is_fill {
                false => PolygonMode::Line,
                true => PolygonMode::Fill,
            },
        ));
        let id = obj.id;

        self.objs.insert(id, obj);

        id
    }

    pub fn add_rectangle(
        &mut self,
        topdown: Point2D,
        h: f32,
        w: f32,
        color: ColorRGBA,
        is_fill: bool,
    ) -> usize {
        let obj = RenderObject::new(Mesh::new(
            vec![
                Vertex2D::new(topdown.clone(), color.clone()),
                Vertex2D::new(topdown.clone_move_x(w), color.clone()),
                Vertex2D::new(topdown.clone_move(w, h), color.clone()),
                Vertex2D::new(topdown.clone_move_y(h), color.clone()),
                Vertex2D::new(topdown.clone(), color.clone()),
            ],
            match is_fill {
                false => PrimitiveTopology::LineStrip,
                true => PrimitiveTopology::TriangleStrip,
            },
            match is_fill {
                false => PolygonMode::Line,
                true => PolygonMode::Fill,
            },
        ));

        let id = obj.id;
        self.objs.insert(id, obj);

        id
    }

    pub fn add_square(&mut self, topdown: Point2D, l: f32, color: ColorRGBA, fill: bool) -> usize {
        self.add_rectangle(topdown, l, l, color, fill)
    }
}

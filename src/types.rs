use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct VulkanVertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}
impl_vertex!(VulkanVertex2D, position, color);

#[derive(Clone, Debug, Copy)]
pub struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x: x, y: y }
    }

    pub fn get_x(&self) -> f32 {
        self.x
    }

    pub fn get_y(&self) -> f32 {
        self.y
    }

    pub fn set_x(&mut self, value: f32) {
        self.x = value;
    }

    pub fn set_y(&mut self, value: f32) {
        self.y = value;
    }

    pub fn pmove(&mut self, delta_x: f32, delta_y: f32) {
        self.x += delta_x;
        self.y += delta_y;
    }

    pub fn move_x(&mut self, delta_x: f32) {
        self.x += delta_x;
    }

    pub fn move_y(&mut self, delta_y: f32) {
        self.y += delta_y;
    }

    pub fn rotate(&mut self, cp: Point2D, alpha: f32) {
        // taking relative center
        let delta_x = self.x - cp.x;
        let delta_y = self.y - cp.y;
        // rotating vertex
        self.x = delta_x * alpha.cos() - delta_y * alpha.sin();
        self.y = delta_x * alpha.sin() + delta_y * alpha.cos();
        // moving to center
        self.pmove(cp.x, cp.y);
    }
    pub fn scale(&mut self, multiplicator: f32) {
        self.x *= multiplicator;
        self.y *= multiplicator;
    }

    pub fn clone_move(&self, delta_x: f32, delta_y: f32) -> Self {
        Self {
            x: self.x + delta_x,
            y: self.y + delta_y,
        }
    }

    pub fn clone_move_x(&self, delta_x: f32) -> Self {
        Self {
            x: self.x + delta_x,
            y: self.y,
        }
    }

    pub fn clone_move_y(&self, delta_y: f32) -> Self {
        Self {
            x: self.x,
            y: self.y + delta_y,
        }
    }

    pub fn get_slice(&self) -> [f32; 2] {
        [self.x, self.y]
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ColorRGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl ColorRGBA {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: r,
            g: g,
            b: b,
            a: a,
        }
    }

    pub fn get_r(&self) -> f32 {
        self.r
    }

    pub fn get_g(&self) -> f32 {
        self.g
    }

    pub fn get_b(&self) -> f32 {
        self.b
    }

    pub fn get_a(&self) -> f32 {
        self.a
    }

    pub fn set_r(&mut self, value: f32) {
        self.r = value;
    }

    pub fn set_g(&mut self, value: f32) {
        self.g = value;
    }

    pub fn set_b(&mut self, value: f32) {
        self.b = value;
    }

    pub fn set_a(&mut self, value: f32) {
        self.a = value;
    }

    pub fn get_slice(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Vertex2D {
    position: Point2D,
    color: ColorRGBA,
}

impl Vertex2D {
    pub fn new(position: Point2D, color: ColorRGBA) -> Self {
        Self {
            position: position,
            color: color,
        }
    }

    pub fn get_position(&self) -> Point2D {
        self.position
    }

    pub fn set_position(&mut self, value: Point2D) {
        self.position = value
    }

    pub fn pmove(&mut self, delta_x: f32, delta_y: f32) {
        self.position.pmove(delta_x, delta_y);
    }

    pub fn rotate(&mut self, cp: Point2D, alpha: f32) {
        self.position.rotate(cp, alpha);
    }

    pub fn scale(&mut self, x: f32) {
        self.scale(x);
    }

    pub fn get_color(&self) -> ColorRGBA {
        self.color
    }

    pub fn set_color(&mut self, value: ColorRGBA) {
        self.color = value
    }

    pub fn get_vulkan_vertex(&self) -> VulkanVertex2D {
        VulkanVertex2D {
            position: self.position.get_slice(),
            color: self.color.get_slice(),
        }
    }
}

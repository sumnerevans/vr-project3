// Flight

use flight::{Error, Light, PbrMesh, Texture};
use flight::draw::{DrawParams, Painter, PbrMaterial, PbrStyle, SolidStyle};
use flight::load;
use flight::mesh::*;
use flight::vr::{ViveController, VrMoment, primary, secondary};

// GFX
use gfx::{self, Factory};
use gfx::traits::FactoryExt;

// nalgebra
use nalgebra::{Point2, Point3, Rotation3, SimilarityMatrix3, Translation3, Vector3, self as na};

// Physics
use ncollide::shape::{Cuboid, Plane};
use nphysics3d::detection::constraint::Constraint;
use nphysics3d::object::{RigidBody, RigidBodyHandle};
use nphysics3d::world::World;

use std::path::Path;
use std::time::Instant;

// Constants
pub const NEAR_PLANE: f64 = 0.1;
pub const FAR_PLANE: f64 = 1000.;
pub const BACKGROUND: [f32; 4] = [0., 0., 0., 1.0];
const PI: f32 = ::std::f32::consts::PI;
const PI2: f32 = 2. * PI;
const DEG: f32 = PI2 / 360.;

pub struct App<R: gfx::Resources> {
    solid: Painter<R, SolidStyle<R>>,
    objects: Vec<(RigidBodyHandle<f32>, PbrMesh<R>)>,
    pbr: Painter<R, PbrStyle<R>>,
    grid: Mesh<R, VertC, ()>,
    controller: PbrMesh<R>,
    snow_block: PbrMesh<R>,
    snowman: PbrMesh<R>,
    last_time: Option<Instant>,
    primary: ViveController,
    secondary: ViveController,
    world: World<f32>,
}

fn grid_lines(count: i32, size: Vector3<f32>) -> MeshSource<VertC, ()> {
    let mut lines = Vec::new();
    let base_color = [1., 1., 1.];
    let light_color = [0.2, 0.2, 0.5];
    let rad = size / 2.;
    let mult = size / count as f32;

    for a in 0..(count + 1) {
        for b in 0..(count + 1) {
            let line_color = if a % 2 == 0 && b % 2 == 0 {
                [base_color; 3]
            } else {
                [light_color; 3]
            };
            let a = a as f32 * mult - rad;
            let b = b as f32 * mult - rad;
            lines.push(VertC {
                pos: [-rad.x, a.y, b.z],
                color: line_color[0],
            });
            lines.push(VertC {
                pos: [rad.x, a.y, b.z],
                color: line_color[0],
            });
            lines.push(VertC {
                pos: [a.x, -rad.y, b.z],
                color: line_color[1],
            });
            lines.push(VertC {
                pos: [a.x, rad.y, b.z],
                color: line_color[1],
            });
            lines.push(VertC {
                pos: [a.x, b.y, -rad.z],
                color: line_color[2],
            });
            lines.push(VertC {
                pos: [a.x, b.y, rad.z],
                color: line_color[2],
            });
        }
    }

    MeshSource {
        verts: lines,
        inds: Indexing::All,
        prim: Primitive::LineList,
        mat: (),
    }
}

fn load_simple_object<P, R, F>(f: &mut F,
                               path: P,
                               albedo: [u8; 4])
                               -> Result<Mesh<R, VertNTT, PbrMaterial<R>>, Error>
    where P: AsRef<Path>,
          R: gfx::Resources,
          F: gfx::Factory<R>
{
    use gfx::format::*;
    Ok(load::wavefront_file(path)
        ?
        .compute_tan()
        .with_material(PbrMaterial {
            normal: Texture::<_, (R8_G8_B8_A8, Unorm)>::uniform_value(f, albedo)?,
            albedo: Texture::<_, (R8_G8_B8_A8, Srgb)>::uniform_value(f, [0x60, 0x60, 0x60, 0xFF])?,
            metalness: Texture::<_, (R8, Unorm)>::uniform_value(f, 0x00)?,
            roughness: Texture::<_, (R8, Unorm)>::uniform_value(f, 0x20)?,
        })
        .upload(f))
}

impl<R: gfx::Resources> App<R> {
    pub fn new<F: Factory<R> + FactoryExt<R>>(factory: &mut F) -> Result<Self, Error> {
        // Setup Painters
        let mut solid = Painter::new(factory)?;
        solid.setup(factory, Primitive::LineList)?;
        solid.setup(factory, Primitive::TriangleList)?;

        let mut pbr: Painter<_, PbrStyle<_>> = Painter::new(factory)?;
        pbr.setup(factory, Primitive::TriangleList)?;

        // Set static world physics
        let mut world = World::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

        // Floor Plane
        let floor = Plane::new(Vector3::new(0.0, 1.0, 0.0));
        let mut floor_rb = RigidBody::new_static(floor, 0.1, 0.6);
        floor_rb.set_margin(0.00001);
        world.add_rigid_body(floor_rb);

        let snow_block = load::object_directory(factory, "assets/snow-block/")?;

        // TODO: REMOVE, need to do controllers
        let block = Cuboid::new(Vector3::new(0.15, 0.15, 0.3));
        let mut block_rb = RigidBody::new_dynamic(block, 100., 0.0, 0.8);
        block_rb.set_margin(0.00001);
        block_rb.set_translation(Translation3::new(1.5, 1.0, 1.5));

        let block = Cuboid::new(Vector3::new(0.15, 0.15, 0.3));
        let mut block_rb2 = RigidBody::new_dynamic(block, 100., 0.0, 0.8);
        block_rb2.set_margin(0.00001);
        block_rb2.set_translation(Translation3::new(1.5, 2.0, 1.5));

        let objs = vec![(world.add_rigid_body(block_rb), snow_block.clone()),
                        (world.add_rigid_body(block_rb2), snow_block.clone())];

        // Construct App
        Ok(App {
            solid: solid,
            pbr: pbr,
            objects: objs,
            grid: grid_lines(8, Vector3::new(8., 8., 8.)).upload(factory),
            controller: load_simple_object(factory,
                                           "assets/controller.obj",
                                           [0x80, 0x80, 0xFF, 0xFF])?,
            snowman: load::object_directory(factory, "assets/snowman/")?,
            snow_block: snow_block,
            last_time: None,
            primary: ViveController {
                is: primary(),
                pad: Point2::new(1., 0.),
                ..Default::default()
            },
            secondary: ViveController { is: secondary(), ..Default::default() },
            world: world,
        })
    }

    pub fn draw<C: gfx::CommandBuffer<R>>(&mut self, ctx: &mut DrawParams<R, C>, vrm: &VrMoment) {
        let dt = if let Some(last) = self.last_time {
            let elapsed = last.elapsed();
            elapsed.as_secs() as f64 + (elapsed.subsec_nanos() as f64 * 1e-9)
        } else {
            0.
        };
        self.last_time = Some(Instant::now());

        match (self.primary.update(vrm), self.secondary.update(vrm)) {
            (Ok(_), Ok(_)) => (),
            _ => warn!("A not vive-like controller is connected"),
        }

        // Clear targets
        ctx.encoder.clear_depth(&ctx.depth, FAR_PLANE as f32);
        ctx.encoder.clear(&ctx.color,
                          [BACKGROUND[0].powf(1. / 2.2),
                           BACKGROUND[1].powf(1. / 2.2),
                           BACKGROUND[2].powf(1. / 2.2),
                           BACKGROUND[3]]);

        // Controller light
        let cont_light = if self.secondary.connected {
            Light {
                pos: self.secondary.pose * Point3::new(0., 0., -0.1),
                color: [0.6, 0.6, 0.6, 10. * self.secondary.trigger as f32],
            }
        } else {
            Default::default()
        };

        // Config PBR lights
        self.pbr.cfg(|s| {
            s.ambient([0.2, 0.2, 0.2, 1.0]);
            s.lights(&[Light {
                           pos: vrm.stage * Point3::new(4., 8., 4.),
                           color: [0.9, 0.8, 0.7, 10.],
                       },
                       Light {
                           pos: vrm.stage * Point3::new(-4., 8., 4.),
                           color: [0.9, 0.8, 0.7, 10.],
                       },
                       Light {
                           pos: vrm.stage * Point3::new(-4., 8., -4.),
                           color: [0.9, 0.8, 0.7, 10.],
                       },
                       Light {
                           pos: vrm.stage * Point3::new(4., 8., -4.),
                           color: [0.9, 0.8, 0.7, 10.],
                       },
                       cont_light]);
        });

        // Draw grid
        self.solid.draw(ctx, vrm.stage * Translation3::new(0., 4., 0.), &self.grid);

        // Draw snowmen
        let snowman1_mtx = vrm.stage * Translation3::new(2., 0., 2.);
        let snowman2_mtx = vrm.stage * Translation3::new(-2., 0., 2.);
        let snowman3_mtx = vrm.stage * Translation3::new(-2., 0., -2.);
        let snowman4_mtx = vrm.stage * Translation3::new(2., 0., -2.);
        self.pbr.draw(ctx, snowman1_mtx, &self.snowman);
        self.pbr.draw(ctx, snowman2_mtx, &self.snowman);
        self.pbr.draw(ctx, snowman3_mtx, &self.snowman);
        self.pbr.draw(ctx, snowman4_mtx, &self.snowman);

        // PHYSICS ===========================================================
        self.world.step(dt as f32);

        // Draw the snow blocks
        for block in &self.objects {
            let block_pos = na::convert(vrm.stage * (*block.0.borrow().position()));
            self.pbr.draw(ctx, block_pos, &self.snow_block);
        }

        // Draw controllers
        for cont in vrm.controllers() {
            self.pbr.draw(ctx, na::convert(cont.pose), &self.controller);
        }
    }
}

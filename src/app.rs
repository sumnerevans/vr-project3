// Flight

use flight::{Error, Light, PbrMesh, Texture};
use flight::draw::{DrawParams, Painter, PbrMaterial, PbrStyle, SolidStyle};
use flight::load;
use flight::mesh::*;
use flight::vr::{Trackable, ViveController, VrMoment, primary, secondary};

// GFX
use gfx::{self, Factory};
use gfx::traits::FactoryExt;

// nalgebra
use nalgebra::{Point2, Point3, Rotation3, Similarity3, Translation3, Vector3, self as na};

// Physics
use nalgebra::geometry::Isometry3;
use ncollide::query::Ray;
use ncollide::shape::{Cuboid, Plane};
use ncollide::world::CollisionGroups;
use nphysics3d::detection::constraint::Constraint;
use nphysics3d::object::{RigidBody, RigidBodyCollisionGroups, RigidBodyHandle, WorldObject};
use nphysics3d::world::World;
use num_traits::bounds::Bounded;

// Standard Library
use std::f32;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

// Constants
pub const NEAR_PLANE: f64 = 0.1;
pub const FAR_PLANE: f64 = 1000.;
pub const BACKGROUND: [f32; 4] = [0., 0., 0., 1.0];
pub const TRIGGER_THRESHOLD: f64 = 0.5;
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
    red_ray: Mesh<R, VertC, ()>,
    blue_ray: Mesh<R, VertC, ()>,
    grabbed: Option<RigidBody<f32>>,
}

fn make_ray(color: [f32; 3]) -> MeshSource<VertC, ()> {
    MeshSource {
        verts: vec![VertC {
                        pos: [0., 0., 0.],
                        color: color,
                    },
                    VertC {
                        pos: [0., 0., -4.],
                        color: color,
                    }],
        inds: Indexing::All,
        prim: Primitive::LineList,
        mat: (),
    }
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
        let floor = Plane::new(Vector3::new(0., 1., 0.));
        let mut floor_rb = RigidBody::new_static(floor, 0.1, 0.6);
        floor_rb.set_margin(0.00001);
        world.add_rigid_body(floor_rb);

        let snow_block = load::object_directory(factory, "assets/snow-block/")?;
        let objs = Vec::new();

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
            red_ray: make_ray([1., 0., 0.]).upload(factory),
            blue_ray: make_ray([0., 0., 1.]).upload(factory),
            grabbed: None,
        })
    }

    fn handle_physics(&mut self) {
        // Calculate dt
        let dt = if let Some(last) = self.last_time {
            let elapsed = last.elapsed();
            elapsed.as_secs() as f64 + (elapsed.subsec_nanos() as f64 * 1e-9)
        } else {
            0.
        };
        self.last_time = Some(Instant::now());

        // Step the physics world
        self.world.step(dt.min(0.1) as f32);
    }

    pub fn draw<C: gfx::CommandBuffer<R>>(&mut self, ctx: &mut DrawParams<R, C>, vrm: &VrMoment) {
        match (self.primary.update(vrm), self.secondary.update(vrm)) {
            (Ok(_), Ok(_)) => (),
            _ => warn!("A not vive-like controller is connected"),
        }

        self.handle_physics();

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

        // Draw snowmen
        let snowmen_locations = vec![Translation3::new(2., 0., 2.),
                                     Translation3::new(-2., 0., 2.),
                                     Translation3::new(-2., 0., -2.),
                                     Translation3::new(2., 0., -2.)];
        for loc in snowmen_locations {
            self.pbr.draw(ctx, vrm.stage * loc, &self.snowman);
        }

        // Draw the snow blocks
        for block in &self.objects {
            let block_pos = na::convert(vrm.stage * (*block.0.borrow().position()));
            self.pbr.draw(ctx, block_pos, &self.snow_block);
        }

        // Draw the currently grabbed block
        if let Some(ref g) = self.grabbed {
            let grabbed_pos = na::convert(vrm.stage * g.position());
            self.pbr.draw(ctx, grabbed_pos, &self.snow_block);
        }

        // Draw grid
        self.solid.draw(ctx, vrm.stage * Translation3::new(0., 4., 0.), &self.grid);

        // CONTROLLERS ========================================================
        if !self.primary.connected || !self.secondary.connected {
            warn!("Controller(s) Disconnected");
        }

        // Handle Controller Events
        let stage_inv: Isometry3<f32> = na::try_convert(vrm.stage.try_inverse().unwrap()).unwrap();
        let pointing_at = |controller: &ViveController, world: &World<f32>| {
            let ray = Ray::new(stage_inv * controller.origin(),
                               stage_inv * controller.pointing());
            let all_groups = &CollisionGroups::new();

            // Track minimum value
            let mut mintoi = Bounded::max_value();
            let mut closest_body = None;
            for (b, inter) in world.collision_world()
                .interferences_with_ray(&ray, all_groups) {
                if inter.toi < mintoi {
                    if let &WorldObject::RigidBody(ref rb) = &b.data {
                        mintoi = inter.toi;
                        closest_body = Some(rb.clone());
                    }
                }
            }

            (closest_body, Some(mintoi))
        };

        let primary_pressed = self.primary.trigger > TRIGGER_THRESHOLD;
        let secondary_pressed = self.secondary.trigger > TRIGGER_THRESHOLD;

        // Determine what the controllers are are pointed at
        let primary_point_at = if self.primary.connected {
            pointing_at(&self.primary, &self.world)
        } else {
            (None, None)
        };

        let secondary_point_at = if self.secondary.connected {
            pointing_at(&self.secondary, &self.world)
        } else {
            (None, None)
        };

        // Handle spawning/moving of blocks
        if primary_pressed && secondary_pressed {
            // let lerp_vec = stage_inv * 0.5 *
            //                (self.primary.pose.translation.vector +
            //                 self.secondary.pose.translation.vector);
            // let lerp_trans = Translation3::from_vector(lerp_vec);
            // let lerp_rot =
            // self.primary.pose.rotation.slerp(&self.secondary.pose.rotation, 0.5);
            // let mid_controller = Isometry3::from_parts(lerp_trans, lerp_rot);
            if let Some(ref mut g) = self.grabbed {
                g.set_transformation(stage_inv * self.primary.pose);
            } else {
                match (primary_point_at.0, secondary_point_at.0) {
                    (None, None) => {
                        // The controllers are not pointed at any block and they are pointed
                        // generally downward.
                        let down = Vector3::new(0., -1., 0.);
                        if self.primary.pointing().dot(&down).acos() < PI / 4. &&
                           self.primary.pointing().dot(&down).acos() < PI / 4. {
                            let block = Cuboid::new(Vector3::new(0.15, 0.15, 0.3));
                            let mut block = RigidBody::new_dynamic(block, 100., 0.0, 0.8);
                            block.set_margin(0.00001);
                            block.set_transformation(stage_inv * self.primary.pose);
                            self.grabbed = Some(block);
                        }
                    }
                    (Some(p), Some(s)) => {
                        if Rc::ptr_eq(&p, &s) {
                            self.world.remove_rigid_body(&p);
                            self.objects.retain(|o| !Rc::ptr_eq(&p, &o.0));
                            self.grabbed = Some(p.borrow().clone());
                        }
                    }
                    _ => {}
                };
            }
        } else if !primary_pressed && !secondary_pressed {
            if let Some(g) = self.grabbed.take() {
                self.objects.push((self.world.add_rigid_body(g), self.snow_block.clone()));
            }
        }

        // Draw controllers
        let mut draw_controller = |controller: &ViveController, pressed, dist: f32| {
            self.pbr.draw(ctx, na::convert(controller.pose), &self.controller);
            let ray = if pressed {
                &self.red_ray
            } else {
                &self.blue_ray
            };
            let sim = Similarity3::from_isometry(controller.pose(), dist.min(5.).max(0.2));
            self.solid.draw(ctx, na::convert(sim), ray);
        };

        if primary_point_at.1.is_some() {
            draw_controller(&self.primary, primary_pressed, primary_point_at.1.unwrap());
        }

        if secondary_point_at.1.is_some() {
            draw_controller(&self.secondary,
                            secondary_pressed,
                            secondary_point_at.1.unwrap());
        }
    }
}

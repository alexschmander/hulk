use std::f32::consts::FRAC_PI_2;

use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, ImageSampler, ImageType},
    prelude::*,
};
use mujoco_rs::prelude::{MjSpec, MjtGeom, MjtJoint, SpecItem};

use super::visual::ObjectVisualAssets;
use crate::{
    bevy_mujoco::{MjcfObject, MujocoBody, MujocoWorld},
    parameters::{BallParameters, CurrentSimulatorParameters},
};

const BALL_BASE_COLOR: &str = "textures/football_base_color.png";
const BALL_NORMAL_MAP: &str = "textures/football_normal.png";

#[derive(Component)]
pub struct Ball;

/// Live balls in insertion order, independent of entity index reuse.
#[derive(Default, Resource)]
pub struct SpawnedBalls(pub Vec<Entity>);

pub fn first_position(world: &MujocoWorld, balls: &SpawnedBalls) -> color_eyre::Result<[f64; 3]> {
    let entity = balls
        .0
        .first()
        .ok_or_else(|| color_eyre::eyre::eyre!("No ball in scene"))?;
    let data = world.data();
    let body = data
        .body(&format!("object_{}_ball", entity.to_bits()))
        .ok_or_else(|| color_eyre::eyre::eyre!("Ball not compiled"))?;
    let p = body.view(data).xpos;
    Ok([p[0], p[1], p[2]])
}

pub fn record_spawn(event: On<Add<Ball>>, mut balls: ResMut<SpawnedBalls>) {
    balls.0.push(event.entity);
}

pub fn record_removal(event: On<Remove<Ball>>, mut balls: ResMut<SpawnedBalls>) {
    balls.0.retain(|entity| *entity != event.entity);
}

pub struct BallAssets {
    mesh: Handle<Mesh>,
    solid: Handle<StandardMaterial>,
    radius: f32,
    parameters: BallParameters,
}

impl BallAssets {
    pub fn load(world: &mut World) -> Self {
        let current = world.resource::<CurrentSimulatorParameters>();
        let radius = current.field_dimensions.ball_radius;
        let parameters = current.parameters.ball.clone();
        let mut load = |path: &str, srgb| {
            let bytes = std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("assets")
                    .join(path),
            )
            .expect("read ball texture");
            let image = Image::from_buffer(
                &bytes,
                ImageType::Extension("png"),
                CompressedImageFormats::NONE,
                srgb,
                ImageSampler::default(),
                RenderAssetUsages::default(),
            )
            .expect("decode ball texture");
            world.resource_mut::<Assets<Image>>().add(image)
        };
        let base_color_texture = load(BALL_BASE_COLOR, true);
        let normal_map_texture = load(BALL_NORMAL_MAP, false);
        let mesh = world.resource_mut::<Assets<Mesh>>().add(ball_mesh(radius));
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        Self {
            mesh,
            solid: materials.add(StandardMaterial {
                base_color_texture: Some(base_color_texture.clone()),
                normal_map_texture: Some(normal_map_texture.clone()),
                perceptual_roughness: 0.8,
                ..default()
            }),
            radius,
            parameters,
        }
    }

    fn mjcf_object(&self) -> MjcfObject {
        let radius = self.radius as f64;
        let parameters = self.parameters.clone();
        MjcfObject::from_factory(move || ball_spec(radius, &parameters), "ball")
            .with_free_joint("ball_free_joint")
            .grounded()
    }

    fn set_parameters(
        &mut self,
        radius: f32,
        parameters: BallParameters,
        meshes: &mut Assets<Mesh>,
    ) {
        if self.radius.to_bits() != radius.to_bits() {
            meshes
                .insert(self.mesh.id(), ball_mesh(radius))
                .expect("ball mesh should exist");
        }
        self.radius = radius;
        self.parameters = parameters;
    }
}

pub fn spawn(commands: &mut Commands, assets: &BallAssets, transform: Transform) -> Entity {
    let mut ball = commands.spawn_empty();
    let entity = ball.id();
    ball.insert((
        Ball,
        assets.mjcf_object(),
        MujocoBody::new(entity, "ball"),
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.solid.clone()),
        transform,
    ));
    entity
}

pub fn update_ball_dimensions(
    parameters: Res<CurrentSimulatorParameters>,
    mut assets: ResMut<ObjectVisualAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut balls: Query<&mut MjcfObject, With<Ball>>,
) {
    let radius = parameters.field_dimensions.ball_radius;
    let ball_parameters = &parameters.parameters.ball;
    if assets.ball.radius.to_bits() == radius.to_bits()
        && assets.ball.parameters == *ball_parameters
    {
        return;
    }

    assets
        .ball
        .set_parameters(radius, ball_parameters.clone(), &mut meshes);
    for mut object in &mut balls {
        *object = assets.ball.mjcf_object();
    }
}

pub(crate) fn ball_spec(radius: f64, parameters: &BallParameters) -> Result<MjSpec, String> {
    let mut spec = MjSpec::new();
    let body = spec.world_body_mut().add_body();
    body.set_name("ball").map_err(|error| error.to_string())?;

    let joint = body.add_joint();
    joint
        .set_name("ball_free_joint")
        .map_err(|error| error.to_string())?;
    joint
        .with_type(MjtJoint::mjJNT_FREE)
        .with_damping([parameters.joint_damping as f64, 0.0, 0.0])
        .with_frictionloss(parameters.joint_friction_loss as f64);

    let geom = body.add_geom();
    geom.set_name("ball").map_err(|error| error.to_string())?;
    geom.with_type(MjtGeom::mjGEOM_SPHERE)
        .with_size([radius, 0.0, 0.0])
        .with_mass(parameters.mass as f64)
        .with_friction(parameters.friction.map(f64::from))
        .with_solref(parameters.solref.map(f64::from))
        .with_solimp(parameters.solimp.map(f64::from))
        .with_priority(1)
        .with_condim(6);

    Ok(spec)
}

fn ball_mesh(radius: f32) -> Mesh {
    let mut mesh = Sphere::new(radius)
        .mesh()
        .uv(64, 32)
        .transformed_by(Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)));
    mesh.generate_tangents()
        .expect("UV sphere should support tangent generation");
    mesh
}

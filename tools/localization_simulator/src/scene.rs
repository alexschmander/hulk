use bevy::{
    asset::RenderAssetUsages,
    mesh::{PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use nalgebra::{Isometry3, Matrix3, Rotation3, UnitQuaternion, Vector3};
use types::{
    field_dimensions::FieldDimensions,
    field_marks::{FieldMark, field_marks_from_field_dimensions},
};

const TRUTH: Color = Color::srgb(0.0, 0.9, 0.95);
const BACKEND: Color = Color::srgb(1.0, 0.82, 0.05);
const LIVE: Color = Color::srgb(1.0, 0.05, 0.75);

#[derive(Clone, Default, Resource)]
pub(crate) struct SceneState {
    pub revision: u64,
    pub truth: Option<Isometry3<f32>>,
    pub backend: Option<Isometry3<f32>>,
    pub live: Option<Isometry3<f32>>,
    pub truth_history: Vec<Isometry3<f32>>,
    pub backend_history: Vec<Isometry3<f32>>,
    pub live_history: Vec<Isometry3<f32>>,
}

#[derive(Clone, Copy, Component)]
enum PoseKind {
    Truth,
    Backend,
    Live,
}

#[derive(Component)]
struct Trail(PoseKind);

pub(crate) fn configure(app: &mut App) {
    app.insert_resource(SceneState::default())
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 450.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (position_camera_once, update_markers, update_trails),
        );
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let dimensions = FieldDimensions::SPL_2025;
    let green = material(&mut materials, Color::srgb(0.035, 0.32, 0.11), false);
    let white = material(&mut materials, Color::WHITE, true);

    commands.spawn((
        PointLight {
            intensity: 2_000.0,
            range: 18.0,
            ..default()
        },
        Transform::from_xyz(0.0, 7.0, 0.0),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            dimensions.length + 2.0 * dimensions.border_strip_width,
            0.025,
            dimensions.width + 2.0 * dimensions.border_strip_width,
        ))),
        MeshMaterial3d(green),
        Transform::from_xyz(0.0, -0.015, 0.0),
    ));

    let half_length = dimensions.length / 2.0;
    let mut segments = Vec::new();
    for mark in field_marks_from_field_dimensions(&dimensions) {
        match mark {
            FieldMark::Line { line, .. } => {
                segments.push(([line.0.x(), line.0.y()], [line.1.x(), line.1.y()]))
            }
            FieldMark::Circle { center, radius } => {
                for index in 0..48 {
                    let start = std::f32::consts::TAU * index as f32 / 48.0;
                    let end = std::f32::consts::TAU * (index + 1) as f32 / 48.0;
                    segments.push((
                        [
                            center.x() + radius * start.cos(),
                            center.y() + radius * start.sin(),
                        ],
                        [
                            center.x() + radius * end.cos(),
                            center.y() + radius * end.sin(),
                        ],
                    ));
                }
            }
        }
    }
    for (start, end) in segments {
        spawn_line(
            &mut commands,
            &mut meshes,
            white.clone(),
            start,
            end,
            dimensions.line_width,
        );
    }

    for sign in [-1.0_f32, 1.0] {
        for side in [-1.0_f32, 1.0] {
            commands.spawn((
                Mesh3d(meshes.add(Cylinder::new(dimensions.goal_post_diameter / 2.0, 0.8))),
                MeshMaterial3d(white.clone()),
                Transform::from_xyz(
                    sign * half_length,
                    0.4,
                    -side * (dimensions.goal_inner_width + dimensions.goal_post_diameter) / 2.0,
                ),
            ));
        }
    }

    for (kind, color) in [
        (PoseKind::Truth, TRUTH),
        (PoseKind::Backend, BACKEND),
        (PoseKind::Live, LIVE),
    ] {
        let pose = commands
            .spawn((kind, Transform::default(), Visibility::Hidden))
            .id();
        commands.entity(pose).with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.28, 0.16, 0.2))),
                MeshMaterial3d(material(&mut materials, color, false)),
                Transform::default(),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.26, 0.07, 0.07))),
                MeshMaterial3d(material(&mut materials, color, true)),
                Transform::from_xyz(0.25, 0.0, 0.0),
            ));
        });
        commands.spawn((
            Trail(kind),
            Mesh3d(meshes.add(empty_line_mesh())),
            MeshMaterial3d(material(&mut materials, color, true)),
        ));
    }
}

fn material(
    materials: &mut Assets<StandardMaterial>,
    color: Color,
    unlit: bool,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        unlit,
        perceptual_roughness: 0.9,
        ..default()
    })
}

fn spawn_line(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
) {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(dx.hypot(dy), 0.012, width))),
        MeshMaterial3d(material),
        Transform::from_xyz((start[0] + end[0]) / 2.0, 0.008, -(start[1] + end[1]) / 2.0)
            .with_rotation(Quat::from_rotation_y(dy.atan2(dx))),
    ));
}

fn position_camera_once(
    mut positioned: Local<bool>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
    if *positioned {
        return;
    }
    for mut camera in &mut cameras {
        *camera = Transform::from_xyz(6.5, 7.5, 8.5).looking_at(Vec3::ZERO, Vec3::Y);
        *positioned = true;
    }
}

fn update_markers(
    state: Res<SceneState>,
    mut markers: Query<(&PoseKind, &mut Transform, &mut Visibility)>,
) {
    if !state.is_changed() {
        return;
    }
    for (kind, mut transform, mut visibility) in &mut markers {
        let pose = match kind {
            PoseKind::Truth => state.truth.as_ref(),
            PoseKind::Backend => state.backend.as_ref(),
            PoseKind::Live => state.live.as_ref(),
        };
        if let Some(pose) = pose {
            *transform = bevy_transform(pose);
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

fn update_trails(
    state: Res<SceneState>,
    trails: Query<(&Trail, &Mesh3d)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut previous: Local<(u64, [usize; 3])>,
) {
    if !state.is_changed() {
        return;
    }
    for (trail, mesh_handle) in &trails {
        let poses = match trail.0 {
            PoseKind::Truth => &state.truth_history,
            PoseKind::Backend => &state.backend_history,
            PoseKind::Live => &state.live_history,
        };
        if let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) {
            let index = match trail.0 {
                PoseKind::Truth => 0,
                PoseKind::Backend => 1,
                PoseKind::Live => 2,
            };
            let rebuild = previous.0 != state.revision
                || poses.len() < previous.1[index]
                || mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_none();
            if rebuild {
                let mut positions = Vec::with_capacity(poses.len().saturating_sub(1) * 2);
                for pair in poses.windows(2) {
                    positions.push(bevy_position(pair[0].translation.vector));
                    positions.push(bevy_position(pair[1].translation.vector));
                }
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            } else if poses.len() > previous.1[index] {
                let Some(VertexAttributeValues::Float32x3(positions)) =
                    mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
                else {
                    continue;
                };
                for current in previous.1[index].max(1)..poses.len() {
                    positions.push(bevy_position(poses[current - 1].translation.vector));
                    positions.push(bevy_position(poses[current].translation.vector));
                }
            }
            previous.1[index] = poses.len();
        }
    }
    previous.0 = state.revision;
}

fn empty_line_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::LineList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
}

fn bevy_transform(pose: &Isometry3<f32>) -> Transform {
    let basis = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
    let rotation = basis * pose.rotation.to_rotation_matrix().matrix() * basis.transpose();
    let quaternion =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation));
    let quaternion = quaternion.quaternion();
    Transform {
        translation: Vec3::from(bevy_position(pose.translation.vector)),
        rotation: Quat::from_xyzw(quaternion.i, quaternion.j, quaternion.k, quaternion.w),
        ..default()
    }
}

fn bevy_position(position: Vector3<f32>) -> [f32; 3] {
    [position.x, position.z, -position.y]
}

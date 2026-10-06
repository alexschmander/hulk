//! Paused robot placement through world-axis translation and rotation handles.
use bevy::{
    light::NotShadowCaster,
    picking::{
        backend::ray::{RayId, RayMap},
        pointer::PointerId,
    },
    prelude::*,
    transform::TransformSystems,
};

use super::{
    ball_interaction::BallSelection,
    field::FieldDropTarget,
    object::{ObjectKind, ObjectPart},
    palette::WorldCamera,
};
use crate::bevy_mujoco::{MujocoWorld, SetObjectPose, SimulationMode};

#[derive(Default, Resource)]
pub struct RobotSelection {
    selected: Option<Entity>,
    drag: Option<RobotDrag>,
}

impl RobotSelection {
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }
}

#[derive(Clone, Copy, Component)]
enum HandleKind {
    Translate(Vec3),
    Rotate(Vec3),
}

#[derive(Component)]
struct GizmoRoot;

struct RobotDrag {
    robot: Entity,
    handle: Entity,
    pointer: PointerId,
    camera: Entity,
    geometry: DragGeometry,
}

struct DragGeometry {
    start: Transform,
    kind: HandleKind,
    plane_normal: Vec3,
    grab: Vec3,
}

impl DragGeometry {
    fn new(start: Transform, kind: HandleKind, ray: Ray3d) -> Option<Self> {
        let plane_normal = match kind {
            HandleKind::Translate(axis) => {
                // A plane containing the axis, facing the pointer as closely as possible.
                (ray.direction.as_vec3() - axis * ray.direction.dot(axis)).try_normalize()?
            }
            HandleKind::Rotate(axis) => axis,
        };
        let grab = plane_point(ray, start.translation, plane_normal)?;
        if matches!(kind, HandleKind::Rotate(_)) && grab.distance_squared(start.translation) < 1e-6
        {
            return None;
        }
        Some(Self {
            start,
            kind,
            plane_normal,
            grab,
        })
    }

    fn pose(&self, ray: Ray3d) -> Option<Transform> {
        let point = plane_point(ray, self.start.translation, self.plane_normal)?;
        let mut pose = self.start;
        match self.kind {
            HandleKind::Translate(axis) => {
                pose.translation += axis * (point - self.grab).dot(axis);
            }
            HandleKind::Rotate(axis) => {
                let from = (self.grab - self.start.translation).try_normalize()?;
                let to = (point - self.start.translation).try_normalize()?;
                let angle = axis.dot(from.cross(to)).atan2(from.dot(to));
                pose.rotation =
                    (Quat::from_axis_angle(axis, angle) * self.start.rotation).normalize();
            }
        }
        Some(pose)
    }
}

fn plane_point(ray: Ray3d, origin: Vec3, normal: Vec3) -> Option<Vec3> {
    // Edge-on handles cannot be dragged reliably. Keep the current pose instead.
    if ray.direction.dot(normal).abs() < 1e-4 {
        return None;
    }
    let distance = ray.intersect_plane(origin, InfinitePlane3d::new(normal))?;
    let point = ray.get_point(distance);
    point.is_finite().then_some(point)
}

pub struct RobotInteractionPlugin;

impl Plugin for RobotInteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RobotSelection>()
            .add_systems(Startup, setup)
            .add_observer(select_robot)
            .add_observer(start_drag)
            .add_observer(drag_robot)
            .add_observer(end_drag)
            .add_observer(cancel_drag)
            .add_systems(Update, release_drag)
            .add_systems(PostUpdate, update_gizmo.before(TransformSystems::Propagate));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let shaft = meshes.add(Cylinder::new(0.018, 0.65));
    let tip = meshes.add(Cone {
        radius: 0.055,
        height: 0.16,
    });
    let ring = meshes.add(Torus {
        minor_radius: 0.012,
        major_radius: 0.58,
    });
    let root = commands
        .spawn((GizmoRoot, Transform::default(), Visibility::Hidden))
        .id();
    for (axis, color) in [
        (Vec3::X, Color::srgb(0.95, 0.18, 0.15)),
        (Vec3::Y, Color::srgb(0.2, 0.9, 0.3)),
        (Vec3::Z, Color::srgb(0.2, 0.45, 1.0)),
    ] {
        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        });
        let rotation = Quat::from_rotation_arc(Vec3::Y, axis);
        for (mesh, translation, kind) in [
            (shaft.clone(), axis * 0.425, HandleKind::Translate(axis)),
            (tip.clone(), axis * 0.83, HandleKind::Translate(axis)),
            (ring.clone(), Vec3::ZERO, HandleKind::Rotate(axis)),
        ] {
            commands.spawn((
                ChildOf(root),
                kind,
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(translation).with_rotation(rotation),
                Pickable::default(),
                NotShadowCaster,
            ));
        }
    }
}

fn select_robot(
    event: On<PointerPress>,
    targets: Query<(Option<&ObjectPart>, Has<FieldDropTarget>, Has<HandleKind>)>,
    objects: Query<&ObjectKind>,
    mode: Res<SimulationMode>,
    mut selection: ResMut<RobotSelection>,
    mut balls: ResMut<BallSelection>,
) {
    if event.button != PointerButton::Primary || *mode != SimulationMode::Paused {
        return;
    }
    let target = event.original_event_target();
    let Ok((part, is_field, is_handle)) = targets.get(target) else {
        return;
    };
    if is_handle {
        return;
    }
    let owner = part.map_or(target, |part| part.0);
    if objects.get(owner) == Ok(&ObjectKind::Robot) {
        selection.selected = Some(owner);
        balls.clear_selection();
    } else if is_field || objects.get(owner) == Ok(&ObjectKind::Ball) {
        selection.selected = None;
    }
}

fn start_drag(
    mut event: On<PointerDragStart>,
    handles: Query<&HandleKind>,
    cameras: Query<(), With<WorldCamera>>,
    rays: Res<RayMap>,
    mode: Res<SimulationMode>,
    physics: Res<MujocoWorld>,
    mut selection: ResMut<RobotSelection>,
) {
    if event.button != PointerButton::Primary
        || *mode != SimulationMode::Paused
        || selection.is_dragging()
    {
        return;
    }
    let handle = event.original_event_target();
    let Ok(kind) = handles.get(handle) else {
        return;
    };
    let Some(robot) = selection.selected else {
        return;
    };
    let Some(pose) = physics.object_pose(robot) else {
        return;
    };
    if !cameras.contains(event.hit.camera) {
        return;
    }
    let Some(ray) = rays
        .map
        .get(&RayId::new(event.hit.camera, event.pointer.id))
    else {
        return;
    };
    let Some(geometry) = DragGeometry::new(pose, *kind, *ray) else {
        return;
    };
    selection.drag = Some(RobotDrag {
        robot,
        handle,
        pointer: event.pointer.id,
        camera: event.hit.camera,
        geometry,
    });
    event.propagate(false);
}

fn drag_robot(
    mut event: On<PointerDrag>,
    rays: Res<RayMap>,
    mode: Res<SimulationMode>,
    selection: Res<RobotSelection>,
    mut poses: MessageWriter<SetObjectPose>,
) {
    let Some(drag) = &selection.drag else {
        return;
    };
    if *mode != SimulationMode::Paused
        || event.button != PointerButton::Primary
        || event.pointer.id != drag.pointer
        || event.original_event_target() != drag.handle
    {
        return;
    }
    if let Some(ray) = rays.map.get(&RayId::new(drag.camera, drag.pointer))
        && let Some(transform) = drag.geometry.pose(*ray)
    {
        poses.write(SetObjectPose {
            object: drag.robot,
            transform,
        });
    }
    event.propagate(false);
}

fn end_drag(event: On<PointerDragEnd>, mut selection: ResMut<RobotSelection>) {
    if event.button == PointerButton::Primary
        && selection
            .drag
            .as_ref()
            .is_some_and(|drag| drag.pointer == event.pointer.id)
    {
        selection.drag = None;
    }
}

fn cancel_drag(
    event: On<PointerCancel>,
    mut selection: ResMut<RobotSelection>,
    mut poses: MessageWriter<SetObjectPose>,
) {
    if selection
        .drag
        .as_ref()
        .is_some_and(|drag| drag.pointer == event.pointer.id)
        && let Some(drag) = selection.drag.take()
    {
        poses.write(SetObjectPose {
            object: drag.robot,
            transform: drag.geometry.start,
        });
    }
}

fn release_drag(
    mode: Res<SimulationMode>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    physics: Res<MujocoWorld>,
    mut selection: ResMut<RobotSelection>,
    mut poses: MessageWriter<SetObjectPose>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        if let Some(drag) = selection.drag.take() {
            poses.write(SetObjectPose {
                object: drag.robot,
                transform: drag.geometry.start,
            });
        } else {
            selection.selected = None;
        }
    }
    if selection
        .selected
        .is_some_and(|entity| !physics.contains_object(entity))
    {
        selection.selected = None;
    }
    if *mode != SimulationMode::Paused
        || selection.drag.as_ref().is_some_and(|drag| {
            !physics.contains_object(drag.robot)
                || (drag.pointer == PointerId::Mouse && !buttons.pressed(MouseButton::Left))
                || windows.iter().all(|window| !window.focused)
        })
    {
        selection.drag = None;
    }
}

fn update_gizmo(
    mode: Res<SimulationMode>,
    selection: Res<RobotSelection>,
    physics: Res<MujocoWorld>,
    mut root: Single<(&mut Transform, &mut Visibility), With<GizmoRoot>>,
) {
    if *mode == SimulationMode::Paused
        && let Some(pose) = selection
            .selected
            .and_then(|entity| physics.object_pose(entity))
    {
        root.0.translation = pose.translation;
        *root.1 = Visibility::Visible;
    } else {
        *root.1 = Visibility::Hidden;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: Vec3, target: Vec3) -> Ray3d {
        Ray3d::new(origin, Dir3::new(target - origin).unwrap())
    }

    #[test]
    fn translation_preserves_other_axes_and_rotation() {
        let start = Transform::from_xyz(1.0, 0.8, 2.0).with_rotation(Quat::from_rotation_z(0.3));
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let eye = start.translation + Vec3::new(3.0, 4.0, 5.0);
            let grab = start.translation + axis * 0.5;
            let drag =
                DragGeometry::new(start, HandleKind::Translate(axis), ray(eye, grab)).unwrap();
            let moved = drag.pose(ray(eye, grab + axis * 1.25)).unwrap();
            assert!(
                moved
                    .translation
                    .abs_diff_eq(start.translation + axis * 1.25, 1e-5)
            );
            assert_eq!(moved.rotation, start.rotation);
        }
    }

    #[test]
    fn rotation_uses_world_axes_around_robot_center() {
        let start = Transform::from_xyz(2.0, 0.8, 3.0).with_rotation(Quat::from_rotation_x(0.4));
        for (axis, radial) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
            let eye = start.translation + axis * 4.0;
            let angle = -0.7;
            let rotation = Quat::from_axis_angle(axis, angle);
            let drag = DragGeometry::new(
                start,
                HandleKind::Rotate(axis),
                ray(eye, start.translation + radial),
            )
            .unwrap();
            let moved = drag
                .pose(ray(eye, start.translation + rotation * radial))
                .unwrap();
            assert_eq!(moved.translation, start.translation);
            assert!(
                moved
                    .rotation
                    .abs_diff_eq((rotation * start.rotation).normalize(), 1e-5)
            );
        }
    }

    #[test]
    fn paused_pose_edits_update_physics_without_rewinding_time_or_changing_joints() {
        use crate::bevy_mujoco::{MjcfObject, MujocoWorldPlugin};
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
        app.insert_resource(SimulationMode::Paused);
        let robot = app
            .world_mut()
            .spawn((
                MjcfObject::new(
                    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"),
                    "Trunk",
                )
                .with_free_joint("world_joint")
                .grounded(),
                Transform::default(),
            ))
            .id();
        app.update();
        let (before, joints, time) = {
            let world = app.world().resource::<MujocoWorld>();
            (
                world.object_pose(robot).unwrap(),
                world.data().qpos()[7..].to_vec(),
                world.data().time(),
            )
        };
        let target = Transform {
            translation: before.translation + Vec3::new(0.7, 0.2, -0.4),
            rotation: Quat::from_euler(EulerRot::YXZ, 0.6, 0.2, -0.1),
            ..default()
        };
        app.world_mut().write_message(SetObjectPose {
            object: robot,
            transform: target,
        });
        app.update();
        let world = app.world().resource::<MujocoWorld>();
        let actual = world.object_pose(robot).unwrap();
        assert!(actual.translation.abs_diff_eq(target.translation, 1e-5));
        assert!(actual.rotation.abs_diff_eq(target.rotation, 1e-5));
        assert_eq!(&world.data().qpos()[7..], joints.as_slice());
        assert!(world.data().qvel()[..6].iter().all(|v| *v == 0.0));
        assert_eq!(world.data().time(), time);
        // The same editing command is ignored once the simulation is running.
        app.insert_resource(SimulationMode::Running);
        app.world_mut().write_message(SetObjectPose {
            object: robot,
            transform: before,
        });
        app.world_mut().run_schedule(Update);
        let world = app.world().resource::<MujocoWorld>();
        assert!(
            world
                .object_pose(robot)
                .unwrap()
                .translation
                .abs_diff_eq(target.translation, 1e-5)
        );
    }

    #[test]
    fn edge_on_and_behind_camera_drags_are_ignored() {
        assert!(
            DragGeometry::new(
                Transform::default(),
                HandleKind::Translate(Vec3::Y),
                Ray3d::new(Vec3::Y, Dir3::NEG_Y)
            )
            .is_none()
        );
        assert!(
            DragGeometry::new(
                Transform::default(),
                HandleKind::Rotate(Vec3::Y),
                Ray3d::new(Vec3::Z, Dir3::X)
            )
            .is_none()
        );
        assert!(plane_point(Ray3d::new(Vec3::Y, Dir3::Y), Vec3::ZERO, Vec3::Y).is_none());
    }
}

use crate::RobotId;
use bevy::{
    camera::primitives::Aabb,
    camera_controller::pan_orbit_camera::prelude::PanOrbitCamera,
    ecs::system::SystemState,
    picking::mesh_picking::ray_cast::{MeshRayCast, MeshRayCastSettings},
    prelude::*,
};
use eframe::egui::{self, Color32, Key, Pos2, Rect, Stroke, Ui};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::{
    bevy_mujoco::{SharedPhysics, SimulationMode},
    scene::{ball::Ball, object::ObjectPart, robot::RobotHead},
    simulation::ControlledRobot,
};

pub fn setup_scene(
    mut cameras: Query<(&mut PanOrbitCamera, &mut Transform, &mut Camera), Added<PanOrbitCamera>>,
    mut lights: Query<&mut Transform, (With<DirectionalLight>, Without<Camera>)>,
) {
    for (mut orbit, mut transform, mut camera) in &mut cameras {
        // egui_bevy uses this component to identify its render target camera.
        // Keep it attached while the simulator owns camera movement.
        orbit.enabled_motion.orbit = false;
        orbit.enabled_motion.pan = false;
        orbit.enabled_motion.zoom = false;
        *transform = Transform::from_xyz(0.0, 4.0, 11.0).looking_at(Vec3::ZERO, Vec3::Y);
        camera.clear_color = ClearColorConfig::Default;
        for mut light in &mut lights {
            *light = Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y);
        }
    }
}

#[derive(Default)]
pub struct Viewport {
    captured: Option<(egui::Context, Arc<CaptureActivity>)>,
    selected: Option<Entity>,
    warp_position: Option<Pos2>,
    drag: Option<Drag>,
    error: Option<String>,
    /// Player labels painted last frame; clicking one selects that robot in Twix.
    labels: Vec<(Rect, RobotId)>,
    clicked_label: Option<RobotId>,
}
// The panel may disappear while the mouse is captured. Release it even when this
// viewport no longer receives ui() calls; the plugin keeps only a weak reference.
struct CaptureActivity {
    active: AtomicBool,
    frame: AtomicU64,
}
#[derive(Default)]
struct CaptureCleanup(Weak<CaptureActivity>);
impl egui::Plugin for CaptureCleanup {
    fn debug_name(&self) -> &'static str {
        "Simulator mouse capture"
    }
    fn on_end_pass(&mut self, ui: &mut Ui) {
        if let Some(activity) = self.0.upgrade()
            && (activity.frame.load(Ordering::Relaxed) != ui.ctx().cumulative_frame_nr()
                || !ui.input(|input| input.focused))
            && activity.active.swap(false, Ordering::Relaxed)
        {
            release_cursor(ui.ctx());
        }
    }
}
// Winit cannot lock the cursor on X11 or Windows. Confine and recenter there;
// use native relative-pointer locking on Wayland and macOS.
fn needs_cursor_warp() -> bool {
    cfg!(target_os = "windows")
        || (cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_none())
}
fn release_cursor(context: &egui::Context) {
    context.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
    context.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
}
struct Drag {
    entity: Entity,
    geometry: DragGeometry,
    resume: SimulationMode,
}
#[derive(Clone, Copy)]
enum Handle {
    Translate(Vec3),
    Rotate(Vec3),
    Ball,
}
struct DragGeometry {
    pose: Transform,
    handle: Handle,
    normal: Vec3,
    grab: Vec3,
}
impl DragGeometry {
    fn new(pose: Transform, handle: Handle, ray: Ray3d) -> Option<Self> {
        let normal = match handle {
            Handle::Translate(axis) => {
                (*ray.direction - axis * ray.direction.dot(axis)).try_normalize()?
            }
            Handle::Rotate(axis) => axis,
            Handle::Ball => Vec3::Y,
        };
        let grab = plane_hit(ray, pose.translation, normal)?;
        Some(Self {
            pose,
            handle,
            normal,
            grab,
        })
    }
    fn pose(&self, ray: Ray3d) -> Option<Transform> {
        let hit = plane_hit(ray, self.pose.translation, self.normal)?;
        let mut pose = self.pose;
        match self.handle {
            Handle::Translate(axis) => pose.translation += axis * (hit - self.grab).dot(axis),
            Handle::Ball => pose.translation += hit - self.grab,
            Handle::Rotate(axis) => {
                let from = (self.grab - pose.translation).try_normalize()?;
                let to = (hit - pose.translation).try_normalize()?;
                let angle = axis.dot(from.cross(to)).atan2(from.dot(to));
                pose.rotation = Quat::from_axis_angle(axis, angle) * pose.rotation;
            }
        }
        Some(pose)
    }
}
fn pointer_motion(events: &[egui::Event], warp: Option<Pos2>) -> egui::Vec2 {
    let Some(warp) = warp else {
        return egui::Vec2::ZERO;
    };
    let mut previous = warp;
    let mut delta = egui::Vec2::ZERO;
    for event in events {
        if let egui::Event::PointerMoved(position) = event {
            // Native windows round egui coordinates to pixels.
            if position.distance(warp) > 1.0 {
                delta += *position - previous;
            }
            previous = *position;
        }
    }
    delta
}

fn plane_hit(ray: Ray3d, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denominator = ray.direction.dot(normal);
    if denominator.abs() < 1e-5 {
        return None;
    }
    let distance = (point - ray.origin).dot(normal) / denominator;
    (distance >= 0.0).then(|| ray.get_point(distance))
}

impl Viewport {
    pub fn toolbar(&mut self, ui: &mut Ui, compact: bool) {
        let icon = egui_material_icons::icons::ICON_3D_ROTATION.codepoint;
        let text = if compact {
            icon.to_owned()
        } else {
            format!("{icon} Fly camera")
        };
        let response = ui
            .add(egui::Button::selectable(self.captured.is_some(), text).shortcut_text("M"))
            .on_hover_text(
                "Capture the mouse to fly the camera; M or Esc releases it.\n\
                 Click a ball to drag it, or point at it and press Delete.\n\
                 Click a robot for its translation arrows and rotation rings.",
            );
        response.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::WidgetType::Button,
                ui.is_enabled(),
                self.captured.is_some(),
                "Fly camera (M)",
            )
        });
        if response.clicked() {
            self.capture(ui.ctx(), self.captured.is_none());
        }
    }
    pub fn captured(&self) -> bool {
        self.captured.is_some()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    fn capture(&mut self, context: &egui::Context, capture: bool) {
        self.warp_position = None;
        context.send_viewport_cmd(egui::ViewportCommand::CursorGrab(if capture {
            if needs_cursor_warp() {
                egui::CursorGrab::Confined
            } else {
                egui::CursorGrab::Locked
            }
        } else {
            egui::CursorGrab::None
        }));
        context.send_viewport_cmd(egui::ViewportCommand::CursorVisible(!capture));
        self.captured = capture.then(|| {
            let activity = Arc::new(CaptureActivity {
                active: AtomicBool::new(true),
                frame: AtomicU64::new(context.cumulative_frame_nr()),
            });
            context.with_plugin(|plugin: &mut CaptureCleanup| plugin.0 = Arc::downgrade(&activity));
            (context.clone(), activity)
        });
    }
    pub fn input(&mut self, ui: &mut Ui, rect: Rect, world: &mut World) {
        if let Some((_, activity)) = &self.captured {
            activity
                .frame
                .store(ui.ctx().cumulative_frame_nr(), Ordering::Relaxed);
            if !activity.active.load(Ordering::Relaxed) {
                self.captured = None;
            }
        }
        let hovered = ui.rect_contains_pointer(rect);
        let (toggle, escape, focused) =
            ui.input(|i| (i.key_pressed(Key::M), i.key_pressed(Key::Escape), i.focused));
        if self.captured.is_some() && (escape || !focused || toggle) {
            self.capture(ui.ctx(), false);
        } else if hovered && toggle && !ui.ctx().text_edit_focused() {
            self.capture(ui.ctx(), true);
        }
        if self.captured.is_some() {
            if needs_cursor_warp() {
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorPosition(rect.center()));
            }
            self.finish_drag(world);
            let mut cameras = world.query_filtered::<&mut Transform, With<Camera3d>>();
            if let Ok(mut camera) = cameras.single_mut(world) {
                ui.input(|i| {
                    let delta = if needs_cursor_warp() {
                        // X11 also reports cursor warps as motion. Use screen positions
                        // and discard our own recenter event, otherwise the camera spins.
                        pointer_motion(&i.events, self.warp_position)
                    } else {
                        i.events
                            .iter()
                            .filter_map(|event| match event {
                                egui::Event::MouseMoved(delta) => Some(*delta),
                                _ => None,
                            })
                            .fold(egui::Vec2::ZERO, |sum, delta| sum + delta)
                    };
                    let (yaw, pitch, _) = camera.rotation.to_euler(EulerRot::YXZ);
                    camera.rotation = Quat::from_euler(
                        EulerRot::YXZ,
                        yaw - delta.x * 0.003,
                        (pitch - delta.y * 0.003).clamp(-1.54, 1.54),
                        0.0,
                    );
                    let key = |key| f32::from(i.key_down(key));
                    let local =
                        Vec3::new(key(Key::D) - key(Key::A), 0.0, key(Key::S) - key(Key::W));
                    let direction = camera.rotation * local + Vec3::Y * (key(Key::E) - key(Key::Q));
                    camera.translation += direction.normalize_or_zero()
                        * i.stable_dt.min(0.05)
                        * if i.modifiers.shift { 10.0 } else { 3.0 };
                });
            }
            self.warp_position = Some(rect.center());
            return;
        }
        if !ui.input(|i| i.pointer.primary_down()) || !focused {
            self.finish_drag(world);
        }
        let Some(view) = CameraView::new(world, rect) else {
            return;
        };
        let Some(pointer) = ui.input(|i| i.pointer.interact_pos()) else {
            return;
        };
        let ray = view.ray(pointer);
        let label = self
            .labels
            .iter()
            .rev()
            .find(|(label, _)| label.contains(pointer))
            .map(|(_, id)| *id);
        if hovered && ui.input(|i| i.pointer.primary_pressed()) && label.is_some() {
            self.clicked_label = label;
        } else if hovered && ui.input(|i| i.pointer.primary_pressed()) {
            let handle = self.selected.and_then(|entity| {
                world.get::<ControlledRobot>(entity)?;
                let pose = world
                    .resource::<SharedPhysics>()
                    .lock()
                    .object_pose(entity)?;
                gizmo(&view, pose)
                    .into_iter()
                    .filter_map(|(handle, points, _)| {
                        let distance = points
                            .windows(2)
                            .map(|line| segment_distance(pointer, line[0], line[1]))
                            .fold(f32::INFINITY, f32::min);
                        (distance < 8.0).then_some((distance, handle))
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, handle)| handle)
            });
            if let Some(handle) = handle {
                self.start_drag(world, self.selected.unwrap(), handle, ray);
            } else {
                self.selected = pick(world, ray);
                if let Some(entity) = self.selected
                    && world.get::<Ball>(entity).is_some()
                {
                    self.start_drag(world, entity, Handle::Ball, ray);
                }
            }
        }
        if let Some(drag) = &self.drag
            && let Some(pose) = drag.geometry.pose(ray)
        {
            self.error = world
                .resource::<SharedPhysics>()
                .lock()
                .set_object_pose(drag.entity, pose)
                .err();
        }
        if hovered
            && !ui.ctx().text_edit_focused()
            && ui.input(|i| i.key_pressed(Key::Delete))
            && let Some(entity) = self.selected
            && world.get::<Ball>(entity).is_some()
            && world.resource::<crate::team::Team>().referee().is_none()
        {
            self.finish_drag(world);
            world.despawn(entity);
            self.selected = None;
        }
    }
    fn start_drag(&mut self, world: &World, entity: Entity, handle: Handle, ray: Ray3d) {
        let mut physics = world.resource::<SharedPhysics>().lock();
        if let Some(pose) = physics.object_pose(entity)
            && let Some(geometry) = DragGeometry::new(pose, handle, ray)
        {
            self.drag = Some(Drag {
                entity,
                geometry,
                resume: physics.mode,
            });
            physics.mode = SimulationMode::Paused;
        }
    }
    fn finish_drag(&mut self, world: &World) {
        if let Some(drag) = self.drag.take() {
            world.resource::<SharedPhysics>().lock().mode = drag.resume;
        }
    }
    /// A robot whose scene label was clicked, for Twix to select its namespace.
    pub fn take_selection(&mut self) -> Option<RobotId> {
        self.clicked_label.take()
    }
    pub fn paint(&mut self, ui: &Ui, rect: Rect, world: &mut World, selected: Option<RobotId>) {
        self.paint_scene(ui, rect, world, selected);
        if self.captured.is_some() {
            capture_legend(ui, rect);
        }
    }
    fn paint_scene(&mut self, ui: &Ui, rect: Rect, world: &mut World, selected: Option<RobotId>) {
        self.labels.clear();
        if let Some(view) = CameraView::new(world, rect) {
            let members = world.resource::<crate::team::Team>().members();
            let mut heads =
                world.query_filtered::<(&ObjectPart, &Aabb, &GlobalTransform), With<RobotHead>>();
            let label_size = egui::vec2(46.0, 28.0);
            for (part, bounds, transform) in heads.iter(world) {
                let Some(member) = members.iter().find(|member| member.entity == Some(part.0))
                else {
                    continue;
                };
                if let Some(head) = view.project_bounds(bounds, transform) {
                    // Keep a four-point gap below the selection rim, regardless of
                    // camera distance, viewing angle, or the robot's head pose.
                    let clearance = 4.0
                        + if Some(member.id) == selected {
                            5.0
                        } else {
                            0.0
                        };
                    let position =
                        egui::pos2(head.center().x, head.top() - clearance - label_size.y / 2.0);
                    let color = crate::widgets::jersey_color(member, ui.visuals());
                    let led = member
                        .io
                        .led_color()
                        .map_or(Color32::GRAY, |led| Color32::from_rgb(led.r, led.g, led.b));
                    let rect = Rect::from_center_size(position, label_size);
                    let painter = ui
                        .painter()
                        .with_clip_rect(rect.intersect(view.rect).expand(5.0));
                    if Some(member.id) == selected {
                        painter.rect_stroke(
                            rect.expand(3.0),
                            7.0,
                            Stroke::new(2.0, ui.visuals().selection.stroke.color),
                            egui::StrokeKind::Outside,
                        );
                    }
                    let visible = rect.intersect(view.rect);
                    if visible.is_positive() && self.captured.is_none() {
                        self.labels.push((visible, member.id));
                        ui.interact(visible, ui.id().with(member.id), egui::Sense::hover())
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text(format!(
                                "Select {} in Twix",
                                crate::robot_namespace(member.id)
                            ));
                    }
                    painter.rect_filled(rect, 5.0, color);
                    // Keep white jerseys legible against field lines and goals.
                    painter.rect_stroke(
                        rect,
                        5.0,
                        Stroke::new(1.0, Color32::from_black_alpha(150)),
                        egui::StrokeKind::Inside,
                    );
                    let text = if u16::from(color.r()) + u16::from(color.g()) + u16::from(color.b())
                        > 380
                    {
                        Color32::BLACK
                    } else {
                        Color32::WHITE
                    };
                    let accent = Rect::from_min_max(
                        egui::pos2(rect.left() + 5.0, rect.bottom() - 5.0),
                        egui::pos2(rect.right() - 5.0, rect.bottom() - 2.0),
                    );
                    painter.rect_filled(accent.expand(1.0), 2.0, Color32::from_black_alpha(180));
                    painter.rect_filled(accent, 1.0, led);
                    painter.text(
                        position - egui::vec2(0.0, 2.0),
                        egui::Align2::CENTER_CENTER,
                        crate::autoref::short(member.id),
                        egui::FontId::proportional(20.0),
                        text,
                    );
                }
            }
        }
        let Some(entity) = self.selected else {
            return;
        };
        let Some(view) = CameraView::new(world, rect) else {
            return;
        };
        let Some(pose) = world.resource::<SharedPhysics>().lock().object_pose(entity) else {
            return;
        };
        let painter = ui.painter().with_clip_rect(rect);
        if world.get::<ControlledRobot>(entity).is_some() {
            for (_, points, color) in gizmo(&view, pose) {
                painter.add(egui::Shape::line(points, Stroke::new(3.0, color)));
            }
        } else if let Some(center) = view.project(pose.translation) {
            let radius = world
                .resource::<crate::parameters::CurrentSimulatorParameters>()
                .field_dimensions
                .ball_radius;
            let edge = view.project(pose.translation + view.pose.right() * radius);
            if let Some(edge) = edge {
                painter.circle_stroke(
                    center,
                    center.distance(edge) + 3.0,
                    Stroke::new(2.0, Color32::YELLOW),
                );
            }
        }
    }
}
/// Shortcut legend while the mouse is captured and the cursor is hidden.
fn capture_legend(ui: &Ui, rect: Rect) {
    egui::Area::new(ui.id().with("capture_legend"))
        .pivot(egui::Align2::CENTER_BOTTOM)
        .fixed_pos(rect.center_bottom() - egui::vec2(0.0, 12.0))
        .constrain_to(rect)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            ui.style_mut().interaction.selectable_labels = false;
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (index, (keys, action)) in [
                        (&["W", "A", "S", "D"][..], "move"),
                        (&["Q", "E"], "down and up"),
                        (&["Shift"], "faster"),
                        (&["M", "Esc"], "release"),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if index > 0 {
                            ui.add_space(10.0);
                        }
                        ui.spacing_mut().item_spacing.x = 3.0;
                        for key in keys {
                            crate::widgets::keycap(ui, key);
                        }
                        ui.add_space(2.0);
                        ui.label(action);
                    }
                });
            });
        });
}
impl Drop for Viewport {
    fn drop(&mut self) {
        if let Some((context, _)) = self.captured.take() {
            release_cursor(&context);
        }
    }
}
struct CameraView {
    pose: Transform,
    rect: Rect,
    scale: f32,
}
impl CameraView {
    fn new(world: &mut World, rect: Rect) -> Option<Self> {
        let (pose, projection) = world
            .query_filtered::<(&Transform, &Projection), With<Camera3d>>()
            .single(world)
            .ok()?;
        let Projection::Perspective(projection) = projection else {
            return None;
        };
        Some(Self {
            pose: *pose,
            rect,
            scale: (projection.fov / 2.0).tan(),
        })
    }
    fn ray(&self, point: Pos2) -> Ray3d {
        let x = (point.x - self.rect.center().x) * 2.0 / self.rect.height() * self.scale;
        let y = (self.rect.center().y - point.y) * 2.0 / self.rect.height() * self.scale;
        Ray3d::new(
            self.pose.translation,
            Dir3::new(self.pose.rotation * Vec3::new(x, y, -1.0)).unwrap(),
        )
    }
    fn project_bounds(&self, bounds: &Aabb, transform: &GlobalTransform) -> Option<Rect> {
        let mut rect = Rect::NOTHING;
        let min = bounds.min();
        let max = bounds.max();
        for x in [min.x, max.x] {
            for y in [min.y, max.y] {
                for z in [min.z, max.z] {
                    let point = transform.transform_point(Vec3::new(x, y, z));
                    rect.extend_with(self.project(point)?);
                }
            }
        }
        Some(rect)
    }
    fn project(&self, point: Vec3) -> Option<Pos2> {
        let local = self.pose.rotation.inverse() * (point - self.pose.translation);
        if local.z >= -0.01 {
            return None;
        }
        let scale = self.rect.height() / (-2.0 * local.z * self.scale);
        Some(self.rect.center() + egui::vec2(local.x, -local.y) * scale)
    }
}
fn pick(world: &mut World, ray: Ray3d) -> Option<Entity> {
    let owners: std::collections::HashMap<_, _> = world
        .query::<(Entity, Option<&ObjectPart>, Has<Ball>)>()
        .iter(world)
        .filter_map(|(entity, part, ball)| {
            part.filter(|part| world.get::<ControlledRobot>(part.0).is_some())
                .map(|part| (entity, part.0))
                .or_else(|| ball.then_some((entity, entity)))
        })
        .collect();
    let mut state = SystemState::<MeshRayCast>::new(world);
    let mut raycast = state.get_mut(world).ok()?;
    let filter = |entity| owners.contains_key(&entity);
    let settings = MeshRayCastSettings::default().with_filter(&filter);
    raycast
        .cast_ray(ray, &settings)
        .first()
        .and_then(|(entity, _)| owners.get(entity).copied())
}
fn gizmo(view: &CameraView, pose: Transform) -> Vec<(Handle, Vec<Pos2>, Color32)> {
    let mut handles = Vec::new();
    for (axis, color) in [
        (Vec3::X, Color32::LIGHT_RED),
        (Vec3::Y, Color32::LIGHT_GREEN),
        (Vec3::Z, Color32::LIGHT_BLUE),
    ] {
        if let (Some(start), Some(end)) = (
            view.project(pose.translation),
            view.project(pose.translation + axis * 0.8),
        ) {
            let direction = (end - start).normalized();
            let side = egui::vec2(-direction.y, direction.x);
            handles.push((
                Handle::Translate(axis),
                vec![
                    start,
                    end,
                    end - direction * 10.0 + side * 5.0,
                    end,
                    end - direction * 10.0 - side * 5.0,
                ],
                color,
            ));
        }
        let tangent = if axis == Vec3::Y { Vec3::X } else { Vec3::Y };
        let bitangent = axis.cross(tangent);
        let points: Option<Vec<_>> = (0..=64)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 64.0;
                view.project(
                    pose.translation + (tangent * angle.cos() + bitangent * angle.sin()) * 0.58,
                )
            })
            .collect();
        if let Some(points) = points {
            handles.push((Handle::Rotate(axis), points, color));
        }
    }
    handles
}
fn segment_distance(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let line = b - a;
    let t = ((point - a).dot(line) / line.length_sq().max(1e-6)).clamp(0.0, 1.0);
    point.distance(a + t * line)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ray(origin: Vec3, target: Vec3) -> Ray3d {
        Ray3d::new(origin, Dir3::new(target - origin).unwrap())
    }
    #[test]
    fn cursor_recentering_does_not_rotate_the_camera() {
        let center = egui::pos2(400.03125, 300.03125);
        let rounded = egui::pos2(400.0, 300.0);
        assert_eq!(
            pointer_motion(
                &[
                    egui::Event::PointerMoved(rounded),
                    egui::Event::PointerMoved(egui::pos2(430.0, 301.0)),
                    egui::Event::PointerMoved(rounded),
                ],
                Some(center)
            ),
            egui::vec2(30.0, 1.0)
        );
        let center = rounded;
        let events = [
            egui::Event::PointerMoved(center),
            egui::Event::PointerMoved(center + egui::vec2(10.0, -5.0)),
            egui::Event::PointerMoved(center),
        ];
        assert_eq!(
            pointer_motion(&events, Some(center)),
            egui::vec2(10.0, -5.0)
        );
        assert_eq!(
            pointer_motion(&[egui::Event::PointerMoved(center)], Some(center)),
            egui::Vec2::ZERO
        );
    }
    #[test]
    fn ball_drag_preserves_height_and_grab_offset() {
        let pose = Transform::from_xyz(1.0, 0.12, 2.0);
        let camera = Vec3::new(0.0, 4.0, 6.0);
        let grab = pose.translation + Vec3::X * 0.1;
        let drag = DragGeometry::new(pose, Handle::Ball, ray(camera, grab)).unwrap();
        let moved = drag
            .pose(ray(camera, grab + Vec3::new(2.0, 0.0, -1.0)))
            .unwrap();
        assert!(
            moved
                .translation
                .abs_diff_eq(Vec3::new(3.0, 0.12, 1.0), 1e-5)
        );
    }
    #[test]
    fn gizmo_translation_constrains_axis_and_rotation_preserves_position() {
        let pose = Transform::from_xyz(0.0, 1.0, 0.0);
        let camera = Vec3::new(0.0, 4.0, 6.0);
        let drag = DragGeometry::new(
            pose,
            Handle::Translate(Vec3::X),
            ray(camera, pose.translation),
        )
        .unwrap();
        let moved = drag
            .pose(ray(camera, pose.translation + Vec3::new(2.0, 0.0, 0.0)))
            .unwrap();
        assert!(
            moved
                .translation
                .abs_diff_eq(Vec3::new(2.0, 1.0, 0.0), 1e-5)
        );
        let drag = DragGeometry::new(
            pose,
            Handle::Rotate(Vec3::Y),
            ray(camera, pose.translation + Vec3::X),
        )
        .unwrap();
        let moved = drag.pose(ray(camera, pose.translation + Vec3::Z)).unwrap();
        assert_eq!(moved.translation, pose.translation);
        assert!((moved.rotation * Vec3::X).abs_diff_eq(Vec3::Z, 1e-5));
    }
    #[test]
    fn projection_and_picking_use_the_same_viewport_coordinates() {
        let view = CameraView {
            pose: Transform::from_xyz(0.0, 4.0, 11.0).looking_at(Vec3::ZERO, Vec3::Y),
            rect: Rect::from_min_size(egui::pos2(17.0, 123.0), egui::vec2(800.0, 600.0)),
            scale: (std::f32::consts::FRAC_PI_4 / 2.0).tan(),
        };
        for point in [Vec3::ZERO, Vec3::new(1.0, 0.0, 2.0)] {
            let ray = view.ray(view.project(point).unwrap());
            assert!((*ray.direction).abs_diff_eq((point - ray.origin).normalize(), 1e-5));
        }
        assert!(plane_hit(Ray3d::new(Vec3::Y, Dir3::X), Vec3::ZERO, Vec3::Y).is_none());
    }
}

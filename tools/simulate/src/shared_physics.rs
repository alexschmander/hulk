//! Nonblocking panel snapshots and edits for the mutex-owned physics world.
use super::{MujocoWorld, SimulationMode, from_mujoco};
use bevy::prelude::*;
use mujoco_rs::prelude::MjtObj;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

type Edit = Box<dyn FnOnce(&mut MujocoWorld) -> Result<(), String> + Send>;

#[derive(Resource, Clone, Default)]
pub struct SharedPhysics(Arc<Shared>);
#[derive(Default)]
struct Shared {
    world: Mutex<MujocoWorld>,
    view: Mutex<Arc<PhysicsView>>,
    requested: AtomicBool,
    edits: Mutex<Vec<Edit>>,
    pending_edits: AtomicBool,
    error: Mutex<Option<String>>,
}

pub(crate) struct PhysicsView {
    pub mode: SimulationMode,
    #[cfg(test)]
    pub time: f64,
    bodies: HashMap<String, BodyPose>,
    objects: HashMap<Entity, BodyPose>,
}
#[derive(Clone, Copy)]
struct BodyPose {
    position: [f64; 3],
    rotation: [f64; 4],
}
impl BodyPose {
    fn transform(self) -> Transform {
        from_mujoco(self.position, self.rotation)
    }
}
impl Default for PhysicsView {
    fn default() -> Self {
        Self {
            mode: SimulationMode::Paused,
            #[cfg(test)]
            time: 0.0,
            bodies: HashMap::new(),
            objects: HashMap::new(),
        }
    }
}
impl PhysicsView {
    fn capture(world: &MujocoWorld) -> Self {
        let data = world.data();
        let bodies: HashMap<_, _> = (0..data.xpos().len())
            .filter_map(|id| {
                Some((
                    data.model().id_to_name(MjtObj::mjOBJ_BODY, id)?.to_owned(),
                    BodyPose {
                        position: data.xpos()[id],
                        rotation: data.xquat()[id],
                    },
                ))
            })
            .collect();
        let objects = world
            .objects
            .iter()
            .filter_map(|(&entity, binding)| Some((entity, *bodies.get(&binding.root_body)?)))
            .collect();
        Self {
            mode: world.mode,
            #[cfg(test)]
            time: data.time(),
            bodies,
            objects,
        }
    }
    pub fn body_pose(&self, name: &str) -> Option<Transform> {
        self.bodies.get(name).copied().map(BodyPose::transform)
    }
    pub fn object_pose(&self, entity: Entity) -> Option<Transform> {
        self.objects.get(&entity).copied().map(BodyPose::transform)
    }
    pub fn motion_pose(
        &self,
        entity: Entity,
    ) -> Option<(nalgebra::Isometry3<f32>, nalgebra::Point3<f32>)> {
        let root = self.objects.get(&entity)?;
        let prefix = format!("object_{}_", entity.to_bits());
        let left = self.bodies.get(&format!("{prefix}left_foot_link"))?;
        let right = self.bodies.get(&format!("{prefix}right_foot_link"))?;
        let trunk = crate::robot_io::body_transform(root.position, root.rotation);
        Some((
            crate::robot_io::ground_pose(trunk, left.position, right.position),
            nalgebra::Point3::from(root.position.map(|v| v as f32)),
        ))
    }
}
impl SharedPhysics {
    pub fn try_lock(&self) -> Option<MutexGuard<'_, MujocoWorld>> {
        match self.0.world.try_lock() {
            Ok(world) => Some(world),
            Err(std::sync::TryLockError::WouldBlock) => None,
            Err(error) => panic!("physics mutex poisoned: {error}"),
        }
    }
    /// Blocking worker/headless access. Panel code uses `view` and `edit`.
    pub fn lock(&self) -> MutexGuard<'_, MujocoWorld> {
        self.0.world.lock().unwrap()
    }

    /// Return the last completed view immediately. A busy worker fulfills the
    /// refresh request at the end of its step, so rendering cannot starve it.
    pub(crate) fn view(&self) -> Arc<PhysicsView> {
        self.0.requested.store(true, Ordering::Release);
        if let Some(world) = self.try_lock() {
            self.publish_view_if_requested(&world);
        }
        self.0.view.lock().unwrap().clone()
    }
    pub(crate) fn publish_view_if_requested(&self, world: &MujocoWorld) {
        if self.0.requested.load(Ordering::Acquire)
            && self.0.requested.swap(false, Ordering::AcqRel)
        {
            // Build outside the view mutex. Readers only hold it to clone an Arc.
            let view = Arc::new(PhysicsView::capture(world));
            let previous = std::mem::replace(&mut *self.0.view.lock().unwrap(), view);
            // Old maps can be deallocated without holding up a reader.
            drop(previous);
        }
    }
    /// Apply immediately if idle (also supports apps without a worker), or hand
    /// the edit to the worker. No caller waits for the physics mutex.
    pub(crate) fn edit(
        &self,
        edit: impl FnOnce(&mut MujocoWorld) -> Result<(), String> + Send + 'static,
    ) {
        {
            let mut edits = self.0.edits.lock().unwrap();
            edits.push(Box::new(edit));
            self.0.pending_edits.store(true, Ordering::Release);
        }
        if let Some(mut world) = self.try_lock() {
            self.apply_edits(&mut world);
            self.publish_view_if_requested(&world);
        }
    }
    pub(crate) fn apply_edits(&self, world: &mut MujocoWorld) {
        if !self.0.pending_edits.load(Ordering::Acquire) {
            return;
        }
        let edits = {
            let mut edits = self.0.edits.lock().unwrap();
            // Clear while holding the queue mutex, so a concurrent producer's
            // notification cannot be lost after this batch is drained.
            self.0.pending_edits.store(false, Ordering::Release);
            std::mem::take(&mut *edits)
        };
        if edits.is_empty() {
            return;
        }
        self.0.requested.store(true, Ordering::Release);
        for edit in edits {
            if let Err(error) = edit(world) {
                world.mode = SimulationMode::Paused;
                self.0.error.lock().unwrap().get_or_insert(error);
                break;
            }
        }
    }
    pub(crate) fn set_mode(&self, mode: SimulationMode) {
        self.edit(move |world| {
            world.mode = mode;
            Ok(())
        });
    }
    pub(crate) fn toggle_mode(&self) {
        self.edit(|world| {
            world.mode = match world.mode {
                SimulationMode::Paused => SimulationMode::Running,
                SimulationMode::Running => SimulationMode::Paused,
            };
            Ok(())
        });
    }
    /// Synchronous completion for headless scene-edit RPCs, never the paint path.
    pub(crate) fn finish_edits(&self) -> Result<(), String> {
        let mut world = self.lock();
        self.apply_edits(&mut world);
        self.publish_view_if_requested(&world);
        self.take_edit_error().map_or(Ok(()), Err)
    }
    pub(crate) fn take_edit_error(&self) -> Option<String> {
        self.0.error.lock().unwrap().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_edits_are_not_lost_while_draining() {
        use std::sync::atomic::AtomicUsize;
        let shared = SharedPhysics::default();
        let count = Arc::new(AtomicUsize::new(0));
        let finished = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let shared = shared.clone();
                let count = count.clone();
                let finished = &finished;
                scope.spawn(move || {
                    for _ in 0..100 {
                        let count = count.clone();
                        shared.edit(move |_| {
                            count.fetch_add(1, Ordering::Relaxed);
                            Ok(())
                        });
                    }
                    finished.fetch_add(1, Ordering::Release);
                });
            }
            while finished.load(Ordering::Acquire) < 2 {
                let mut world = shared.lock();
                shared.apply_edits(&mut world);
                drop(world);
                std::thread::yield_now();
            }
        });
        shared.finish_edits().unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn queued_edits_preserve_order_and_report_errors() {
        let shared = SharedPhysics::default();
        let previous = shared.view();
        let mut world = shared.lock();
        shared.set_mode(SimulationMode::Running);
        shared.toggle_mode();
        shared.toggle_mode();
        assert_eq!(shared.view().mode, SimulationMode::Paused);
        shared.apply_edits(&mut world);
        shared.publish_view_if_requested(&world);
        assert_eq!(shared.view().mode, SimulationMode::Running);
        assert_eq!(previous.mode, SimulationMode::Paused);
        shared.edit(|_| Err("rejected scene edit".into()));
        shared.apply_edits(&mut world);
        shared.publish_view_if_requested(&world);
        assert_eq!(shared.view().mode, SimulationMode::Paused);
        assert_eq!(
            shared.take_edit_error().as_deref(),
            Some("rejected scene edit")
        );
        assert!(shared.take_edit_error().is_none());
        shared.edit(|_| Err("failed queued RPC edit".into()));
        drop(world);
        assert_eq!(
            shared.finish_edits(),
            Err("failed queued RPC edit".to_owned())
        );
        assert!(shared.finish_edits().is_ok());
    }
}

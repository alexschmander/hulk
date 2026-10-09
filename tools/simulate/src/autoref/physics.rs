use super::{Ball, Engine, Robot, Snapshot, engine::Effect};
use crate::{RobotId, bevy_mujoco::MujocoWorld, team::Member};
use bevy::prelude::*;
use mujoco_rs::prelude::MjtObj;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

#[derive(Default)]
pub(super) struct Observer {
    generation: Option<u64>,
    owners: Vec<Option<RobotId>>,
    legs: BTreeMap<RobotId, Vec<usize>>,
}
impl Observer {
    fn snapshot(&mut self, world: &MujocoWorld, members: &[Member]) -> Snapshot {
        let data = world.data();
        let model = data.model();
        if self.generation != Some(world.generation) {
            let roots: BTreeMap<_, _> = members
                .iter()
                .filter_map(|m| {
                    let entity = m.entity?;
                    Some((
                        model.name_to_id(
                            MjtObj::mjOBJ_BODY,
                            &format!("object_{}_Trunk", entity.to_bits()),
                        )? as i32,
                        m.id,
                    ))
                })
                .collect();
            self.owners = model
                .geom_bodyid()
                .iter()
                .map(|body| roots.get(&model.body_rootid()[*body as usize]).copied())
                .collect();
            self.legs.clear();
            for (joint, body) in model.jnt_bodyid().iter().enumerate() {
                let Some(id) = roots.get(&model.body_rootid()[*body as usize]) else {
                    continue;
                };
                let name = model.id_to_name(MjtObj::mjOBJ_JOINT, joint).unwrap_or("");
                if ["Hip", "Knee", "Ankle"]
                    .iter()
                    .any(|part| name.contains(part))
                {
                    self.legs
                        .entry(*id)
                        .or_default()
                        .push(model.jnt_dofadr()[joint] as usize);
                }
            }
            self.generation = Some(world.generation);
        }
        let ball = crate::observations::balls(world)
            .into_iter()
            .next()
            .map(|b| Ball {
                entity: b.id,
                position: b.position,
                velocity: b.velocity,
                epoch: world.object_epoch(Entity::from_bits(b.id)),
            });
        let ball_body = ball
            .as_ref()
            .and_then(|b| {
                model.name_to_id(MjtObj::mjOBJ_BODY, &format!("object_{}_ball", b.entity))
            })
            .map(|id| id as i32);
        let mut contacts = BTreeSet::new();
        let mut feet = BTreeMap::<RobotId, Vec<[f64; 2]>>::new();
        for (index, contact) in data.contact().iter().enumerate() {
            if contact.geom[0] < 0 || contact.geom[1] < 0 || data.contact_force(index)[0] < 0.01 {
                continue;
            }
            for (first, other) in [
                (contact.geom[0] as usize, contact.geom[1] as usize),
                (contact.geom[1] as usize, contact.geom[0] as usize),
            ] {
                let Some(id) = self.owners[first] else {
                    continue;
                };
                let other_body = model.geom_bodyid()[other];
                if Some(other_body) == ball_body {
                    contacts.insert(id);
                }
                if other_body == 0 {
                    feet.entry(id)
                        .or_default()
                        .push([contact.pos[0], contact.pos[1]]);
                }
            }
        }
        let robots = members
            .iter()
            .filter_map(|member| {
                let entity = member.entity?;
                let body = data
                    .body(&format!("object_{}_Trunk", entity.to_bits()))?
                    .view(data);
                let velocity = data
                    .joint(&format!("object_{}_world_joint", entity.to_bits()))?
                    .view(data)
                    .qvel;
                let position = [body.xpos[0], body.xpos[1], body.xpos[2]];
                Some(Robot {
                    id: member.id,
                    position,
                    feet: feet
                        .remove(&member.id)
                        .unwrap_or_else(|| vec![[position[0], position[1]]]),
                    speed: velocity[0].hypot(velocity[1]),
                    leg_speed: self
                        .legs
                        .get(&member.id)
                        .into_iter()
                        .flatten()
                        .map(|&dof| data.qvel()[dof].abs())
                        .fold(0.0, f64::max),
                    fallen: position[2] < 0.4 || body.xmat[8] < 0.5,
                    penalized: member.io.primary.get_latest().is_some_and(|state| {
                        *state == types::primary_state::PrimaryState::Penalized
                    }),
                })
            })
            .collect();
        Snapshot {
            ball,
            robots,
            contacts,
        }
    }
    pub fn update(
        &mut self,
        engine: &mut Engine,
        world: &mut MujocoWorld,
        members: &[Member],
        dt: Duration,
    ) -> Result<(), String> {
        let snapshot = self.snapshot(world, members);
        for effect in engine.update(&snapshot, dt) {
            match effect {
                Effect::Ball(p) => {
                    if let Some(ball) = world.balls.first().copied() {
                        world.ground_object(
                            ball,
                            Transform::from_xyz(p[0] as f32, 0.0, -p[1] as f32),
                        )?;
                    }
                }
                Effect::Robot(id, p, yaw) => {
                    if let Some(member) = members.iter().find(|m| m.id == id)
                        && let Some(entity) = member.entity
                    {
                        let binding = crate::robot_io::RobotBinding::new(
                            world.data(),
                            &format!("object_{}_", entity.to_bits()),
                        )
                        .map_err(|e| e.to_string())?;
                        binding.reset_joints(world.data_mut());
                        world.ground_object(
                            entity,
                            Transform::from_xyz(p[0] as f32, 0.0, -p[1] as f32)
                                .with_rotation(Quat::from_rotation_y(yaw as f32)),
                        )?;
                    }
                }
                Effect::Whistle => {
                    for member in members {
                        member.io.whistle();
                    }
                }
            }
        }
        Ok(())
    }
}

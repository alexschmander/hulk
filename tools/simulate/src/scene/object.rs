use bevy::prelude::*;

use crate::bevy_mujoco::MjcfObject;

#[derive(Component)]
pub struct ObjectPart(pub Entity);

pub fn cleanup_parts(
    mut removed: RemovedComponents<MjcfObject>,
    parts: Query<(Entity, &ObjectPart)>,
    mut commands: Commands,
) {
    for owner in removed.read() {
        for (entity, part) in &parts {
            if part.0 == owner {
                commands.entity(entity).despawn();
            }
        }
    }
}

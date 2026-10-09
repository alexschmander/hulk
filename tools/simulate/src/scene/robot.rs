use std::{collections::HashMap, path::Path};

use bevy::{asset::RenderAssetUsages, mesh::PrimitiveTopology, prelude::*};

use super::object::ObjectPart;
use crate::bevy_mujoco::{MjcfObject, MujocoBody};

#[derive(Component)]
pub(crate) struct RobotHead;

const ROBOT_MJCF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml");
const MESH_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/meshes");

#[derive(Clone, Copy)]
enum RobotMaterial {
    Silver,
    Black,
    Metal,
    Logo,
}

struct Link {
    body: &'static str,
    mesh: &'static str,
    material: RobotMaterial,
}

const LINKS: &[Link] = &[
    Link {
        body: "Trunk",
        mesh: "Trunk.STL",
        material: RobotMaterial::Silver,
    },
    Link {
        body: "Trunk",
        mesh: "K1logo.STL",
        material: RobotMaterial::Logo,
    },
    Link {
        body: "Head_1",
        mesh: "Head_1.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "Head_2",
        mesh: "Head_2.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "Left_Arm_1",
        mesh: "Left_Arm_1.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Arm_2",
        mesh: "Left_Arm_2.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Arm_3",
        mesh: "Left_Arm_3.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "left_hand_link",
        mesh: "Left_Arm_4.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Arm_1",
        mesh: "Right_Arm_1.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Arm_2",
        mesh: "Right_Arm_2.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Arm_3",
        mesh: "Right_Arm_3.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "right_hand_link",
        mesh: "Right_Arm_4.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Hip_Pitch",
        mesh: "Left_Hip_Pitch.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Hip_Roll",
        mesh: "Left_Hip_Roll.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Hip_Yaw",
        mesh: "Left_Hip_Yaw.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Left_Shank",
        mesh: "Left_Shank.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "Left_Ankle_Cross",
        mesh: "Left_Ankle_Cross.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "left_foot_link",
        mesh: "Left_Foot.STL",
        material: RobotMaterial::Silver,
    },
    Link {
        body: "Right_Hip_Pitch",
        mesh: "Right_Hip_Pitch.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Hip_Roll",
        mesh: "Right_Hip_Roll.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Hip_Yaw",
        mesh: "Right_Hip_Yaw.STL",
        material: RobotMaterial::Metal,
    },
    Link {
        body: "Right_Shank",
        mesh: "Right_Shank.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "Right_Ankle_Cross",
        mesh: "Right_Ankle_Cross.STL",
        material: RobotMaterial::Black,
    },
    Link {
        body: "right_foot_link",
        mesh: "Right_Foot.STL",
        material: RobotMaterial::Silver,
    },
];

#[derive(Resource)]
pub struct RobotAssets {
    meshes: HashMap<&'static str, Handle<Mesh>>,
    silver: Handle<StandardMaterial>,
    black: Handle<StandardMaterial>,
    metal: Handle<StandardMaterial>,
    logo: Handle<StandardMaterial>,
}

impl FromWorld for RobotAssets {
    fn from_world(world: &mut World) -> Self {
        let meshes = LINKS
            .iter()
            .map(|link| {
                let path = Path::new(MESH_DIRECTORY).join(link.mesh);
                let mesh = load_binary_stl(&path)
                    .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()));
                (link.mesh, world.resource_mut::<Assets<Mesh>>().add(mesh))
            })
            .collect();
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        let material = |color: Color,
                        metallic: f32,
                        roughness: f32,
                        alpha: f32,
                        materials: &mut Assets<StandardMaterial>| {
            materials.add(StandardMaterial {
                base_color: color.with_alpha(alpha),
                metallic,
                perceptual_roughness: roughness,
                alpha_mode: if alpha < 1.0 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
                ..default()
            })
        };

        Self {
            meshes,
            silver: material(Color::srgb(0.8, 0.8, 0.8), 0.0, 0.5, 1.0, &mut materials),
            black: material(Color::srgb(0.1, 0.1, 0.1), 0.0, 0.5, 1.0, &mut materials),
            metal: material(Color::srgb(0.1, 0.1, 0.1), 0.1, 0.9, 1.0, &mut materials),
            logo: material(
                Color::srgb(0.792_156_9, 0.819_607_85, 0.933_333_34),
                0.0,
                0.5,
                1.0,
                &mut materials,
            ),
        }
    }
}

impl RobotAssets {
    fn material(&self, material: RobotMaterial) -> Handle<StandardMaterial> {
        match material {
            RobotMaterial::Silver => self.silver.clone(),
            RobotMaterial::Black => self.black.clone(),
            RobotMaterial::Metal => self.metal.clone(),
            RobotMaterial::Logo => self.logo.clone(),
        }
    }
}

pub fn spawn(commands: &mut Commands, assets: &RobotAssets, transform: Transform) -> Entity {
    let owner = commands
        .spawn((
            MjcfObject::new(ROBOT_MJCF, "Trunk")
                .with_free_joint("world_joint")
                .grounded(),
            transform,
        ))
        .id();

    for link in LINKS {
        let mut part = commands.spawn((
            Name::new(link.mesh),
            ObjectPart(owner),
            MujocoBody::new(owner, link.body),
            Mesh3d(assets.meshes[link.mesh].clone()),
            MeshMaterial3d(assets.material(link.material)),
            Transform::default(),
            Visibility::Hidden,
        ));
        if link.body == "Head_2" {
            part.insert(RobotHead);
        }
    }

    owner
}

fn load_binary_stl(path: &Path) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() < 84 {
        return Err("file is too short to be a binary STL".to_owned());
    }

    let triangle_count =
        u32::from_le_bytes(bytes[80..84].try_into().expect("slice has length 4")) as usize;
    let expected_length = 84 + triangle_count * 50;
    if bytes.len() < expected_length {
        return Err(format!(
            "expected {expected_length} bytes, got {}",
            bytes.len()
        ));
    }

    let mut positions = Vec::with_capacity(triangle_count * 3);
    let mut normals = Vec::with_capacity(triangle_count * 3);
    let mut offset = 84;

    for _ in 0..triangle_count {
        let normal = convert(read_vec3(&bytes, offset));
        offset += 12;
        let mut triangle = [Vec3::ZERO; 3];
        for vertex in &mut triangle {
            *vertex = convert(read_vec3(&bytes, offset));
            offset += 12;
        }
        offset += 2;

        let normal = normal.try_normalize().unwrap_or_else(|| {
            (triangle[1] - triangle[0])
                .cross(triangle[2] - triangle[0])
                .normalize_or_zero()
        });
        positions.extend(triangle.map(|vertex| vertex.to_array()));
        normals.extend([normal.to_array(); 3]);
    }

    Ok(Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals))
}

fn read_vec3(bytes: &[u8], offset: usize) -> [f32; 3] {
    [
        read_f32(bytes, offset),
        read_f32(bytes, offset + 4),
        read_f32(bytes, offset + 8),
    ]
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("slice has length 4"),
    )
}

fn convert([x, y, z]: [f32; 3]) -> Vec3 {
    Vec3::new(x, z, -y)
}

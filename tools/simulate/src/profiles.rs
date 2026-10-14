use std::{future::Future, pin::Pin, sync::Arc};

use color_eyre::Result;
use ros_z::context::Context;
use serde::{Deserialize, Serialize};

/// Ordered by robotics coverage; each profile also runs the real nodes of those before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    #[default]
    MotionBehavior,
    Filtering,
    BodyStateOdometry,
    Localization,
}

impl Profile {
    pub const ALL: [Self; 4] = [
        Self::MotionBehavior,
        Self::Filtering,
        Self::BodyStateOdometry,
        Self::Localization,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::MotionBehavior => "Motion & behavior",
            Self::Filtering => "Ball and obstacle filtering",
            Self::BodyStateOdometry => "Body state & odometry",
            Self::Localization => "Localization",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::MotionBehavior => "Perfect ball, obstacles, body geometry and field pose.",
            Self::Filtering => {
                "Real ball and obstacle filters; perfect body geometry, odometry and field pose."
            }
            Self::BodyStateOdometry => {
                "Real body estimation and odometry; perfect field pose and synthetic detections."
            }
            Self::Localization => {
                "Real localization from synthetic detections and camera motion; no images required."
            }
        }
    }

    pub(crate) fn filtering(self) -> bool {
        self != Self::MotionBehavior
    }
    pub(crate) fn body_state(self) -> bool {
        matches!(self, Self::BodyStateOdometry | Self::Localization)
    }
    pub(crate) fn localization(self) -> bool {
        self == Self::Localization
    }

    /// Real node groups launched for every robot, in launch order.
    pub(crate) fn node_groups(self) -> impl Iterator<Item = &'static NodeGroup> {
        NODE_GROUPS
            .iter()
            .filter(move |group| group.profile <= self)
    }
    pub(crate) fn nodes(self) -> impl Iterator<Item = &'static NodeSpec> {
        self.node_groups().flat_map(|group| group.nodes)
    }

    /// Whether the simulator owns a reference's production topic in this profile.
    pub(crate) fn publishes(self, topic: &str) -> bool {
        REFERENCES
            .iter()
            .find(|reference| reference.topic == topic)
            .unwrap_or_else(|| panic!("{topic} is not a simulator reference"))
            .production(self)
            .is_some()
    }
}

pub(crate) struct NodeGroup {
    pub label: &'static str,
    /// The first profile that runs these nodes.
    pub profile: Profile,
    pub nodes: &'static [NodeSpec],
}
const NODE_GROUPS: [NodeGroup; 5] = [
    NodeGroup {
        label: "Behavior, motion & hardware",
        profile: Profile::MotionBehavior,
        nodes: COMMON,
    },
    NodeGroup {
        label: "HSL communication & game state",
        profile: Profile::MotionBehavior,
        nodes: NETWORK,
    },
    NodeGroup {
        label: "Ball and obstacle filtering",
        profile: Profile::Filtering,
        nodes: FILTERING,
    },
    NodeGroup {
        label: "Body state & odometry",
        profile: Profile::BodyStateOdometry,
        nodes: BODY_STATE,
    },
    NodeGroup {
        label: "Localization",
        profile: Profile::Localization,
        nodes: LOCALIZATION,
    },
];

/// A physical reference, always published under `ground_truth/`.
pub(crate) struct ReferenceSpec {
    pub topic: &'static str,
    pub meaning: &'static str,
    production: Production,
}
#[derive(Clone, Copy)]
enum Production {
    /// Comparison only.
    Never,
    /// Ideal input until the given profile runs the real producer.
    Before(Profile),
    /// Synthetic measurement for real nodes from the given profile on.
    From(Profile),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Supply {
    /// Truth replaces the output of a production node that does not run.
    Perfect,
    /// Simulated measurements stand in for perception that does not run.
    Synthetic,
}
impl ReferenceSpec {
    /// How the simulator also publishes this reference to its production topic, if at all.
    pub fn production(&self, profile: Profile) -> Option<Supply> {
        match self.production {
            Production::Before(real) if profile < real => Some(Supply::Perfect),
            Production::From(first) if profile >= first => Some(Supply::Synthetic),
            _ => None,
        }
    }
}
macro_rules! references {
    ($($topic:literal: $production:expr, $meaning:literal;)*) => {
        &[$(ReferenceSpec { topic: $topic, meaning: $meaning, production: $production }),*]
    };
}
pub(crate) const REFERENCES: &[ReferenceSpec] = {
    use {Production::*, Profile::*};
    references! {
        "ground_to_field": Before(Localization), "Exact planar field pose";
        "ball_filter/ball_position": Before(Filtering), "First scene ball, whether visible or not";
        "visual_kick/ball_position": Before(Filtering), "First scene ball, for visual kicks";
        "obstacles": Before(Filtering), "Goal posts and every other robot";
        "robot_to_ground": Before(BodyStateOdometry), "Physical body/ground geometry";
        "ground_to_robot": Before(BodyStateOdometry), "Physical ground/body geometry";
        "robot_kinematics": Before(BodyStateOdometry), "MuJoCo link transforms";
        "support_foot_state": Before(BodyStateOdometry), "Support foot from physical contacts";
        "camera_matrix": Before(BodyStateOdometry), "Exact camera geometry and intrinsics";
        "camera_geometry": Before(BodyStateOdometry), "Exact body-relative camera geometry";
        "inputs/odometry": Before(BodyStateOdometry), "Exact planar movement since the odometry origin";
        "detected_objects": From(Filtering), "Visible objects as image boxes; the robot receives optional noise";
        "visual_odometry/current_left_camera_to_visual_odometer": From(Localization), "Accumulated camera motion, standing in for stereo VO";
        "visual_odometry/current_left_camera_to_previous_left_camera": Never, "Exact camera-motion delta";
        "balls": Never, "Every scene ball with a stable identity";
    }
};

type Runner = fn(Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;
pub(crate) struct NodeSpec {
    pub name: &'static str,
    pub run: Runner,
}
macro_rules! nodes {
    ($($node:ident),* $(,)?) => { &[$(NodeSpec { name: stringify!($node), run: $node::run_boxed }),*] };
}
const COMMON: &[NodeSpec] = &[
    NodeSpec {
        name: "led_handler",
        run: led_handler::run_boxed,
    },
    NodeSpec {
        name: "head_motion",
        run: head_motion::node::run_boxed,
    },
    NodeSpec {
        name: "behavior_node",
        run: behavior_node::run_boxed,
    },
    NodeSpec {
        name: "ball_state_composer",
        run: ball_state_composer::run_boxed,
    },
    NodeSpec {
        name: "rule_obstacle_composer",
        run: rule_obstacle_composer::run_boxed,
    },
    NodeSpec {
        name: "motion",
        run: motion::run_boxed,
    },
    NodeSpec {
        name: "motion_inference",
        run: motion_inference::run_boxed,
    },
    NodeSpec {
        name: "hardware_interface",
        run: hardware_interface::run_boxed,
    },
    NodeSpec {
        name: "fall_detection",
        run: fall_detection::run_boxed,
    },
    NodeSpec {
        name: "safe_pose_checker",
        run: safe_pose_checker::run_boxed,
    },
    NodeSpec {
        name: "button_event_bridge",
        run: button_event_bridge::run_boxed,
    },
    NodeSpec {
        name: "button_event_handler",
        run: button_event_handler::run_boxed,
    },
    NodeSpec {
        name: "global_parameter_provider",
        run: global_parameter_provider::run_boxed,
    },
    NodeSpec {
        name: "low_state_bridge",
        run: low_state_bridge::run_boxed,
    },
    NodeSpec {
        name: "whistle_filter",
        run: whistle_filter::run_boxed,
    },
    NodeSpec {
        name: "world_to_field_provider",
        run: world_to_field_provider::run_boxed,
    },
];
pub(crate) const NETWORK: &[NodeSpec] = nodes![
    message_handler,
    message_filter,
    game_controller_filter,
    game_controller_state_filter,
    primary_state_filter,
    player_states_receiver,
    team_ball_filter
];
const FILTERING: &[NodeSpec] = nodes![
    ball_filter,
    visual_kick_ball_selector,
    obstacle_filter,
    search_suggestor
];
const BODY_STATE: &[NodeSpec] = nodes![
    kinematics_provider,
    support_foot_estimator,
    ground_provider,
    camera_matrix_calculator,
    odometry
];
const LOCALIZATION: &[NodeSpec] = nodes![field_mark_association, localization_3d, localization_2d];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn profiles_are_cumulative_and_keep_all_common_nodes() {
        let mut previous = HashSet::new();
        for (profile, count) in Profile::ALL.into_iter().zip([23, 27, 32, 35]) {
            let names: HashSet<_> = profile.nodes().map(|node| node.name).collect();
            assert_eq!(names.len(), count);
            assert!(previous.is_subset(&names));
            for required in [
                "low_state_bridge",
                "led_handler",
                "whistle_filter",
                "world_to_field_provider",
                "message_handler",
                "button_event_handler",
            ] {
                assert!(names.contains(required));
            }
            previous = names;
        }
    }

    #[test]
    fn references_keep_each_profiles_production_owners() {
        let names: HashSet<_> = REFERENCES.iter().map(|reference| reference.topic).collect();
        assert_eq!(names.len(), REFERENCES.len());
        for profile in Profile::ALL {
            for (topics, perfect) in [
                (
                    &[
                        "ball_filter/ball_position",
                        "visual_kick/ball_position",
                        "obstacles",
                    ][..],
                    !profile.filtering(),
                ),
                (
                    &[
                        "robot_to_ground",
                        "ground_to_robot",
                        "robot_kinematics",
                        "support_foot_state",
                        "camera_matrix",
                        "camera_geometry",
                        "inputs/odometry",
                    ],
                    !profile.body_state(),
                ),
                (&["ground_to_field"], !profile.localization()),
            ] {
                for topic in topics {
                    assert_eq!(profile.publishes(topic), perfect, "{profile:?}: {topic}");
                }
            }
            assert_eq!(profile.publishes("detected_objects"), profile.filtering());
            assert_eq!(
                profile.publishes("visual_odometry/current_left_camera_to_visual_odometer"),
                profile.localization()
            );
            assert!(!profile.publishes("balls"));
        }
    }
}

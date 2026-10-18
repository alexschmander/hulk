use std::{future::Future, pin::Pin, sync::Arc};

use color_eyre::Result;
use ros_z::context::Context;
use serde::{Deserialize, Serialize};

/// Presets retain their saved names; custom selections enable estimators independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    #[default]
    MotionBehavior,
    Filtering,
    BodyStateOdometry,
    Localization,
    Custom(Estimators),
}

impl Profile {
    pub const PRESETS: [Self; 4] = [
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
            Self::Custom(_) => "Custom",
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
            Self::Custom(_) => "Independent estimators; disabled groups receive simulator truth.",
        }
    }

    pub fn estimators(self) -> Estimators {
        match self {
            Self::MotionBehavior => Estimators::default(),
            Self::Filtering => Estimators {
                filtering: true,
                ..Default::default()
            },
            Self::BodyStateOdometry => Estimators {
                filtering: true,
                body_state: true,
                localization: false,
            },
            Self::Localization => Estimators {
                filtering: true,
                body_state: true,
                localization: true,
            },
            Self::Custom(estimators) => estimators,
        }
    }

    /// Reuse a preset name whenever the effective selection matches it.
    pub fn from_estimators(estimators: Estimators) -> Self {
        Self::PRESETS
            .into_iter()
            .find(|preset| preset.estimators() == estimators)
            .unwrap_or(Self::Custom(estimators))
    }

    pub(crate) fn filtering(self) -> bool {
        self.estimators().filtering
    }
    pub(crate) fn body_state(self) -> bool {
        self.estimators().body_state
    }
    pub(crate) fn localization(self) -> bool {
        self.estimators().localization
    }
    pub(crate) fn detections(self) -> bool {
        self.filtering() || self.localization()
    }

    /// Real node groups launched for every robot, in launch order.
    pub(crate) fn node_groups(self) -> impl Iterator<Item = &'static NodeGroup> {
        NODE_GROUPS.iter().filter(move |group| {
            group
                .estimator
                .is_none_or(|estimator| estimator.enabled(self.estimators()))
        })
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

/// Real estimation pipelines. Disabled pipelines are replaced at their output topics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Estimators {
    pub filtering: bool,
    pub body_state: bool,
    pub localization: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum Estimator {
    Filtering,
    BodyState,
    Localization,
}
impl Estimator {
    pub fn enabled(self, estimators: Estimators) -> bool {
        match self {
            Self::Filtering => estimators.filtering,
            Self::BodyState => estimators.body_state,
            Self::Localization => estimators.localization,
        }
    }
}

pub(crate) struct NodeGroup {
    pub label: &'static str,
    /// None for the behavior, motion, protection and game/team stack that always runs.
    pub estimator: Option<Estimator>,
    pub nodes: &'static [NodeSpec],
}
const NODE_GROUPS: [NodeGroup; 5] = [
    NodeGroup {
        label: "Behavior, motion & hardware",
        estimator: None,
        nodes: COMMON,
    },
    NodeGroup {
        label: "HSL communication & game state",
        estimator: None,
        nodes: NETWORK,
    },
    NodeGroup {
        label: "Ball and obstacle filtering",
        estimator: Some(Estimator::Filtering),
        nodes: FILTERING,
    },
    NodeGroup {
        label: "Body state & odometry",
        estimator: Some(Estimator::BodyState),
        nodes: BODY_STATE,
    },
    NodeGroup {
        label: "Localization",
        estimator: Some(Estimator::Localization),
        nodes: LOCALIZATION,
    },
];

/// A physical reference declared under `ground_truth/`, published when observed.
pub(crate) struct ReferenceSpec {
    pub topic: &'static str,
    pub meaning: &'static str,
    production: Production,
}
#[derive(Clone, Copy)]
enum Production {
    /// Comparison only.
    Never,
    /// Ideal input while the corresponding real estimator is disabled.
    Unless(Estimator),
    /// Synthetic object observations needed by object filtering or localization.
    Detections,
    /// Synthetic camera motion needed by localization.
    VisualOdometry,
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
            Production::Unless(estimator) if !estimator.enabled(profile.estimators()) => {
                Some(Supply::Perfect)
            }
            Production::Detections if profile.detections() => Some(Supply::Synthetic),
            Production::VisualOdometry if profile.localization() => Some(Supply::Synthetic),
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
    use {Estimator::*, Production::*};
    references! {
        "ground_to_field": Unless(Localization), "Exact planar field pose";
        "ball_filter/ball_position": Unless(Filtering), "First scene ball, whether visible or not";
        "visual_kick/ball_position": Unless(Filtering), "First scene ball, for visual kicks";
        "obstacles": Unless(Filtering), "Goal posts and every other robot";
        "robot_to_ground": Unless(BodyState), "Physical body/ground geometry";
        "ground_to_robot": Unless(BodyState), "Physical ground/body geometry";
        "robot_kinematics": Unless(BodyState), "MuJoCo link transforms";
        "support_foot_state": Unless(BodyState), "Support foot from physical contacts";
        "camera_matrix": Unless(BodyState), "Exact camera geometry and intrinsics";
        "camera_geometry": Unless(BodyState), "Exact body-relative camera geometry";
        "inputs/odometry": Unless(BodyState), "Exact planar movement since the odometry origin";
        "detected_objects": Detections, "Visible objects as image boxes; the robot receives optional noise";
        "visual_odometry/current_left_camera_to_visual_odometer": VisualOdometry, "Accumulated camera motion, standing in for stereo VO";
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
        name: "search_suggestor",
        run: search_suggestor::run_boxed,
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
const FILTERING: &[NodeSpec] = nodes![ball_filter, visual_kick_ball_selector, obstacle_filter];
const BODY_STATE: &[NodeSpec] = nodes![
    kinematics_provider,
    support_foot_estimator,
    ground_provider,
    camera_matrix_calculator,
    odometry
];
const LOCALIZATION: &[NodeSpec] = nodes![field_mark_association, localization_3d, localization_2d];

#[cfg(test)]
pub(crate) fn test_profiles() -> [Profile; 8] {
    std::array::from_fn(|bits| {
        Profile::from_estimators(Estimators {
            filtering: bits & 1 != 0,
            body_state: bits & 2 != 0,
            localization: bits & 4 != 0,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn presets_are_cumulative_and_keep_all_common_nodes() {
        let mut previous = HashSet::new();
        for (profile, count) in Profile::PRESETS.into_iter().zip([24, 27, 32, 35]) {
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
                "search_suggestor",
            ] {
                assert!(names.contains(required));
            }
            previous = names;
        }
    }

    #[test]
    fn all_estimator_combinations_preserve_production_owners() {
        let names: HashSet<_> = REFERENCES.iter().map(|reference| reference.topic).collect();
        assert_eq!(names.len(), REFERENCES.len());
        for profile in test_profiles() {
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
            assert_eq!(
                profile.publishes("detected_objects"),
                profile.filtering() || profile.localization()
            );
            assert_eq!(
                profile.publishes("visual_odometry/current_left_camera_to_visual_odometer"),
                profile.localization()
            );
            assert!(!profile.publishes("balls"));
            assert!(
                !profile.publishes("visual_odometry/current_left_camera_to_previous_left_camera")
            );
            let nodes: Vec<_> = profile.nodes().map(|node| node.name).collect();
            assert_eq!(nodes.iter().collect::<HashSet<_>>().len(), nodes.len());
            assert!(nodes.contains(&"search_suggestor"));
            for (node, expected) in [
                ("ball_filter", profile.filtering()),
                ("visual_kick_ball_selector", profile.filtering()),
                ("obstacle_filter", profile.filtering()),
                ("kinematics_provider", profile.body_state()),
                ("support_foot_estimator", profile.body_state()),
                ("ground_provider", profile.body_state()),
                ("camera_matrix_calculator", profile.body_state()),
                ("odometry", profile.body_state()),
                ("field_mark_association", profile.localization()),
                ("localization_3d", profile.localization()),
                ("localization_2d", profile.localization()),
            ] {
                assert_eq!(nodes.contains(&node), expected, "{profile:?}: {node}");
            }
        }
    }
    #[test]
    fn preset_serialization_stays_compatible_and_custom_choices_round_trip() {
        for (profile, name) in Profile::PRESETS.into_iter().zip([
            "motion_behavior",
            "filtering",
            "body_state_odometry",
            "localization",
        ]) {
            let saved = serde_json::Value::String(name.into());
            assert_eq!(serde_json::to_value(profile).unwrap(), saved);
            assert_eq!(serde_json::from_value::<Profile>(saved).unwrap(), profile);
            assert_eq!(Profile::from_estimators(profile.estimators()), profile);
        }
        for profile in test_profiles() {
            assert_eq!(
                serde_json::from_value::<Profile>(serde_json::to_value(profile).unwrap()).unwrap(),
                profile
            );
        }
    }
}

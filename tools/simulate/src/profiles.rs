use std::{future::Future, pin::Pin, sync::Arc};

use color_eyre::Result;
use ros_z::context::Context;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
            Self::Filtering => "Filtering",
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

    pub(crate) fn nodes(
        self,
        controller: ControllerSource,
    ) -> impl Iterator<Item = &'static NodeSpec> {
        COMMON
            .iter()
            .chain(NETWORK)
            .chain((controller == ControllerSource::Local).then_some(&CONTROLLER))
            .chain(FILTERING.iter().filter(move |_| self.filtering()))
            .chain(BODY_STATE.iter().filter(move |_| self.body_state()))
            .chain(LOCALIZATION.iter().filter(move |_| self.localization()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerSource {
    #[default]
    Local,
    External,
}

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
const CONTROLLER: NodeSpec = NodeSpec {
    name: "controller_handler",
    run: controller_handler::run_boxed,
};
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
            let names: HashSet<_> = profile
                .nodes(ControllerSource::Local)
                .map(|node| node.name)
                .collect();
            assert_eq!(names.len(), count);
            assert!(previous.is_subset(&names));
            for required in [
                "low_state_bridge",
                "controller_handler",
                "whistle_filter",
                "world_to_field_provider",
                "message_handler",
                "button_event_handler",
            ] {
                assert!(names.contains(required));
            }
            assert_eq!(profile.nodes(ControllerSource::External).count(), count - 1);
            previous = names;
        }
    }
}

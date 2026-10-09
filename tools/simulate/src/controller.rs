//! One physical gamepad, routed to the selected robot with a fresh Start edge.
use crate::{
    robotics::Configuration,
    team::{Member, RobotId},
};
use color_eyre::Result;
use ros_z::prelude::*;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use types::controller_input::{Button, ControllerInput};

pub(crate) struct Controller {
    selected: Arc<Mutex<Option<RobotId>>>,
    additions: tokio::sync::mpsc::UnboundedSender<Member>,
    task: tokio_util::task::AbortOnDropHandle<Result<()>>,
    runtime: tokio::runtime::Handle,
    failure: Option<String>,
    source: Arc<Context>,
    reader: tokio_util::task::AbortOnDropHandle<Result<()>>,
}
struct Target {
    context: Arc<Context>,
    publisher: Publisher<ControllerInput>,
    behavior: ros_z::cache::Cache<behavior_node::node::Blackboard>,
    suppress_start: bool,
}
impl Drop for Target {
    fn drop(&mut self) {
        let _ = self.context.shutdown();
    }
}
impl Controller {
    pub async fn new(configuration: &Configuration, members: Vec<Member>) -> Result<Self> {
        let mut builder = ContextBuilder::default()
            .with_namespace("/simulator/gamepad")
            .with_json("namespace", crate::ZENOH_NAMESPACE);
        if let Some(router) = &configuration.router {
            builder = builder.with_router_endpoint(router)?;
        }
        let source = Arc::new(builder.build().await?);
        let node = source.create_node("gamepad_router").build().await?;
        let input = node
            .subscriber::<ControllerInput>("inputs/controller_input")
            .cache(1)
            .build()
            .await?;
        let reader = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(
            controller_handler::run_boxed(source.clone()),
        ));
        let selected = Arc::new(Mutex::new(Some(RobotId::FIRST)));
        let selection = selected.clone();
        let (additions, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        for member in members {
            additions.send(member)?;
        }
        let router = configuration.router.clone();
        let task = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
            let mut targets = BTreeMap::new();
            let mut gate = Gate::default();
            let mut timer = tokio::time::interval(Duration::from_millis(20));
            loop {
                tokio::select! {
                    member = receiver.recv() => {
                        let Some(member) = member else { break; };
                        let result: Result<Target> = async {
                            let mut builder = ContextBuilder::default().with_namespace(crate::team::robot_namespace(member.id))
                                .with_json("namespace", crate::team::transport_scope(member.id));
                            if let Some(router) = &router { builder = builder.with_router_endpoint(router)?; }
                            let context = Arc::new(builder.build().await?);
                            let node = context.create_node("simulator_gamepad").build().await?;
                            let publisher = node.publisher("inputs/controller_input").build().await?;
                            let behavior = node.subscriber("behavior/blackboard").cache(1).build().await?;
                            Ok(Target { context, publisher, behavior, suppress_start: false })
                        }.await;
                        targets.insert(member.id, result?);
                    }
                    _ = timer.tick() => {
                        let input = input.get_after(ros_z::time::Clock::wallclock().now() - Duration::from_millis(250)).map(|v| v.as_ref().clone()).unwrap_or_default();
                        let selected = *selection.lock().unwrap();
                        let fresh_start = gate.update(selected, &input);
                        for (&number, target) in &mut targets {
                            if !input.is_pressed(Button::Start) { target.suppress_start = false; }
                            let mut output = if Some(number) == selected && gate.armed { input.clone() } else { ControllerInput::default() };
                            if Some(number) == selected && gate.armed && fresh_start
                                && target.behavior.get_latest().is_some_and(|state| state.remote_control_enabled) {
                                // Re-enable input routing without toggling an already enabled robot off.
                                target.suppress_start = true;
                            }
                            if target.suppress_start { output.buttons.retain(|button| button.name != Button::Start); }
                            target.publisher.publish(&output).await?;
                        }
                    }
                }
            }
            Ok(())
        }));
        Ok(Self {
            runtime: tokio::runtime::Handle::current(),
            failure: None,
            selected,
            additions,
            task,
            source,
            reader,
        })
    }
    pub fn poll(&mut self) -> Option<String> {
        if self.failure.is_none() {
            let result = if self.task.is_finished() {
                Some(("routing", self.runtime.block_on(&mut self.task)))
            } else if self.reader.is_finished() {
                Some(("capture", self.runtime.block_on(&mut self.reader)))
            } else {
                None
            };
            if let Some((name, result)) = result {
                self.failure = Some(match result {
                    Ok(Ok(())) => format!(
                        "Local gamepad {name} stopped; restart with External controller if host input is unavailable"
                    ),
                    Ok(Err(error)) => format!("Local gamepad {name} failed: {error:#}"),
                    Err(error) => format!("Local gamepad {name} task failed: {error}"),
                });
            }
        }
        self.failure.clone()
    }
    pub fn select(&self, id: Option<RobotId>) {
        *self.selected.lock().unwrap() = id;
    }
    pub fn add(&self, member: Member) {
        let _ = self.additions.send(member);
    }
}
impl Drop for Controller {
    fn drop(&mut self) {
        self.task.abort();
        self.reader.abort();
        let _ = self.source.shutdown();
    }
}
#[derive(Default)]
struct Gate {
    selected: Option<RobotId>,
    armed: bool,
    last_start: bool,
    start_edge: bool,
}
impl Gate {
    fn update(&mut self, selected: Option<RobotId>, input: &ControllerInput) -> bool {
        let start = input.connected && input.is_pressed(Button::Start);
        if selected != self.selected || !input.connected {
            self.selected = selected;
            self.armed = false;
            self.last_start = start;
        }
        self.start_edge = start && !self.last_start;
        self.last_start = start;
        let fresh = self.start_edge && !self.armed;
        if fresh {
            self.armed = true;
        }
        fresh
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn switching_requires_start_release_and_a_new_press() {
        let mut gate = Gate::default();
        let mut input = ControllerInput {
            connected: true,
            ..Default::default()
        };
        gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 1,
            }),
            &input,
        );
        input
            .buttons
            .push(types::controller_input::ControllerButton {
                name: Button::Start,
                pressed: true,
                value: 1.0,
            });
        assert!(gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 1
            }),
            &input
        ));
        assert!(gate.armed);
        assert!(!gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 2
            }),
            &input
        ));
        assert!(!gate.armed);
        input.buttons.clear();
        gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 2,
            }),
            &input,
        );
        input
            .buttons
            .push(types::controller_input::ControllerButton {
                name: Button::Start,
                pressed: true,
                value: 1.0,
            });
        assert!(gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 2
            }),
            &input
        ));
        gate.update(
            Some(RobotId {
                team: crate::TeamId::Hulks,
                number: 2,
            }),
            &ControllerInput::default(),
        );
        assert!(!gate.armed);
        input.buttons.clear();
        let own = Some(RobotId {
            team: crate::TeamId::Hulks,
            number: 1,
        });
        let opponent = Some(RobotId {
            team: crate::TeamId::Opponents,
            number: 1,
        });
        gate.update(own, &input);
        input
            .buttons
            .push(types::controller_input::ControllerButton {
                name: Button::Start,
                pressed: true,
                value: 1.0,
            });
        assert!(gate.update(own, &input));
        assert!(!gate.update(opponent, &input));
        assert!(!gate.armed);
        input.buttons.clear();
        gate.update(opponent, &input);
        input
            .buttons
            .push(types::controller_input::ControllerButton {
                name: Button::Start,
                pressed: true,
                value: 1.0,
            });
        assert!(gate.update(opponent, &input));
    }
}

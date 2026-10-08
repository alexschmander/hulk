//! K1 body buttons enter through the same raw transport as real hardware.
use std::time::{Duration, Instant};

use booster::{ButtonEventMsg, ButtonEventType};
use color_eyre::{Result, eyre::eyre};
use eframe::egui;
use types::primary_state::PrimaryState;

use crate::robotics::Robotics;

pub async fn publish(session: &zenoh::Session, button: i32, event: ButtonEventType) -> Result<()> {
    let bytes =
        cdr::serialize::<_, _, cdr::CdrLe>(&ButtonEventMsg { button, event }, cdr::Infinite)?;
    session
        .put("rt/button_event", bytes)
        .await
        .map_err(|error| eyre!("{error}"))
}

// One second is the simulator's hold threshold, not a firmware timing guarantee.
const HOLD: Duration = Duration::from_secs(1);

#[derive(Default)]
pub struct BodyButtons {
    held: Option<(i32, Instant, bool)>,
}
impl BodyButtons {
    pub fn cancel(&mut self, io: &Robotics) -> Result<()> {
        if let Some((button, _, long)) = self.held.take() {
            if long {
                io.button_event(button, ButtonEventType::LongPressEnd)?;
            }
            io.button_event(button, ButtonEventType::PressUp)?;
        }
        Ok(())
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, io: &Robotics) -> Result<()> {
        let mut rects = [egui::Rect::NOTHING; 3];
        ui.spacing_mut().item_spacing.x = 2.0;
        for (button, label, help) in [
            (0, "F1", "Tap to enter Damping"),
            (
                1,
                "Stand",
                "Tap for Prepare; hold 1 second and release for Initial",
            ),
            (
                2,
                "Walk",
                "Hold 1 second and release to enter Playing from Initial",
            ),
        ] {
            let response = ui
                .add(egui::Button::new(label).min_size(egui::vec2(36.0, 0.0)))
                .on_hover_text(help);
            rects[button as usize] = response.rect;
            if self.held.is_none() {
                if response.is_pointer_button_down_on() {
                    io.button_event(button, ButtonEventType::PressDown)?;
                    self.held = Some((button, Instant::now(), false));
                } else if response.clicked() {
                    press(io, button, false)?;
                }
            }
        }
        ui.spacing_mut().item_spacing.x = ui.style().spacing.item_spacing.x;
        if let Some((button, started, long)) = &mut self.held {
            if !*long && started.elapsed() >= HOLD {
                io.button_event(*button, ButtonEventType::LongPressStart)?;
                io.button_event(*button, ButtonEventType::LongPressHold)?;
                *long = true;
            }
            // Fills while the press counts toward the long-press threshold.
            let rect = rects[*button as usize].shrink2(egui::vec2(3.0, 2.0));
            let progress = (started.elapsed().as_secs_f32() / HOLD.as_secs_f32()).min(1.0);
            let bar = egui::Rect::from_min_max(
                egui::pos2(rect.left(), rect.bottom() - 2.0),
                egui::pos2(rect.left() + rect.width() * progress, rect.bottom()),
            );
            let visuals = ui.visuals();
            let color = if *long {
                visuals.selection.stroke.color
            } else {
                visuals.selection.bg_fill
            };
            ui.painter().rect_filled(bar, 1.0, color);
            ui.ctx().request_repaint();
            if ui.input(|input| !input.pointer.primary_down() || !input.focused) {
                release(io, *button, *long)?;
                self.held = None;
            }
        }
        Ok(())
    }
}
fn release(io: &Robotics, button: i32, long: bool) -> Result<()> {
    if long {
        io.button_event(button, ButtonEventType::LongPressEnd)?;
    }
    io.button_event(button, ButtonEventType::PressUp)?;
    if !long {
        io.button_event(button, ButtonEventType::SingleClick)?;
    }
    Ok(())
}
fn press(io: &Robotics, button: i32, long: bool) -> Result<()> {
    io.button_event(button, ButtonEventType::PressDown)?;
    if long {
        io.button_event(button, ButtonEventType::LongPressStart)?;
        io.button_event(button, ButtonEventType::LongPressHold)?;
    }
    release(io, button, long)
}

/// Boot through the real button handler and safe-pose check. No primary-state override
/// survives startup, so subsequent buttons and Game Controller packets own the state.
#[derive(Default)]
pub struct Startup {
    started: Option<Instant>,
    last_press: Option<Instant>,
}
impl Startup {
    pub fn advance(&mut self, io: &Robotics) -> Result<bool, String> {
        let started = self.started.get_or_insert_with(Instant::now);
        let primary = io.primary.get_latest().map(|state| *state);
        if primary
            .is_some_and(|state| !matches!(state, PrimaryState::Damping | PrimaryState::Prepare))
        {
            return Ok(true);
        }
        if started.elapsed() > Duration::from_secs(10) {
            return Err(format!(
                "Startup could not reach Initial: primary={primary:?}, safe_pose={:?}",
                io.safe_pose.get_latest()
            ));
        }
        if io.safe_pose.get_latest().is_none()
            || (primary == Some(PrimaryState::Prepare)
                && io.safe_pose.get_latest().is_none_or(|safe| !*safe))
            || self
                .last_press
                .is_some_and(|last| last.elapsed() < Duration::from_millis(250))
        {
            return Ok(false);
        }
        if let Some(primary) = primary {
            press(io, 1, primary == PrimaryState::Prepare)
                .map_err(|error| format!("Startup button: {error:#}"))?;
            self.last_press = Some(Instant::now());
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::prelude::*;
    use std::{path::PathBuf, sync::Arc};
    use types::buttons::{ButtonPressType, Buttons};

    #[tokio::test(flavor = "multi_thread")]
    async fn raw_body_events_reach_the_real_handler_and_primary_state_filter() {
        let context = Arc::new(
            crate::robotics::scoped_transport(
                ContextBuilder::default()
                    .with_namespace("/button_test")
                    .disable_multicast_scouting()
                    .with_connect_endpoints(std::iter::empty::<&str>())
                    .with_listen_endpoints(["tcp/127.0.0.1:0"])
                    .with_parameter_layers([
                        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters/base")
                    ]),
            )
            .build()
            .await
            .unwrap(),
        );
        let node = context.create_node("test").build().await.unwrap();
        let buttons = node
            .subscriber::<Buttons<Option<ButtonPressType>>>("buttons")
            .build()
            .await
            .unwrap();
        let primary = node
            .subscriber::<PrimaryState>("primary_state")
            .build()
            .await
            .unwrap();
        let safe = node
            .publisher::<bool>("is_safe_pose")
            .build()
            .await
            .unwrap();
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(button_event_bridge::run_boxed(context.clone()));
        tasks.spawn(button_event_handler::run_boxed(context.clone()));
        tasks.spawn(primary_state_filter::run_boxed(context.clone()));
        tokio::time::timeout(Duration::from_secs(10), async {
            assert_eq!(primary.recv().await.unwrap(), PrimaryState::Damping);
            // Acknowledge discovery through the real bridge/handler, then start assertions.
            loop {
                safe.publish(&true).await.unwrap();
                publish(context.session(), 0, ButtonEventType::PressDown)
                    .await
                    .unwrap();
                if tokio::time::timeout(Duration::from_millis(50), buttons.recv())
                    .await
                    .is_ok()
                {
                    break;
                }
            }
            // Prove PressDown alone does not change primary state; this also acknowledges
            // the safe-pose cache before sending Stand.
            loop {
                safe.publish(&true).await.unwrap();
                publish(context.session(), 0, ButtonEventType::PressDown)
                    .await
                    .unwrap();
                if let Ok(Ok(state)) =
                    tokio::time::timeout(Duration::from_millis(50), primary.recv()).await
                {
                    assert_eq!(state, PrimaryState::Damping);
                    break;
                }
            }
            for (button, long, expected) in [
                (1, false, PrimaryState::Prepare),
                (1, true, PrimaryState::Initial),
                (2, true, PrimaryState::Playing),
                (0, false, PrimaryState::Damping),
            ] {
                publish(context.session(), button, ButtonEventType::PressDown)
                    .await
                    .unwrap();
                if long {
                    publish(context.session(), button, ButtonEventType::LongPressStart)
                        .await
                        .unwrap();
                    publish(context.session(), button, ButtonEventType::LongPressHold)
                        .await
                        .unwrap();
                    publish(context.session(), button, ButtonEventType::LongPressEnd)
                        .await
                        .unwrap();
                }
                publish(context.session(), button, ButtonEventType::PressUp)
                    .await
                    .unwrap();
                loop {
                    let event = buttons.recv().await.unwrap();
                    if let Some(press) = event[button] {
                        assert!(matches!(
                            (press, long),
                            (ButtonPressType::Long, true) | (ButtonPressType::Short, false)
                        ));
                        break;
                    }
                }
                loop {
                    if primary.recv().await.unwrap() == expected {
                        break;
                    }
                }
            }
        })
        .await
        .unwrap();
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        context.shutdown().unwrap();
    }
}

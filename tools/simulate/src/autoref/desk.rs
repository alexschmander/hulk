//! The running panel's view of the embedded referee: scoreboard, match controls and the
//! referee desk. Legality and effects stay in [`Engine`]; this module only presents them.
use super::{
    calls::{self, Call, PENALTIES, RESTARTS},
    engine::{Engine, Event, Origin},
    side,
};
use crate::{RobotId, TeamId};
use eframe::egui::{
    self, Color32, CornerRadius, Margin, Rect, Response, RichText, Sense, Stroke, Ui, WidgetInfo,
    WidgetType,
};
use egui_material_icons::icons;
use game_controller_core::{
    timer::{SignedDuration, Timer},
    types::{Game, Penalty, PenaltyCall, Phase, PlayerNumber, SetPlay, Side, State},
};
use std::time::Duration;

/// How long a new decision stays marked in the log and the scene.
const FRESH: Duration = Duration::from_millis(2500);

/// Jersey colors from the embedded controller's team parameters: HULKs blue, opponents red.
pub fn team_color(visuals: &egui::Visuals, team: TeamId) -> Color32 {
    match (team, visuals.dark_mode) {
        (TeamId::Hulks, true) => Color32::from_rgb(92, 168, 255),
        (TeamId::Hulks, false) => Color32::from_rgb(25, 103, 210),
        (TeamId::Opponents, true) => Color32::from_rgb(255, 112, 112),
        (TeamId::Opponents, false) => Color32::from_rgb(200, 48, 48),
    }
}

/// The GameController's lamp: green while the ball is in play, amber while robots position.
fn state_color(visuals: &egui::Visuals, game: &Game) -> Color32 {
    let dark = visuals.dark_mode;
    if game.stopped {
        visuals.error_fg_color
    } else {
        match game.state {
            State::Playing if dark => Color32::from_rgb(76, 195, 138),
            State::Playing => Color32::from_rgb(28, 138, 82),
            State::Ready | State::Set | State::Timeout if dark => Color32::from_rgb(232, 176, 74),
            State::Ready | State::Set | State::Timeout => Color32::from_rgb(168, 112, 0),
            State::Initial | State::Finished => visuals.weak_text_color(),
        }
    }
}

fn halftime(game: &Game) -> bool {
    (game.phase == Phase::FirstHalf && game.state == State::Finished)
        || (game.phase == Phase::SecondHalf
            && game.state == State::Initial
            && game.secondary_timer.get_remaining().is_positive())
}

fn clock(remaining: SignedDuration) -> String {
    let seconds = remaining.whole_seconds();
    format!(
        "{}{}:{:02}",
        if seconds < 0 { "−" } else { "" },
        seconds.abs() / 60,
        seconds.abs() % 60
    )
}

fn timer_running(timer: &Timer) -> bool {
    matches!(timer, Timer::Started { .. })
}

/// Score, period, match clock, state and restart, followed by the match controls.
pub fn match_strip(ui: &mut Ui, engine: &mut Engine, desk: &mut bool, compact: bool) {
    let game = engine.core.get_game(false).clone();
    ui.horizontal_wrapped(|ui| {
        score_bug(ui, &game);
        ui.add_space(6.0);
        let break_time = halftime(&game);
        ui.label(
            RichText::new(if break_time {
                "Halftime"
            } else {
                match game.phase {
                    Phase::FirstHalf => "1st half",
                    Phase::SecondHalf => "2nd half",
                    Phase::FirstExtraHalf => "1st extra half",
                    Phase::SecondExtraHalf => "2nd extra half",
                    Phase::PenaltyShootout => "Shootout",
                }
            })
            .weak(),
        );
        let remaining = if break_time {
            game.secondary_timer.get_remaining()
        } else {
            game.primary_timer.get_remaining()
        };
        ui.label(
            RichText::new(clock(remaining))
                .monospace()
                .size(15.0)
                .strong(),
        )
        .on_hover_text(if break_time {
            "Time until the second half"
        } else {
            "Match time remaining; it runs with simulation time"
        });
        if !break_time {
            ui.add_space(6.0);
            state_lamp(ui, &game);
        }
        crate::widgets::trailing(ui, if compact { 120.0 } else { 330.0 }, |ui| {
            let toggle = ui
                .add(
                    egui::Button::selectable(
                        *desk,
                        if compact {
                            icons::ICON_VIEW_SIDEBAR.codepoint.to_owned()
                        } else {
                            format!("{} Referee desk", icons::ICON_VIEW_SIDEBAR.codepoint)
                        },
                    )
                    .shortcut_text(if compact { "" } else { "R" }),
                )
                .on_hover_text("Team and player calls with the decision log (R)");
            toggle.widget_info(|| {
                WidgetInfo::selected(WidgetType::Button, true, *desk, "Referee desk (R)")
            });
            if toggle.clicked() {
                *desk = !*desk;
            }
            if let Some(next) = Call::next(&game) {
                call_button(ui, engine, next, compact, Some("N"));
            }
            let stop = if game.stopped {
                Call::ResumePlay
            } else {
                Call::StopPlay
            };
            call_button(ui, engine, stop, compact, Some("Shift+Space"));
        });
    });
}

/// Team names flank the score; their color rails are the jersey colors robots see.
fn score_bug(ui: &mut Ui, game: &Game) {
    let visuals = ui.visuals().clone();
    egui::Frame::new()
        .fill(visuals.extreme_bg_color)
        .stroke(visuals.widgets.noninteractive.bg_stroke)
        .corner_radius(4)
        .inner_margin(Margin::symmetric(3, 2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 7.0;
            ui.horizontal(|ui| {
                let rail = |ui: &mut Ui, team: TeamId| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(4.0, 22.0), Sense::hover());
                    ui.painter().rect_filled(
                        rect,
                        CornerRadius::same(2),
                        team_color(&visuals, team),
                    );
                };
                let name = |ui: &mut Ui, team: TeamId| {
                    let kicking = game.kicking_side == Some(side(team))
                        && game.phase != Phase::PenaltyShootout;
                    let label = ui.label(RichText::new(calls::name(team)).strong());
                    if kicking {
                        let (rect, response) =
                            ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
                        ui.painter()
                            .circle_filled(rect.center(), 3.5, visuals.strong_text_color());
                        response.on_hover_text(kicking_role(game));
                    }
                    label
                };
                rail(ui, TeamId::Hulks);
                name(ui, TeamId::Hulks);
                ui.label(
                    RichText::new(format!(
                        "{}  :  {}",
                        game.teams[Side::Home].score,
                        game.teams[Side::Away].score
                    ))
                    .size(19.0)
                    .strong()
                    .color(visuals.strong_text_color()),
                );
                name(ui, TeamId::Opponents);
                rail(ui, TeamId::Opponents);
            });
        })
        .response
        .widget_info(|| {
            WidgetInfo::labeled(
                WidgetType::Label,
                true,
                format!(
                    "HULKs {}, Opponents {}",
                    game.teams[Side::Home].score,
                    game.teams[Side::Away].score
                ),
            )
        });
}

fn kicking_role(game: &Game) -> &'static str {
    if game.set_play == SetPlay::KickOff || matches!(game.state, State::Initial | State::Timeout) {
        "kicks off"
    } else {
        "takes the restart"
    }
}

fn state_lamp(ui: &mut Ui, game: &Game) {
    let color = state_color(ui.visuals(), game);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.5, color);
    let state = if game.stopped {
        "Stopped"
    } else {
        match game.state {
            State::Initial => "Initial",
            State::Ready => "Ready",
            State::Set => "Set",
            State::Playing => "Playing",
            State::Finished if game.phase == Phase::FirstHalf => "Half finished",
            State::Finished => "Full time",
            State::Timeout => "Timeout",
        }
    };
    ui.label(RichText::new(state).color(color).strong())
        .on_hover_text(if game.stopped {
            "A referee stopped play; robots must stand still until it resumes"
        } else {
            "GameController state sent to every robot"
        });
    if game.set_play != SetPlay::NoSetPlay {
        let restart = calls::set_play_label(game.set_play);
        ui.label(match game.kicking_side {
            Some(side) => format!("{restart} for {}", calls::name(super::team(side))),
            None => format!("Neutral {}", restart.to_lowercase()),
        });
    }
    if timer_running(&game.secondary_timer) && !matches!(game.state, State::Initial) {
        ui.label(
            RichText::new(clock(game.secondary_timer.get_remaining()))
                .monospace()
                .weak(),
        )
        .on_hover_text(match game.state {
            State::Ready => "Time left to reach positions",
            State::Timeout => "Time left in the timeout",
            _ => "Time left for the restart",
        });
    }
}

/// A button for one call, disabled with the core's reason when it is not allowed now.
fn call_button(
    ui: &mut Ui,
    engine: &mut Engine,
    call: Call,
    compact: bool,
    shortcut: Option<&str>,
) -> Response {
    let allowed = engine.allowed(call);
    let game = engine.core.get_game(false);
    let icon = match call {
        Call::StopPlay => Some(icons::ICON_BACK_HAND.codepoint),
        Call::ResumePlay => Some(icons::ICON_PLAY_ARROW.codepoint),
        Call::Whistle => Some(icons::ICON_SPORTS.codepoint),
        Call::Kickoff | Call::Set | Call::BallFree | Call::SecondHalf => {
            Some(icons::ICON_SKIP_NEXT.codepoint)
        }
        _ => None,
    };
    let label = call.label();
    let text = match (icon, compact) {
        (Some(icon), true) => icon.to_owned(),
        (Some(icon), false) => format!("{icon} {label}"),
        (None, _) => label.clone(),
    };
    let mut button = egui::Button::new(text);
    if call == Call::ResumePlay {
        // A stopped match needs attention; make its way out the obvious action.
        button = button.fill(ui.visuals().selection.bg_fill);
    }
    if let Some(shortcut) = shortcut.filter(|_| !compact) {
        button = button.shortcut_text(RichText::new(shortcut).weak());
    }
    let hint = match shortcut {
        Some(shortcut) => format!("{} ({shortcut})", call.hint(game)),
        None => call.hint(game),
    };
    let response = ui
        .add_enabled(allowed, button)
        .on_hover_text(hint)
        .on_disabled_hover_text(call.refusal(game));
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, allowed, &label));
    if response.clicked() {
        let _ = engine.call(call);
    }
    response
}

/// Keyboard calls from the running panel: Shift+Space stops or resumes play, N advances.
pub fn shortcut(engine: &mut Engine, stop: bool, next: bool) {
    let game = engine.core.get_game(false);
    let call = if stop {
        Some(if game.stopped {
            Call::ResumePlay
        } else {
            Call::StopPlay
        })
    } else if next {
        Call::next(game)
    } else {
        None
    };
    // Refusals are logged and shown like any other call.
    if let Some(call) = call {
        let _ = engine.call(call);
    }
}

/// The desk lists every match, team and player call next to the decision log.
/// Returns a player the operator selected.
pub fn desk(
    ui: &mut Ui,
    engine: &mut Engine,
    robots: &[RobotId],
    selected: Option<RobotId>,
) -> Option<RobotId> {
    let mut select = None;
    ui.spacing_mut().item_spacing.y = 6.0;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Match").strong());
        crate::widgets::trailing(ui, 150.0, |ui| {
            ui.checkbox(&mut engine.enabled, "Automatic calls")
                .on_hover_text(
                    "Judge goals, exits, positioning and stuck play from physics.\n\
                     Off keeps penalty handling and the network roundtrip for manual calls.",
                );
        });
    });
    ui.horizontal_wrapped(|ui| {
        let game = engine.core.get_game(false);
        let stop = if game.stopped {
            Call::ResumePlay
        } else {
            Call::StopPlay
        };
        let next = Call::next(game);
        call_button(ui, engine, stop, false, None);
        if let Some(next) = next {
            call_button(ui, engine, next, false, None);
        }
    });
    ui.horizontal_wrapped(|ui| {
        for call in [
            Call::DroppedBall,
            Call::FinishHalf,
            Call::AddMinute,
            Call::RefereeTimeout,
        ] {
            call_button(ui, engine, call, false, None);
        }
    });
    for team in TeamId::ALL {
        ui.add_space(8.0);
        team_section(ui, engine, team, robots, selected, &mut select);
    }
    ui.add_space(8.0);
    ui.label(RichText::new("Decisions").strong());
    log(ui, engine);
    select
}

fn team_section(
    ui: &mut Ui,
    engine: &mut Engine,
    team: TeamId,
    robots: &[RobotId],
    selected: Option<RobotId>,
    select: &mut Option<RobotId>,
) {
    let color = team_color(ui.visuals(), team);
    let top = ui.cursor().top();
    let left = ui.cursor().left();
    let inner = ui
        .horizontal(|ui| {
            ui.add_space(10.0);
            ui.label(RichText::new(calls::name(team)).strong().color(color));
            let game = engine.core.get_game(false);
            if game.kicking_side == Some(side(team)) && game.phase != Phase::PenaltyShootout {
                ui.weak(kicking_role(game));
            }
            crate::widgets::trailing(ui, 130.0, |ui| {
                award_menu(ui, engine, team);
                call_button(ui, engine, Call::Goal(team), false, None);
            });
        })
        .response;
    let mut bottom = inner.rect.bottom();
    let mut players: Vec<_> = robots
        .iter()
        .filter(|id| id.team == team)
        .copied()
        .collect();
    players.sort();
    if players.is_empty() {
        ui.horizontal(|ui| {
            ui.add_space(10.0);
            ui.weak("No players on the field");
        });
    }
    for id in players {
        let row = ui.horizontal(|ui| {
            ui.add_space(10.0);
            if badge(ui, id, selected == Some(id)).clicked() {
                *select = Some(id);
            }
            penalty_status(ui, engine.core.get_game(false), id);
            crate::widgets::trailing(ui, 110.0, |ui| {
                penalty_menu(ui, engine, id, true);
                release_button(ui, engine, id);
            });
        });
        bottom = row.response.rect.bottom();
    }
    // The team's rail ties its calls and players together.
    ui.painter().rect_filled(
        Rect::from_min_max(egui::pos2(left, top), egui::pos2(left + 3.0, bottom)),
        CornerRadius::same(1),
        color,
    );
}

fn award_menu(ui: &mut Ui, engine: &mut Engine, team: TeamId) {
    let response = ui.menu_button("Award", |ui| {
        for set_play in RESTARTS {
            if call_button(ui, engine, Call::Restart(team, set_play), false, None).clicked() {
                ui.close();
            }
        }
        ui.separator();
        if call_button(ui, engine, Call::Timeout(team), false, None).clicked() {
            ui.close();
        }
    });
    response.response.on_hover_text(format!(
        "Award a restart or timeout to {}",
        calls::name(team)
    ));
}

/// Robot badge as in the scene; clicking selects the robot in Twix.
fn badge(ui: &mut Ui, id: RobotId, selected: bool) -> Response {
    let size = egui::vec2(30.0, 20.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let visuals = ui.visuals();
    let color = team_color(visuals, id.team);
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(4), color.gamma_multiply(0.22));
    painter.rect_stroke(
        rect,
        CornerRadius::same(4),
        Stroke::new(1.0, color),
        egui::StrokeKind::Inside,
    );
    if selected || response.has_focus() {
        painter.rect_stroke(
            rect.expand(2.0),
            CornerRadius::same(6),
            visuals.selection.stroke,
            egui::StrokeKind::Outside,
        );
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        calls::short(id),
        egui::FontId::proportional(13.0),
        visuals.strong_text_color(),
    );
    let response = response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(format!("Select {} in Twix", crate::robot_namespace(id)));
    response
        .widget_info(|| WidgetInfo::selected(WidgetType::Button, true, selected, calls::short(id)));
    response
}

/// The player's penalty and the time it still has to serve.
pub fn penalty_status(ui: &mut Ui, game: &Game, id: RobotId) {
    let player = &game.teams[side(id.team)][PlayerNumber::new(id.number)];
    if player.penalty == Penalty::NoPenalty {
        return;
    }
    let color = if player.penalty == Penalty::SentOff {
        ui.visuals().error_fg_color
    } else {
        ui.visuals().warn_fg_color
    };
    ui.label(RichText::new(calls::penalty_label(player.penalty)).color(color));
    if timer_running(&player.penalty_timer) {
        ui.label(
            RichText::new(clock(player.penalty_timer.get_remaining()))
                .monospace()
                .weak(),
        )
        .on_hover_text("Penalty time left; it runs in Ready and Playing");
    } else if !matches!(player.penalty, Penalty::SentOff | Penalty::Substitute) {
        ui.weak("leaving").on_hover_text(
            "The referee moves the robot beside the field, then starts its penalty time",
        );
    }
}

/// Penalty calls for one player; calls the core refuses now stay visible but disabled.
pub fn penalty_menu(ui: &mut Ui, engine: &mut Engine, id: RobotId, compact: bool) {
    let text = if compact {
        icons::ICON_GAVEL.codepoint.to_owned()
    } else {
        format!("{} Penalize", icons::ICON_GAVEL.codepoint)
    };
    // Cards stay possible for every player still in the match.
    let open = !matches!(
        calls::player_penalty(engine.core.get_game(false), id),
        Penalty::SentOff | Penalty::Substitute
    );
    ui.add_enabled_ui(open, |ui| {
        ui.menu_button(text, |ui| {
            ui.label(RichText::new(format!("Penalize {}", calls::short(id))).strong());
            for call in PENALTIES {
                if call == PenaltyCall::Caution {
                    ui.separator();
                }
                if call_button(ui, engine, Call::Penalize(id, call), false, None).clicked() {
                    ui.close();
                }
            }
        })
        .response
        .on_hover_text(format!("Penalize {}", calls::short(id)))
        .on_disabled_hover_text(format!("{} cannot be penalized now", calls::short(id)));
    });
}

/// Release a penalized player early; shown only while it serves a releasable penalty.
pub fn release_button(ui: &mut Ui, engine: &mut Engine, id: RobotId) {
    if engine.allowed(Call::Release(id)) {
        call_button(ui, engine, Call::Release(id), false, None);
    }
}

fn origin_icon(ui: &mut Ui, event: &Event) {
    let (icon, color, hint) = match (event.origin, event.accepted) {
        (_, false) => (
            icons::ICON_BLOCK.codepoint,
            ui.visuals().error_fg_color,
            "Not applied",
        ),
        (Origin::Operator, true) => (
            icons::ICON_PERSON.codepoint,
            ui.visuals().strong_text_color(),
            "Your call",
        ),
        (Origin::Automatic, true) => (
            icons::ICON_SPORTS.codepoint,
            ui.visuals().weak_text_color(),
            "Automatic referee",
        ),
    };
    ui.label(RichText::new(icon).color(color))
        .on_hover_text(hint);
}

fn event_time(event: &Event) -> String {
    let seconds = event.time.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

/// Newest decisions first; new ones are briefly marked so a click visibly lands.
fn log(ui: &mut Ui, engine: &Engine) {
    if engine.events.is_empty() {
        ui.weak("Calls appear here once the match starts.");
        return;
    }
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        for event in engine.events.iter().rev() {
            let fresh = 1.0 - event.logged.elapsed().as_secs_f32() / FRESH.as_secs_f32();
            let background = ui.painter().add(egui::Shape::Noop);
            let row = ui.horizontal(|ui| {
                ui.label(RichText::new(event_time(event)).monospace().weak())
                    .on_hover_text("Simulation time");
                origin_icon(ui, event);
                let text = RichText::new(&event.text);
                ui.add(
                    egui::Label::new(if event.accepted {
                        text
                    } else {
                        text.color(ui.visuals().error_fg_color)
                    })
                    .wrap(),
                );
            });
            if fresh > 0.0 {
                ui.painter().set(
                    background,
                    egui::Shape::rect_filled(
                        row.response.rect.expand(2.0),
                        CornerRadius::same(3),
                        ui.visuals().selection.bg_fill.gamma_multiply(0.35 * fresh),
                    ),
                );
            }
        }
    });
}

/// Confirms an operator call over the scene, also when the desk is closed.
pub fn toast(ui: &Ui, rect: Rect, engine: &Engine) {
    let Some(event) = engine
        .events
        .back()
        .filter(|event| event.origin == Origin::Operator && event.logged.elapsed() < FRESH)
    else {
        return;
    };
    egui::Area::new(ui.id().with("referee_toast"))
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(rect.center_top() + egui::vec2(0.0, 12.0))
        .constrain_to(rect)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    origin_icon(ui, event);
                    let text = RichText::new(&event.text);
                    ui.label(if event.accepted {
                        text.strong()
                    } else {
                        text.color(ui.visuals().error_fg_color)
                    });
                });
            });
        });
}

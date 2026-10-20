use std::sync::Arc;

use color_eyre::Result;
use eframe::{
    emath::Align2,
    epaint::{Color32, FontId, Stroke},
};

use coordinate_systems::{Field, Ground};
use hsl_network_messages::{PlayerNumber, TeamColor, TeamState};
use linear_algebra::{Point2, Vector2, vector};
use nalgebra::Matrix2;
use ros_z_debug::{SampleRecord, TopicObservation};
use types::{
    field_dimensions::FieldDimensions,
    game_controller_state::GameControllerState,
    tracked_objects::{HumanoidRole, HumanoidTrack, TrackStatus, TrackedObjects},
};

use crate::{backend::RobotBackend, panels::map::layer::Layer};
use twix_visualization::twix_painter::TwixPainter;

use super::tactics_board::{
    CHALK, INK, MAXIMUM_DRAWN_STANDARD_DEVIATION, odometry_to_ground_rotation, pixels,
};

/// People are told apart by their square shape; their color stays neutral.
const PERSON: Color32 = Color32::from_rgb(0xd9, 0xdc, 0xd3);
const ROBOT_RADIUS: f32 = 0.2;
/// Motion arrows end where the object will be after this many seconds.
const MOTION_HORIZON: f32 = 1.0;
const MINIMUM_DRAWN_SPEED: f32 = 0.15;
/// Teammate probability above which an unconfirmed identity is hinted with a question mark.
const IDENTITY_HINT_PROBABILITY: f32 = 0.5;

/// Robots and people of the humanoid filter as pucks and squares, teammates colored by the game
/// controller's jersey colors.
pub struct HumanoidFilter {
    humanoids: TopicObservation<TrackedObjects<HumanoidTrack>>,
    game_controller_state: TopicObservation<Option<GameControllerState>>,
}

impl Layer<Ground> for HumanoidFilter {
    const NAME: &'static str = "Humanoid Filter";

    fn new(backend: Arc<RobotBackend>) -> Self {
        let _runtime_handle = backend.runtime_handle().enter();
        let observer = backend.observer();

        Self {
            humanoids: observer
                .observe_typed("humanoid_filter/tracks")
                .expect("failed to construct humanoid tracks observer")
                .spawn(),
            game_controller_state: observer
                .observe_typed("game_controller_state")
                .expect("failed to construct game controller state observer")
                .spawn(),
        }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Ground>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        let colors = TeamColors::new(self.game_controller_state().as_ref());
        if let Some(SampleRecord {
            value: humanoids, ..
        }) = self.humanoids.latest().as_deref()
        {
            paint_humanoid_tracks(painter, humanoids, &colors);
        }
        Ok(())
    }

    /// Places each message with the ground-to-field transform of its own time, which the filter
    /// publishes with its tracks; messages without localization are left out.
    fn paint_in_field(
        &self,
        painter: &TwixPainter<Field>,
        _field_dimensions: &FieldDimensions,
    ) -> Option<Result<()>> {
        let colors = TeamColors::new(self.game_controller_state().as_ref());
        if let Some(SampleRecord {
            value: humanoids, ..
        }) = self.humanoids.latest().as_deref()
            && let Some(ground_to_field) = humanoids.ground_to_field
        {
            paint_humanoid_tracks(
                &painter.transform_painter(ground_to_field.inverse()),
                humanoids,
                &colors,
            );
        }
        Some(Ok(()))
    }
}

impl HumanoidFilter {
    fn game_controller_state(&self) -> Option<GameControllerState> {
        self.game_controller_state
            .latest()
            .and_then(|record| record.value.clone())
    }
}

/// How sure the filter is about a humanoid, shown by fill and outline rather than color.
#[derive(Clone, Copy, PartialEq)]
enum Appearance {
    /// Detected within the last second: a solid puck.
    Seen,
    /// Confident but not detected for a while: a faded puck.
    OutOfSight,
    /// Not yet confident: a faint puck.
    Unconfirmed,
    /// Known only from team messages: a thick ring.
    FromTeamMessages,
}

impl Appearance {
    fn of(track: &HumanoidTrack) -> Self {
        if track.remote_only {
            return Self::FromTeamMessages;
        }
        match track.status {
            TrackStatus::Confirmed => Self::Seen,
            TrackStatus::Dormant => Self::OutOfSight,
            TrackStatus::Tentative => Self::Unconfirmed,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Shape {
    Puck,
    Square,
}

/// Jersey colors of both teams, from the game controller when it is known.
struct TeamColors {
    hulks: Option<TeamState>,
    opponents: Option<TeamState>,
}

impl TeamColors {
    fn new(game_controller_state: Option<&GameControllerState>) -> Self {
        Self {
            hulks: game_controller_state.map(|state| state.hulks_team.clone()),
            opponents: game_controller_state.map(|state| state.opponent_team.clone()),
        }
    }

    fn of(&self, role: HumanoidRole) -> Color32 {
        match role {
            HumanoidRole::Human => PERSON,
            HumanoidRole::Teammate(player_number) => {
                self.hulks.as_ref().map_or(jersey(TeamColor::Blue), |team| {
                    team_jersey(team, Some(player_number))
                })
            }
            HumanoidRole::Opponent => self
                .opponents
                .as_ref()
                .map_or(jersey(TeamColor::Red), |team| team_jersey(team, None)),
        }
    }
}

fn paint_humanoid_tracks(
    painter: &TwixPainter<Ground>,
    humanoids: &TrackedObjects<HumanoidTrack>,
    colors: &TeamColors,
) {
    let rotation = odometry_to_ground_rotation(humanoids);
    // Back to front: uncertainty, motion, pucks, then labels, so text is never covered.
    for track in &humanoids.tracks {
        if Appearance::of(track) != Appearance::Unconfirmed {
            let covariance = rotation
                * track.state.covariance.fixed_view::<2, 2>(0, 0).into_owned()
                * rotation.transpose();
            paint_uncertainty(
                painter,
                track.position_in_ground,
                covariance,
                colors.of(track.role),
            );
        }
    }
    for track in &humanoids.tracks {
        if Appearance::of(track) == Appearance::Seen {
            paint_motion(painter, track.position_in_ground, track.velocity_in_ground);
        }
    }
    for track in &humanoids.tracks {
        let shape = if track.role == HumanoidRole::Human {
            Shape::Square
        } else {
            Shape::Puck
        };
        paint_marker(
            painter,
            track.position_in_ground,
            shape,
            colors.of(track.role),
            Appearance::of(track),
        );
    }
    for track in &humanoids.tracks {
        let (label, hint) = match track.role {
            HumanoidRole::Teammate(player_number) => (number(player_number), false),
            HumanoidRole::Opponent => match likely_teammate(track) {
                Some(player_number) => (format!("{}?", number(player_number)), true),
                None => continue,
            },
            HumanoidRole::Human => continue,
        };
        let fill = colors.of(track.role);
        let color = match Appearance::of(track) {
            Appearance::Seen if !hint => text_color_on(fill),
            Appearance::Seen => text_color_on(fill).gamma_multiply(0.8),
            _ => CHALK,
        };
        let halo = Appearance::of(track) != Appearance::Seen;
        paint_text(
            painter,
            track.position_in_ground,
            Align2::CENTER_CENTER,
            &label,
            13.0,
            color,
            halo,
        );
    }
}

/// The teammate an unconfirmed robot most likely is, if that is more likely than not.
fn likely_teammate(track: &HumanoidTrack) -> Option<PlayerNumber> {
    track
        .teammate_probabilities
        .iter()
        .filter(|belief| belief.probability >= IDENTITY_HINT_PROBABILITY)
        .max_by(|a, b| a.probability.total_cmp(&b.probability))
        .map(|belief| belief.player_number)
}

fn number(player_number: PlayerNumber) -> String {
    player_number.to_string()
}

fn paint_marker(
    painter: &TwixPainter<Ground>,
    position: Point2<Ground>,
    shape: Shape,
    color: Color32,
    appearance: Appearance,
) {
    let (fill, ring) = match appearance {
        Appearance::Seen => (color, Stroke::new(pixels(painter, 2.0), CHALK)),
        Appearance::OutOfSight => (
            color.gamma_multiply(0.4),
            Stroke::new(pixels(painter, 1.5), CHALK.gamma_multiply(0.6)),
        ),
        Appearance::Unconfirmed => (
            color.gamma_multiply(0.15),
            Stroke::new(pixels(painter, 1.0), CHALK.gamma_multiply(0.5)),
        ),
        Appearance::FromTeamMessages => (
            Color32::TRANSPARENT,
            Stroke::new(pixels(painter, 1.0), CHALK.gamma_multiply(0.8)),
        ),
    };
    match shape {
        Shape::Puck => {
            if appearance == Appearance::FromTeamMessages {
                // A thick ring in the jersey color between two chalk hairlines.
                let width = pixels(painter, 4.0);
                painter.circle_stroke(
                    position,
                    ROBOT_RADIUS - width / 2.0,
                    Stroke::new(width, color),
                );
                painter.circle_stroke(position, ROBOT_RADIUS - width, ring);
            }
            painter.circle(position, ROBOT_RADIUS, fill, ring);
        }
        Shape::Square => {
            let half = vector![ROBOT_RADIUS * 0.85, ROBOT_RADIUS * 0.85];
            painter.rect_filled(position - half, position + half, fill);
            painter.rect_stroke(position - half, position + half, ring);
        }
    }
}

/// A chalk arrow to where the object will be after the motion horizon.
fn paint_motion(
    painter: &TwixPainter<Ground>,
    position: Point2<Ground>,
    velocity: Vector2<Ground>,
) {
    let speed = velocity.norm();
    if speed < MINIMUM_DRAWN_SPEED {
        return;
    }
    let direction = velocity / speed;
    let start = position + direction * ROBOT_RADIUS;
    let end = position + velocity * MOTION_HORIZON;
    if (end - start).dot(&direction) <= 0.0 {
        return;
    }
    let stroke = Stroke::new(pixels(painter, 2.0), CHALK);
    painter.line_segment(start, end, stroke);
    let head = 0.12_f32.min((end - start).norm());
    let back = -direction * head;
    let side = vector![-direction.y(), direction.x()] * head * 0.5;
    painter.line_segment(end, end + back + side, stroke);
    painter.line_segment(end, end + back - side, stroke);
}

/// One standard deviation as a faint wash with a hairline edge.
fn paint_uncertainty(
    painter: &TwixPainter<Ground>,
    position: Point2<Ground>,
    covariance: Matrix2<f32>,
    color: Color32,
) {
    let largest_variance = covariance.symmetric_eigenvalues().max();
    if !largest_variance.is_finite() || largest_variance.sqrt() > MAXIMUM_DRAWN_STANDARD_DEVIATION {
        return;
    }
    painter.covariance(
        position,
        covariance,
        Stroke::new(pixels(painter, 1.0), color.gamma_multiply(0.35)),
        color.gamma_multiply(0.08),
    );
}

/// Text in screen pixels; with a halo of ink where it sits on the field instead of a puck.
fn paint_text(
    painter: &TwixPainter<Ground>,
    position: Point2<Ground>,
    align: Align2,
    text: &str,
    size: f32,
    color: Color32,
    halo: bool,
) {
    if halo {
        let offset = pixels(painter, 1.0);
        for (x, y) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
            painter.floating_text(
                position + vector![x * offset, y * offset],
                align,
                text.to_string(),
                FontId::proportional(size),
                INK,
            );
        }
    }
    painter.floating_text(
        position,
        align,
        text.to_string(),
        FontId::proportional(size),
        color,
    );
}

/// Ink or chalk, whichever contrasts more with the fill.
fn text_color_on(fill: Color32) -> Color32 {
    let linear = |channel: u8| {
        let value = channel as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    let luminance =
        0.2126 * linear(fill.r()) + 0.7152 * linear(fill.g()) + 0.0722 * linear(fill.b());
    // Equal contrast against ink and chalk lies near this luminance.
    if luminance > 0.18 { INK } else { CHALK }
}

fn team_jersey(team: &TeamState, player_number: Option<PlayerNumber>) -> Color32 {
    let is_goal_keeper = player_number.is_some() && player_number == team.goal_keeper_player_number;
    jersey(if is_goal_keeper {
        team.goal_keeper_color.clone()
    } else {
        team.field_player_color.clone()
    })
}

/// Jersey colors, stepped for contrast and color-blind separation on the dark field where a
/// validated step exists.
fn jersey(color: TeamColor) -> Color32 {
    match color {
        TeamColor::Blue => Color32::from_rgb(0x39, 0x87, 0xe5),
        TeamColor::Red => Color32::from_rgb(0xe6, 0x67, 0x67),
        TeamColor::Yellow => Color32::from_rgb(0xc9, 0x85, 0x00),
        TeamColor::Black => Color32::from_rgb(0x2b, 0x2b, 0x2a),
        TeamColor::White => Color32::from_rgb(0xf0, 0xef, 0xec),
        TeamColor::Green => Color32::from_rgb(0x00, 0x83, 0x00),
        TeamColor::Orange => Color32::from_rgb(0xd9, 0x59, 0x26),
        TeamColor::Purple => Color32::from_rgb(0x90, 0x85, 0xe9),
        TeamColor::Brown => Color32::from_rgb(0x8a, 0x5a, 0x2b),
        TeamColor::Gray => Color32::from_rgb(0x8c, 0x8c, 0x8c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_on_light_jerseys_use_ink() {
        assert_eq!(text_color_on(jersey(TeamColor::White)), INK);
        assert_eq!(text_color_on(jersey(TeamColor::Yellow)), INK);
        assert_eq!(text_color_on(jersey(TeamColor::Black)), CHALK);
        assert_eq!(text_color_on(jersey(TeamColor::Green)), CHALK);
    }
}

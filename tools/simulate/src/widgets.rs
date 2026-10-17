//! Presentational pieces shared by the Simulator panel's setup and running views.
use eframe::egui::{
    self, Color32, CornerRadius, Margin, Pos2, Rect, Response, RichText, Sense, Stroke, StrokeKind,
    Ui, UiBuilder, WidgetInfo, WidgetType,
};
use egui_material_icons::icons;

use crate::{FieldConfiguration, Profile};

/// A toolbar button whose accessible name stays readable when only its icon is shown.
pub fn action(ui: &mut Ui, icon: &str, label: &str, compact: bool) -> Response {
    let text = if compact {
        icon.to_owned()
    } else {
        format!("{icon} {label}")
    };
    let response = ui.add(egui::Button::new(text));
    let response = if compact {
        response.on_hover_text(label)
    } else {
        response
    };
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, ui.is_enabled(), label));
    response
}

/// The emphasized action of a view, styled like Twix's timeline transport.
pub fn primary(ui: &mut Ui, icon: &str, label: &str) -> Response {
    let response = ui.add(
        egui::Button::new(format!("{icon} {label}"))
            .fill(ui.visuals().selection.bg_fill)
            .min_size(egui::vec2(0.0, 26.0)),
    );
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, ui.is_enabled(), label));
    response
}

/// Places trailing controls at the right edge, wrapping first when the row is full.
pub fn trailing<R>(ui: &mut Ui, width: f32, content: impl FnOnce(&mut Ui) -> R) -> R {
    if ui.available_width() < width {
        ui.end_row();
    }
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), content)
        .inner
}

pub fn error_banner(ui: &mut Ui, message: &str) {
    let color = ui.visuals().error_fg_color;
    egui::Frame::new()
        .fill(color.gamma_multiply(0.12))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.5)))
        .corner_radius(4)
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                ui.label(RichText::new(icons::ICON_ERROR.codepoint).color(color));
                ui.add(
                    egui::Label::new(
                        RichText::new(message).color(ui.visuals().strong_text_color()),
                    )
                    .wrap(),
                );
            });
        });
}

/// A key legend chip, e.g. for the fly camera's shortcuts.
pub fn keycap(ui: &mut Ui, key: &str) {
    egui::Frame::new()
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(3)
        .inner_margin(Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.label(RichText::new(key).size(11.0).strong());
        });
}

/// Select a cumulative profile preset.
fn profile_ladder(ui: &mut Ui, selected: &mut Profile) -> Response {
    let index = Profile::PRESETS
        .iter()
        .position(|profile| profile == selected);
    let mut changed = None;
    let mut response = ui
        .vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for (step, profile) in Profile::PRESETS.into_iter().enumerate() {
                let included = index.is_some_and(|index| step <= index);
                let current = index == Some(step);
                let last = step + 1 == Profile::PRESETS.len();
                let row = ui.scope_builder(UiBuilder::new().sense(Sense::click()), |ui| {
                    let hovered = ui.response().hovered();
                    let rail = ui.cursor().left() + 8.0;
                    let background = ui.painter().add(egui::Shape::Noop);
                    ui.horizontal_top(|ui| {
                        ui.add_space(22.0);
                        ui.vertical(|ui| {
                            ui.add_space(3.0);
                            let label = RichText::new(profile.label());
                            ui.add(
                                egui::Label::new(if current { label.strong() } else { label })
                                    .selectable(false),
                            );
                            ui.add(
                                egui::Label::new(
                                    RichText::new(profile.description()).size(11.5).weak(),
                                )
                                .selectable(false)
                                .wrap(),
                            );
                            ui.add_space(6.0);
                        });
                    });
                    let rect = ui.min_rect();
                    let visuals = ui.visuals();
                    let accent = accent(visuals);
                    let quiet = visuals.weak_text_color();
                    let painter = ui.painter();
                    if hovered {
                        painter.set(
                            background,
                            egui::Shape::rect_filled(
                                Rect::from_x_y_ranges(ui.max_rect().x_range(), rect.y_range()),
                                4,
                                visuals.widgets.hovered.weak_bg_fill.gamma_multiply(0.5),
                            ),
                        );
                    }
                    let dot = Pos2::new(
                        rail,
                        rect.top() + 3.0 + ui.text_style_height(&egui::TextStyle::Body) / 2.0,
                    );
                    if !last {
                        let next_included = index.is_some_and(|index| step < index);
                        painter.line_segment(
                            [
                                dot + egui::vec2(0.0, 6.0),
                                Pos2::new(rail, rect.bottom() + 3.0),
                            ],
                            Stroke::new(
                                2.0,
                                if next_included {
                                    accent
                                } else {
                                    quiet.gamma_multiply(0.5)
                                },
                            ),
                        );
                    }
                    if included {
                        painter.circle_filled(dot, 5.0, accent);
                    } else {
                        painter.circle_stroke(dot, 4.5, Stroke::new(1.5, quiet));
                    }
                    if current {
                        painter.circle_stroke(dot, 8.0, Stroke::new(1.5, accent));
                    }
                });
                let response = row.response;
                if response.has_focus() {
                    ui.painter().rect_stroke(
                        response.rect,
                        4,
                        ui.visuals().selection.stroke,
                        StrokeKind::Inside,
                    );
                }
                response.widget_info(|| {
                    WidgetInfo::selected(
                        WidgetType::RadioButton,
                        ui.is_enabled(),
                        current,
                        profile.label(),
                    )
                });
                if response.clicked() {
                    changed = Some(profile);
                }
            }
        })
        .response;
    if let Some(profile) = changed {
        *selected = profile;
        response.mark_changed();
    }
    response
}

/// Profile presets, with node and ground-truth details for the selection.
/// The popover ignores outside clicks so configurations can be compared live.
pub fn profile_picker(ui: &mut Ui, selected: &mut Profile) -> Response {
    let id = ui.make_persistent_id("profile_details");
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    let top = ui.cursor().top();
    let info = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.label(RichText::new("Profile").strong());
            let label = "Nodes and ground truth";
            let response = ui
                .add(egui::Button::selectable(open, icons::ICON_INFO.codepoint).small())
                .on_hover_text(format!("{label} of the selected profile"));
            response
                .widget_info(|| WidgetInfo::labeled(WidgetType::Button, ui.is_enabled(), label));
            response
        })
        .inner;
    if info.clicked() {
        egui::Popup::toggle_id(ui.ctx(), id);
    }
    let selection = profile_ladder(ui, selected);
    let anchor = Rect::from_x_y_ranges(selection.rect.x_range(), top..=selection.rect.bottom());
    let profile = *selected;
    egui::Popup::new(id, ui.ctx().clone(), anchor, ui.layer_id())
        .open_memory(None)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .align(egui::RectAlign::RIGHT_START)
        .align_alternatives(&[egui::RectAlign::LEFT_START, egui::RectAlign::BOTTOM_START])
        .gap(8.0)
        .width(400.0)
        .show(|ui| {
            // Areas default to 400 points high, too little for the larger profiles.
            let height = ui.ctx().content_rect().height() * 0.75;
            ui.set_width(400.0);
            ui.set_max_height(height);
            ui.horizontal(|ui| {
                ui.label(RichText::new(profile.label()).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(icons::ICON_CLOSE.codepoint).frame(false))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        egui::Popup::close_id(ui.ctx(), id);
                    }
                });
            });
            ui.label(
                RichText::new(format!(
                    "{} real nodes run per robot",
                    profile.nodes().count()
                ))
                .size(11.5)
                .weak(),
            );
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| profile_details(ui, profile));
        });
    selection
}

/// What a profile launches per robot, read from the same tables that start it.
fn profile_details(ui: &mut Ui, profile: Profile) {
    use crate::profiles::{REFERENCES, Supply};
    let small = |text: &str| RichText::new(text).size(11.5);
    let note = |ui: &mut Ui, text: &str| {
        ui.add(egui::Label::new(small(text).weak()).wrap());
    };
    let topic = |ui: &mut Ui, topic: &str| {
        ui.add(egui::Label::new(RichText::new(topic).monospace().size(11.0)).extend())
    };
    let heading = |ui: &mut Ui, text: &str| {
        ui.add_space(10.0);
        ui.label(RichText::new(text).strong());
    };
    // Keeps wrapped rows clear of the scroll bar.
    ui.set_max_width(ui.available_width() - 10.0);
    ui.spacing_mut().item_spacing.y = 4.0;
    ui.label(RichText::new("Real nodes").strong());
    for group in profile.node_groups() {
        let selected_estimator = group.estimator.is_some();
        let rail = ui.painter().add(egui::Shape::Noop);
        let block = ui
            .horizontal_top(|ui| {
                ui.add_space(10.0);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    ui.label(small(group.label));
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        for node in group.nodes {
                            topic(ui, node.name);
                        }
                    });
                })
            })
            .response;
        let color = if selected_estimator {
            accent(ui.visuals())
        } else {
            ui.visuals().weak_text_color().gamma_multiply(0.5)
        };
        ui.painter().set(
            rail,
            egui::Shape::line_segment(
                [
                    Pos2::new(block.rect.left() + 2.0, block.rect.top() + 2.0),
                    Pos2::new(block.rect.left() + 2.0, block.rect.bottom() - 2.0),
                ],
                Stroke::new(2.0, color),
            ),
        );
        block.on_hover_text(if selected_estimator {
            "Enabled by the estimator selection"
        } else {
            "Always runs, including in custom selections"
        });
    }
    note(
        ui,
        "Once per session, controller_handler routes the gamepad to the selected robot.",
    );

    for (supply, title, absent) in [
        (
            Supply::Perfect,
            "Ground truth fed to the robot",
            "None. Real nodes estimate everything from simulated measurements.",
        ),
        (
            Supply::Synthetic,
            "Synthetic measurements",
            "None. No real node consumes detections in this profile.",
        ),
    ] {
        heading(ui, title);
        let mut supplied = REFERENCES
            .iter()
            .filter(|reference| reference.production(profile) == Some(supply))
            .peekable();
        if supplied.peek().is_none() {
            note(ui, absent);
        }
        for reference in supplied {
            ui.horizontal_wrapped(|ui| {
                topic(ui, reference.topic);
                ui.add(egui::Label::new(small(reference.meaning).weak()));
            });
        }
    }
    ui.add_space(4.0);
    note(
        ui,
        "Every profile also simulates raw Booster sensors, buttons, whistle detections and camera calibration.",
    );

    heading(ui, "Comparison references");
    note(
        ui,
        &format!(
            "All {} are published under ground_truth/ in every profile. Hover one for its meaning.",
            REFERENCES.len()
        ),
    );
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        for reference in REFERENCES {
            topic(ui, reference.topic).on_hover_text(reference.meaning);
        }
    });
}

/// Twix's selection color, darkened in light mode where its fill is too pale for thin marks.
fn accent(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        visuals.selection.bg_fill
    } else {
        visuals.selection.stroke.color
    }
}

/// One choice of a segmented control; every option keeps its frame so the group reads as one.
pub fn segment(ui: &mut Ui, selected: bool, label: &str) -> Response {
    ui.add(
        egui::Button::selectable(selected, label)
            .frame_when_inactive(true)
            .min_size(egui::vec2(32.0, 24.0)),
    )
}

/// Top-down sketch of the selected field, with the initial spawn slots for up to five players.
pub fn field_preview(
    ui: &mut Ui,
    field: &FieldConfiguration,
    robots: u8,
    opponents: u8,
    opponent: &crate::Opponent,
    max_height: f32,
) -> Response {
    let dimensions = &field.dimensions;
    let margin = dimensions
        .border_strip_width
        .max(dimensions.goal_depth + 0.2);
    let extent = egui::vec2(
        dimensions.length + 2.0 * margin,
        dimensions.width + 2.0 * dimensions.border_strip_width.max(0.4),
    );
    let scale = (ui.available_width() / extent.x)
        .min(max_height / extent.y)
        .max(1.0);
    let (rect, response) = ui.allocate_exact_size(extent * scale, Sense::hover());
    let dark = ui.visuals().dark_mode;
    let turf = if dark {
        Color32::from_rgb(28, 84, 44)
    } else {
        Color32::from_rgb(62, 136, 76)
    };
    let surround = if dark {
        Color32::from_rgb(22, 68, 36)
    } else {
        Color32::from_rgb(54, 122, 67)
    };
    let chalk = Color32::from_rgba_unmultiplied(238, 242, 236, 225);
    let line = Stroke::new((dimensions.line_width * scale).max(1.0), chalk);
    let at = |x: f32, y: f32| rect.center() + egui::vec2(x, -y) * scale;
    let area = |x0: f32, y0: f32, x1: f32, y1: f32| Rect::from_two_pos(at(x0, y0), at(x1, y1));
    let painter = ui.painter_at(rect);
    let (half_length, half_width) = (dimensions.length / 2.0, dimensions.width / 2.0);

    painter.rect_filled(rect, CornerRadius::same(6), surround);
    painter.rect_filled(
        area(-half_length, -half_width, half_length, half_width),
        0,
        turf,
    );
    painter.rect_stroke(
        area(-half_length, -half_width, half_length, half_width),
        0,
        line,
        StrokeKind::Middle,
    );
    painter.line_segment([at(0.0, -half_width), at(0.0, half_width)], line);
    painter.circle_stroke(
        at(0.0, 0.0),
        dimensions.center_circle_diameter / 2.0 * scale,
        line,
    );
    painter.circle_filled(at(0.0, 0.0), line.width.max(1.5), chalk);
    for sign in [-1.0, 1.0] {
        let goal_line = sign * half_length;
        for (length, width) in [
            (
                dimensions.penalty_area_length,
                dimensions.penalty_area_width,
            ),
            (
                dimensions.goal_box_area_length,
                dimensions.goal_box_area_width,
            ),
        ] {
            painter.rect_stroke(
                area(
                    goal_line,
                    -width / 2.0,
                    goal_line - sign * length,
                    width / 2.0,
                ),
                0,
                line,
                StrokeKind::Middle,
            );
        }
        painter.circle_filled(
            at(goal_line - sign * dimensions.penalty_marker_distance, 0.0),
            (dimensions.penalty_marker_size / 2.0 * scale).max(1.5),
            chalk,
        );
        let goal = area(
            goal_line,
            -dimensions.goal_inner_width / 2.0,
            goal_line + sign * dimensions.goal_depth,
            dimensions.goal_inner_width / 2.0,
        );
        painter.rect_filled(goal, 0, Color32::from_black_alpha(50));
        painter.rect_stroke(goal, 0, Stroke::new(line.width, chalk), StrokeKind::Middle);
    }

    let radius = (0.3 * scale).clamp(7.0, 11.0);
    for (away, count) in [(false, robots), (true, opponents)] {
        for index in 0..5 {
            let pose = crate::team::on_field_side(crate::team::spawn_pose(dimensions, index), away);
            let center = at(pose.translation.x, -pose.translation.z);
            let forward = pose.rotation * bevy::math::Vec3::X;
            let heading = egui::vec2(forward.x, forward.z).normalized();
            let number = (index + 1).to_string();
            if index < count {
                painter.line_segment(
                    [center + heading * radius, center + heading * (radius + 5.0)],
                    Stroke::new(2.0, chalk),
                );
                painter.circle_filled(
                    center,
                    radius,
                    if away {
                        Color32::from_rgb(255, 170, 95)
                    } else {
                        Color32::WHITE
                    },
                );
                painter.circle_stroke(
                    center,
                    radius,
                    Stroke::new(1.5, Color32::from_black_alpha(140)),
                );
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    number,
                    egui::FontId::proportional(radius * 1.25),
                    Color32::BLACK,
                );
            } else {
                painter.circle_stroke(
                    center,
                    radius - 1.0,
                    Stroke::new(1.25, Color32::from_white_alpha(150)),
                );
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    number,
                    egui::FontId::proportional(radius * 1.15),
                    Color32::from_white_alpha(170),
                );
            }
        }
    }
    response.widget_info(|| {
        WidgetInfo::labeled(
            WidgetType::Other,
            true,
            format!(
                "{} × {} m field with {robots} HULKs and {opponents} {}",
                meters(dimensions.length),
                meters(dimensions.width),
                opponent.team_name(crate::TeamId::Opponents)
            ),
        )
    });
    response
}

pub fn meters(value: f32) -> String {
    let rounded = (value * 100.0).round() / 100.0;
    format!("{rounded}")
}

/// Use the received jersey assignment for both teams, including the designated goalkeeper.
pub(crate) fn jersey_color(member: &crate::team::Member, visuals: &egui::Visuals) -> Color32 {
    use hsl_network_messages::TeamColor;
    match member.io.jersey_color(member.id.number) {
        Some(TeamColor::Blue) => Color32::from_rgb(35, 105, 220),
        Some(TeamColor::Red) => Color32::from_rgb(210, 45, 45),
        Some(TeamColor::Yellow) => Color32::from_rgb(245, 210, 35),
        Some(TeamColor::Black) => Color32::from_rgb(25, 25, 25),
        Some(TeamColor::White) => Color32::from_rgb(240, 240, 240),
        Some(TeamColor::Green) => Color32::from_rgb(35, 170, 75),
        Some(TeamColor::Orange) => Color32::from_rgb(240, 135, 35),
        Some(TeamColor::Purple) => Color32::from_rgb(145, 70, 190),
        Some(TeamColor::Brown) => Color32::from_rgb(130, 85, 50),
        Some(TeamColor::Gray) => Color32::from_gray(145),
        None => crate::autoref::team_color(visuals, member.id.team),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        context: &egui::Context,
        profile: &mut Profile,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        context.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(760.0, 900.0))),
                events,
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.set_max_width(320.0);
                    profile_picker(ui, profile);
                });
            },
        )
    }

    fn text<'a>(
        output: &'a egui::FullOutput,
        label: &'a str,
    ) -> impl Iterator<Item = &'a egui::epaint::TextShape> {
        output
            .shapes
            .iter()
            .filter_map(move |shape| match &shape.shape {
                egui::epaint::Shape::Text(text) if text.galley.text() == label => Some(text),
                _ => None,
            })
    }

    fn click_label(
        context: &egui::Context,
        profile: &mut Profile,
        output: &egui::FullOutput,
        label: &str,
    ) -> egui::FullOutput {
        let text = text(output, label)
            .next()
            .unwrap_or_else(|| panic!("missing {label}"));
        let pos = text.pos + text.galley.size() / 2.0;
        let mut events = vec![egui::Event::PointerMoved(pos)];
        events.extend([true, false].map(|pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }));
        frame(context, profile, events);
        frame(context, profile, vec![])
    }

    #[test]
    fn profile_presets_update_selection_and_details() {
        let context = egui::Context::default();
        egui_material_icons::initialize(&context);
        context.global_style_mut(|style| style.animation_time = 0.0);
        let mut profile = Profile::default();
        frame(&context, &mut profile, vec![]);
        let output = frame(&context, &mut profile, vec![]);
        let output = click_label(&context, &mut profile, &output, "Body state & odometry");
        assert_eq!(profile, Profile::BodyStateOdometry);
        let output = click_label(&context, &mut profile, &output, icons::ICON_INFO.codepoint);
        assert!(
            text(&output, "32 real nodes run per robot")
                .next()
                .is_some()
        );
        assert!(text(&output, "search_suggestor").next().is_some());
        assert!(text(&output, "kinematics_provider").next().is_some());
        assert!(text(&output, "ball_filter").next().is_some());
        let output = click_label(
            &context,
            &mut profile,
            &output,
            "Ball and obstacle filtering",
        );
        assert_eq!(profile, Profile::Filtering);
        assert!(
            text(&output, "27 real nodes run per robot")
                .next()
                .is_some()
        );
        // The information popover stays open while selecting another profile.
        click_label(&context, &mut profile, &output, "Motion & behavior");
        assert_eq!(profile, Profile::MotionBehavior);
    }
}

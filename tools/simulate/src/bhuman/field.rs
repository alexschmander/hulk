use std::path::Path;

use color_eyre::{Result, eyre::ensure};

/// Keep native symbolic field lines and tactical configuration, changing the
/// dimensional inputs to match the host's field. The pinned checkout stays read-only.
pub(super) fn write_field_configuration(
    source: &Path,
    directory: &Path,
    field: &crate::FieldConfiguration,
) -> Result<()> {
    let f = field.dimensions;
    let x = f.length * 500.0;
    let y = f.width * 500.0;
    let post = f.goal_post_diameter * 500.0;
    let values = [
        ("xPosOpponentFieldBorder", x + f.border_strip_width * 1000.0),
        ("xPosOpponentGoal", x + f.goal_depth * 1000.0),
        ("xPosOpponentGoalPost", x + f.line_width * 500.0),
        ("xPosOpponentGoalLine", x),
        ("xPosOpponentGoalArea", x - f.goal_box_area_length * 1000.0),
        (
            "xPosOpponentPenaltyArea",
            x - f.penalty_area_length * 1000.0,
        ),
        (
            "xPosOpponentPenaltyMark",
            x - f.penalty_marker_distance * 1000.0,
        ),
        ("yPosLeftFieldBorder", y + f.border_strip_width * 1000.0),
        ("yPosLeftReturnFromPenalty", y + 500.0),
        ("yPosLeftTouchline", y),
        ("yPosLeftPenaltyArea", f.penalty_area_width * 500.0),
        ("yPosLeftGoalArea", f.goal_box_area_width * 500.0),
        ("yPosLeftGoal", f.goal_inner_width * 500.0 + post),
        ("fieldLinesWidth", f.line_width * 1000.0),
        ("centerCircleRadius", f.center_circle_diameter * 500.0),
        ("goalPostRadius", post),
        ("goalHeight", field.goal_height * 1000.0),
        ("markRadius", f.penalty_marker_size * 500.0),
        ("cornerArcRadius", f.corner_arc_radius * 1000.0),
    ];
    let mut template =
        std::fs::read_to_string(source.join("Config/Locations/Default/fieldDimensions.cfg"))?;
    for (key, value) in values {
        template = replace_assignment(&template, key, &value.to_string())?;
    }
    template = replace_assignment(
        &template,
        "hasCornerArcs",
        if f.corner_arc_radius > 0.0 {
            "true"
        } else {
            "false"
        },
    )?;
    std::fs::write(directory.join("fieldDimensions.cfg"), template)?;
    let ball =
        std::fs::read_to_string(source.join("Config/Locations/Default/K1/ballSpecification.cfg"))?;
    std::fs::write(
        directory.join("ballSpecification.cfg"),
        replace_assignment(&ball, "radius", &(f.ball_radius * 1000.0).to_string())?,
    )?;
    Ok(())
}

fn replace_assignment(template: &str, key: &str, value: &str) -> Result<String> {
    let mut replaced = false;
    let lines: Vec<_> = template
        .lines()
        .map(|line| {
            if line
                .split_once('=')
                .is_some_and(|(name, _)| name.trim() == key)
            {
                replaced = true;
                format!("{key} = {value};")
            } else {
                line.to_owned()
            }
        })
        .collect();
    ensure!(replaced, "missing native field parameter {key}");
    Ok(lines.join("\n") + "\n")
}

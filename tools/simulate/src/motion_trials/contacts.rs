//! Physics-step strike measurements, independent of policy observations.
use crate::robot_io::Data;
use mujoco_rs::prelude::MjtObj;
use nalgebra::{Matrix3, Vector3};
use ros_z::time::Time;

struct Body {
    position: [f64; 3],
    rotation: [f64; 9],
    angular: [f64; 3],
    linear: [f64; 3],
}

impl Body {
    fn capture(data: &Data, id: usize) -> Self {
        let velocity = data.object_velocity(MjtObj::mjOBJ_XBODY, id, false);
        Self {
            position: data.xpos()[id],
            rotation: data.xmat()[id],
            angular: velocity[..3].try_into().unwrap(),
            linear: velocity[3..].try_into().unwrap(),
        }
    }

    fn point(&self, local: Vector3<f64>) -> Vector3<f64> {
        Vector3::from(self.position) + Matrix3::from_row_slice(&self.rotation) * local
    }

    fn velocity_at(&self, point: Vector3<f64>) -> Vector3<f64> {
        Vector3::from(self.linear)
            + Vector3::from(self.angular).cross(&(point - Vector3::from(self.position)))
    }
}

struct Contact {
    foot: Option<usize>,
    body_name: String,
    point: [f64; 3],
    force: f64,
}

struct Frame {
    nanos: i64,
    ball: Body,
    feet: [Body; 2],
    trunk: Body,
    contacts: Vec<Contact>,
}

pub(super) struct Audit {
    ball: usize,
    ball_geom: i32,
    feet: [usize; 2],
    trunk: usize,
    frames: Vec<Frame>,
}

impl Audit {
    pub fn new(data: &Data, robot: &str, ball: &str) -> Self {
        let id = |kind, name: String| data.model().name_to_id(kind, &name).unwrap();
        Self {
            ball: id(MjtObj::mjOBJ_BODY, format!("{ball}ball")),
            ball_geom: id(MjtObj::mjOBJ_GEOM, format!("{ball}ball")) as i32,
            feet: ["left_foot_link", "right_foot_link"]
                .map(|name| id(MjtObj::mjOBJ_BODY, format!("{robot}{name}"))),
            trunk: id(MjtObj::mjOBJ_BODY, format!("{robot}Trunk")),
            frames: Vec::new(),
        }
    }

    pub fn capture(&mut self, data: &Data, time: Time) {
        let contacts = data
            .contact()
            .iter()
            .enumerate()
            .filter_map(|(index, contact)| {
                if !contact.geom.contains(&self.ball_geom) {
                    return None;
                }
                let other = *contact.geom.iter().find(|&&id| id != self.ball_geom)?;
                if other < 0 {
                    return None;
                }
                let body = data.model().geom_bodyid()[other as usize] as usize;
                let force = data.contact_force(index)[0];
                if body == 0 || force <= 1.0 {
                    return None;
                }
                Some(Contact {
                    foot: self.feet.iter().position(|&id| id == body),
                    body_name: data
                        .model()
                        .id_to_name(MjtObj::mjOBJ_BODY, body)
                        .unwrap()
                        .to_owned(),
                    point: contact.pos,
                    force,
                })
            })
            .collect();
        self.frames.push(Frame {
            nanos: time.as_nanos(),
            ball: Body::capture(data, self.ball),
            feet: self.feet.map(|id| Body::capture(data, id)),
            trunk: Body::capture(data, self.trunk),
            contacts,
        });
    }

    pub fn finish(&self, start: Time, scenario: &str, goal_x: f64) -> serde_json::Value {
        let start = start.as_nanos();
        let mut strikes = Vec::new();
        let mut previous = [0; 3];
        for (index, frame) in self.frames.iter().enumerate() {
            if frame.nanos < start {
                continue;
            }
            let Some(contact) = frame
                .contacts
                .iter()
                .max_by(|a, b| a.force.total_cmp(&b.force))
            else {
                continue;
            };
            let group = contact.foot.unwrap_or(2);
            if frame.nanos.saturating_sub(previous[group]) < 200_000_000 {
                continue;
            }
            previous[group] = frame.nanos;
            let Some(foot) = contact.foot else {
                strikes.push(serde_json::json!({"time":(frame.nanos-start) as f64*1e-9,"non_foot_body":contact.body_name,"force":contact.force,"point":contact.point,"ball":frame.ball.position}));
                continue;
            };
            let ball = Vector3::from(frame.ball.position);
            let axis = if scenario == "kickoff" && ball.xy().norm() < 0.5 {
                nalgebra::vector![1.0, 1.0, 0.0].normalize()
            } else if matches!(scenario, "behavior" | "kickoff") {
                nalgebra::vector![goal_x - ball.x, -ball.y, 0.0].normalize()
            } else {
                Vector3::x()
            };
            let lateral = nalgebra::vector![-axis.y, axis.x, 0.0];
            let foot_body = &frame.feet[foot];
            let local = Matrix3::from_row_slice(&foot_body.rotation).transpose()
                * (Vector3::from(contact.point) - Vector3::from(foot_body.position));
            let before = &self.frames[index.saturating_sub(1)];
            let point = before.feet[foot].point(local);
            let velocity = before.feet[foot].velocity_at(point);
            let swing = velocity - before.trunk.velocity_at(point);
            let relative = velocity - Vector3::from(before.ball.linear);
            let radial = ball - Vector3::from(contact.point);
            let mut peak = (f64::NEG_INFINITY, frame.nanos);
            let mut peak_swing = f64::NEG_INFINITY;
            let mut through = f64::NEG_INFINITY;
            let mut timeline = Vec::new();
            for sample in self.frames.iter().filter(|sample| {
                sample.nanos >= frame.nanos.saturating_sub(300_000_000)
                    && sample.nanos <= frame.nanos + 200_000_000
            }) {
                let p = sample.feet[foot].point(local);
                let v = sample.feet[foot].velocity_at(p);
                let s = v - sample.trunk.velocity_at(p);
                if sample.nanos < frame.nanos {
                    if v.dot(&axis) > peak.0 {
                        peak = (v.dot(&axis), sample.nanos);
                    }
                    peak_swing = peak_swing.max(s.dot(&axis));
                } else if sample.nanos <= frame.nanos + 150_000_000 {
                    through = through.max((p - ball).dot(&axis));
                }
                timeline.push(serde_json::json!({
                    "ms":(sample.nanos-frame.nanos) as f64*1e-6,
                    "point":p.as_slice(),"velocity":v.as_slice(),"swing_velocity":s.as_slice(),
                    "ball":sample.ball.position,"ball_velocity":sample.ball.linear,
                    "force":sample.contacts.iter().filter(|c|c.foot==Some(foot)).map(|c|c.force).sum::<f64>(),
                }));
            }
            strikes.push(serde_json::json!({
                "time":(frame.nanos-start) as f64*1e-9,"foot":if foot==0 {"left"} else {"right"},
                "ball":frame.ball.position,"contact":contact.point,"foot_local":local.as_slice(),
                "preimpact_speed":velocity.norm(),"preimpact_forward_speed":velocity.dot(&axis),
                "preimpact_swing_forward_speed":swing.dot(&axis),"closing_speed":relative.dot(&axis),
                "peak_forward_speed":peak.0,"peak_swing_forward_speed":peak_swing,
                "speed_ratio":velocity.dot(&axis)/peak.0,"swing_speed_ratio":swing.dot(&axis)/peak_swing,
                "peak_lead_ms":(frame.nanos-peak.1) as f64*1e-6,
                "lateral_offset_mm":radial.dot(&lateral)*1000.0,"vertical_offset_mm":radial.z*1000.0,
                "centre_offset_mm":radial.cross(&axis).norm()*1000.0,
                "incidence_degrees":radial.normalize().dot(&relative.normalize()).clamp(-1.0,1.0).acos().to_degrees(),
                "velocity_line_offset_mm":radial.cross(&relative.normalize()).norm()*1000.0,
                "followthrough_past_original_centre_mm":through*1000.0,
                "timeline":timeline,
            }));
        }
        if let Ok(path) = std::env::var("MOTION_TRIAL_CONTACT_OUTPUT") {
            std::fs::write(
                path,
                serde_json::to_vec(
                    &serde_json::json!({"strikes":strikes,"physics_steps":self.frames.len()}),
                )
                .unwrap(),
            )
            .unwrap();
        }
        // Keep detailed physics histories in the optional artifact, not the test log.
        for strike in &mut strikes {
            strike.as_object_mut().unwrap().remove("timeline");
        }
        let result = serde_json::json!({"strikes":strikes,"physics_steps":self.frames.len()});
        eprintln!("MOTION_CONTACTS {result}");
        let index = std::env::var("MOTION_TRIAL_CONTACT_INDEX")
            .map(|index| index.parse::<usize>().unwrap())
            .unwrap_or(0);
        let strike = result["strikes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|strike| strike["foot"].is_string())
            .nth(index);
        if std::env::var("MOTION_TRIAL_REQUIRE_FOOT_FIRST").as_deref() == Ok("1") {
            assert!(
                result["strikes"][0]["foot"].is_string(),
                "ball hit another body before the foot: {result}"
            );
        }
        if let Ok(minimum) = std::env::var("MOTION_TRIAL_MIN_CONTACT_SPEED_RATIO") {
            let minimum: f64 = minimum.parse().unwrap();
            let ratio = strike.expect("no foot contact")["speed_ratio"]
                .as_f64()
                .unwrap();
            assert!(
                ratio >= minimum,
                "late foot contact: {ratio:.3} of pre-impact swing peak, expected {minimum:.3}"
            );
        }
        if let Ok(minimum) = std::env::var("MOTION_TRIAL_MIN_FOLLOWTHROUGH_MM") {
            let minimum: f64 = minimum.parse().unwrap();
            let through = strike.expect("no foot contact")["followthrough_past_original_centre_mm"]
                .as_f64()
                .unwrap();
            assert!(
                through >= minimum,
                "foot stopped at {through:.1} mm past the ball's original centre, expected {minimum:.1}"
            );
        }
        if let Ok(maximum) = std::env::var("MOTION_TRIAL_MAX_CONTACT_LINE_OFFSET_MM") {
            let maximum: f64 = maximum.parse().unwrap();
            let offset = strike.expect("no foot contact")["velocity_line_offset_mm"]
                .as_f64()
                .unwrap();
            assert!(
                offset <= maximum,
                "contact trajectory missed the ball centre by {offset:.1} mm, expected at most {maximum:.1}"
            );
        }
        result
    }
}

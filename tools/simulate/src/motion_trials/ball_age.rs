//! Join original ball observation time to the exact kick service payload.
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use coordinate_systems::Ground;
use motion_inference::{
    inference::InferenceCommand,
    trace::{Event, Record},
};
use ros_z::{
    qos::{QosProfile, QosReliability},
    time::Time,
};
use types::{
    ball_detection::BallPercept,
    ball_position::BallPosition,
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
};

struct Stage {
    kind: &'static str,
    source: Time,
    received: Time,
    seen: Option<Time>,
}
struct Behavior {
    time: Time,
    observed: Option<Time>,
    visual: bool,
    tracking_state: Option<Time>,
    tracking_observation: Option<Time>,
}
#[derive(Default)]
struct Samples {
    stages: Vec<Stage>,
    behavior: BTreeMap<Time, Behavior>,
    motion: Vec<motion::trace::KickInput>,
}
pub(super) struct Audit {
    samples: Arc<Mutex<Samples>>,
    task: tokio::task::JoinHandle<()>,
}
impl Audit {
    pub fn new(runtime: &tokio::runtime::Runtime, io: &crate::robotics::Robotics) -> Self {
        let node = io.node();
        assert!(
            node.clock().is_logical(),
            "ball-age audit requires the simulator clock"
        );
        let trace_qos = QosProfile {
            reliability: QosReliability::BestEffort,
            ..Default::default()
        };
        let (behavior, motion, detections, percepts, visual) = runtime.block_on(async {
            (
                node.subscriber::<behavior_node::kick::InputTrace>(
                    behavior_node::kick::INPUT_TRACE_TOPIC,
                )
                .qos(trace_qos.clone())
                .build()
                .await
                .unwrap(),
                node.subscriber::<motion::trace::KickInput>(motion::trace::KICK_INPUT_TOPIC)
                    .qos(trace_qos)
                    .build()
                    .await
                    .unwrap(),
                node.subscriber::<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>("detected_objects")
                    .build()
                    .await
                    .unwrap(),
                node.subscriber::<Vec<BallPercept>>("ball_filter/ball_percepts")
                    .build()
                    .await
                    .unwrap(),
                node.subscriber::<Option<BallPosition<Ground>>>("visual_kick/ball_position")
                    .build()
                    .await
                    .unwrap(),
            )
        });
        let samples = Arc::new(Mutex::new(Samples::default()));
        let output = samples.clone();
        let clock = node.clock().clone();
        let task = runtime.spawn(async move {
            loop {
                let stage = tokio::select! {
                    value = behavior.recv_with_metadata() => {
                        let value = value.unwrap().message;
                        output.lock().unwrap().behavior.insert(value.frame_time, Behavior {
                            time:value.frame_time, observed:value.observation_time, visual:value.visual,
                            tracking_state:value.tracking_cache_stamp, tracking_observation:value.tracking_observation_time,
                        });
                        continue;
                    }
                    value = motion.recv_with_metadata() => {
                        output.lock().unwrap().motion.push(value.unwrap().message);
                        continue;
                    }
                    value = detections.recv_with_metadata() => {
                        let value=value.unwrap();
                        Stage{kind:"detections",source:value.message.time,seen:Some(value.message.time),received:clock.now()}
                    }
                    value = percepts.recv_with_metadata() => {
                        let value=value.unwrap();
                        Stage{kind:"percepts",source:value.source_time,seen:Some(value.source_time),received:clock.now()}
                    }
                    value = visual.recv_with_metadata() => {
                        let value=value.unwrap();
                        Stage{kind:"visual",source:value.source_time,seen:value.message.map(|ball|ball.last_seen),received:clock.now()}
                    }
                };
                output.lock().unwrap().stages.push(stage);
            }
        });
        Self { samples, task }
    }

    pub fn finish(
        &self,
        records: &[Arc<Record>],
        start: Time,
        end: Time,
        first_contact: Time,
    ) -> serde_json::Value {
        let samples = self.samples.lock().unwrap();
        let mut requests = HashMap::new();
        let mut previous_request = None;
        let mut inputs = 0;
        let mut unmatched = 0;
        let mut rows = Vec::new();
        let ms = |later: Time, earlier: Time| (later.as_nanos() - earlier.as_nanos()) as f64 * 1e-6;
        for record in records {
            match &record.event {
                Event::Request(command) => {
                    if let InferenceCommand::Kick(command) = command {
                        requests.insert(
                            record.request_id.clone(),
                            (
                                record.time,
                                previous_request,
                                serde_json::to_value(command).unwrap(),
                            ),
                        );
                    }
                    previous_request = Some(record.time);
                }
                Event::Input {
                    policy,
                    inference_time,
                    sources,
                    ..
                } if matches!(
                    policy,
                    motion_inference::config::Policy::Kick
                        | motion_inference::config::Policy::SoftKick
                ) && *inference_time >= start
                    && *inference_time <= end =>
                {
                    inputs += 1;
                    let Some((requested, preceding_request, command)) =
                        requests.get(&record.request_id)
                    else {
                        unmatched += 1;
                        continue;
                    };
                    // Motion has one outstanding inference. Match its exact payload,
                    // not the latest behavior command at the later inference time.
                    let Some(motion) = samples
                        .motion
                        .iter()
                        .rev()
                        .find(|motion| motion.prepared_at <= *requested)
                    else {
                        unmatched += 1;
                        continue;
                    };
                    // A dropped record must not reuse an earlier identical command.
                    if preceding_request.is_some_and(|time| motion.prepared_at <= time)
                        || serde_json::to_value(motion.command).unwrap() != *command
                    {
                        unmatched += 1;
                        continue;
                    };
                    let Some(behavior) = samples.behavior.get(&motion.command_time) else {
                        unmatched += 1;
                        continue;
                    };
                    let Some(observed) = behavior.observed else {
                        unmatched += 1;
                        continue;
                    };
                    let stage_age = |kind, source| {
                        samples
                            .stages
                            .iter()
                            .find(|s| s.kind == kind && s.source == source)
                            .map(|s| ms(s.received, s.source))
                    };
                    let visual = samples.stages.iter().rev().find(|s| {
                        s.kind == "visual" && s.seen == Some(observed) && s.source <= behavior.time
                    });
                    let detection = samples
                        .stages
                        .iter()
                        .find(|s| s.kind == "detections" && s.source == observed);
                    let percept = samples
                        .stages
                        .iter()
                        .find(|s| s.kind == "percepts" && s.source == observed);
                    rows.push(serde_json::json!({
                        "inference_ns":inference_time.as_nanos(),"observation_ns":observed.as_nanos(),"behavior_ns":behavior.time.as_nanos(),"motion_ns":motion.prepared_at.as_nanos(),
                        "source":if behavior.visual {"visual"} else {"tracked_or_held"},
                        "before_contact":*inference_time<first_contact,
                        "observation_age_ms":ms(*inference_time,observed),
                        "age_at_behavior_ms":ms(behavior.time,observed),
                        "behavior_to_motion_ms":ms(motion.prepared_at,behavior.time),
                        "motion_to_request_ms":ms(*requested,motion.prepared_at),
                        "request_to_input_ms":ms(*inference_time,*requested),
                        "sensor_age_ms":sources.as_ref().map(|s|ms(*inference_time,s.sensor.source_time)),
                        "detection_delivery_age_ms":stage_age("detections",observed),
                        "percept_delivery_age_ms":stage_age("percepts",observed),
                        "detection_to_percept_ms":detection.zip(percept).map(|(d,p)|ms(p.received,d.received)),
                        "visual_source_stamp_age_ms":visual.map(|s|ms(s.source,observed)),
                        "visual_delivery_age_ms":visual.map(|s|ms(s.received,observed)),
                        "tracked_state_age_ms":behavior.tracking_state.map(|t|ms(*inference_time,t)),
                        "tracked_last_observation_age_ms":behavior.tracking_observation.map(|t|ms(*inference_time,t)),
                    }));
                }
                _ => {}
            }
        }
        let stats = |key: &str, before_only: bool| {
            let mut values: Vec<_> = rows
                .iter()
                .filter(|r| !before_only || r["before_contact"] == true)
                .filter_map(|r| r[key].as_f64())
                .collect();
            values.sort_by(f64::total_cmp);
            serde_json::json!({"count":values.len(),"min":values.first(),"median":values.get(values.len()/2),"p95":values.get(values.len()*95/100),"max":values.last()})
        };
        let keys = [
            "observation_age_ms",
            "age_at_behavior_ms",
            "behavior_to_motion_ms",
            "motion_to_request_ms",
            "request_to_input_ms",
            "sensor_age_ms",
            "detection_delivery_age_ms",
            "percept_delivery_age_ms",
            "detection_to_percept_ms",
            "visual_source_stamp_age_ms",
            "visual_delivery_age_ms",
            "tracked_state_age_ms",
            "tracked_last_observation_age_ms",
        ];
        let statistics: serde_json::Map<_, _> = keys
            .into_iter()
            .map(|key| (key.to_owned(), stats(key, true)))
            .collect();
        let result = serde_json::json!({"inputs":inputs,"matched":rows.len(),"unmatched":unmatched,"visual":rows.iter().filter(|r|r["source"]=="visual").count(),"before_contact":rows.iter().filter(|r|r["before_contact"]==true).count(),"statistics":statistics,"all_observation_age_ms":stats("observation_age_ms",false)});
        eprintln!("MOTION_BALL_AGE {result}");
        if let Ok(path) = std::env::var("MOTION_TRIAL_BALL_AGE_OUTPUT") {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&serde_json::json!({"summary":result,"samples":rows}))
                    .unwrap(),
            )
            .unwrap();
        }
        assert!(
            rows.iter().filter(|r| r["before_contact"] == true).count() >= 10,
            "too few matched pre-contact ball-age samples: {result}"
        );
        assert!(
            unmatched * 20 <= inputs,
            "ball-age trace lost more than 5% of kick inputs: {result}"
        );
        assert!(
            rows.iter()
                .all(|r| r["observation_age_ms"].as_f64().unwrap() >= 0.0),
            "future original ball timestamp"
        );
        if let Ok(maximum) = std::env::var("MOTION_TRIAL_MAX_BALL_AGE_MS") {
            let maximum: f64 = maximum.parse().unwrap();
            let p95 = result["statistics"]["observation_age_ms"]["p95"]
                .as_f64()
                .unwrap();
            assert!(
                p95 <= maximum,
                "ball observation p95 age {p95:.1} ms exceeds {maximum:.1} ms"
            );
        }
        result
    }
}
impl Drop for Audit {
    fn drop(&mut self) {
        self.task.abort();
    }
}

//! Replays the inputs of the humanoid and stationary object filters from a simulator recording.
//!
//! The simulator's synthetic detections do not depend on the filters, so replaying a recording
//! through changed filter code or parameters is equivalent to rerunning the match, but
//! deterministic and much faster. Outputs are written as JSON lines next to the recorded ground
//! truth, one file per topic.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use clap::Parser;
use color_eyre::{
    Result,
    eyre::{Context, ContextCompat, bail},
};
use coordinate_systems::{Field, Ground, Odometry};
use hsl_network_messages::PlayerNumber;
use linear_algebra::{Isometry2, Point2, Pose2, point};
use object_tracking::multi_bernoulli::{ScanReport, Track};
use projection::{Projection, camera_matrix::CameraMatrix};
use ros_z::{SerdeCdrCodec, message::WireDecoder, time::Time};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use types::{
    field_dimensions::{FieldDimensions, Half, Side},
    game_controller_state::GameControllerState,
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    players::Players,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    world_state::PlayerState,
};

#[derive(Parser)]
struct Arguments {
    /// Recording directory of one robot, containing `recording-segments.json`.
    recording: PathBuf,
    /// Directory for the replayed outputs; omit with `--list-topics`.
    output: Option<PathBuf>,
    /// Parameter root containing `base/<node>.json5`.
    #[arg(long, default_value = "etc/parameters")]
    parameters: PathBuf,
    /// JSON merge patch applied to the humanoid filter parameters.
    #[arg(long)]
    humanoid_patch: Option<String>,
    /// JSON merge patch applied to the stationary object filter parameters.
    #[arg(long)]
    stationary_patch: Option<String>,
    /// Prefix of the output file names, matched by the analysis scripts.
    #[arg(long, default_value = "replay")]
    target: String,
    /// Delay between the stamp of an input and the time the node processes it.
    #[arg(long, default_value_t = 30)]
    processing_delay_ms: u64,
    /// Print the recorded topics with message counts and exit.
    #[arg(long)]
    list_topics: bool,
    /// Write a copy of the recording that keeps only the topics the replay reads, into this
    /// directory, instead of replaying.
    #[arg(long)]
    extract: Option<PathBuf>,
    /// Shift of odometry sample times relative to images, in milliseconds, to test latencies.
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    odometry_offset_ms: i64,
    /// Extra delay of localization estimates, in milliseconds, to test the pairing with odometry
    /// at the latencies of a robot.
    #[arg(long, default_value_t = 0)]
    localization_delay_ms: u64,
    /// Location whose field dimensions the recording was made on.
    #[arg(long, default_value = "hsl_middle")]
    location: String,
    /// Also write ground residuals of robot detections against the ground truth.
    #[arg(long)]
    diagnostics: bool,
}

/// Ground truth published by the simulator; mirrors `simulate::behavior_inputs::HumanoidTruth`.
#[derive(Deserialize, Serialize)]
struct HumanoidTruth {
    team: TruthTeam,
    player_number: u8,
    is_teammate: bool,
    position_in_ground: Point2<Ground>,
    position_in_field: Point2<Field>,
}

#[derive(Deserialize, Serialize)]
enum TruthTeam {
    Hulks,
    Opponents,
}

/// Leading field of `ros_z_streams::Announcement`; the future map keys data by this time.
#[derive(Deserialize)]
struct Announcement {
    time: Time,
}

struct Sample<T> {
    /// Simulation time at which the recorder received the message.
    receipt: Time,
    /// Source time of the publisher.
    stamp: Time,
    value: T,
}

/// Messages of one topic ordered by receipt time.
struct Stream<T>(Vec<Sample<T>>);

impl<T> Stream<T> {
    /// Sample whose stamp is closest to `time`, regardless of when it was received.
    fn nearest_sample(&self, time: Time) -> Option<&Sample<T>> {
        let index = self.0.partition_point(|sample| sample.stamp < time);
        [index.checked_sub(1), Some(index)]
            .into_iter()
            .flatten()
            .filter_map(|index| self.0.get(index))
            .min_by_key(|sample| sample.stamp.abs_diff(time))
    }

    fn received_until(&self, now: Time) -> &[Sample<T>] {
        &self.0[..self.0.partition_point(|sample| sample.receipt <= now)]
    }

    fn latest(&self, now: Time) -> Option<&T> {
        self.received_until(now).last().map(|sample| &sample.value)
    }

    /// Received message whose stamp is closest to `time`, assuming stamps increase.
    fn nearest(&self, now: Time, time: Time) -> Option<&T> {
        let received = self.received_until(now);
        let index = received.partition_point(|sample| sample.stamp < time);
        [index.checked_sub(1), Some(index)]
            .into_iter()
            .flatten()
            .filter_map(|index| received.get(index))
            .min_by_key(|sample| sample.stamp.abs_diff(time))
            .map(|sample| &sample.value)
    }
}

struct Recording {
    odometry_announcements: Stream<Announcement>,
    detected_objects_announcements: Stream<Announcement>,
    odometry: Stream<Pose2<Odometry>>,
    detected_objects: Stream<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>,
    camera_matrix: Stream<TimeWrapper<CameraMatrix>>,
    ground_to_field: Stream<Isometry2<Ground, Field>>,
    player_number: Stream<PlayerNumber>,
    primary_state: Stream<PrimaryState>,
    game_controller_state: Stream<Option<GameControllerState>>,
    player_states: Stream<Players<Option<TimeWrapper<PlayerState>>>>,
    humanoid_truth: Stream<Vec<HumanoidTruth>>,
}

/// Raw messages of all segments by topic.
type RawMessages = BTreeMap<String, Vec<(u64, u64, Vec<u8>)>>;

/// Topics the replay decodes.
const TOPICS: &[&str] = &[
    "inputs/odometry/announce",
    "detected_objects/announce",
    "inputs/odometry",
    "detected_objects",
    "camera_matrix",
    "ground_to_field",
    "player_number",
    "primary_state",
    "game_controller_state",
    "player_states",
    "ground_truth/humanoids",
];

/// Copies the messages of the replayed topics from all segments into a single segment.
fn extract(directory: &Path, output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let mut writer =
        mcap::Writer::new(BufWriter::new(File::create(output.join("recording.mcap"))?))?;
    for file in segment_files(directory)? {
        let file_name = file;
        let file = File::open(directory.join(&file_name)).wrap_err_with(|| file_name.clone())?;
        // SAFETY: recordings are finalized and not modified while extracting.
        let mapped = unsafe { memmap2::Mmap::map(&file)? };
        // Schemas of one name may differ between segments, so channels are registered per segment.
        let mut channels = BTreeMap::new();
        for message in mcap::MessageStream::new(&mapped)? {
            let Some(message) = complete(message, &file_name)? else {
                break;
            };
            let channel = &message.channel;
            if !TOPICS.contains(&channel.topic.as_str()) {
                continue;
            }
            let channel_id = match channels.get(&channel.id) {
                Some(id) => *id,
                None => {
                    let schema_id = match &channel.schema {
                        Some(schema) => {
                            writer.add_schema(&schema.name, &schema.encoding, &schema.data)?
                        }
                        None => 0,
                    };
                    let id = writer.add_channel(
                        schema_id,
                        &channel.topic,
                        &channel.message_encoding,
                        &channel.metadata,
                    )?;
                    channels.insert(channel.id, id);
                    id
                }
            };
            writer.write_to_known_channel(
                &mcap::records::MessageHeader {
                    channel_id,
                    sequence: message.sequence,
                    log_time: message.log_time,
                    publish_time: message.publish_time,
                },
                &message.data,
            )?;
        }
    }
    writer.finish()?;
    std::fs::write(
        output.join("recording-segments.json"),
        json!({ "files": ["recording.mcap"], "finalized": true }).to_string(),
    )?;
    Ok(())
}

/// Segment files of a recording; robot logs without a segment list have a single file.
/// A message of a recording, or `None` at the end of a recording that was cut off, e.g. when the
/// robot was switched off while writing.
fn complete<'a>(
    message: mcap::McapResult<mcap::Message<'a>>,
    file: &str,
) -> Result<Option<mcap::Message<'a>>> {
    match message {
        Ok(message) => Ok(Some(message)),
        Err(error @ (mcap::McapError::UnexpectedEof | mcap::McapError::UnexpectedEoc)) => {
            eprintln!("{file}: recording ends early ({error}), using the messages before");
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

fn segment_files(directory: &Path) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Segments {
        files: Vec<String>,
    }
    let list = directory.join("recording-segments.json");
    if !list.exists() && directory.join("recording.mcap").exists() {
        return Ok(vec!["recording.mcap".to_owned()]);
    }
    let segments: Segments = serde_json::from_reader(
        File::open(list).wrap_err("recording directory lacks recording-segments.json")?,
    )?;
    Ok(segments.files)
}

/// Reads the messages of all segments, keeping only topics accepted by `keep`.
fn read_raw(directory: &Path, keep: impl Fn(&str) -> bool) -> Result<RawMessages> {
    let mut messages = RawMessages::new();
    for file in segment_files(directory)? {
        let file_name = file;
        let file = File::open(directory.join(&file_name)).wrap_err_with(|| file_name.clone())?;
        // SAFETY: recordings are finalized and not modified while replaying.
        let mapped = unsafe { memmap2::Mmap::map(&file)? };
        for message in mcap::MessageStream::new(&mapped)? {
            let Some(message) = complete(message, &file_name)? else {
                break;
            };
            if !keep(&message.channel.topic) {
                continue;
            }
            messages
                .entry(message.channel.topic.clone())
                .or_default()
                .push((
                    message.log_time,
                    message.publish_time,
                    message.data.into_owned(),
                ));
        }
    }
    Ok(messages)
}

fn decode<T: DeserializeOwned>(messages: &RawMessages, name: &str) -> Result<Stream<T>> {
    let Some(raw) = messages.get(name) else {
        bail!("topic {name} is not recorded")
    };
    let mut samples = raw
        .iter()
        .map(|(log_time, publish_time, data)| {
            Ok(Sample {
                receipt: Time::from_nanos(*log_time as i64),
                stamp: Time::from_nanos(*publish_time as i64),
                value: SerdeCdrCodec::<T>::deserialize(data)
                    .wrap_err_with(|| format!("failed to decode {name}"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    samples.sort_by_key(|sample| sample.receipt);
    Ok(Stream(samples))
}

impl<T> Stream<T> {
    /// Announced time of each sample, matched by the stamp of the announcing publication.
    fn announced_times<'a>(
        &'a self,
        announcements: &Stream<Announcement>,
    ) -> impl Iterator<Item = (Time, &'a T)> {
        let announced: BTreeMap<Time, Time> = announcements
            .0
            .iter()
            .map(|sample| (sample.stamp, sample.value.time))
            .collect();
        self.0.iter().map(move |sample| {
            (
                announced
                    .get(&sample.stamp)
                    .copied()
                    .unwrap_or(sample.stamp),
                &sample.value,
            )
        })
    }
}

impl Recording {
    /// Odometry, detections and camera matrices are required; robot logs may lack the others,
    /// e.g. without game controller, teammates or ground truth.
    fn read(messages: &RawMessages) -> Result<Self> {
        Ok(Self {
            odometry_announcements: decode_optional(messages, "inputs/odometry/announce")?,
            detected_objects_announcements: decode_optional(messages, "detected_objects/announce")?,
            odometry: decode(messages, "inputs/odometry")?,
            detected_objects: decode(messages, "detected_objects")?,
            camera_matrix: decode(messages, "camera_matrix")?,
            ground_to_field: decode_optional(messages, "ground_to_field")?,
            player_number: decode_optional(messages, "player_number")?,
            primary_state: decode_optional(messages, "primary_state")?,
            game_controller_state: decode_optional(messages, "game_controller_state")?,
            player_states: decode_optional(messages, "player_states")?,
            humanoid_truth: decode_optional(messages, "ground_truth/humanoids")?,
        })
    }
}

fn decode_optional<T: DeserializeOwned>(messages: &RawMessages, name: &str) -> Result<Stream<T>> {
    if messages.contains_key(name) {
        decode(messages, name)
    } else {
        Ok(Stream(Vec::new()))
    }
}

/// Inputs as the node's caches hold them at `now`.
struct Inputs<'a> {
    recording: &'a Recording,
    now: Time,
}

impl humanoid_filter::Inputs for Inputs<'_> {
    fn primary_state(&self) -> Option<PrimaryState> {
        self.recording.primary_state.latest(self.now).copied()
    }

    fn game_controller_state(&self) -> Option<GameControllerState> {
        self.recording
            .game_controller_state
            .latest(self.now)
            .cloned()
            .flatten()
    }

    fn player_number(&self) -> Option<PlayerNumber> {
        self.recording.player_number.latest(self.now).copied()
    }

    fn camera_matrix_nearest(&self, time: Time) -> Option<TimeWrapper<CameraMatrix>> {
        self.recording
            .camera_matrix
            .nearest(self.now, time)
            .cloned()
    }
}

impl stationary_object_filter::Inputs for Inputs<'_> {
    fn primary_state(&self) -> Option<PrimaryState> {
        humanoid_filter::Inputs::primary_state(self)
    }

    fn camera_matrix_nearest(&self, time: Time) -> Option<TimeWrapper<CameraMatrix>> {
        humanoid_filter::Inputs::camera_matrix_nearest(self, time)
    }
}

fn merge_patch(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(target), Value::Object(patch)) => {
            for (key, value) in patch {
                if value.is_null() {
                    target.remove(key);
                } else {
                    merge_patch(target.entry(key.clone()).or_insert(Value::Null), value);
                }
            }
        }
        (target, patch) => *target = patch.clone(),
    }
}

fn load_parameters<T: DeserializeOwned>(root: &Path, node: &str, patch: Option<&str>) -> Result<T> {
    let path = root.join("base").join(format!("{node}.json5"));
    let mut value: Value = json5::from_str(
        &std::fs::read_to_string(&path).wrap_err_with(|| path.display().to_string())?,
    )?;
    if let Some(patch) = patch {
        merge_patch(&mut value, &serde_json::from_str(patch)?);
    }
    serde_json::from_value(value).wrap_err_with(|| format!("invalid {node} parameters"))
}

fn load_field_dimensions(root: &Path, location: &str) -> Result<FieldDimensions> {
    let mut value = Value::Null;
    for path in [
        root.join("base/global.json5"),
        root.join("location").join(location).join("global.json5"),
    ] {
        let layer: Value = json5::from_str(
            &std::fs::read_to_string(&path).wrap_err_with(|| path.display().to_string())?,
        )?;
        if let Some(dimensions) = layer.get("field_dimensions") {
            merge_patch(&mut value, dimensions);
        }
    }
    serde_json::from_value(value).wrap_err("invalid field dimensions")
}

/// Goal posts in ground coordinates for each recorded field pose.
///
/// Recordings made before the simulator's away-side fix carry mirrored goal post truth, so the
/// truth is derived from the field pose, which is exact in the filtering profile.
fn goal_post_truth(
    ground_to_field: &Stream<Isometry2<Ground, Field>>,
    field: &FieldDimensions,
) -> Stream<Vec<Obstacle>> {
    Stream(
        ground_to_field
            .0
            .iter()
            .map(|sample| {
                let field_to_ground = sample.value.inverse();
                let posts = [Half::Own, Half::Opponent]
                    .into_iter()
                    .flat_map(|half| {
                        [Side::Left, Side::Right].map(|side| {
                            Obstacle::goal_post(
                                field_to_ground * field.goal_post(half, side),
                                field.goal_post_diameter / 2.0,
                            )
                        })
                    })
                    .collect();
                Sample {
                    receipt: sample.receipt,
                    stamp: sample.stamp,
                    value: posts,
                }
            })
            .collect(),
    )
}

/// Output file in the row format of the simulator's exports.
struct Output(BufWriter<File>);

impl Output {
    fn create(directory: &Path, target: &str, topic: &str) -> Result<Self> {
        let name = format!("{target}__{}.jsonl", topic.replace('/', "__"));
        Ok(Self(BufWriter::new(File::create(directory.join(name))?)))
    }

    fn write(&mut self, time: Time, value: &impl Serialize) -> Result<()> {
        serde_json::to_writer(
            &mut self.0,
            &json!({
                "selected_time_ns": time.as_nanos().to_string(),
                "value_status": "present",
                "value": value,
            }),
        )?;
        self.0.write_all(b"\n")?;
        Ok(())
    }

    fn write_stream<T: Serialize>(&mut self, stream: &Stream<T>) -> Result<()> {
        for sample in &stream.0 {
            self.write(sample.receipt, &sample.value)?;
        }
        Ok(())
    }
}

/// Ground positions of robot detections, projected like the filter does, with the distance to the
/// nearest true robot and the stamp offsets of the camera matrix and truth that were used.
fn write_detection_residuals(recording: &Recording, output: &Path, target: &str) -> Result<()> {
    let mut residuals = Output::create(output, target, "diagnostics/robot_detections")?;
    let everything = Time::from_nanos(i64::MAX);
    for (time, detected_objects) in recording
        .detected_objects
        .announced_times(&recording.detected_objects_announcements)
    {
        let Some(camera_matrix) = recording.camera_matrix.nearest(everything, time) else {
            continue;
        };
        let truth = recording.humanoid_truth.nearest_sample(time);
        let camera_pose = camera_matrix.inner.ground_to_camera.inverse().as_pose();
        let detections: Vec<_> = detected_objects
            .inner
            .iter()
            .filter(|object| object.label == RobocupObjectLabel::Robot)
            .filter_map(|object| {
                let area = object.bounding_box.area;
                let bottom = point![(area.min.x() + area.max.x()) / 2.0, area.max.y()];
                let ground = camera_matrix.inner.pixel_to_ground(bottom).ok()?;
                let nearest = truth.map(|truth| {
                    truth
                        .value
                        .iter()
                        .map(|humanoid| (humanoid.position_in_ground - ground).norm())
                        .fold(f32::INFINITY, f32::min)
                });
                Some(json!({
                    "position_in_ground": ground,
                    "confidence": object.bounding_box.confidence,
                    "nearest_truth_distance": nearest,
                    "box_height": area.max.y() - area.min.y(),
                    "bottom_y": area.max.y(),
                }))
            })
            .collect();
        let ground_to_field = recording
            .ground_to_field
            .nearest_sample(time)
            .map(|sample| sample.value);
        // Other objects with the ground position of their box bottom, as goal posts are anchored.
        let others: Vec<_> = detected_objects
            .inner
            .iter()
            .filter(|object| object.label != RobocupObjectLabel::Robot)
            .map(|object| {
                let area = object.bounding_box.area;
                let bottom = point![(area.min.x() + area.max.x()) / 2.0, area.max.y()];
                let ground = camera_matrix.inner.pixel_to_ground(bottom).ok();
                let ray = camera_matrix.inner.ground_to_camera.inverse()
                    * camera_matrix.inner.bearing(bottom);
                json!({
                    "label": object.label,
                    "azimuth": ray.y().atan2(ray.x()),
                    "depression": (-ray.z()).atan2(ray.x().hypot(ray.y())),
                    "confidence": object.bounding_box.confidence,
                    "box": [area.min.x(), area.min.y(), area.max.x(), area.max.y()],
                    "bottom_in_ground": ground,
                    "bottom_in_field": ground.zip(ground_to_field).map(|(ground, pose)| pose * ground),
                })
            })
            .collect();
        residuals.write(
            time,
            &json!({
                "camera_matrix_offset_ms": signed_milliseconds(camera_matrix.time, time),
                "truth_offset_ms": truth.map(|truth| signed_milliseconds(truth.stamp, time)),
                "detections": detections,
                "other_objects": others,
                "camera_in_ground": camera_pose.position().xy(),
                "camera_height": camera_pose.position().z(),
                "image_size": camera_matrix.inner.image_size,
                "field_of_view": camera_matrix.inner.field_of_view,
            }),
        )?;
    }
    Ok(())
}

/// New stationary tracks that survived their first scan, with the closest track of the same class
/// that existed before the scan, to tell duplicates from new objects.
fn write_stationary_births<A>(
    output: &mut Output,
    time: Time,
    report: &ScanReport,
    previous: &[Track<2, A>],
    current: &[Track<2, A>],
    class_of: impl Fn(&A) -> usize,
    detection_probability_of: impl Fn(&A) -> f32,
) -> Result<()> {
    let births: Vec<_> = report
        .births
        .iter()
        .filter_map(|id| current.iter().find(|track| track.id == *id))
        .filter(|track| track.existence >= 0.1)
        .map(|birth| {
            let class = class_of(&birth.attributes);
            let estimate = birth.estimate();
            let nearest = previous
                .iter()
                .filter(|track| class_of(&track.attributes) == class)
                .map(|track| {
                    let other = track.estimate();
                    let difference = estimate.mean - other.mean;
                    let covariance = estimate.covariance + other.covariance;
                    let mahalanobis = covariance
                        .try_inverse()
                        .map(|inverse| (difference.transpose() * inverse * difference)[0]);
                    (difference.norm(), mahalanobis, track)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(distance, mahalanobis, track)| {
                    json!({
                        "distance": distance,
                        "squared_mahalanobis": mahalanobis,
                        "existence": track.existence,
                        "detection_probability": detection_probability_of(&track.attributes),
                        "position_standard_deviation": track.estimate().covariance.trace().sqrt(),
                    })
                });
            json!({
                "class": class,
                "existence": birth.existence,
                "position_standard_deviation": estimate.covariance.trace().sqrt(),
                "nearest_same_class": nearest,
            })
        })
        .collect();
    if !births.is_empty() {
        output.write(time, &births)?;
    }
    Ok(())
}

fn signed_milliseconds(time: Time, reference: Time) -> f64 {
    let magnitude = time.abs_diff(reference).as_secs_f64() * 1e3;
    if time < reference {
        -magnitude
    } else {
        magnitude
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let arguments = Arguments::parse();
    if let Some(output) = &arguments.extract {
        return extract(&arguments.recording, output);
    }
    let messages = read_raw(&arguments.recording, |topic| {
        arguments.list_topics || TOPICS.contains(&topic)
    })?;
    if arguments.list_topics {
        for (topic, samples) in &messages {
            println!("{:8} {topic}", samples.len());
        }
        return Ok(());
    }
    let output = arguments
        .output
        .wrap_err("an output directory is required")?;
    std::fs::create_dir_all(&output)?;
    let humanoid_parameters: humanoid_filter::Parameters = load_parameters(
        &arguments.parameters,
        "humanoid_filter",
        arguments.humanoid_patch.as_deref(),
    )?;
    let stationary_parameters: stationary_object_filter::Parameters = load_parameters(
        &arguments.parameters,
        "stationary_object_filter",
        arguments.stationary_patch.as_deref(),
    )?;
    let recording = Recording::read(&messages)?;
    drop(messages);

    let mut items = BTreeMap::<Time, (Option<Pose2<Odometry>>, Option<_>)>::new();
    for (time, odometry) in recording
        .odometry
        .announced_times(&recording.odometry_announcements)
    {
        let time = Time::from_nanos(time.as_nanos() + arguments.odometry_offset_ms * 1_000_000);
        items.entry(time).or_default().0 = Some(*odometry);
    }
    for (time, detected_objects) in recording
        .detected_objects
        .announced_times(&recording.detected_objects_announcements)
    {
        items.entry(time).or_default().1 = Some(detected_objects.clone());
    }

    let target = &arguments.target;
    if arguments.diagnostics {
        write_detection_residuals(&recording, &output, target)?;
    }
    let mut humanoid_tracks = Output::create(&output, target, "humanoid_filter/tracks")?;
    let mut humanoid_reports = Output::create(&output, target, "humanoid_filter/scan_report")?;
    let mut stationary_tracks = Output::create(&output, target, "stationary_object_filter/tracks")?;
    let mut stationary_reports =
        Output::create(&output, target, "stationary_object_filter/scan_report")?;
    let mut births = arguments
        .diagnostics
        .then(|| Output::create(&output, target, "diagnostics/stationary_births"))
        .transpose()?;

    let delay = Duration::from_millis(arguments.processing_delay_ms);
    let mut humanoid = humanoid_filter::Processor::default();
    let mut stationary = stationary_object_filter::Processor::default();
    let mut player_states = recording.player_states.0.iter().peekable();
    let mut ground_to_field = recording.ground_to_field.0.iter().peekable();
    for (time, item) in items {
        let now = time.saturating_add(delay);
        let inputs = Inputs {
            recording: &recording,
            now,
        };
        // Like the nodes, receive localization with the time of its image as soon as it arrives.
        let localization_delay = Duration::from_millis(arguments.localization_delay_ms);
        while let Some(sample) = ground_to_field
            .next_if(|sample| sample.receipt.saturating_add(localization_delay) <= now)
        {
            humanoid.receive_ground_to_field(sample.stamp, sample.value);
            stationary.receive_ground_to_field(sample.stamp, sample.value);
        }
        while let Some(sample) = player_states.next_if(|sample| sample.receipt <= now) {
            humanoid.receive_player_states(
                &sample.value,
                recording.player_number.latest(sample.receipt).copied(),
            );
        }
        if let Some(outputs) =
            humanoid.process([(time, item.clone())], &inputs, &humanoid_parameters)
        {
            for report in &outputs.reports {
                humanoid_reports.write(now, report)?;
            }
            if let Some(tracks) = &outputs.tracks {
                humanoid_tracks.write(now, tracks)?;
            }
        }
        let previous_stationary = births
            .is_some()
            .then(|| stationary.filter().tracks.tracks.clone());
        if let Some(outputs) = stationary.process([(time, item)], &inputs, &stationary_parameters) {
            for report in &outputs.reports {
                stationary_reports.write(now, report)?;
                if let (Some(births), Some(previous)) = (&mut births, &previous_stationary) {
                    write_stationary_births(
                        births,
                        now,
                        report,
                        previous,
                        &stationary.filter().tracks.tracks,
                        |attributes| attributes.class.most_likely(),
                        |attributes| attributes.detection_probability,
                    )?;
                }
            }
            if let Some(tracks) = &outputs.tracks {
                stationary_tracks.write(now, tracks)?;
            }
        }
    }

    Output::create(&output, target, "ground_truth/humanoids")?
        .write_stream(&recording.humanoid_truth)?;
    Output::create(
        &output,
        target,
        "ground_truth/stationary_object_filter/obstacles",
    )?
    .write_stream(&goal_post_truth(
        &recording.ground_to_field,
        &load_field_dimensions(&arguments.parameters, &arguments.location)?,
    ))?;
    Output::create(&output, target, "game_controller_state")?
        .write_stream(&recording.game_controller_state)?;
    Ok(())
}

use std::{path::PathBuf, str::FromStr, time::Duration};

use clap::{Parser, Subcommand};
use color_eyre::{
    Result,
    eyre::{Context, ContextCompat},
};
use detection::ExecutionProvider;
use detection_replay::{DetectionThresholds, ModelRunConfig, ModelRunState, Recording, run_model};
use tracing_subscriber::EnvFilter;

mod app;
mod snapshot;
mod timeline;

#[derive(Debug, Parser)]
#[command(about = "Prerender and compare detection models on a ROS-Z MCAP recording")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build the random-access frame proxy and import recorded detections.
    Index {
        recording: PathBuf,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[command(flatten)]
        frames: FrameRangeArguments,
    },
    /// Run one or more models through the actual ROS-Z detection node.
    Prerender {
        recording: PathBuf,
        #[arg(long, value_name = "LABEL=MODEL", required = true)]
        model: Vec<ModelArgument>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 0.05)]
        confidence: f32,
        #[arg(long, default_value_t = 0.4)]
        iou: f32,
        /// ONNX Runtime execution provider.
        #[arg(long, value_enum, default_value = "cuda")]
        provider: ProviderArgument,
        /// GPU device index used by TensorRT or CUDA.
        #[arg(long, default_value_t = 0, value_name = "INDEX")]
        gpu: u32,
        /// Stop after this many frames in the selected range. Useful for provider smoke tests.
        #[arg(long)]
        limit: Option<usize>,
        #[command(flatten)]
        frames: FrameRangeArguments,
    },
    /// Open the synchronized multi-model viewer.
    View {
        recording: PathBuf,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[command(flatten)]
        frames: FrameRangeArguments,
    },
    /// Decode every cached prediction in the range and verify timestamps and Hydra heads.
    Verify {
        recording: PathBuf,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Verify only the run with this cache key or label.
        #[arg(long)]
        run: Option<String>,
        #[command(flatten)]
        frames: FrameRangeArguments,
    },
    /// Export one cached frame with the same Hydra overlay conventions as the viewer.
    Snapshot {
        recording: PathBuf,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        run: String,
        #[arg(long)]
        frame: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        details: bool,
        #[arg(long, default_value_t = 0.25)]
        object_confidence: f32,
        #[arg(long, default_value_t = 0.5)]
        anchor_presence: f32,
        #[arg(long, default_value_t = 0.39)]
        line_confidence: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ProviderArgument {
    Automatic,
    #[value(name = "tensorrt")]
    TensorRt,
    Cuda,
    Cpu,
}

impl ProviderArgument {
    fn execution_provider(self) -> ExecutionProvider {
        match self {
            Self::Automatic => ExecutionProvider::Automatic,
            Self::TensorRt => ExecutionProvider::TensorRt,
            Self::Cuda => ExecutionProvider::Cuda,
            Self::Cpu => ExecutionProvider::Cpu,
        }
    }

    fn uses_nvidia(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

#[derive(Clone, Copy, Debug, clap::Args)]
struct FrameRangeArguments {
    /// First frame ID to process or display.
    #[arg(long, default_value_t = 0)]
    start_frame: usize,
    /// Last frame ID to process or display, inclusive.
    #[arg(long)]
    end_frame: Option<usize>,
}

impl FrameRangeArguments {
    fn resolve(self, frame_count: usize) -> Result<(usize, usize)> {
        let end = self
            .end_frame
            .unwrap_or_else(|| frame_count.saturating_sub(1));
        if self.start_frame > end {
            color_eyre::eyre::bail!(
                "start frame {} is greater than end frame {end}",
                self.start_frame
            );
        }
        if end >= frame_count {
            color_eyre::eyre::bail!("end frame {end} is out of range for {frame_count} frames");
        }
        Ok((self.start_frame, end))
    }
}

#[derive(Clone, Debug)]
struct ModelArgument {
    label: String,
    path: PathBuf,
}

impl FromStr for ModelArgument {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (label, path) = value
            .split_once('=')
            .ok_or_else(|| "model must use LABEL=MODEL syntax".to_string())?;
        if label.trim().is_empty() || path.trim().is_empty() {
            return Err("model label and path must not be empty".to_string());
        }
        Ok(Self {
            label: label.trim().to_string(),
            path: PathBuf::from(path.trim()),
        })
    }
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    configure_gpu_visibility(&arguments.command);
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    match arguments.command {
        Command::Index {
            recording,
            cache_dir,
            frames,
        } => {
            let recording_path = resolve_recording(recording);
            let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&recording_path));
            let recording = Recording::open(&recording_path, &cache_dir)?;
            report_recording_warning(&recording);
            let (start_frame, end_frame) = frames.resolve(recording.frame_count())?;
            let recorded_predictions = recording
                .load_runs()?
                .into_iter()
                .find(|run| run.key == "recorded")
                .map(|run| run.available_count(start_frame..=end_frame))
                .unwrap_or_default();
            println!(
                "indexed frames {}..={} ({} of {}) and {} aligned recorded prediction frames from {} into {}",
                start_frame,
                end_frame,
                end_frame - start_frame + 1,
                recording.frame_count(),
                recorded_predictions,
                recording_path.display(),
                cache_dir.display()
            );
        }
        Command::Prerender {
            recording,
            model,
            cache_dir,
            confidence,
            iou,
            provider,
            gpu,
            limit,
            frames,
        } => {
            let recording_path = resolve_recording(recording);
            let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&recording_path));
            let recording = Recording::open(&recording_path, &cache_dir)?;
            report_recording_warning(&recording);
            let (start_frame, end_frame) = frames.resolve(recording.frame_count())?;
            let thresholds = DetectionThresholds {
                minimum_candidate_confidence: confidence,
                maximum_intersection_over_union: iou,
            }
            .validate()?;
            let ort_result = ort::init()
                .commit()
                .wrap_err("failed to initialize ONNX Runtime");
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .wrap_err("failed to build replay runtime")?;

            let prerender_result = ort_result.and_then(|_| {
                for model in model {
                    let mut config = ModelRunConfig::new(model.label.clone(), model.path);
                    config.thresholds = thresholds;
                    config.provider = provider.execution_provider();
                    config.gpu_device_id = gpu;
                    config.frame_limit = limit;
                    config.start_frame = start_frame;
                    config.end_frame = Some(end_frame);
                    config.output_timeout = Duration::from_secs(120);
                    let mut last_reported = usize::MAX;
                    let manifest = runtime.block_on(run_model(&recording, config, |progress| {
                        if progress.completed_frames == progress.target_frames
                            || progress.completed_frames / 100 != last_reported / 100
                        {
                            println!(
                                "{}: {}/{} frames",
                                progress.label, progress.completed_frames, progress.target_frames
                            );
                            last_reported = progress.completed_frames;
                        }
                    }))?;
                    println!(
                        "{}: {:?} ({}/{} frames)",
                        manifest.label,
                        manifest.state,
                        manifest.completed_frame_count,
                        end_frame - start_frame + 1
                    );
                }
                Ok(())
            });
            runtime.shutdown_timeout(Duration::from_secs(2));
            prerender_result?;
        }
        Command::View {
            recording,
            cache_dir,
            frames,
        } => {
            let recording_path = resolve_recording(recording);
            let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&recording_path));
            let recording = Recording::open(recording_path, cache_dir)?;
            let (start_frame, end_frame) = frames.resolve(recording.frame_count())?;
            app::run(recording, start_frame, end_frame)?;
        }
        Command::Verify {
            recording,
            cache_dir,
            run,
            frames,
        } => {
            let recording_path = resolve_recording(recording);
            let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&recording_path));
            let recording = Recording::open(recording_path, cache_dir)?;
            report_recording_warning(&recording);
            let (start_frame, end_frame) = frames.resolve(recording.frame_count())?;
            verify_cached_runs(&recording, start_frame, end_frame, run.as_deref())?;
        }
        Command::Snapshot {
            recording,
            cache_dir,
            run,
            frame,
            output,
            raw,
            details,
            object_confidence,
            anchor_presence,
            line_confidence,
        } => {
            let recording_path = resolve_recording(recording);
            let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&recording_path));
            let recording = Recording::open(recording_path, cache_dir)?;
            if frame >= recording.frame_count() {
                color_eyre::eyre::bail!(
                    "frame {frame} is out of range for {} frames",
                    recording.frame_count()
                );
            }
            snapshot::export(
                &recording,
                &run,
                frame,
                &output,
                raw,
                details,
                object_confidence,
                anchor_presence,
                line_confidence,
            )?;
            println!("wrote {}", output.display());
        }
    }

    Ok(())
}

fn configure_gpu_visibility(command: &Command) {
    let Command::Prerender { provider, gpu, .. } = command else {
        return;
    };
    if !provider.uses_nvidia() {
        return;
    }

    // SAFETY: this runs at process startup, before ONNX Runtime initialization or
    // creation of the Tokio worker threads that may load CUDA.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", gpu.to_string()) };
}

fn verify_cached_runs(
    recording: &Recording,
    start_frame: usize,
    end_frame: usize,
    selected_run: Option<&str>,
) -> Result<()> {
    let (runs, load_errors) = recording.load_runs_with_errors()?;
    if selected_run.is_none() && !load_errors.is_empty() {
        color_eyre::eyre::bail!(
            "{} cached runs failed inspection: {}",
            load_errors.len(),
            load_errors.join("; ")
        );
    }
    if selected_run.is_some() {
        for error in &load_errors {
            eprintln!("warning: ignored unrelated invalid cached run: {error}");
        }
    }
    let runs = runs
        .into_iter()
        .filter(|run| {
            selected_run.is_none_or(|selected| run.key == selected || run.label == selected)
        })
        .collect::<Vec<_>>();
    if runs.is_empty() {
        color_eyre::eyre::bail!(
            "no cached run matched {}",
            selected_run.unwrap_or("the selection")
        );
    }
    for run in runs {
        let expected_frames = end_frame - start_frame + 1;
        if selected_run.is_some() {
            let manifest = run.manifest.as_ref().wrap_err_with(|| {
                format!("selected run {} is not a model-run manifest", run.key)
            })?;
            if manifest.state != ModelRunState::Complete {
                color_eyre::eyre::bail!(
                    "selected run {} is {:?}, not complete",
                    run.key,
                    manifest.state
                );
            }
            let available = run.available_count(start_frame..=end_frame);
            if available != expected_frames {
                color_eyre::eyre::bail!(
                    "selected run {} covers {available} of {expected_frames} requested frames",
                    run.key
                );
            }
        }
        let mut frame = start_frame;
        let mut prediction_frames = 0_usize;
        let mut object_rows = 0_usize;
        let mut empty_object_frames = 0_usize;
        let mut hydra_messages = 0_usize;
        let mut missing_hydra_messages = 0_usize;
        let mut empty_line_frames = 0_usize;
        let mut anchors = 0_usize;
        let mut lines = 0_usize;
        let mut raw_object_rows = 0_usize;
        let mut raw_line_rows = 0_usize;
        while frame <= end_frame {
            if !run.is_available(frame) {
                frame += 1;
                continue;
            }
            let chunk = run.prediction_chunk(frame)?;
            let chunk_end = chunk
                .start_frame
                .saturating_add(chunk.predictions.len().saturating_sub(1))
                .min(end_frame);
            for current in frame..=chunk_end {
                let offset = current - chunk.start_frame;
                let Some(prediction) = &chunk.predictions[offset] else {
                    continue;
                };
                let expected_timestamp = recording.frames()[current].timestamp_nanos;
                if prediction.frame_index != current
                    || prediction.timestamp_nanos != expected_timestamp
                {
                    color_eyre::eyre::bail!(
                        "run {} frame {current} disagrees with the recording index",
                        run.key
                    );
                }
                prediction_frames += 1;
                object_rows += prediction.objects.len();
                empty_object_frames += usize::from(prediction.objects.is_empty());
                match &chunk.hydra_supplements[offset] {
                    Some(supplement) => {
                        if supplement.object_anchors.len() != prediction.objects.len() {
                            color_eyre::eyre::bail!(
                                "run {} frame {current} has {} anchors for {} objects",
                                run.key,
                                supplement.object_anchors.len(),
                                prediction.objects.len()
                            );
                        }
                        if supplement.raw_objects.len() != 300 || supplement.raw_lines.len() != 300
                        {
                            color_eyre::eyre::bail!(
                                "run {} frame {current} does not contain all 300 raw rows per Hydra head",
                                run.key
                            );
                        }
                        hydra_messages += 1;
                        empty_line_frames += usize::from(supplement.lines.is_empty());
                        anchors += supplement.object_anchors.iter().flatten().count();
                        lines += supplement.lines.len();
                        raw_object_rows += supplement.raw_objects.len();
                        raw_line_rows += supplement.raw_lines.len();
                    }
                    None => missing_hydra_messages += 1,
                }
            }
            frame = chunk_end.saturating_add(1);
        }
        if selected_run.is_some()
            && (prediction_frames != expected_frames
                || hydra_messages != expected_frames
                || missing_hydra_messages != 0)
        {
            color_eyre::eyre::bail!(
                "selected run {} decoded {prediction_frames}/{expected_frames} predictions and {hydra_messages}/{expected_frames} Hydra messages ({missing_hydra_messages} missing)",
                run.key
            );
        }
        let model_blake3 = run.manifest.as_ref().map_or_else(
            || "recorded".to_string(),
            |manifest| hex_digest(&manifest.model_hash),
        );
        println!(
            "run={} label={:?} model_blake3={} prediction_frames={} object_rows={} empty_object_frames={} hydra_messages={} missing_hydra_messages={} anchors={} line_rows={} empty_line_frames={} raw_object_rows={} raw_line_rows={}",
            run.key,
            run.label,
            model_blake3,
            prediction_frames,
            object_rows,
            empty_object_frames,
            hydra_messages,
            missing_hydra_messages,
            anchors,
            lines,
            empty_line_frames,
            raw_object_rows,
            raw_line_rows,
        );
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write;

    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    )
}

fn report_recording_warning(recording: &Recording) {
    if let Some(warning) = &recording.index().tail_warning {
        eprintln!("warning: {warning}");
    }
}

fn resolve_recording(path: PathBuf) -> PathBuf {
    if path.is_dir() {
        path.join("recording.mcap")
    } else {
        path
    }
}

fn default_cache_dir(recording: &std::path::Path) -> PathBuf {
    recording.with_extension("detection-replay-cache")
}

//! Simulator-owned diagnostics. No robotics node needs to know about capture or recording.
mod logs;
mod recording;

use color_eyre::{Result, eyre::ensure};
use ros_z::{prelude::*, time::Clock};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::Ordering},
};
use tracing_subscriber::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, ros_z::Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub topics: Vec<String>,
    pub queue_capacity: usize,
    /// Error=1, Warn=2, Info=3, Debug=4, Trace=5.
    pub capture_level: u8,
}
#[derive(Clone, Default)]
pub struct Health(Arc<Mutex<Option<String>>>);
impl Health {
    pub fn error(&self, error: impl Into<String>) {
        self.0.lock().unwrap().get_or_insert_with(|| error.into());
    }
    pub fn get(&self) -> Option<String> {
        self.0.lock().unwrap().clone()
    }
}
thread_local! { static DISPATCH: RefCell<Option<tracing::dispatcher::DefaultGuard>> = const { RefCell::new(None) }; }
pub struct RobotRuntime(Option<tokio::runtime::Runtime>);
impl RobotRuntime {
    pub fn new(capture: logs::Capture) -> Result<Self> {
        let dispatcher = tracing::Dispatch::new(tracing_subscriber::registry().with(capture));
        Ok(Self(Some(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("simulator-robot")
                .enable_all()
                .on_thread_start(move || {
                    DISPATCH.with(|guard| {
                        *guard.borrow_mut() = Some(tracing::dispatcher::set_default(&dispatcher))
                    })
                })
                .on_thread_stop(|| {
                    DISPATCH.with(|guard| {
                        guard.borrow_mut().take();
                    })
                })
                .build()?,
        )))
    }
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.0.as_ref().unwrap().handle().clone()
    }
}
impl Drop for RobotRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            runtime.shutdown_background();
        }
    }
}
pub struct Startup {
    pub capture: logs::Capture,
    receiver: tokio::sync::mpsc::Receiver<logs::LogRecord>,
    file: File,
    pub directory: PathBuf,
}
impl Startup {
    pub fn new(directory: PathBuf, clock: Clock) -> Result<Self> {
        std::fs::create_dir_all(&directory)?;
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join("logs.jsonl"))?;
        let (capture, receiver) = logs::Capture::new(clock);
        Ok(Self {
            capture,
            receiver,
            file,
            directory,
        })
    }
}
pub struct Telemetry {
    capture: logs::Capture,
    logs: logs::Logs,
    recording: recording::Recording,
    pub health: Health,
    pub directory: PathBuf,
}
impl Telemetry {
    pub async fn start(
        startup: Startup,
        context: &Arc<Context>,
        namespace: &str,
        layers: &[PathBuf],
    ) -> Result<Self> {
        let node = Arc::new(context.create_node("simulator_recording").build().await?);
        let parameters = node.bind_parameter_as::<Parameters>("simulator_recording")?;
        let configured = parameters.snapshot().typed().clone();
        let expected = configured.clone();
        parameters.add_validation_hook(move |candidate| {
            if *candidate == expected { Ok(()) } else { Err("Recording settings are fixed for this session; edit the location overlay before starting".into()) }
        })?;
        let parameters = configured;
        ensure!(
            (1..=5).contains(&parameters.capture_level),
            "capture_level must be 1 through 5"
        );
        ensure!(
            (1..=65536).contains(&parameters.queue_capacity),
            "queue_capacity must be 1 through 65536"
        );
        startup
            .capture
            .level
            .store(parameters.capture_level, Ordering::Relaxed);
        for (index, layer) in layers.iter().enumerate() {
            if !layer.is_dir() {
                continue;
            }
            let destination = startup.directory.join("parameters").join(index.to_string());
            std::fs::create_dir_all(&destination)?;
            for entry in std::fs::read_dir(layer)? {
                let entry = entry?;
                if entry.path().extension().is_some_and(|e| e == "json5") {
                    std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
                }
            }
        }
        let health = Health::default();
        let recording = recording::Recording::start(
            context,
            node.clone(),
            namespace,
            &startup.directory,
            &parameters,
            health.clone(),
        )
        .await?;
        let logs = logs::Logs::start(&node, startup.file, startup.receiver, health.clone()).await?;
        Ok(Self {
            capture: startup.capture,
            logs,
            recording,
            health,
            directory: startup.directory,
        })
    }
    pub fn status(&self) -> Option<String> {
        if let Some(error) = self.health.get() {
            return Some(error);
        }
        if self.recording.is_finished() {
            return Some("MCAP recording stopped unexpectedly".into());
        }
        let logs = self.capture.dropped.load(Ordering::Relaxed);
        let samples = self.recording.dropped.load(Ordering::Relaxed);
        (logs + samples > 0)
            .then(|| format!("Dropped {logs} log entries and {samples} recording samples"))
    }
    pub async fn finish(self) -> Option<String> {
        self.capture.level.store(0, Ordering::Relaxed);
        let dropped_samples = self.recording.dropped.clone();
        if let Err(error) = self.logs.finish().await {
            self.health.error(format!("Log finalization: {error:#}"));
        }
        if let Err(error) = self.recording.finish().await {
            self.health.error(format!("MCAP finalization: {error:#}"));
        }
        let summary = serde_json::json!({"error": self.health.get(), "dropped_samples": dropped_samples.load(Ordering::Relaxed), "dropped_logs": self.capture.dropped.load(Ordering::Relaxed)});
        if let Err(error) = std::fs::write(
            self.directory.join("finished.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        ) {
            self.health
                .error(format!("Saving final recording status: {error}"));
        }
        self.health.get()
    }
}

pub fn session(configuration: &crate::Configuration, clock: &Clock) -> Result<PathBuf> {
    let root = std::env::var_os("SIMULATOR_RECORDING_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs/simulator"));
    std::fs::create_dir_all(&root)?;
    let directory = tempfile::Builder::new()
        .prefix(&format!("{}-", clock.now().as_nanos()))
        .tempdir_in(root)?
        .keep();
    let git = |arguments: &[&str]| {
        std::process::Command::new("git")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
    };
    let revision = git(&["rev-parse", "HEAD"])
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let dirty = git(&["status", "--porcelain"])
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    if let Some(changes) = git(&["diff", "HEAD", "--binary"]) {
        std::fs::write(directory.join("source.patch"), changes.stdout)?;
    }
    let mut models = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(&configuration.model_directory)? {
        let entry = entry?;
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "onnx")
        {
            models.insert(
                entry.file_name().to_string_lossy().into_owned(),
                sha256(&entry.path())?,
            );
        }
    }
    std::fs::write(
        directory.join("session.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "revision": revision, "dirty": dirty, "model_sha256": models, "profile": configuration.profile, "location": configuration.location,
            "initial_robot_count": configuration.robot_count, "initial_opponent_count": configuration.opponent_count,
            "model_directory": configuration.model_directory, "simulation_start": clock.now().as_nanos(),
            "field": configuration.field_configuration, "referee": configuration.referee, "competition": configuration.competition,
            "clock": "MCAP log_time is simulation time; publish_time preserves each publisher's clock; _recording/receipt preserves wall arrival times",
        }))?,
    )?;
    Ok(directory)
}

pub fn robot_directory(session: &Path, namespace: &str) -> PathBuf {
    session.join(namespace.trim_matches('/'))
}

static FINALIZERS: Mutex<Vec<std::thread::JoinHandle<Option<String>>>> = Mutex::new(Vec::new());
pub(crate) fn finalize(work: impl FnOnce() -> Option<String> + Send + 'static) {
    FINALIZERS.lock().unwrap().push(std::thread::spawn(work));
}
/// Wait for outstanding recording writers. Call off the UI thread after dropping the simulator.
pub fn finish_recordings() -> Vec<String> {
    let handles = std::mem::take(&mut *FINALIZERS.lock().unwrap());
    handles
        .into_iter()
        .filter_map(|handle| match handle.join() {
            Ok(error) => error,
            Err(_) => Some("Recording finalizer panicked".into()),
        })
        .collect()
}

#[cfg(test)]
mod tests;

fn sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

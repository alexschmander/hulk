use super::{clock::SharedClock, rosz};
use color_eyre::{
    Result,
    eyre::{bail, ensure, eyre},
};
use rmcp::{
    ServerHandler, ServiceExt,
    model::*,
    service::{RequestContext, RoleServer},
};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::Mutex as AsyncMutex,
};
use tokio_util::sync::CancellationToken;

pub(super) const CONTRACT: &str =
    include_str!("../../../../docs/tooling/simulator_mcp_contract.schema.json");
pub(super) fn id() -> String {
    uuid::Uuid::new_v4().to_string()
}
pub(super) fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key].as_str().ok_or_else(|| eyre!("Missing {key}"))
}
pub(super) fn wall() -> i64 {
    ros_z::time::Clock::wallclock().now().as_nanos()
}
pub(super) fn error(code: &str, message: impl ToString, effect: &str) -> Value {
    json!({"error":{"code":code,"message":message.to_string(),"retryable":false,"effect":effect}})
}

pub(super) struct Worker {
    rpc: AsyncMutex<BufReader<UnixStream>>,
    child: AsyncMutex<tokio::process::Child>,
    pub clock: Arc<SharedClock>,
    pub origin: i64,
    pub configuration: Value,
    pub id: String,
    pub last: Mutex<Value>,
    pub stopped: std::sync::atomic::AtomicBool,
    pub advance: AsyncMutex<()>,
    _directory: tempfile::TempDir,
}
impl Worker {
    pub async fn call(&self, method: &str, arguments: Value) -> Result<Value> {
        ensure!(
            !self.stopped.load(std::sync::atomic::Ordering::Acquire),
            "Session stopped"
        );
        let mut rpc = self.rpc.lock().await;
        let mut line = String::new();
        let transport = tokio::time::timeout(Duration::from_secs(60), async {
            rpc.get_mut()
                .write_all(
                    format!("{}\n", json!({"method":method,"arguments":arguments})).as_bytes(),
                )
                .await?;
            ensure!(
                rpc.read_line(&mut line).await? != 0,
                "Simulator worker exited"
            );
            ensure!(
                line.len() <= 16 * 1024 * 1024,
                "Worker response exceeds transport limit"
            );
            Ok::<_, color_eyre::Report>(())
        })
        .await;
        if !matches!(transport, Ok(Ok(()))) {
            self.stopped
                .store(true, std::sync::atomic::Ordering::Release);
            let message = match transport {
                Ok(Err(e)) => format!("{e:#}"),
                _ => "Simulator worker response timed out".into(),
            };
            {
                let mut last = self.last.lock().unwrap();
                last["state"] = json!("failed");
                last["error"] = json!(message);
            }
            let mut child = self.child.lock().await;
            let _ = child.kill().await;
            let _ = child.wait().await;
            bail!("{message}");
        }
        let reply: Value = serde_json::from_str(&line)?;
        if let Some(error) = reply.get("error") {
            bail!("{}", error.as_str().unwrap_or("Worker failed"));
        }
        let result = reply["result"].clone();
        if method == "status" || method == "control" {
            *self.last.lock().unwrap() = result.clone();
        }
        Ok(result)
    }
    pub fn time(&self) -> i64 {
        self.clock.load() - self.origin
    }
    pub async fn stop(&self) -> Result<Value> {
        let result = self.call("stop", json!({})).await;
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        let mut child = self.child.lock().await;
        let exit_status = match tokio::time::timeout(Duration::from_secs(30), child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                child.kill().await?;
                child.wait().await?
            }
        };
        if !exit_status.success() {
            let mut last = self.last.lock().unwrap();
            last["state"] = json!("failed");
            last["error"] = json!(format!("Worker exited with {exit_status}"));
            bail!("Worker exited with {exit_status}");
        }
        let mut last = self.last.lock().unwrap();
        if result.is_ok() {
            last["state"] = json!("stopped");
        }
        result.map(|_| last.clone())
    }
}
#[derive(Clone)]
pub(super) struct Operation {
    pub value: Arc<Mutex<Value>>,
    pub cancel: CancellationToken,
}
impl Operation {
    fn new(tool: &str) -> Self {
        Self {
            value: Arc::new(Mutex::new(
                json!({"operation_id":id(),"tool":tool,"state":"queued","effect":"none"}),
            )),
            cancel: CancellationToken::new(),
        }
    }
    pub fn stage(&self, stage: &str) {
        let mut value = self.value.lock().unwrap();
        value["state"] = json!("running");
        value["stage"] = json!(stage);
    }
    pub fn snapshot(&self) -> Value {
        self.value.lock().unwrap().clone()
    }
    pub fn check(&self) -> Result<()> {
        ensure!(!self.cancel.is_cancelled(), "Operation cancelled");
        Ok(())
    }
}
pub(super) struct FrozenQuery {
    pub created: Instant,
    pub arguments: Value,
    pub rows: Vec<Value>,
    pub metadata: Value,
    pub collection: String,
}
pub(super) struct State {
    pub sessions: AsyncMutex<HashMap<String, Arc<Worker>>>,
    pub targets: AsyncMutex<HashMap<String, Arc<rosz::Target>>>,
    pub observations: AsyncMutex<HashMap<String, rosz::Observation>>,
    pub queries: Mutex<HashMap<String, FrozenQuery>>,
    pub operations: Mutex<HashMap<String, Operation>>,
    requests: AsyncMutex<HashMap<String, (Value, Value)>>,
    start: AsyncMutex<()>,
    pub artifacts: PathBuf,
    pub artifact_index: Mutex<HashMap<String, PathBuf>>,
    contract: Value,
    tools: Vec<Tool>,
    validators: HashMap<String, jsonschema::Validator>,
}
#[derive(Clone)]
struct Server(Arc<State>);
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().build()).with_server_info(Implementation::new("hulk-simulator",env!("CARGO_PKG_VERSION"))).with_instructions("Start an owned simulator, discover targets and ROS-Z schemas, observe topics before advancing, and inspect operation_status. Sessions start paused. MCAP history is opt-in. Video arms a future interval and never advances the simulation.")
    }
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, rmcp::ErrorData> {
        Ok(ListToolsResult {
            tools: self.0.tools.clone(),
            ..Default::default()
        })
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResponse, rmcp::ErrorData> {
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        let output = match self.0.dispatch(request.name.as_ref(), arguments).await {
            Ok(value) => value,
            Err(e) => error("request_failed", format!("{e:#}"), "none"),
        };
        Ok(if output.get("error").is_some() {
            CallToolResult::structured_error(output)
        } else {
            CallToolResult::structured(output)
        }
        .into())
    }
    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ReadResourceResponse, rmcp::ErrorData> {
        let path = self
            .0
            .artifact_index
            .lock()
            .unwrap()
            .get(&request.uri)
            .cloned()
            .ok_or_else(|| rmcp::ErrorData::invalid_params("Unknown artifact", None))?;
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
        use base64::Engine;
        Ok(
            ReadResourceResult::new(vec![ResourceContents::BlobResourceContents {
                uri: request.uri,
                mime_type: Some(mime(&path).into()),
                blob: base64::engine::general_purpose::STANDARD.encode(bytes),
                meta: None,
            }])
            .into(),
        )
    }
}
pub(super) fn schema(contract: &Value, reference: &Value) -> Value {
    if let Some(reference) = reference.get("$ref").and_then(Value::as_str) {
        return schema(
            contract,
            contract.pointer(reference.trim_start_matches('#')).unwrap(),
        );
    }
    match reference {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), schema(contract, value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(|v| schema(contract, v)).collect()),
        _ => reference.clone(),
    }
}
pub(super) async fn run() -> Result<()> {
    let contract: Value = serde_json::from_str(CONTRACT)?;
    let mut validators = HashMap::new();
    let mut tools = Vec::new();
    for tool in contract["x-tools"].as_array().unwrap() {
        let name = string(tool, "name")?;
        let input = schema(&contract, &tool["input_schema"]);
        validators.insert(name.to_owned(), jsonschema::validator_for(&input)?);
        let mut definition = Tool::new(
            name.to_owned(),
            string(tool, "description")?.to_owned(),
            Arc::new(input.as_object().unwrap().clone()),
        );
        definition.output_schema = Some(Arc::new(
            schema(&contract, &tool["output_schema"])
                .as_object()
                .unwrap()
                .clone(),
        ));
        tools.push(definition);
    }
    let artifacts = std::env::var_os("SIMULATOR_MCP_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs/simulator-mcp").join(id()));
    std::fs::create_dir_all(&artifacts)?;
    let artifacts = artifacts.canonicalize()?;
    let state = Arc::new(State {
        sessions: Default::default(),
        targets: Default::default(),
        observations: Default::default(),
        queries: Default::default(),
        operations: Default::default(),
        requests: Default::default(),
        start: Default::default(),
        artifacts,
        artifact_index: Default::default(),
        contract,
        tools,
        validators,
    });
    let service = Server(state.clone())
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    for operation in state.operations.lock().unwrap().values() {
        operation.cancel.cancel();
    }
    let sessions: Vec<_> = state.sessions.lock().await.values().cloned().collect();
    for session in sessions {
        if !session.stopped.load(std::sync::atomic::Ordering::Acquire) {
            let _ = session.stop().await;
        }
    }
    Ok(())
}
impl State {
    async fn dispatch(self: &Arc<Self>, tool: &str, args: Value) -> Result<Value> {
        self.validators
            .get(tool)
            .ok_or_else(|| eyre!("Unknown tool {tool}"))?
            .validate(&args)
            .map_err(|e| eyre!(e.to_string()))?;
        if let Some(request) = args["request_id"].as_str() {
            let signature = json!({"tool":tool,"arguments":args});
            let mut requests = self.requests.lock().await;
            if let Some((old, value)) = requests.get(request) {
                ensure!(
                    old == &signature,
                    "request_id reused with different arguments"
                );
                return Ok(value.clone());
            }
            ensure!(
                requests.len() < 100_000,
                "Request ledger capacity reached; start a new server"
            );
            let result = self.begin(tool, args.clone()).await?;
            requests.insert(request.into(), (signature, result.clone()));
            Ok(result)
        } else {
            self.begin(tool, args).await
        }
    }
    async fn begin(self: &Arc<Self>, tool: &str, args: Value) -> Result<Value> {
        match tool {
            "server_info" => return Ok(self.info()),
            "operation_status" => {
                let op = self
                    .operations
                    .lock()
                    .unwrap()
                    .get(string(&args, "operation_id")?)
                    .cloned()
                    .ok_or_else(|| eyre!("Unknown operation"))?;
                let deadline =
                    Instant::now() + Duration::from_millis(args["wait_ms"].as_u64().unwrap_or(0));
                loop {
                    let value = op.snapshot();
                    if !matches!(value["state"].as_str(), Some("queued" | "running"))
                        || Instant::now() >= deadline
                    {
                        return Ok(value);
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
            "operation_cancel" => {
                let op = self
                    .operations
                    .lock()
                    .unwrap()
                    .get(string(&args, "operation_id")?)
                    .cloned()
                    .ok_or_else(|| eyre!("Unknown operation"))?;
                op.cancel.cancel();
                return Ok(op.snapshot());
            }
            "simulator_status" => {
                let worker = self.worker(&args).await?;
                return self.session_state(&worker).await;
            }
            "scene_read" => {
                let worker = self.worker(&args).await?;
                return Ok(scene(worker.call("scene", json!({})).await?));
            }
            "rosz_graph"
            | "rosz_schema"
            | "rosz_observe"
            | "rosz_unobserve"
            | "rosz_parameter_read"
            | "logs_read" => return rosz::read_tool(self, tool, &args).await,
            "rosz_read" if !args["allow_recording_fallback"].as_bool().unwrap_or(false) => {
                return rosz::read(self, &args).await;
            }
            _ => {}
        }
        let operation = Operation::new(tool);
        let snapshot = operation.snapshot();
        {
            let mut operations = self.operations.lock().unwrap();
            ensure!(
                operations.len() < 2048,
                "Operation capacity reached; start a new server"
            );
            operations.insert(
                snapshot["operation_id"].as_str().unwrap().into(),
                operation.clone(),
            );
        }
        let state = self.clone();
        let tool = tool.to_owned();
        tokio::spawn(async move {
            operation.stage("starting");
            let result = state.execute(&tool, args, &operation).await;
            let mut value = operation.value.lock().unwrap();
            match result {
                Ok(result) => {
                    value["state"] = json!(if operation.cancel.is_cancelled() {
                        "cancelled"
                    } else {
                        "succeeded"
                    });
                    value["effect"] = json!(if matches!(
                        tool.as_str(),
                        "rosz_wait" | "rosz_read" | "view_render" | "video_capture"
                    ) {
                        "none"
                    } else {
                        "applied"
                    });
                    value["result"] = result;
                }
                Err(e) => {
                    value["state"] = json!(if operation.cancel.is_cancelled() {
                        "cancelled"
                    } else {
                        "failed"
                    });
                    let effect = if matches!(
                        tool.as_str(),
                        "rosz_wait"
                            | "rosz_read"
                            | "view_render"
                            | "video_capture"
                            | "simulator_start"
                    ) {
                        "none"
                    } else {
                        "unknown"
                    };
                    value["effect"] = json!(effect);
                    value["error"] =
                        error("operation_failed", format!("{e:#}"), effect)["error"].clone();
                }
            }
        });
        Ok(snapshot)
    }
    pub async fn worker(&self, args: &Value) -> Result<Arc<Worker>> {
        self.sessions
            .lock()
            .await
            .get(string(args, "session_id")?)
            .cloned()
            .ok_or_else(|| eyre!("Unknown session"))
    }
    pub async fn target(&self, args: &Value) -> Result<Arc<rosz::Target>> {
        self.targets
            .lock()
            .await
            .get(string(args, "target_id")?)
            .cloned()
            .ok_or_else(|| eyre!("Unknown target"))
    }
    async fn start(&self, args: &Value, operation: &Operation) -> Result<Value> {
        let _guard = self.start.lock().await;
        ensure!(
            self.sessions
                .lock()
                .await
                .values()
                .all(|worker| worker.stopped.load(std::sync::atomic::Ordering::Acquire)),
            "A simulator is already active"
        );
        operation.check()?;
        operation.stage("initializing");
        let directory = tempfile::tempdir()?;
        let socket = directory.path().join("worker.sock");
        let listener = UnixListener::bind(&socket)?;
        let clock_path = directory.path().join("clock");
        let clock = Arc::new(SharedClock::open(&clock_path)?);
        let mut command = tokio::process::Command::new(std::env::current_exe()?);
        command
            .arg("--worker")
            .arg(&socket)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        if let Some(root) = args["configuration"]["recording_root"].as_str() {
            command.env("SIMULATOR_RECORDING_ROOT", root);
        }
        let mut child = command.spawn()?;
        let (stream, _) =
            tokio::time::timeout(Duration::from_secs(10), listener.accept()).await??;
        let mut rpc = BufReader::new(stream);
        let session = id();
        rpc.get_mut().write_all(format!("{}\n",json!({"method":"start","arguments":{"session_id":session,"configuration":args["configuration"],"clock_path":clock_path}})).as_bytes()).await?;
        let mut line = String::new();
        tokio::select! {result=tokio::time::timeout(Duration::from_secs(180),rpc.read_line(&mut line))=>{ensure!(result??!=0,"Simulator exited during startup");},_=operation.cancel.cancelled()=>{child.kill().await?;bail!("Startup cancelled");}}
        let response: Value = serde_json::from_str(&line)?;
        ensure!(
            response.get("error").is_none(),
            "Simulator startup: {}",
            response["error"]
        );
        let status = response["result"].clone();
        let worker = Arc::new(Worker {
            rpc: AsyncMutex::new(rpc),
            child: AsyncMutex::new(child),
            clock,
            origin: string(&status, "clock_origin_ns")?.parse()?,
            configuration: args["configuration"].clone(),
            id: session.clone(),
            last: Mutex::new(status.clone()),
            stopped: Default::default(),
            advance: Default::default(),
            _directory: directory,
        });
        let mut targets = Vec::new();
        for target in status["targets"].as_array().unwrap() {
            let target =
                rosz::Target::connect(worker.clone(), target, string(&status, "endpoint")?).await?;
            targets.push(Arc::new(target));
        }
        if operation.cancel.is_cancelled() {
            worker.stop().await?;
            bail!("Startup cancelled");
        }
        self.targets.lock().await.extend(
            targets
                .into_iter()
                .map(|target| (target.id.clone(), target)),
        );
        self.sessions.lock().await.insert(session, worker.clone());
        self.session_state(&worker).await
    }
    pub async fn session_state(&self, worker: &Worker) -> Result<Value> {
        let raw = if worker.stopped.load(std::sync::atomic::Ordering::Acquire) {
            worker.last.lock().unwrap().clone()
        } else {
            match worker.call("status", json!({})).await {
                Ok(raw) => raw,
                Err(_) => worker.last.lock().unwrap().clone(),
            }
        };
        let targets: Vec<_> = raw["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|target| {
                let mut target = target.clone();
                target
                    .as_object_mut()
                    .unwrap()
                    .remove("recording_directory");
                target["live"] = json!(raw["state"] != "stopped" && raw["state"] != "failed");
                target
            })
            .collect();
        Ok(
            json!({"session_id":worker.id,"state":raw["state"],"simulation_time_ns":raw["simulation_time_ns"],"physics_step_ns":raw["timestep_ns"],"scene_revision":raw["scene_revision"],"targets":targets,"configuration":worker.configuration,"ready":raw["state"]=="paused" || raw["state"]=="running","diagnostics":raw["error"].as_str().into_iter().collect::<Vec<_>>(),"rendering_available":worker.configuration["rendering"]!="disabled","recording_status":if raw["state"]=="stopped"{"finalized"}else if raw["state"]=="failed"{"failed"}else{"recording"},"active_operation_ids":self.operations.lock().unwrap().values().filter_map(|op|{let v=op.snapshot();(v["state"]=="running").then(||v["operation_id"].clone())}).collect::<Vec<_>>()}),
        )
    }
    async fn execute(
        self: &Arc<Self>,
        tool: &str,
        args: Value,
        operation: &Operation,
    ) -> Result<Value> {
        operation.check()?;
        match tool {
            "simulator_start" => self.start(&args, operation).await,
            "simulator_control" => {
                let worker = self.worker(&args).await?;
                let command = &args["command"];
                if command["kind"] == "stop" {
                    worker.stop().await?;
                    return Ok(
                        json!({"session":self.session_state(&worker).await?,"stop_reason":"requested"}),
                    );
                }
                if command["kind"] != "advance" {
                    worker.call("control", command.clone()).await?;
                    return Ok(
                        json!({"session":self.session_state(&worker).await?,"stop_reason":"requested"}),
                    );
                }
                let _guard = worker
                    .advance
                    .try_lock()
                    .map_err(|_| eyre!("Advance already in progress"))?;
                let initial = worker.time();
                operation.stage("advancing");
                let mut condition = match command.get("stop_when") {
                    Some(condition) => Some(rosz::ConditionWatch::new(self, condition).await?),
                    None => None,
                };
                worker.call("control", command.clone()).await?;
                let deadline = Instant::now()
                    + Duration::from_millis(command["wall_timeout_ms"].as_u64().unwrap_or(30000));
                let mut reason = if command.get("steps").is_some() {
                    "steps_reached"
                } else {
                    "duration_reached"
                };
                let result = async {
                    loop {
                        if operation.cancel.is_cancelled() {
                            reason = "requested";
                            break;
                        }
                        if Instant::now() >= deadline {
                            reason = "timeout";
                            break;
                        }
                        if let Some(condition) = condition.as_mut()
                            && condition.evaluate()?.0
                        {
                            reason = "condition_met";
                            break;
                        }
                        let status = worker.call("status", json!({})).await?;
                        if status["state"] == "failed" {
                            bail!("Physics worker failed: {}", status["error"]);
                        }
                        if status["state"] == "paused" {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    Ok::<_, color_eyre::Report>(())
                }
                .await;
                worker.call("control", json!({"kind":"pause"})).await?;
                result?;
                let session = self.session_state(&worker).await?;
                Ok(
                    json!({"advanced_steps":(worker.time()-initial)/string(&session,"physics_step_ns")?.parse::<i64>()?,"session":session,"stop_reason":reason}),
                )
            }
            "scene_edit" => {
                let worker = self.worker(&args).await?;
                let result = worker.call("edit", args.clone()).await?;
                let status = worker.call("status", json!({})).await?;
                for target in status["targets"].as_array().unwrap() {
                    let key = string(target, "target_id")?;
                    if !self.targets.lock().await.contains_key(key) {
                        let target = rosz::Target::connect(
                            worker.clone(),
                            target,
                            string(&status, "endpoint")?,
                        )
                        .await?;
                        self.targets
                            .lock()
                            .await
                            .insert(target.id.clone(), Arc::new(target));
                    }
                }
                Ok(scene(result))
            }
            "simulator_input" => {
                let worker = self.worker(&args).await?;
                let mut details = worker.call("input", args["input"].clone()).await?;
                let input_id = details["id"].clone();
                while details["done"] != true {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    details = worker
                        .call(
                            "input_status",
                            json!({"id":input_id,"cancel":operation.cancel.is_cancelled()}),
                        )
                        .await?;
                }
                Ok(
                    json!({"accepted":true,"applied_at_simulation_ns":worker.time().to_string(),"details":details}),
                )
            }
            "rosz_wait" => {
                let deadline = Instant::now()
                    + Duration::from_millis(args["wall_timeout_ms"].as_u64().unwrap());
                let mut condition = rosz::ConditionWatch::new(self, &args["condition"]).await?;
                operation.stage("observing");
                loop {
                    operation.check()?;
                    let (matched, samples) = condition.evaluate()?;
                    if matched || Instant::now() >= deadline {
                        return Ok(json!({"matched":matched,"samples":samples}));
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
            "rosz_read" => {
                operation.stage("checkpointing");
                rosz::read_cancellable(self, &args, Some(operation.cancel.clone())).await
            }
            "rosz_parameter_write" | "rosz_publish" | "rosz_call" => {
                rosz::write_tool(self, tool, &args, operation).await
            }
            "view_render" | "video_capture" => {
                super::visual::execute(self, tool, &args, operation).await
            }
            _ => bail!("Unknown operation {tool}"),
        }
    }
    fn configuration_schema(&self) -> Value {
        let mut value = schema(
            &self.contract,
            &json!({"$ref":"#/$defs/SimulatorConfiguration"}),
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters");
        if let Ok(locations) = crate::FieldConfiguration::locations(&root) {
            value["properties"]["location"]["examples"] = json!(locations);
        }
        value
    }
    fn info(&self) -> Value {
        json!({"contract_version":"draft-1","capabilities":["owned_headless_sessions","rosz_graph","rosz_state","rosz_parameters","rosz_publish","rosz_services","retained_history","active_recording_fallback","scene_render","plots","video","logs"],"limits":{"active_sessions":1,"observations":128,"observation_bytes":268435456,"read_samples":10000,"read_bytes":1048576,"operations":2048,"request_ids":100000},"configuration_schema":self.configuration_schema(),"simulator_input_schemas":{"input":schema(&self.contract,&self.contract["$defs"]["simulator_input_input"]["properties"]["input"]),"controller":{"encoding":"serde-json","example":types::controller_input::ControllerInput::default(),"canonical_schema":<types::controller_input::ControllerInput as ros_z::Message>::schema()},"referee":crate::autoref::command_schemas()},"rendering":{"backend":"MuJoCo EGL","on_demand":true,"camera_images":"Only available when a robot publishes an actual image"},"prerequisites":["MuJoCo 3.9 shared library","ONNX Runtime and motion models","EGL for scene rendering"]})
    }
    pub fn artifact(&self, path: &Path, description: &str) -> Result<Value> {
        let uri = format!("simulator-artifact://{}", id());
        self.artifact_index
            .lock()
            .unwrap()
            .insert(uri.clone(), path.to_owned());
        Ok(
            json!({"uri":uri,"mime_type":mime(path),"byte_length":std::fs::metadata(path)?.len(),"description":description}),
        )
    }
}
fn scene(mut raw: Value) -> Value {
    let objects = raw.as_object_mut().unwrap().remove("objects").unwrap();
    raw.as_object_mut().unwrap().remove("field");
    raw["entities"] = objects;
    raw
}
pub(super) fn mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

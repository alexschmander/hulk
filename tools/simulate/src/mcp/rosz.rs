use super::server::{FrozenQuery, Operation, State, Worker, id, string, wall};
use color_eyre::{
    Result,
    eyre::{bail, ensure, eyre},
};
use ros_z::{
    dynamic::{ByteRenderPolicy, DynamicJsonRenderPolicy, DynamicPayload, ValuePath},
    prelude::*,
    time::Time,
};
use ros_z_debug::{CachedSubscription, CachedSubscriptionBuilder, RetentionClock, RetentionPolicy};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) const JSON: DynamicJsonRenderPolicy = DynamicJsonRenderPolicy {
    tag_optionals: true,
    wide_integers_as_strings: true,
    bytes: ByteRenderPolicy::FullArray,
    non_finite_float: ros_z::dynamic::NonFiniteFloatRenderPolicy::Object,
};
pub(super) struct Target {
    pub id: String,
    pub namespace: String,
    pub node: Arc<Node>,
    pub worker: Arc<Worker>,
    pub directory: PathBuf,
    _context: Arc<Context>,
}
impl Target {
    pub async fn connect(worker: Arc<Worker>, target: &Value, endpoint: &str) -> Result<Self> {
        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace(string(target, "namespace")?)
                .with_router_endpoint(endpoint)?
                .disable_multicast_scouting()
                .with_json("namespace", string(target, "transport_scope")?)
                .build()
                .await?,
        );
        let node = Arc::new(
            context
                .create_node(&format!("mcp_{}", id().replace('-', "_")))
                .build()
                .await?,
        );
        Ok(Self {
            id: string(target, "target_id")?.into(),
            namespace: string(target, "namespace")?.into(),
            directory: PathBuf::from(string(target, "recording_directory")?),
            node,
            worker,
            _context: context,
        })
    }
    pub fn absolute(&self, name: &str) -> Result<String> {
        let name = if name.starts_with('/') {
            name.to_owned()
        } else {
            format!("{}/{}", self.namespace, name)
        };
        ensure!(
            name.starts_with(&format!("{}/", self.namespace)),
            "Name is outside target namespace"
        );
        Ok(name)
    }
}
pub(super) struct Observation {
    pub target: String,
    pub topics: HashMap<String, CachedSubscription<DynamicPayload>>,

    pub bytes: usize,
}
fn nonzero(value: u64) -> Result<NonZeroUsize> {
    NonZeroUsize::new(value.try_into()?).ok_or_else(|| eyre!("Limit must be positive"))
}
async fn subscribe(
    target: &Target,
    topic: &str,
    retention: RetentionPolicy,
) -> Result<CachedSubscription<DynamicPayload>> {
    let clock = target.worker.clock.clone();
    CachedSubscriptionBuilder::new(target.node.clone(), target.absolute(topic)?)?
        .retention(retention)
        .receipt_clock(Arc::new(move || Time::from_nanos(clock.load())))
        .build_dynamic()
        .await
        .map_err(Into::into)
}
pub(super) fn type_info(info: &ros_z::TypeInfo) -> Value {
    json!({"name":info.name,"schema_hash":info.hash.to_hash_string()})
}
fn parameter_client(
    target: &Target,
    args: &Value,
) -> Result<ros_z::parameter::remote::RemoteParameterClient> {
    Ok(ros_z::parameter::remote::RemoteParameterClient::new(
        target.node.clone(),
        target.absolute(string(args, "node")?)?,
    )?)
}
pub(super) async fn read_tool(state: &Arc<State>, tool: &str, args: &Value) -> Result<Value> {
    if tool == "rosz_unobserve" {
        let observation = state
            .observations
            .lock()
            .await
            .remove(string(args, "observation_id")?)
            .ok_or_else(|| eyre!("Unknown observation"))?;
        drop(observation);
        return Ok(json!({"released":true}));
    }
    if let Some(cursor) = args["cursor"].as_str() {
        return page(state, args, cursor);
    }
    let target = state.target(args).await?;
    match tool {
        "rosz_graph" => {
            let graph = target.node.graph().lock();
            let prefix = args["name_prefix"].as_str().unwrap_or("");
            let kind = args["kind"].as_str().unwrap_or("all");
            let mut rows = Vec::new();
            if kind == "all" || kind == "nodes" {
                rows.extend(
                    graph
                        .nodes()
                        .filter(|n| n.fully_qualified_name().starts_with(prefix))
                        .map(|n| json!({"node":{"name":n.name,"namespace":n.namespace}})),
                );
            }
            if kind != "nodes" {
                rows.extend(graph.endpoints().filter(|e|e.topic.starts_with(prefix)).filter(|e|kind == "all" || match e.kind {ros_z::entity::EndpointKind::Publisher|ros_z::entity::EndpointKind::Subscription=>kind=="topics",_=>kind=="services"}).map(|e|json!({"endpoint":{"name":e.topic,"node":e.node.name,"namespace":e.node.namespace,"kind":match e.kind{ros_z::entity::EndpointKind::Publisher=>"publisher",ros_z::entity::EndpointKind::Subscription=>"subscription",ros_z::entity::EndpointKind::Service=>"service",ros_z::entity::EndpointKind::Client=>"client"},"type_info":type_info(&e.type_info),"qos":{"encoded":e.qos.encode()}}})));
            }
            rows.sort_by_key(Value::to_string);
            paginate(
                state,
                args,
                rows,
                json!({"revision":graph.revision(),"observed_at_wall_ns":wall().to_string()}),
                "graph",
            )
        }

        "rosz_schema" => {
            let name = target.absolute(string(args, "name")?)?;
            match args["kind"].as_str().unwrap() {
                "topic" => {
                    let discovered = target
                        .node
                        .discover_topic_schema(&name, Duration::from_secs(2))
                        .await?;
                    if let Some(hash) = args["schema_hash"].as_str() {
                        ensure!(
                            hash == discovered.schema_hash.to_hash_string(),
                            "Schema hash changed"
                        );
                    }
                    let example = DynamicPayload::default_for_schema(discovered.schema.clone())?;
                    Ok(
                        json!({"type_info":type_info(&discovered.type_info()),"value_schema":{"encoding":"ros-z-lossless-json","canonical":discovered.schema,"wide_integers":"decimal strings","bytes":"arrays","enums":"variant_index, variant_name, payload","optionals":"None is null; Some is {\"$type\":\"some\",\"value\":...}"},"examples":[ros_z::dynamic::dynamic_payload_to_json(&example,JSON)]}),
                    )
                }
                "service" => {
                    let discovered = target
                        .node
                        .discover_service_schema(&name, Duration::from_secs(2))
                        .await?;
                    Ok(
                        json!({"type_info":type_info(&discovered.type_info),"request_schema":{"encoding":"ros-z-lossless-json","canonical":discovered.request},"response_schema":{"encoding":"ros-z-lossless-json","canonical":discovered.response}}),
                    )
                }
                "parameters" => {
                    let client = ros_z::parameter::remote::RemoteParameterClient::new(
                        target.node.clone(),
                        name,
                    )?;
                    let info = client.get_type_info().await?;
                    ensure!(info.success, "{}", info.message);
                    Ok(json!({"type_info":{"name":info.type_name,"schema_hash":info.schema_hash}}))
                }
                _ => unreachable!(),
            }
        }
        "rosz_observe" => {
            let duration = args["retention"]["duration_ms"].as_u64().unwrap_or(60000);
            let samples = args["retention"]["max_samples_per_topic"]
                .as_u64()
                .unwrap_or(16384);
            let bytes = args["retention"]["max_bytes"]
                .as_u64()
                .unwrap_or(64 * 1024 * 1024);
            let clock = args["retention"]["clock"]
                .as_str()
                .unwrap_or("simulation_receipt");
            ensure!(
                samples <= 100_000 && bytes <= 64 * 1024 * 1024,
                "Observation exceeds server limits"
            );
            let mut observations = state.observations.lock().await;
            ensure!(
                observations.len() < 128
                    && observations.values().map(|o| o.bytes).sum::<usize>() + bytes as usize
                        <= 256 * 1024 * 1024,
                "Observation memory budget exhausted"
            );
            let sources = args["sources"].as_array().unwrap();
            let topics: std::collections::BTreeSet<_> = sources
                .iter()
                .map(|s| string(s, "topic"))
                .collect::<Result<_>>()?;
            let policy = RetentionPolicy::time_window_with_max_samples(
                Duration::from_millis(duration),
                nonzero(samples)?,
            )?
            .with_max_bytes(nonzero(bytes / topics.len() as u64)?)
            .with_clock(match clock {
                "source" => RetentionClock::Source,
                "wall_receipt" => RetentionClock::WallReceipt,
                _ => RetentionClock::Receipt,
            });
            let mut subscriptions = HashMap::new();
            for topic in topics {
                subscriptions.insert(topic.to_owned(), subscribe(&target, topic, policy).await?);
            }
            let identifier = id();
            let retention = json!({"duration_ms":duration,"max_samples_per_topic":samples,"max_bytes":bytes,"clock":clock});
            observations.insert(
                identifier.clone(),
                Observation {
                    target: target.id.clone(),
                    topics: subscriptions,
                    bytes: bytes as usize,
                },
            );
            Ok(
                json!({"observation_id":identifier,"state":"ready","retention":retention,"coverage":[]}),
            )
        }
        "rosz_parameter_read" => {
            let client = parameter_client(&target, args)?;
            if let Some(path) = args["path"].as_str() {
                let result = client.get_value(path).await?;
                ensure!(result.success, "{}", result.message);
                Ok(
                    json!({"node":args["node"],"revision":result.revision,"layers":[result.effective_source_layer],"values":serde_json::from_str::<Value>(result.value_json.as_ref())?}),
                )
            } else {
                let snapshot = serde_json::to_value(client.get_snapshot().await?)?;
                ensure!(snapshot["success"] == true, "{}", snapshot["message"]);
                Ok(
                    json!({"node":args["node"],"revision":snapshot["revision"],"layers":snapshot["layers"],"values":serde_json::from_str::<Value>(snapshot["value_json"].as_str().unwrap_or("null"))?}),
                )
            }
        }
        "logs_read" => logs(state, &target, args),
        _ => bail!("Unknown read tool {tool}"),
    }
}
pub(super) async fn write_tool(
    state: &Arc<State>,
    tool: &str,
    args: &Value,
    operation: &Operation,
) -> Result<Value> {
    let target = state.target(args).await?;
    ensure!(
        !target
            .worker
            .stopped
            .load(std::sync::atomic::Ordering::Acquire),
        "Target is no longer live"
    );
    match tool {
        "rosz_parameter_write" => {
            let client = parameter_client(&target, args)?;
            let action = &args["action"];
            let result = match action["kind"].as_str().unwrap() {
                "set" => {
                    let writes=action["writes"].as_array().unwrap().iter().map(|write|serde_json::from_value(json!({"path":write["path"],"value_json":write["value"].to_string(),"target_layer":write["layer"]}))).collect::<std::result::Result<Vec<ros_z::parameter::remote::types::NodeParameterWriteJson>,_>>()?;
                    serde_json::to_value(
                        client
                            .set_json_atomically(writes, action["expected_revision"].as_u64())
                            .await?,
                    )?
                }
                "reset" => serde_json::to_value(
                    client
                        .reset(
                            string(action, "path")?,
                            string(action, "layer")?,
                            action["expected_revision"].as_u64(),
                        )
                        .await?,
                )?,
                "reload" => serde_json::to_value(client.reload().await?)?,
                _ => unreachable!(),
            };
            ensure!(
                result["success"] == true,
                "Parameter update rejected: {}",
                result["message"]
            );
            Ok(
                json!({"node":args["node"],"committed_revision":result["committed_revision"],"changed_paths":result["changed_paths"]}),
            )
        }
        "rosz_publish" => {
            let topic = target.absolute(string(args, "topic")?)?;
            let discovered = target
                .node
                .discover_topic_schema(&topic, Duration::from_secs(2))
                .await?;
            ensure!(
                string(args, "schema_hash")? == discovered.schema_hash.to_hash_string(),
                "Schema hash changed"
            );
            let payload = ros_z::dynamic::json::dynamic_payload_from_json(
                discovered.schema.clone(),
                &args["payload"],
            )?;
            let mut builder =
                target
                    .node
                    .dynamic_publisher(&topic, discovered.type_info(), discovered.schema);
            if let Some(qos) = args.get("qos") {
                builder = builder.qos(ros_z::qos::QosProfile::decode(string(qos, "encoded")?)?);
            }
            let publisher = builder.build().await?;
            operation.check()?;
            operation.stage("publishing");
            let repeat = args["delivery"]["kind"] == "repeat";
            let deadline = Instant::now()
                + Duration::from_millis(args["delivery"]["wall_duration_ms"].as_u64().unwrap_or(0));
            let mut count = 0;
            loop {
                operation.check()?;
                publisher.publish(&payload).await?;
                count += 1;
                if !repeat || Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_secs_f64(
                    1.0 / args["delivery"]["rate_hz"].as_f64().unwrap(),
                ))
                .await;
            }
            Ok(
                json!({"publisher_id":ros_z::attachment::EndpointGlobalId::from(publisher.entity()).to_string(),"published_samples":count}),
            )
        }
        "rosz_call" => {
            let service = target.absolute(string(args, "service")?)?;
            let schema = target
                .node
                .discover_service_schema(&service, Duration::from_secs(2))
                .await?;
            ensure!(
                string(args, "schema_hash")? == schema.type_info.hash.to_hash_string(),
                "Schema hash changed"
            );
            let request = ros_z::dynamic::json::dynamic_payload_from_json(
                schema.request.clone(),
                &args["request"],
            )?;
            let client = target.node.dynamic_service_client(&service, schema).await?;
            operation.check()?;
            operation.stage("dispatched");
            let response = client
                .call(
                    &request,
                    Duration::from_millis(args["wall_timeout_ms"].as_u64().unwrap()),
                )
                .await?;
            Ok(
                json!({"service":args["service"],"schema_hash":args["schema_hash"],"response":ros_z::dynamic::dynamic_payload_to_json(&response,JSON)}),
            )
        }
        _ => bail!("Unknown write tool {tool}"),
    }
}
fn selected(payload: &DynamicPayload, source: &Value) -> Result<(Value, &'static str)> {
    let path: ValuePath = source["path"].as_str().unwrap_or("").parse()?;
    match path.select(payload) {
        Ok(value) => Ok((value.to_json(JSON), "present")),
        Err(error) if error.is_unavailable() => Ok((Value::Null, "absent_path")),
        Err(error) => Err(error.into()),
    }
}
fn sample(
    target: &Target,
    source: &Value,
    record: &ros_z_debug::SampleRecord<DynamicPayload>,
    clock: &str,
) -> Result<Value> {
    if let Some(hash) = source["schema_hash"].as_str() {
        ensure!(
            hash == record.metadata.type_info.hash.to_hash_string(),
            "Observed schema changed"
        );
    }
    let (value, status) = selected(&record.value, source)?;
    let simulation = record.receipt_time.as_nanos() - target.worker.origin;
    let time = match clock {
        "source" => record.source_time.as_nanos(),
        "wall_receipt" => record.wall_receipt_time.as_nanos(),
        _ => simulation,
    };
    Ok(
        json!({"topic":source["topic"],"path":source["path"].as_str().unwrap_or(""),"type_info":type_info(&record.metadata.type_info),"publication_id":format!("{}:{}",target.worker.id,record.publication_id),"source_time_ns":record.source_time.as_nanos().to_string(),"source_clock":"unknown","receipts":[{"collector":"observation","simulation_ns":simulation.to_string(),"wall_ns":record.wall_receipt_time.as_nanos().to_string()}],"selected_time_ns":time.to_string(),"value":value,"value_status":status,"provenance":["observation"]}),
    )
}
fn in_range(sample: &Value, range: &Value) -> Result<bool> {
    let time = string(sample, "selected_time_ns")?.parse::<i64>()?;
    Ok(time >= string(range, "start_ns")?.parse::<i64>()?
        && time < string(range, "end_ns")?.parse::<i64>()?)
}
pub(super) async fn read(state: &Arc<State>, args: &Value) -> Result<Value> {
    read_cancellable(state, args, None).await
}
pub(super) async fn read_cancellable(
    state: &Arc<State>,
    args: &Value,
    cancellation: Option<tokio_util::sync::CancellationToken>,
) -> Result<Value> {
    if let Some(cursor) = args["cursor"].as_str() {
        return page(state, args, cursor);
    }
    let target = state.target(args).await?;
    let latest = args["query"]["kind"] == "latest";
    let range = &args["query"]["range"];
    let clock = range["clock"].as_str().unwrap_or("simulation_receipt");
    if !latest {
        ensure!(
            string(range, "start_ns")?.parse::<i64>()? < string(range, "end_ns")?.parse::<i64>()?,
            "History range must be increasing"
        );
    }
    let sources = args["sources"].as_array().unwrap();
    let mut rows = Vec::new();
    let mut coverage = Vec::new();
    for source in sources {
        let topic = string(source, "topic")?;
        let cached: Vec<_> = state
            .observations
            .lock()
            .await
            .values()
            .filter(|o| o.target == target.id)
            .filter_map(|o| o.topics.get(topic).cloned())
            .collect();
        let observed = !cached.is_empty();
        let mut records = Vec::new();
        if latest {
            ensure!(
                !args["allow_recording_fallback"].as_bool().unwrap_or(false),
                "Latest reads cannot use recordings"
            );
            let transient;
            let caches = if observed {
                &cached
            } else {
                transient = vec![subscribe(&target, topic, RetentionPolicy::LatestOnly).await?];
                &transient
            };
            let deadline = Instant::now()
                + Duration::from_millis(args["query"]["wait_ms"].as_u64().unwrap_or(1000));
            loop {
                records = caches.iter().filter_map(|cache| cache.latest()).collect();
                records.sort_by_key(|s| s.wall_receipt_time);
                if let Some(age) = args["query"]["max_age_ms"].as_u64() {
                    records.retain(|s| {
                        wall().saturating_sub(s.wall_receipt_time.as_nanos())
                            <= age.saturating_mul(1_000_000) as i64
                    });
                }
                if !records.is_empty() || Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            records = records.into_iter().rev().take(1).collect();
        } else {
            for cache in &cached {
                records
                    .extend(cache.window(Time::from_nanos(i64::MIN), Time::from_nanos(i64::MAX)));
            }
        }
        for record in records {
            let value = sample(&target, source, &record, clock)?;
            if latest || in_range(&value, range)? {
                rows.push(value);
            }
        }
        coverage.push(json!({"topic":topic,"path":source["path"].as_str().unwrap_or(""),"subscription_state":if !observed{"not_observed"}else if target.worker.stopped.load(std::sync::atomic::Ordering::Acquire){"closed"}else{"active"},"intervals":if latest{vec![]}else{vec![json!({"range":range,"status":if observed{"unknown"}else{"not_observed"}})]},"loss_status":"unknown","warnings":if observed{vec!["Retention and transport loss may leave gaps; sample timestamps do not prove complete coverage."]}else{vec![]}}));
    }
    let checkpoint = if !latest && args["allow_recording_fallback"] == true {
        let (recorded, checkpoint) =
            super::archive::read(&target, sources, range, cancellation).await?;
        rows.extend(recorded);
        Some(checkpoint)
    } else {
        None
    };
    let mut unique = BTreeMap::<(String, String, String), Value>::new();
    for row in rows {
        let key = (
            string(&row, "topic")?.into(),
            string(&row, "path")?.into(),
            string(&row, "publication_id")?.into(),
        );
        if let Some(old) = unique.get_mut(&key) {
            for receipt in row["receipts"].as_array().unwrap() {
                if !old["receipts"].as_array().unwrap().contains(receipt) {
                    old["receipts"]
                        .as_array_mut()
                        .unwrap()
                        .push(receipt.clone());
                }
            }
            for provenance in row["provenance"].as_array().unwrap() {
                if !old["provenance"].as_array().unwrap().contains(provenance) {
                    old["provenance"]
                        .as_array_mut()
                        .unwrap()
                        .push(provenance.clone());
                }
            }
        } else {
            unique.insert(key, row);
        }
    }
    let mut rows: Vec<_> = unique.into_values().collect();
    rows.sort_by_key(|row| {
        row["selected_time_ns"]
            .as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or_default()
    });
    ensure!(
        rows.len() <= 100_000 && serde_json::to_vec(&rows)?.len() <= 64 * 1024 * 1024,
        "Query exceeds snapshot budget; narrow the time range"
    );
    let query_id = id();
    let mut metadata = json!({"query_id":query_id,"coverage":coverage,"warnings":[]});
    if let Some(checkpoint) = checkpoint {
        metadata["checkpoint"] = checkpoint;
    }
    paginate(state, args, rows, metadata, "samples")
}

// Freeze only paginated results. Ordinary polling must not consume cursor capacity.
fn paginate(
    state: &State,
    args: &Value,
    mut rows: Vec<Value>,
    metadata: Value,
    collection: &str,
) -> Result<Value> {
    let budget = args["max_bytes"].as_u64().unwrap_or(262144) as usize;
    for row in &mut rows {
        if collection == "samples" && serde_json::to_vec(row)?.len() > budget / 2 {
            let path = state.artifacts.join(format!("{}.json", id()));
            std::fs::write(&path, serde_json::to_vec(&row["value"])?)?;
            row.as_object_mut().unwrap().remove("value");
            row["value_status"] = json!("too_large");
            row["artifact"] = state.artifact(&path, "Value exceeding inline response limit")?;
        }
    }
    let key = id();
    let query = FrozenQuery {
        created: Instant::now(),
        arguments: args.clone(),
        rows,
        metadata,
        collection: collection.to_owned(),
    };
    let result = query_page(&query, args, &key, 0)?;
    if !result["next_cursor"].is_null() {
        let mut queries = state.queries.lock().unwrap();
        queries.retain(|_, q| q.created.elapsed() < Duration::from_secs(300));
        let retained_bytes: usize = queries
            .values()
            .map(|q| serde_json::to_vec(&q.rows).map_or(usize::MAX, |b| b.len()))
            .sum();
        ensure!(
            queries.len() < 128
                && retained_bytes.saturating_add(serde_json::to_vec(&query.rows)?.len())
                    <= 64 * 1024 * 1024,
            "Cursor memory budget exhausted; narrow the query or retry after expiry"
        );
        queries.insert(key, query);
    }
    Ok(result)
}
fn page(state: &State, args: &Value, cursor: &str) -> Result<Value> {
    let (key, offset) = cursor
        .rsplit_once(':')
        .ok_or_else(|| eyre!("Invalid cursor"))?;
    let queries = state.queries.lock().unwrap();
    let query = queries
        .get(key)
        .filter(|q| q.created.elapsed() < Duration::from_secs(300))
        .ok_or_else(|| eyre!("Cursor expired"))?;
    let mut signature = args.clone();
    signature.as_object_mut().unwrap().remove("cursor");
    ensure!(
        signature == query.arguments,
        "Cursor belongs to different query arguments"
    );
    query_page(query, args, key, offset.parse()?)
}
fn query_page(query: &FrozenQuery, args: &Value, key: &str, offset: usize) -> Result<Value> {
    ensure!(offset <= query.rows.len(), "Invalid cursor offset");
    let limit = args["limit"]
        .as_u64()
        .unwrap_or(if query.collection == "samples" {
            1000
        } else {
            100
        }) as usize;
    let budget = args["max_bytes"].as_u64().unwrap_or(262144) as usize;
    let mut result = query.metadata.clone();
    let mut bytes = serde_json::to_vec(&result)?.len() + key.len() + 128;
    let mut rows = Vec::new();
    for row in query.rows.iter().skip(offset).take(limit) {
        let size = serde_json::to_vec(row)?.len() + 1;
        if bytes + size > budget {
            ensure!(
                !rows.is_empty(),
                "Response metadata exceeds byte budget; increase max_bytes or request fewer sources"
            );
            break;
        }
        bytes += size;
        rows.push(row.clone());
    }
    let next = offset + rows.len();
    result["next_cursor"] = if next < query.rows.len() {
        json!(format!("{key}:{next}"))
    } else {
        Value::Null
    };
    match query.collection.as_str() {
        "samples" => {
            result["truncated"] = json!(next < query.rows.len());
            result["samples"] = json!(rows);
        }
        "graph" => {
            result["nodes"] = json!(
                rows.iter()
                    .filter_map(|r| r.get("node"))
                    .collect::<Vec<_>>()
            );
            result["endpoints"] = json!(
                rows.iter()
                    .filter_map(|r| r.get("endpoint"))
                    .collect::<Vec<_>>()
            );
        }
        field => result[field] = json!(rows),
    }
    Ok(result)
}

struct WatchedPredicate {
    target: Arc<Target>,
    source: Value,
    predicate: Value,
    cache: CachedSubscription<DynamicPayload>,
    baseline: Option<Value>,
}
pub(super) struct ConditionWatch {
    predicates: Vec<WatchedPredicate>,
    all: bool,
}
impl ConditionWatch {
    pub async fn new(state: &Arc<State>, condition: &Value) -> Result<Self> {
        let mut predicates = Vec::new();
        for predicate in condition["predicates"].as_array().unwrap() {
            let target = state.target(predicate).await?;
            let source = predicate["source"].clone();
            let topic = string(&source, "topic")?;
            let cached = state
                .observations
                .lock()
                .await
                .values()
                .filter(|o| o.target == target.id)
                .find_map(|o| o.topics.get(topic).cloned());
            let cache = match cached {
                Some(cache) => cache,
                None => subscribe(&target, topic, RetentionPolicy::LatestOnly).await?,
            };
            let baseline = cache
                .latest()
                .map(|record| selected(&record.value, &source).map(|(value, _)| value))
                .transpose()?;
            predicates.push(WatchedPredicate {
                target,
                source,
                predicate: predicate.clone(),
                cache,
                baseline,
            });
        }
        Ok(Self {
            predicates,
            all: condition["match"] == "all",
        })
    }
    pub fn evaluate(&mut self) -> Result<(bool, Vec<Value>)> {
        let mut matches = Vec::new();
        let mut samples = Vec::new();
        for watch in &mut self.predicates {
            let Some(record) = watch.cache.latest() else {
                matches.push(false);
                continue;
            };
            if let Some(age) = watch.predicate["max_age_ms"].as_u64() {
                if (wall() as i128 - record.wall_receipt_time.as_nanos() as i128)
                    > age as i128 * 1_000_000
                {
                    matches.push(false);
                    continue;
                }
            }
            let row = sample(&watch.target, &watch.source, &record, "simulation_receipt")?;
            let actual = (row["value_status"] == "present").then_some(&row["value"]);
            let matched = compare(actual, &watch.predicate, watch.baseline.as_ref());
            if watch.baseline.is_none() {
                watch.baseline = actual.cloned();
            }
            matches.push(matched);
            samples.push(row);
        }
        Ok((
            if self.all {
                matches.into_iter().all(|v| v)
            } else {
                matches.into_iter().any(|v| v)
            },
            samples,
        ))
    }
}
fn compare(actual: Option<&Value>, predicate: &Value, baseline: Option<&Value>) -> bool {
    let expected = &predicate["value"];
    match predicate["operator"].as_str().unwrap_or("") {
        "exists" => actual.is_some(),
        "eq" => actual == Some(expected),
        "ne" => actual.is_some_and(|a| a != expected),
        "changed" => actual.zip(baseline).is_some_and(|(a, b)| a != b),
        "lt" | "le" | "gt" | "ge" => {
            // Preserve integer precision, including decimal-encoded i64/u64.
            let integer = |v: &Value| {
                v.as_str()
                    .and_then(|s| s.parse::<i128>().ok())
                    .or_else(|| v.as_i64().map(i128::from))
                    .or_else(|| v.as_u64().map(i128::from))
            };
            let ordering = actual.zip(Some(expected)).and_then(|(a, b)| {
                if let Some((a, b)) = integer(a).zip(integer(b)) {
                    return Some(a.cmp(&b));
                }
                a.as_f64()
                    .zip(b.as_f64())
                    .and_then(|(a, b)| a.partial_cmp(&b))
            });
            ordering.is_some_and(|order| match predicate["operator"].as_str().unwrap() {
                "lt" => order.is_lt(),
                "le" => order.is_le(),
                "gt" => order.is_gt(),
                _ => order.is_ge(),
            })
        }
        _ => false,
    }
}
fn logs(state: &State, target: &Target, args: &Value) -> Result<Value> {
    use std::io::{BufRead, BufReader, Read};
    let file = std::fs::File::open(target.directory.join("logs.jsonl"))?;
    // Freeze at the file length observed at query start; never consume a partial tail line.
    let length = file.metadata()?.len();
    let contains = args["contains"].as_str().unwrap_or("");
    let levels = ["error", "warn", "info", "debug", "trace"];
    let max_level = args["min_level"]
        .as_str()
        .and_then(|l| levels.iter().position(|name| *name == l))
        .unwrap_or(4)
        + 1;
    let mut records = Vec::new();
    let mut bytes = 0;
    for line in BufReader::new(file.take(length)).lines() {
        let line = line?;
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            break;
        };
        let level = record["level"].as_u64().unwrap_or(5) as usize;
        if !line.contains(contains) || level > max_level {
            continue;
        }
        if let Some(targets) = args["targets"].as_array() {
            if !targets.contains(&record["target"]) {
                continue;
            }
        }
        let simulation = record["simulation_time"]
            .as_i64()
            .ok_or_else(|| eyre!("Missing log timestamp"))?
            - target.worker.origin;
        let wall = record["wall_time"]
            .as_i64()
            .ok_or_else(|| eyre!("Missing log timestamp"))?;
        if let Some(range) = args.get("range") {
            let time = match range["clock"].as_str().unwrap() {
                "source" => record["simulation_time"].as_i64().unwrap(),
                "wall_receipt" => wall,
                _ => simulation,
            };
            if time < string(range, "start_ns")?.parse()?
                || time >= string(range, "end_ns")?.parse()?
            {
                continue;
            }
        }
        let entry = json!({"level":levels[level.clamp(1,5)-1],"target":record["target"],"message":record["message"],"fields":serde_json::from_str::<Value>(record["fields"].as_str().unwrap_or("{}"))?,"source_time_ns":record["simulation_time"].as_i64().unwrap().to_string(),"simulation_time_ns":simulation.to_string(),"wall_time_ns":wall.to_string()});
        bytes += serde_json::to_vec(&entry)?.len();
        ensure!(
            bytes <= 16 * 1024 * 1024 && records.len() < 100_000,
            "Log query exceeds budget; narrow its filters"
        );
        records.push(entry);
    }
    paginate(
        state,
        args,
        records,
        json!({"coverage_status":"partial"}),
        "entries",
    )
}
pub(super) fn recorded_sample(
    target: &Target,
    source: &Value,
    message: &mcap::Message<'_>,
    receipt: &crate::telemetry::recording::Receipt,
    range: &Value,
) -> Result<Option<Value>> {
    let schema: ros_z::dynamic::Schema = Arc::new(serde_json::from_slice(
        &message
            .channel
            .schema
            .as_ref()
            .ok_or_else(|| eyre!("Missing recorded schema"))?
            .data,
    )?);
    if let Some(hash) = source["schema_hash"].as_str() {
        ensure!(
            message
                .channel
                .metadata
                .get("ros_z.schema_hash")
                .map(String::as_str)
                == Some(hash),
            "Recorded schema differs from requested schema"
        );
    }
    let payload = ros_z::dynamic::DynamicCdrCodec::decode(&message.data, &schema)?;
    let (value, status) = selected(&payload, source)?;
    let simulation = message.log_time as i64 - target.worker.origin;
    let time = match range["clock"].as_str().unwrap() {
        "source" => message.publish_time as i64,
        "wall_receipt" => receipt.wall_time,
        _ => simulation,
    };
    let sample = json!({"topic":source["topic"],"path":source["path"].as_str().unwrap_or(""),"type_info":{"name":message.channel.schema.as_ref().unwrap().name,"schema_hash":message.channel.metadata.get("ros_z.schema_hash")},"publication_id":format!("{}:{}#{}",target.worker.id,receipt.publisher,receipt.sequence),"source_time_ns":message.publish_time.to_string(),"source_clock":"unknown","receipts":[{"collector":"recording","simulation_ns":simulation.to_string(),"wall_ns":receipt.wall_time.to_string()}],"selected_time_ns":time.to_string(),"value":value,"value_status":status,"provenance":["recording"]});
    Ok(in_range(&sample, range)?.then_some(sample))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditions_preserve_integer_precision_and_absent_paths_are_not_values() {
        assert!(compare(
            Some(&json!("9007199254740993")),
            &json!({"operator":"gt","value":"9007199254740992"}),
            None
        ));
        assert!(!compare(None, &json!({"operator":"ne","value":null}), None));
        assert!(!compare(None, &json!({"operator":"exists"}), None));
        assert!(compare(
            Some(&json!(2)),
            &json!({"operator":"changed"}),
            Some(&json!(1))
        ));
    }
    #[test]
    fn frozen_pages_are_stable_and_bounded() {
        let query = FrozenQuery {
            created: Instant::now(),
            arguments: json!({}),
            rows: vec![json!({"value":1}), json!({"value":2}), json!({"value":3})],
            metadata: json!({"query_id":"test","coverage":[]}),
            collection: "samples".into(),
        };
        let args = json!({"limit":2,"max_bytes":1024});
        let first = query_page(&query, &args, "query", 0).unwrap();
        assert_eq!(first["next_cursor"], "query:2");
        assert_eq!(query_page(&query, &args, "query", 0).unwrap(), first);
        let last = query_page(&query, &args, "query", 2).unwrap();
        assert_eq!(last["samples"], json!([{"value":3}]));
        assert!(last["next_cursor"].is_null());
    }
}

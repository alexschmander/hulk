//! Stream selected history into local artifacts without allocating a pagination snapshot.
use super::{
    archive, rosz,
    server::{Operation, State, id, string},
    verification::Verification,
};
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::{dynamic::DynamicPayload, time::Time};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    io::{BufWriter, Write},
    sync::Arc,
};

type Samples = Vec<Arc<ros_z_debug::SampleRecord<DynamicPayload>>>;
const MAX_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;

struct Output<W: Write> {
    writer: W,
    format: String,
    bytes: u64,
    max_bytes: u64,
    samples: u64,
}
impl<W: Write> Output<W> {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        ensure!(
            self.bytes.saturating_add(bytes.len() as u64) <= self.max_bytes,
            "Export exceeds max_bytes; narrow the range or increase the export limit"
        );
        self.writer.write_all(bytes)?;
        self.bytes += bytes.len() as u64;
        Ok(())
    }
    fn sample(&mut self, row: &Value) -> Result<()> {
        if self.format == "jsonl" {
            self.write(&serde_json::to_vec(row)?)?;
            self.write(b"\n")?;
        } else {
            let fields = [
                "topic",
                "path",
                "publication_id",
                "selected_time_ns",
                "source_time_ns",
                "type_info",
                "receipts",
                "value_status",
                "value",
                "provenance",
            ];
            if self.samples == 0 {
                self.write(format!("{}\n", fields.join(",")).as_bytes())?;
            }
            let line = fields
                .iter()
                .map(|key| {
                    let text = if matches!(*key, "type_info" | "receipts" | "value" | "provenance")
                    {
                        row[key].to_string()
                    } else {
                        row[key].as_str().unwrap_or("").to_owned()
                    };
                    format!("\"{}\"", text.replace('"', "\"\""))
                })
                .collect::<Vec<_>>()
                .join(",")
                + "\n";
            self.write(line.as_bytes())?;
        }
        self.samples += 1;
        Ok(())
    }
}

pub(super) async fn execute(
    state: &Arc<State>,
    args: &Value,
    operation: &Operation,
) -> Result<Value> {
    // A full retained-history snapshot may pin payloads while live caches rotate.
    let _guard = state
        .export
        .try_lock()
        .map_err(|_| eyre!("A history export is already running"))?;
    let target = state.target(args).await?;
    let sources = args["sources"].as_array().unwrap();
    for source in sources {
        target.absolute(string(source, "topic")?)?;
    }
    let range = &args["range"];
    let selection =
        archive::Selection::new(&target.namespace, target.worker.origin, sources, range)?;
    let recording = args["data_source"] == "recording";
    operation.stage(if recording {
        "checkpointing"
    } else {
        "snapshotting"
    });
    let checkpoint = if recording {
        Some(archive::checkpoint(&target).await?)
    } else {
        None
    };
    let mut snapshots = BTreeMap::<String, Samples>::new();
    if !recording {
        let observations = state.observations.lock().await;
        let mut bytes = 0usize;
        for source in sources {
            let topic = string(source, "topic")?;
            let absolute = target.absolute(topic)?;
            if snapshots.contains_key(topic) {
                continue;
            }
            let mut seen = HashSet::new();
            let mut samples = Vec::new();
            for cache in observations
                .values()
                .filter(|o| o.target == target.id)
                .filter_map(|o| o.topics.get(&absolute))
            {
                for record in cache.window(Time::from_nanos(i64::MIN), Time::from_nanos(i64::MAX)) {
                    if seen.insert(record.publication_id) {
                        bytes = bytes.saturating_add(
                            record.retained_bytes + std::mem::size_of_val(record.as_ref()),
                        );
                        ensure!(
                            bytes <= MAX_SNAPSHOT_BYTES,
                            "Observation export snapshot exceeds 256 MiB; export fewer sources"
                        );
                        samples.push(record);
                    }
                }
            }
            snapshots.insert(topic.into(), samples);
        }
    }
    operation.check()?;
    operation.stage("exporting");
    let state = state.clone();
    let args = args.clone();
    let operation = operation.clone();
    tokio::task::spawn_blocking(move || {
        let sources=args["sources"].as_array().unwrap();
        let range=&args["range"];
        let format=args["format"].as_str().unwrap_or("jsonl");
        let mut file=tempfile::NamedTempFile::new_in(&state.artifacts)?;
        let mut output=Output {writer:BufWriter::new(file.as_file_mut()),format:format.into(),bytes:0,max_bytes:args["max_bytes"].as_u64().unwrap_or(1024*1024*1024),samples:0};
        let mut verification=Verification::default();
        let mut decoder=rosz::RecordingDecoder::default();
        let mut emit=|row:Value| -> Result<()> {operation.check()?;verification.observe(&row);output.sample(&row)};
        let scan=if let Some(checkpoint)=&checkpoint {
            Some(archive::scan(&target.directory,checkpoint,&selection,&operation.cancel,|source,message,receipt| {
                if let Some(row)=decoder.sample(&target,source,message,receipt,range)? {emit(row)?;}
                Ok(())
            })?.json(sources))
        }else{
            for source in sources {
                for record in &snapshots[string(source,"topic")?] {
                    operation.check()?;
                    let row=rosz::sample(&target,source,record,string(range,"clock")?)?;
                    if rosz::in_range(&row,range)? {emit(row)?;}
                }
            }
            None
        };
        if format == "csv" && output.samples == 0 {
            output.write(b"topic,path,publication_id,selected_time_ns,source_time_ns,type_info,receipts,value_status,value,provenance\n")?;
        }
        output.writer.flush()?;
        let count=output.samples;
        drop(output);
        operation.check()?;
        let checkpoint=checkpoint.as_ref().map(|c|archive::metadata(&target,c));
        let summary=verification.finish(checkpoint.as_ref(),state.parameter_revisions(&target.id));
        let metadata=json!({"target_id":target.id,"data_source":if recording{"recording"}else{"observations"},
            "range":range,"sources":sources,"sample_count":count,"format":format,
            "order":if recording{"recording ingestion order; no global timestamp sort"}else{"source then retained subscription order; no global timestamp sort"},
            "checkpoint":checkpoint,"archive_scan":scan,"verification":summary,
            "warnings":["Exported samples do not prove complete coverage. Retention, subscription timing, and recording drops may leave gaps."]});
        let stem=id();
        let path=state.artifacts.join(format!("{stem}.{format}"));
        let metadata_path=state.artifacts.join(format!("{stem}.metadata.json"));
        // Publish only completed files. A failed/cancelled export drops its temporary file.
        let mut metadata_file=tempfile::NamedTempFile::new_in(&state.artifacts)?;
        serde_json::to_writer_pretty(metadata_file.as_file_mut(),&metadata)?;
        metadata_file.persist(&metadata_path)?;
        file.persist(&path)?;
        Ok(json!({"artifact":state.artifact(&path,"Exported ROS-Z history")?,
            "metadata_artifact":state.artifact(&metadata_path,"Export coverage and verification evidence")?,
            "sample_count":count,"data_source":metadata["data_source"],"verification":summary,
            "archive_scan":scan,"checkpoint":checkpoint,"warnings":metadata["warnings"]}))
    }).await.map_err(|e|eyre!("History exporter: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streams_beyond_inline_query_limits() {
        let row = json!({"value":"x".repeat(1024)});
        let mut output = Output {
            writer: std::io::sink(),
            format: "jsonl".into(),
            bytes: 0,
            max_bytes: 128 * 1024 * 1024,
            samples: 0,
        };
        for _ in 0..100_001 {
            output.sample(&row).unwrap();
        }
        assert!(output.bytes > 64 * 1024 * 1024);
        assert_eq!(output.samples, 100_001);
    }

    #[test]
    fn jsonl_and_csv_preserve_values_and_enforce_disk_budget() {
        let row = json!({"topic":"a,\"b","path":"","value":{"large":"9007199254740993","text":"line\nbreak"}});
        let mut out = Output {
            writer: Vec::new(),
            format: "jsonl".into(),
            bytes: 0,
            max_bytes: 1024,
            samples: 0,
        };
        out.sample(&row).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&out.writer).unwrap(), row);
        out.max_bytes = out.bytes;
        assert!(out.sample(&row).is_err());
        let mut csv = Output {
            writer: Vec::new(),
            format: "csv".into(),
            bytes: 0,
            max_bytes: 1024,
            samples: 0,
        };
        csv.sample(&row).unwrap();
        let csv = String::from_utf8(csv.writer).unwrap();
        assert!(csv.contains("\"a,\"\"b\""));
        assert!(csv.contains("9007199254740993"));
        assert_eq!(csv.lines().count(), 2);
    }
}

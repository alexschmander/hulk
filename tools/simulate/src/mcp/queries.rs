//! Frozen pages own their budget until completion, explicit release, or expiry.
use super::server::id;
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_QUERIES: usize = 128;
const RETRY_BYTES: usize = 8 * 1024 * 1024;
const TTL: Duration = Duration::from_secs(300);

struct FrozenQuery {
    created: Instant,
    arguments: Value,
    rows: Vec<Value>,
    metadata: Value,
    collection: String,
    bytes: usize,
}
struct CompletedPage {
    key: String,
    cursor: String,
    arguments: Value,
    result: Value,
    created: Instant,
    bytes: usize,
}

#[derive(Default)]
pub(super) struct QueryStore {
    active: HashMap<String, FrozenQuery>,
    completed: VecDeque<CompletedPage>,
}
impl QueryStore {
    fn expire(&mut self) {
        self.active.retain(|_, query| query.created.elapsed() < TTL);
        self.completed.retain(|page| page.created.elapsed() < TTL);
    }
    fn bytes(&self) -> usize {
        self.active.values().map(|q| q.bytes).sum()
    }
    fn retry_bytes(&self) -> usize {
        self.completed.iter().map(|p| p.bytes).sum()
    }

    pub fn usage(&mut self) -> Value {
        self.expire();
        json!({"active_queries":self.active.len(),"snapshot_bytes":self.bytes(),
            "max_queries":MAX_QUERIES,"max_snapshot_bytes":MAX_BYTES,
            "completed_pages":self.completed.len(),"retry_bytes":self.retry_bytes(),
            "max_retry_bytes":RETRY_BYTES,"ttl_seconds":TTL.as_secs()})
    }
    pub fn release(&mut self, key: &str) -> Value {
        self.expire();
        let bytes = self.active.remove(key).map_or(0, |query| query.bytes);
        let before = self.retry_bytes();
        self.completed.retain(|page| page.key != key);
        let freed_bytes = bytes + before - self.retry_bytes();
        json!({"query_id":key,"released":true,"freed_bytes":freed_bytes,"query_budget":self.usage()})
    }
    pub fn insert(
        &mut self,
        args: &Value,
        rows: Vec<Value>,
        mut metadata: Value,
        collection: &str,
    ) -> Result<Value> {
        self.expire();
        let key = id();
        metadata["query_id"] = json!(key);
        let bytes = serde_json::to_vec(&rows)?.len()
            + serde_json::to_vec(&metadata)?.len()
            + serde_json::to_vec(args)?.len();
        let query = FrozenQuery {
            created: Instant::now(),
            arguments: args.clone(),
            rows,
            metadata,
            collection: collection.into(),
            bytes,
        };
        let result = query_page(&query, args, &key, 0)?;
        if !result["next_cursor"].is_null() {
            ensure!(
                self.active.len() < MAX_QUERIES && self.bytes().saturating_add(bytes) <= MAX_BYTES,
                "Cursor memory budget exhausted; finish pages or call query_release; inspect server_info.query_budget"
            );
            self.active.insert(key, query);
        }
        Ok(result)
    }
    pub fn page(&mut self, args: &Value, cursor: &str) -> Result<Value> {
        self.expire();
        let (key, offset) = cursor
            .rsplit_once(':')
            .ok_or_else(|| eyre!("Invalid cursor"))?;
        let mut signature = args.clone();
        signature.as_object_mut().unwrap().remove("cursor");
        if let Some(page) = self.completed.iter().find(|page| page.cursor == cursor) {
            ensure!(
                signature == page.arguments,
                "Cursor belongs to different query arguments"
            );
            return Ok(page.result.clone());
        }
        let query = self
            .active
            .get(key)
            .ok_or_else(|| eyre!("Cursor completed, released, or expired"))?;
        ensure!(
            signature == query.arguments,
            "Cursor belongs to different query arguments"
        );
        let result = query_page(query, args, key, offset.parse()?)?;
        if result["next_cursor"].is_null() {
            let query = self.active.remove(key).unwrap();
            let bytes = serde_json::to_vec(&result)?.len() + serde_json::to_vec(&signature)?.len();
            // Preserve last-page retries without retaining the full snapshot.
            self.completed.push_back(CompletedPage {
                key: key.into(),
                cursor: cursor.into(),
                arguments: signature,
                result: result.clone(),
                created: query.created,
                bytes,
            });
            while self.completed.len() > MAX_QUERIES || self.retry_bytes() > RETRY_BYTES {
                self.completed.pop_front();
            }
        }
        Ok(result)
    }
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
    ensure!(
        bytes <= budget,
        "Response metadata exceeds byte budget; increase max_bytes or request fewer sources"
    );
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
    ensure!(
        serde_json::to_vec(&result)?.len() <= budget,
        "Response exceeds byte budget; increase max_bytes"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_pages_release_snapshots_and_allow_last_page_retries() {
        let mut store = QueryStore::default();
        let args = json!({"limit":1});
        for _ in 0..256 {
            let first = store
                .insert(&args, vec![json!(1), json!(2)], json!({}), "samples")
                .unwrap();
            let cursor = first["next_cursor"].as_str().unwrap();
            assert_eq!(cursor.rsplit_once(':').unwrap().0, first["query_id"]);
            let page_args = json!({"limit":1,"cursor":cursor});
            assert!(store.bytes() > 0);
            let last = store.page(&page_args, cursor).unwrap();
            assert!(last["next_cursor"].is_null());
            assert_eq!(store.bytes(), 0);
            assert_eq!(store.page(&page_args, cursor).unwrap(), last);
        }
        assert_eq!(store.usage()["active_queries"], 0);
        assert_eq!(store.completed.len(), MAX_QUERIES);
        assert!(store.retry_bytes() <= RETRY_BYTES);
    }

    #[test]
    fn release_is_idempotent_and_expiry_reclaims_capacity() {
        let mut store = QueryStore::default();
        let args = json!({"limit":1});
        let first = store
            .insert(&args, vec![json!(1), json!(2)], json!({}), "entries")
            .unwrap();
        let key = first["query_id"].as_str().unwrap();
        assert!(store.release(key)["freed_bytes"].as_u64().unwrap() > 0);
        assert_eq!(store.release(key)["freed_bytes"], 0);
        assert!(
            store
                .page(&args, first["next_cursor"].as_str().unwrap())
                .is_err()
        );
        let first = store
            .insert(&args, vec![json!(1), json!(2)], json!({}), "graph")
            .unwrap();
        store
            .active
            .get_mut(first["query_id"].as_str().unwrap())
            .unwrap()
            .created = Instant::now() - TTL;
        assert_eq!(store.usage()["snapshot_bytes"], 0);
    }

    #[test]
    fn single_pages_do_not_retain_queries_and_wrong_arguments_do_not_release() {
        let mut store = QueryStore::default();
        store
            .insert(&json!({}), vec![json!(1)], json!({}), "samples")
            .unwrap();
        assert_eq!(store.bytes(), 0);
        let first = store
            .insert(
                &json!({"limit":1}),
                vec![json!(1), json!(2)],
                json!({}),
                "samples",
            )
            .unwrap();
        assert!(
            store
                .page(&json!({"limit":2}), first["next_cursor"].as_str().unwrap())
                .is_err()
        );
        assert_eq!(store.active.len(), 1);
    }
}

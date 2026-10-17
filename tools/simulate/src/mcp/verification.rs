//! Evidence about selected publications. Missing prefixes and tails remain unknown.
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const MAX_INTERVALS: usize = 16384;
const MAX_PUBLISHERS: usize = 1024;

#[derive(Default)]
struct Sequences {
    intervals: BTreeMap<i64, i64>,
    samples: u64,
    duplicates: u64,
}
impl Sequences {
    fn insert(&mut self, sequence: i64) {
        self.samples += 1;
        let previous = self
            .intervals
            .range(..=sequence)
            .next_back()
            .map(|(a, b)| (*a, *b));
        if previous.is_some_and(|(_, end)| sequence <= end) {
            self.duplicates += 1;
            return;
        }
        let mut start = sequence;
        if let Some((a, b)) = previous
            && b.checked_add(1) == Some(sequence)
        {
            start = a;
            self.intervals.remove(&a);
        }
        let mut end = sequence;
        if let Some((a, b)) = self
            .intervals
            .range(sequence..)
            .next()
            .map(|(a, b)| (*a, *b))
            && sequence.checked_add(1) == Some(a)
        {
            end = b;
            self.intervals.remove(&a);
        }
        self.intervals.insert(start, end);
    }
    fn missing(&self) -> u64 {
        let mut previous = None;
        let mut gaps = 0;
        for (&start, &end) in &self.intervals {
            if let Some(before) = previous {
                gaps += (start as i128 - before as i128 - 1) as u64;
            }
            previous = Some(end);
        }
        gaps
    }
}
#[derive(Default)]
pub(super) struct Verification {
    publishers: BTreeMap<(String, String), Sequences>,
    schemas: BTreeMap<String, BTreeSet<String>>,
    incomplete: bool,
}
impl Verification {
    pub fn observe(&mut self, row: &Value) {
        let topic = row["topic"].as_str().unwrap_or("");
        let identity = row["publication_id"].as_str().unwrap_or("");
        let Some((publisher, sequence)) = identity.rsplit_once('#') else {
            self.incomplete = true;
            return;
        };
        let Ok(sequence) = sequence.parse::<i64>() else {
            self.incomplete = true;
            return;
        };
        let key = (topic.into(), publisher.into());
        if (!self.publishers.contains_key(&key) && self.publishers.len() >= MAX_PUBLISHERS)
            || self
                .publishers
                .values()
                .map(|p| p.intervals.len())
                .sum::<usize>()
                >= MAX_INTERVALS
        {
            self.incomplete = true;
            return;
        }
        self.publishers.entry(key).or_default().insert(sequence);
        let schemas = self.schemas.entry(topic.into()).or_default();
        if schemas.len() < 256 {
            schemas.insert(
                row["type_info"]["schema_hash"]
                    .as_str()
                    .unwrap_or("unknown")
                    .into(),
            );
        } else {
            self.incomplete = true;
        }
    }
    pub fn finish(&self, checkpoint: Option<&Value>, revisions: Value) -> Value {
        json!({"sequence_scope":"between first and last selected publication per topic and publisher; prefixes and tails are unknown",
            "tracking_complete":!self.incomplete,
            "publishers":self.publishers.iter().map(|((topic,publisher),seq)|json!({"topic":topic,"publisher_id":publisher,
                "selected_rows":seq.samples,"duplicate_rows":seq.duplicates,"first_sequence":seq.intervals.first_key_value().map(|(s,_)|s.to_string()),
                "last_sequence":seq.intervals.last_key_value().map(|(_,s)|s.to_string()),"missing_between_selected":seq.missing(),
                "gap_intervals":seq.intervals.len().saturating_sub(1)})).collect::<Vec<_>>(),
            "schemas":self.schemas.iter().map(|(topic,hashes)|json!({"topic":topic,"schema_hashes":hashes,"multiple_schemas":hashes.len()>1})).collect::<Vec<_>>(),
            "recorders":checkpoint.map(|c|c["recorders"].clone()).unwrap_or(json!([])),
            "parameter_revisions":revisions,
            "parameter_revision_scope":"revisions observed by MCP reads and writes at their observation times; not a complete parameter change history",
            "completeness":"unknown"})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn out_of_order_publications_close_gaps_and_duplicates_do_not_hide_loss() {
        let mut sequences = Sequences::default();
        for sequence in [2, 5, 3, 3, 1] {
            sequences.insert(sequence);
        }
        assert_eq!(sequences.missing(), 1);
        assert_eq!(sequences.duplicates, 1);
        sequences.insert(4);
        assert_eq!(sequences.missing(), 0);
        assert_eq!(sequences.intervals, BTreeMap::from([(1, 5)]));
    }
    #[test]
    fn summaries_separate_publishers_and_report_schema_changes() {
        let mut summary = Verification::default();
        for (publisher, sequence, hash) in [("a", 1, "old"), ("a", 3, "new"), ("b", 2, "new")] {
            summary.observe(&json!({"topic":"test","publication_id":format!("session:{publisher}#{sequence}"),"type_info":{"schema_hash":hash}}));
        }
        let result = summary.finish(None, json!([]));
        assert_eq!(result["publishers"][0]["missing_between_selected"], 1);
        assert_eq!(result["publishers"][1]["missing_between_selected"], 0);
        assert_eq!(result["schemas"][0]["multiple_schemas"], true);
        assert_eq!(result["completeness"], "unknown");
    }
}

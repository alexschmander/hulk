use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use ros_z::time::Time;

use crate::{RetentionPolicy, SampleRecord};

pub struct TimeIndexedHistory<V> {
    entries: TimestampIndex<SampleRecord<V>>,
    policy: RetentionPolicy,
}

struct TimestampIndex<T> {
    entries: BTreeMap<Time, VecDeque<Arc<T>>>,
    len: usize,
    bytes: usize,
    weight: fn(&T) -> usize,
}

impl<T> TimestampIndex<T> {
    fn new(weight: fn(&T) -> usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            len: 0,
            bytes: 0,
            weight,
        }
    }

    fn insert(&mut self, timestamp: Time, value: Arc<T>) {
        self.bytes = self.bytes.saturating_add((self.weight)(&value));
        self.entries.entry(timestamp).or_default().push_back(value);
        self.len += 1;
    }

    fn get_interval(&self, start: Time, end: Time) -> Vec<Arc<T>> {
        if start > end {
            return Vec::new();
        }

        self.entries
            .range(start..=end)
            .flat_map(|(_, values)| values.iter().cloned())
            .collect()
    }

    fn get_all(&self) -> Vec<Arc<T>> {
        self.entries
            .values()
            .flat_map(|value| value.iter().cloned())
            .collect()
    }

    fn get_nearest(&self, time: Time) -> Option<Arc<T>> {
        let nearest_preceding_sample = self.entries.range(..time).next_back();
        let nearest_following_sample = self.entries.range(time..).next();

        match ((nearest_preceding_sample), nearest_following_sample) {
            (Some((_, nearest_preceding_sample)), None) => {
                Some(nearest_preceding_sample.iter().last().unwrap().clone())
            }
            (None, Some((_, nearest_following_sample))) => {
                Some(nearest_following_sample.iter().next().unwrap().clone())
            }
            (None, None) => None,
            (
                Some((nearest_preceding_sample_time, nearest_preceding_sample)),
                Some((nearest_following_sample_time, nearest_following_sample)),
            ) => {
                let distance_to_nearest_preceding_sample =
                    time.duration_since(*nearest_preceding_sample_time);
                let distance_to_nearest_following_sample =
                    nearest_following_sample_time.duration_since(time);

                if distance_to_nearest_preceding_sample < distance_to_nearest_following_sample {
                    Some(nearest_preceding_sample.iter().last().unwrap().clone())
                } else {
                    Some(nearest_following_sample.iter().next().unwrap().clone())
                }
            }
        }
    }

    fn latest_stamp(&self) -> Option<Time> {
        self.entries.keys().next_back().copied()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn remove_older_than(&mut self, minimum: Time) {
        while self
            .entries
            .first_key_value()
            .is_some_and(|(time, _)| *time < minimum)
        {
            self.pop_oldest();
        }
    }

    fn pop_oldest(&mut self) -> Option<Arc<T>> {
        let oldest_timestamp = *self.entries.keys().next()?;
        let values = self.entries.get_mut(&oldest_timestamp)?;
        let value = values.pop_front()?;
        self.len -= 1;
        self.bytes = self.bytes.saturating_sub((self.weight)(&value));

        if values.is_empty() {
            self.entries.remove(&oldest_timestamp);
        }

        Some(value)
    }
}

impl<V> TimeIndexedHistory<V> {
    pub fn new(policy: RetentionPolicy) -> Self {
        Self {
            entries: TimestampIndex::new(|record| record.retained_bytes),
            policy,
        }
    }

    pub fn insert(&mut self, record: Arc<SampleRecord<V>>) {
        let source_time = match self.policy {
            RetentionPolicy::TimeWindow(window) => match window.clock() {
                crate::RetentionClock::Source => record.effective_time,
                crate::RetentionClock::Receipt => record.receipt_time,
                crate::RetentionClock::WallReceipt => record.wall_receipt_time,
            },
            _ => record.effective_time,
        };
        self.entries.insert(source_time, record);

        self.evict();
    }

    pub fn window(&self, start: Time, end: Time) -> Vec<Arc<SampleRecord<V>>> {
        self.entries.get_interval(start, end)
    }

    pub fn get_all(&self) -> Vec<Arc<SampleRecord<V>>> {
        self.entries.get_all()
    }

    pub fn get_nearest(&self, time: Time) -> Option<Arc<SampleRecord<V>>> {
        self.entries.get_nearest(time)
    }

    fn evict(&mut self) {
        let Some(newest_source_time) = self.entries.latest_stamp() else {
            return;
        };

        let (duration, max_samples) = match self.policy {
            RetentionPolicy::LatestOnly => {
                self.entries.remove_older_than(newest_source_time);
                return;
            }
            RetentionPolicy::TimeWindow(window) => (window.duration(), window.max_samples()),
        };

        if let RetentionPolicy::TimeWindow(window) = self.policy
            && let Some(max_bytes) = window.max_bytes()
        {
            while self.entries.bytes > max_bytes.get() {
                if self.entries.pop_oldest().is_none() {
                    break;
                }
            }
        }
        let minimum_source_time = newest_source_time.saturating_sub(duration);
        self.entries.remove_older_than(minimum_source_time);

        if let Some(max_samples) = max_samples {
            while self.entries.len() > max_samples.get() {
                self.entries.pop_oldest();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use ros_z::time::Time;

    use super::TimeIndexedHistory;
    use crate::{
        DEFAULT_TIME_WINDOW_MAX_SAMPLES, RetentionPolicy, SampleMetadata, SampleRecord,
        TopicReference,
    };

    fn test_type_info() -> ros_z::TypeInfo {
        ros_z::TypeInfo::new("test_msgs::DebugValue", ros_z::SchemaHash::zero())
    }

    fn test_publication_id() -> ros_z::pubsub::PublicationId {
        ros_z::pubsub::Received {
            message: (),
            transport_time: None,
            source_time: Time::zero(),
            sequence_number: 1,
            source_global_id: ros_z::EndpointGlobalId::from([7; 16]),
        }
        .publication_id()
    }

    fn test_metadata() -> Arc<SampleMetadata> {
        Arc::new(SampleMetadata {
            topic_reference: TopicReference::new("debug").unwrap(),
            resolved_topic: "/debug".to_string(),
            type_info: test_type_info(),
        })
    }

    fn record(value: i32, source_time_nanos: i64) -> Arc<SampleRecord<i32>> {
        Arc::new(SampleRecord {
            receipt_time: Time::zero(),
            wall_receipt_time: Time::zero(),
            retained_bytes: 8,
            value,
            source_time: Time::from_nanos(source_time_nanos),
            effective_time: Time::from_nanos(source_time_nanos),
            transport_time: None,
            publication_id: test_publication_id(),
            metadata: test_metadata(),
        })
    }

    #[test]
    fn receipt_window_ignores_regressing_source_clock_and_enforces_byte_limit() {
        let mut history = TimeIndexedHistory::new(
            RetentionPolicy::time_window(Duration::from_nanos(10))
                .unwrap()
                .with_clock(crate::RetentionClock::Receipt)
                .with_max_bytes(std::num::NonZeroUsize::new(16).unwrap()),
        );
        for (value, source, receipt) in [(1, 1000, 1), (2, -100, 2), (3, 5000, 3)] {
            let mut sample = record(value, source);
            Arc::get_mut(&mut sample).unwrap().receipt_time = Time::from_nanos(receipt);
            history.insert(sample);
        }
        assert_eq!(
            history
                .get_all()
                .iter()
                .map(|s| s.value)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        let mut sample = record(4, -999);
        Arc::get_mut(&mut sample).unwrap().receipt_time = Time::from_nanos(20);
        history.insert(sample);
        assert_eq!(
            history
                .get_all()
                .iter()
                .map(|s| s.value)
                .collect::<Vec<_>>(),
            vec![4]
        );
    }

    #[test]
    fn time_window_eviction_uses_source_time() {
        let mut history =
            TimeIndexedHistory::new(RetentionPolicy::time_window(Duration::from_secs(1)).unwrap());

        history.insert(record(1, 0));
        history.insert(record(2, 2_000_000_000));

        let values = history.window(Time::from_nanos(0), Time::from_nanos(2_000_000_000));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn max_samples_evicts_oldest_after_time_eviction() {
        let mut history = TimeIndexedHistory::new(
            RetentionPolicy::time_window_with_max_samples(
                Duration::from_secs(10),
                std::num::NonZeroUsize::new(2).unwrap(),
            )
            .unwrap(),
        );

        history.insert(record(1, 1));
        history.insert(record(2, 2));
        history.insert(record(3, 3));

        let values = history.window(Time::from_nanos(0), Time::from_nanos(10));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn duplicate_timestamps_preserve_insertion_order() {
        let mut history =
            TimeIndexedHistory::new(RetentionPolicy::time_window(Duration::from_secs(10)).unwrap());

        history.insert(record(1, 5));
        history.insert(record(2, 5));

        let values = history.window(Time::from_nanos(5), Time::from_nanos(5));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn latest_only_retains_only_latest_source_time_records() {
        let mut history = TimeIndexedHistory::new(RetentionPolicy::LatestOnly);

        history.insert(record(1, 1));
        history.insert(record(2, 3));
        history.insert(record(3, 2));

        let values = history.window(Time::from_nanos(0), Time::from_nanos(3));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn latest_only_preserves_duplicate_newest_timestamp_order() {
        let mut history = TimeIndexedHistory::new(RetentionPolicy::LatestOnly);

        history.insert(record(1, 1));
        history.insert(record(2, 3));
        history.insert(record(3, 3));

        let values = history.window(Time::from_nanos(0), Time::from_nanos(3));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn time_window_eviction_uses_newest_source_time_for_out_of_order_inserts() {
        let mut history =
            TimeIndexedHistory::new(RetentionPolicy::time_window(Duration::from_secs(1)).unwrap());

        history.insert(record(1, 2_000_000_000));
        history.insert(record(2, 0));

        let values = history.window(Time::from_nanos(0), Time::from_nanos(2_000_000_000));
        assert_eq!(
            values.iter().map(|record| record.value).collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn default_time_window_retention_bounds_stalled_source_timestamps() {
        let mut history =
            TimeIndexedHistory::new(RetentionPolicy::time_window(Duration::from_secs(10)).unwrap());

        for value in 0..=DEFAULT_TIME_WINDOW_MAX_SAMPLES {
            history.insert(record(value as i32, 5));
        }

        let values = history.window(Time::from_nanos(5), Time::from_nanos(5));
        assert_eq!(values.len(), DEFAULT_TIME_WINDOW_MAX_SAMPLES);
        assert_eq!(values.first().map(|record| record.value), Some(1));
        assert_eq!(
            values.last().map(|record| record.value),
            Some(DEFAULT_TIME_WINDOW_MAX_SAMPLES as i32)
        );
    }

    #[test]
    fn get_nearest() {
        let mut history =
            TimeIndexedHistory::new(RetentionPolicy::time_window(Duration::from_secs(10)).unwrap());

        for value in 1..=10 {
            history.insert(record(value as i32, value * 100));
        }

        assert_eq!(history.get_nearest(Time::from_nanos(90)).unwrap().value, 1);
        assert_eq!(history.get_nearest(Time::from_nanos(140)).unwrap().value, 1);
        assert_eq!(history.get_nearest(Time::from_nanos(160)).unwrap().value, 2);
        assert_eq!(
            history.get_nearest(Time::from_nanos(1100)).unwrap().value,
            10
        );
    }

    #[test]
    fn get_nearest_on_empty_returns_none() {
        let history = TimeIndexedHistory::<i32>::new(
            RetentionPolicy::time_window(Duration::from_secs(10)).unwrap(),
        );

        assert!(history.get_nearest(Time::from_nanos(1)).is_none());
    }
}

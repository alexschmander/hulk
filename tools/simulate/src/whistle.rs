use std::time::Duration;

use color_eyre::{Result, eyre::ensure};
use ros_z::{context::Context, parameter::NodeParametersExt, time::Time};
use tokio::sync::watch;
use types::{parameters::WhistleDetectionParameters, whistle::Whistle};

// Cover the upstream GameController's normal 500 ms packet interval.
pub(crate) const PULSE_DURATION: Duration = Duration::from_millis(750);

pub(crate) async fn run(context: &Context, pulse: watch::Receiver<Option<Time>>) -> Result<()> {
    let node = context.create_node("simulator_whistle").build().await?;
    let parameters = node.bind_parameter_as::<WhistleDetectionParameters>("whistle_detection")?;
    let publisher = node
        .publisher::<Whistle>("detected_whistle")
        .build()
        .await?;
    loop {
        let snapshot = parameters.snapshot();
        let config = snapshot.typed();
        ensure!(
            config.number_audio_channels > 0 && config.number_audio_channels <= 64,
            "whistle channel count must be between 1 and 64"
        );
        ensure!(
            config.audio_sample_rate > 0 && config.number_audio_samples > 0,
            "whistle logical sample rate and frame size must be positive"
        );
        let period = Duration::from_secs_f64(
            config.number_audio_samples as f64 / config.audio_sample_rate as f64,
        );
        ensure!(
            period >= Duration::from_millis(1) && period <= Duration::from_secs(1),
            "whistle logical frame period must be between 1 ms and 1 s"
        );
        let active = pulse
            .borrow()
            .is_some_and(|until| node.clock().now() < until);
        publisher
            .publish(&Whistle {
                is_detected: vec![active; config.number_audio_channels],
            })
            .await?;
        node.clock().sleep(period).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::prelude::*;
    use std::sync::Arc;
    use std::time::Instant;
    use types::filtered_whistle::FilteredWhistle;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn logical_channel_count_and_filter_threshold_are_respected() {
        let parameters = tempfile::tempdir().unwrap();
        std::fs::write(
            parameters.path().join("whistle_detection.json5"),
            "{number_audio_channels: 1}",
        )
        .unwrap();
        let context = Arc::new(
            ContextBuilder::default()
                .with_mode("peer")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .with_parameter_layers([
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../etc/parameters/base"),
                    parameters.path().to_owned(),
                ])
                .build()
                .await
                .unwrap(),
        );
        let node = context
            .create_node("whistle_fixture")
            .build()
            .await
            .unwrap();
        let raw = node
            .subscriber::<Whistle>("detected_whistle")
            .cache(1)
            .build()
            .await
            .unwrap();
        let filtered = node
            .subscriber::<FilteredWhistle>("filtered_whistle")
            .cache(1)
            .build()
            .await
            .unwrap();
        let filter = tokio::spawn(whistle_filter::run_boxed(context.clone()));
        let (pulse, receiver) = watch::channel(None);
        let publisher_context = context.clone();
        let publisher = tokio::spawn(async move { run(&publisher_context, receiver).await });
        let start = Instant::now();
        while filtered.get_latest().is_none() {
            assert!(start.elapsed() < Duration::from_secs(2));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(raw.get_latest().unwrap().is_detected.len(), 1);
        pulse.send_replace(Some(context.clock().now() + Duration::from_millis(100)));
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            filtered.get_latest().unwrap().last_detection.is_none(),
            "below-threshold pulse was accepted"
        );
        pulse.send_replace(Some(context.clock().now() + PULSE_DURATION));
        let start = Instant::now();
        while !filtered.get_latest().unwrap().is_detected {
            assert!(start.elapsed() < Duration::from_secs(1));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let start = Instant::now();
        while filtered.get_latest().unwrap().is_detected {
            assert!(start.elapsed() < Duration::from_secs(3));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        publisher.abort();
        filter.abort();
        let _ = publisher.await;
        let _ = filter.await;
        context.shutdown().unwrap();
    }
}

use std::collections::BTreeMap;
use std::future::ready;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{boxed::Box, future::Future, pin::Pin};

use color_eyre::Result;

use ros_z::prelude::*;
use ros_z::qos::QosDurability;
use ros2::sensor_msgs::{camera_info::CameraInfo, image::Image};
use serde::{Deserialize, Serialize};
use types::{stereo_camera_info::StereoCameraInfo, stereo_image_pair::StereoImagePair};
use x5_receiver::receiver::{Side, X5Receiver};

const X5_ADDRESS: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 127, 10)), 7654);

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
struct ImageReceiverParameters {
    /// Maximum publication frequency in Hz per image stream. Zero means unlimited.
    #[serde(default)]
    max_frequency: u32,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("image_receiver").build().await?;
    let parameters = node.bind_parameter_as::<ImageReceiverParameters>("image_receiver")?;

    let left_image_pub = node.publisher::<Image>("inputs/left_image").build().await?;
    let right_image_pub = node
        .publisher::<Image>("inputs/right_image")
        .build()
        .await?;
    let camera_info_pub = node
        .publisher::<CameraInfo>("inputs/camera_info")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;
    let stereo_camera_info_pub = node
        .publisher::<StereoCameraInfo>("inputs/stereo_camera_info")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;
    let stereo_image_pair_pub = node
        .publisher::<StereoImagePair>("inputs/stereo_image_pair")
        .build()
        .await?;

    let x5_receiver = X5Receiver::new(X5_ADDRESS);
    let camera_info = x5_receiver.wait_for_camera_info().await;
    let stereo_camera_info = StereoCameraInfo {
        left: camera_info.left_camera_info(),
        right: camera_info.right_camera_info(),
    };

    let mut camera_info_timer = node.clock().interval(Duration::from_secs(1));
    let mut stereo_image_pairer = StereoImagePairer::default();
    let mut left_frame_receiver = x5_receiver.subscribe_image(Side::Left);
    let mut right_frame_receiver = x5_receiver.subscribe_image(Side::Right);
    let mut left_throttle = ImageThrottle::default();
    let mut right_throttle = ImageThrottle::default();
    let mut stereo_throttle = ImageThrottle::default();

    loop {
        let stereo_image_pair = tokio::select! {
            left_image = left_frame_receiver.recv() => {
                let frame_identifier = left_image.header.frame_identifier;
                let image: Image = left_image.into();
                if left_throttle.allow(Instant::now(), parameters.snapshot().typed().max_frequency) {
                    left_image_pub.publish(&image).await?;
                }
                stereo_image_pairer.insert(CameraSide::Left, frame_identifier, image)
            }
            right_image = right_frame_receiver.recv() => {
                let frame_identifier = right_image.header.frame_identifier;
                let image: Image = right_image.into();
                if right_throttle.allow(Instant::now(), parameters.snapshot().typed().max_frequency) {
                    right_image_pub.publish_if_subscribed(|| ready(&image)).await?;
                }
                stereo_image_pairer.insert(CameraSide::Right, frame_identifier, image)
            }
            _ = camera_info_timer.tick() => {
                camera_info_pub
                    .publish(&stereo_camera_info.left)
                    .await?;
                stereo_camera_info_pub
                    .publish(&stereo_camera_info)
                    .await?;
                continue
            }
        };

        if !stereo_image_pair_pub.has_subscribers() {
            stereo_image_pairer.clear();
            continue;
        }

        let Some(stereo_image_pair) = stereo_image_pair else {
            continue;
        };
        // Throttle complete pairs, never their halves, so VO keeps matching frame IDs.
        if stereo_throttle.allow(Instant::now(), parameters.snapshot().typed().max_frequency) {
            stereo_image_pair_pub.publish(&stereo_image_pair).await?;
        }
    }
}

#[derive(Default)]
struct ImageThrottle {
    last_published: Option<Instant>,
}

impl ImageThrottle {
    fn allow(&mut self, now: Instant, max_frequency: u32) -> bool {
        if max_frequency > 0
            && self.last_published.is_some_and(|last| {
                now.duration_since(last).as_secs_f64() < 1.0 / f64::from(max_frequency)
            })
        {
            return false;
        }
        // Use actual publication time to avoid catch-up bursts after a stall.
        self.last_published = Some(now);
        true
    }
}

#[derive(Clone, Copy)]
enum CameraSide {
    Left,
    Right,
}

#[derive(Default)]
struct StereoImagePairer {
    pending_left: BTreeMap<u32, Image>,
    pending_right: BTreeMap<u32, Image>,
    latest_frame_identifier: u32,
}

impl StereoImagePairer {
    const MAX_UNMATCHED_FRAME_AGE: u32 = 8;

    fn insert(
        &mut self,
        side: CameraSide,
        frame_identifier: u32,
        image: Image,
    ) -> Option<StereoImagePair> {
        self.update_latest_frame_identifier(frame_identifier);

        let (remove_from, insert_in) = match side {
            CameraSide::Left => (&mut self.pending_right, &mut self.pending_left),
            CameraSide::Right => (&mut self.pending_left, &mut self.pending_right),
        };

        let Some(other) = remove_from.remove(&frame_identifier) else {
            insert_in.insert(frame_identifier, image);
            self.expire_old_frames();
            return None;
        };

        self.expire_old_frames();

        let (left, right) = match side {
            CameraSide::Left => (image, other),
            CameraSide::Right => (other, image),
        };

        Some(StereoImagePair {
            frame_identifier,
            left,
            right,
        })
    }

    fn clear(&mut self) {
        self.pending_left.clear();
        self.pending_right.clear();
    }

    fn update_latest_frame_identifier(&mut self, frame_identifier: u32) {
        self.latest_frame_identifier = self.latest_frame_identifier.max(frame_identifier);
    }

    fn expire_old_frames(&mut self) {
        let cutoff = self
            .latest_frame_identifier
            .saturating_sub(Self::MAX_UNMATCHED_FRAME_AGE);
        self.pending_left
            .retain(|frame_identifier, _| *frame_identifier >= cutoff);
        self.pending_right
            .retain(|frame_identifier, _| *frame_identifier >= cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros2::{builtin_interfaces::time::Time, std_msgs::header::Header};

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn image_frequency_parameter_loads_and_updates_at_runtime() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "image-receiver-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir_all(&root)?;
        let file = root.join("image_receiver.json5");
        std::fs::write(
            &file,
            include_str!("../../../../etc/parameters/base/image_receiver.json5"),
        )?;
        let context = ContextBuilder::default()
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_parameter_layers([root.clone()])
            .build()
            .await?;
        let node = context.create_node("image_receiver_test").build().await?;
        let parameters = node.bind_parameter_as::<ImageReceiverParameters>("image_receiver")?;
        let layer = root.to_string_lossy().into_owned();
        assert_eq!(parameters.snapshot().typed().max_frequency, 15);

        let mut throttle = ImageThrottle::default();
        let start = Instant::now();
        assert!(throttle.allow(start, parameters.snapshot().typed().max_frequency));
        let next_frame = start + Duration::from_millis(50);
        assert!(!throttle.allow(next_frame, parameters.snapshot().typed().max_frequency));
        parameters.set_json("max_frequency", 20.into(), layer.clone())?;
        assert!(throttle.allow(next_frame, parameters.snapshot().typed().max_frequency));
        parameters.set_json("max_frequency", 0.into(), layer.clone())?;
        assert!(throttle.allow(next_frame, parameters.snapshot().typed().max_frequency));
        for invalid in [(-1).into(), 1.5.into()] {
            assert!(
                parameters
                    .set_json("max_frequency", invalid, layer.clone())
                    .is_err()
            );
            assert_eq!(parameters.snapshot().typed().max_frequency, 0);
        }
        // Existing deployments without this new parameter file remain unlimited.
        std::fs::remove_file(file)?;
        parameters.reload()?;
        assert_eq!(parameters.snapshot().typed().max_frequency, 0);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn throttle_passes_first_frame_and_drops_until_interval_elapsed() {
        let mut throttle = ImageThrottle::default();
        let start = Instant::now();

        assert!(throttle.allow(start, 10));
        assert!(!throttle.allow(start, 10));
        assert!(!throttle.allow(start + Duration::from_millis(99), 10));
        assert!(throttle.allow(start + Duration::from_millis(100), 10));
        assert!(!throttle.allow(start + Duration::from_millis(199), 10));
        assert!(throttle.allow(start + Duration::from_millis(200), 10));
    }

    #[test]
    fn throttle_does_not_accumulate_credit_during_stalls() {
        let mut throttle = ImageThrottle::default();
        let start = Instant::now();

        assert!(throttle.allow(start, 10));
        let after_stall = start + Duration::from_secs(3);
        assert!(throttle.allow(after_stall, 10));
        assert!(!throttle.allow(after_stall, 10));
        assert!(!throttle.allow(after_stall + Duration::from_millis(99), 10));
        assert!(throttle.allow(after_stall + Duration::from_millis(100), 10));
    }

    #[test]
    fn throttle_applies_frequency_changes_and_zero_disables_it() {
        let mut throttle = ImageThrottle::default();
        let start = Instant::now();

        assert!(throttle.allow(start, 10));
        assert!(!throttle.allow(start + Duration::from_millis(50), 10));
        assert!(throttle.allow(start + Duration::from_millis(50), 20));
        assert!(!throttle.allow(start + Duration::from_millis(100), 5));
        assert!(throttle.allow(start + Duration::from_millis(250), 5));
        assert!(throttle.allow(start + Duration::from_millis(250), 0));
        assert!(throttle.allow(start + Duration::from_millis(250), 0));
        assert!(!throttle.allow(start + Duration::from_millis(251), 10));
        assert!(throttle.allow(start + Duration::from_millis(350), 10));
    }

    fn image(frame: u32, side: u8) -> Image {
        Image {
            header: Header {
                stamp: Time {
                    sec: 1,
                    nanosec: frame,
                },
                frame_id: "x5".into(),
            },
            height: 2,
            width: 2,
            encoding: "nv12".into(),
            is_bigendian: 0,
            step: 2,
            data: vec![side; 6].into(),
        }
    }

    #[test]
    fn throttled_pairs_keep_matching_frames_and_original_images_in_either_arrival_order() {
        let mut pairer = StereoImagePairer::default();
        let mut throttle = ImageThrottle::default();
        let start = Instant::now();
        let mut published = Vec::new();

        for frame in 0..5 {
            let now = start + Duration::from_millis(u64::from(frame) * 50);
            let sides = if (frame / 2) % 2 == 0 {
                [CameraSide::Left, CameraSide::Right]
            } else {
                [CameraSide::Right, CameraSide::Left]
            };
            for side in sides {
                let marker = match side {
                    CameraSide::Left => 1,
                    CameraSide::Right => 2,
                };
                if let Some(pair) = pairer.insert(side, frame, image(frame, marker))
                    && throttle.allow(now, 10)
                {
                    published.push(pair);
                }
            }
        }

        assert_eq!(published.len(), 3);
        for (pair, frame) in published.iter().zip([0, 2, 4]) {
            assert_eq!(pair.frame_identifier, frame);
            assert_eq!(pair.left.header.stamp.nanosec, frame);
            assert_eq!(pair.right.header.stamp.nanosec, frame);
            assert_eq!(pair.left.header.stamp.sec, 1);
            assert_eq!(pair.right.header.stamp.sec, 1);
            assert_eq!(pair.left.data.as_ref(), &[1; 6]);
            assert_eq!(pair.right.data.as_ref(), &[2; 6]);
        }
        assert!(pairer.pending_left.is_empty());
        assert!(pairer.pending_right.is_empty());
    }
}

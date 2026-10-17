//! The same paired CDR samples consumed by the production low-state bridge.
use booster::LowState;
use color_eyre::{Result, eyre::eyre};
use ros_z::time::Time;
use ros2::{sensor_msgs::joint_state::JointState, std_msgs::header::Header};

pub(crate) struct Sensors {
    joints: zenoh::pubsub::Publisher<'static>,
    low_state: zenoh::pubsub::Publisher<'static>,
    sample: JointState,
}

impl Sensors {
    pub async fn new(session: &zenoh::Session) -> Result<Self> {
        Ok(Self {
            joints: session
                .declare_publisher("rt/joint_states")
                .await
                .map_err(|e| eyre!("{e}"))?,
            low_state: session
                .declare_publisher("rt/low_state")
                .await
                .map_err(|e| eyre!("{e}"))?,
            sample: JointState {
                header: Header::default(),
                name: crate::robot_io::JOINTS
                    .iter()
                    .map(|name| (*name).into())
                    .collect(),
                position: Vec::with_capacity(crate::robot_io::JOINTS.len()),
                velocity: Vec::with_capacity(crate::robot_io::JOINTS.len()),
                effort: Vec::with_capacity(crate::robot_io::JOINTS.len()),
            },
        })
    }

    pub async fn publish(&mut self, state: &LowState, time: Time) -> Result<()> {
        self.sample.header.stamp = time.to_wallclock().into();
        self.sample.position.clear();
        self.sample.velocity.clear();
        self.sample.effort.clear();
        for motor in &state.motor_state_serial {
            self.sample.position.push(f64::from(motor.position));
            self.sample.velocity.push(f64::from(motor.velocity));
            self.sample.effort.push(f64::from(motor.torque));
        }
        let stamp = cdr::serialize::<_, _, cdr::CdrLe>(&self.sample, cdr::Infinite)?;
        let low = cdr::serialize::<_, _, cdr::CdrLe>(state, cdr::Infinite)?;
        self.joints.put(stamp).await.map_err(|e| eyre!("{e}"))?;
        self.low_state.put(low).await.map_err(|e| eyre!("{e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::prelude::*;
    use std::{sync::Arc, time::Duration};

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn reused_joint_message_replaces_values_and_preserves_earlier_packets() {
        let context = ContextBuilder::default()
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .build()
            .await
            .unwrap();
        let packets = context
            .session()
            .declare_subscriber("rt/joint_states")
            .await
            .unwrap();
        let mut sensors = Sensors::new(context.session()).await.unwrap();
        let mut state = LowState {
            motor_state_serial: vec![booster::MotorState::default(); 22],
            ..Default::default()
        };
        let first = Time::from_nanos(1_700_000_000_000_000_000);
        sensors.publish(&state, first).await.unwrap();
        let original = packets.recv_async().await.unwrap();
        state.motor_state_serial[3].position = 0.75;
        let second = first + Duration::from_millis(2);
        sensors.publish(&state, second).await.unwrap();
        let updated = packets.recv_async().await.unwrap();
        let decode = |sample: zenoh::sample::Sample| -> JointState {
            cdr::deserialize(sample.payload().to_bytes().as_ref()).unwrap()
        };
        let original = decode(original);
        let updated = decode(updated);
        assert_eq!(original.position[3], 0.0);
        assert_eq!(updated.position[3], 0.75);
        assert_eq!(updated.name, original.name);
        assert_eq!(updated.position.len(), 22);
        assert_eq!(updated.velocity.len(), 22);
        assert_eq!(updated.effort.len(), 22);
        let first: ros2::builtin_interfaces::time::Time = first.to_wallclock().into();
        let second: ros2::builtin_interfaces::time::Time = second.to_wallclock().into();
        assert_eq!(original.header.stamp.sec, first.sec);
        assert_eq!(original.header.stamp.nanosec, first.nanosec);
        assert_eq!(updated.header.stamp.sec, second.sec);
        assert_eq!(updated.header.stamp.nanosec, second.nanosec);
        context.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn paired_raw_packets_reach_the_unmodified_bridge_with_the_capture_time() {
        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace("/sensor_test")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .build()
                .await
                .unwrap(),
        );
        let node = context.create_node("observer").build().await.unwrap();
        let output = node
            .subscriber::<LowState>("inputs/low_state")
            .build()
            .await
            .unwrap();
        let serial = node
            .subscriber::<kinematics::joints::Joints<booster::MotorState>>(
                "inputs/serial_motor_states",
            )
            .build()
            .await
            .unwrap();
        let task = tokio::spawn(low_state_bridge::run_boxed(context.clone()));
        let mut sensors = Sensors::new(context.session()).await.unwrap();
        let time = Time::from_nanos(1_700_000_000_123_456_789);
        let state = LowState {
            motor_state_serial: (0..22)
                .map(|i| booster::MotorState {
                    position: i as f32 * 0.1234,
                    velocity: i as f32 * -0.03125,
                    torque: i as f32 * 0.17,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let packet = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                sensors.publish(&state, time).await.unwrap();
                tokio::select! {
                    packet = output.recv_with_metadata() => break packet.unwrap(),
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(packet.source_time, time);
        let serial = tokio::time::timeout(Duration::from_secs(1), serial.recv_with_metadata())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(serial.source_time, time);
        for (expected, actual) in state
            .motor_state_serial
            .iter()
            .zip(packet.into_message().motor_state_serial)
        {
            assert_eq!(expected.position.to_bits(), actual.position.to_bits());
            assert_eq!(expected.velocity.to_bits(), actual.velocity.to_bits());
            assert_eq!(expected.torque.to_bits(), actual.torque.to_bits());
        }
        task.abort();
        let _ = task.await;
        context.shutdown().unwrap();
    }
}

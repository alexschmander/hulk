//! The same paired CDR samples consumed by the production low-state bridge.
use booster::LowState;
use color_eyre::{Result, eyre::eyre};
use ros_z::time::Time;
use ros2::{sensor_msgs::joint_state::JointState, std_msgs::header::Header};

pub(crate) async fn publish(session: &zenoh::Session, state: &LowState, time: Time) -> Result<()> {
    let joints = JointState {
        header: Header {
            stamp: time.to_wallclock().into(),
            frame_id: String::new(),
        },
        name: crate::robot_io::JOINTS
            .iter()
            .map(|name| (*name).into())
            .collect(),
        position: state
            .motor_state_serial
            .iter()
            .map(|s| f64::from(s.position))
            .collect(),
        velocity: state
            .motor_state_serial
            .iter()
            .map(|s| f64::from(s.velocity))
            .collect(),
        effort: state
            .motor_state_serial
            .iter()
            .map(|s| f64::from(s.torque))
            .collect(),
    };
    let stamp = cdr::serialize::<_, _, cdr::CdrLe>(&joints, cdr::Infinite)?;
    let low = cdr::serialize::<_, _, cdr::CdrLe>(state, cdr::Infinite)?;
    session
        .put("rt/joint_states", stamp)
        .await
        .map_err(|e| eyre!("{e}"))?;
    session
        .put("rt/low_state", low)
        .await
        .map_err(|e| eyre!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::prelude::*;
    use std::{sync::Arc, time::Duration};

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
                publish(context.session(), &state, time).await.unwrap();
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

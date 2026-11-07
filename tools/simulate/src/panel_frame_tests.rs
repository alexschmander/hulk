//! Exercises the same state reads and Bevy pose updates as the simulator panel.
use super::*;

#[test]
#[ignore = "requires models and free UDP ports; run alone, optionally under taskset CPU pressure"]
fn panel_snapshots_remain_responsive_at_high_speed() {
    let directory = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let mut session = Session::start(
        &runtime,
        &json!({
            "session_id":"panel-frame-test", "clock_path":directory.path().join("clock"),
            "configuration":{"location":"hsl_middle","profile":"localization","referee":"external",
                "robot_count":5,"opponent_count":5,"rendering":"disabled"}
        }),
    )
    .unwrap();
    session.app.insert_resource(session.team.clone());
    session
        .app
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<crate::scene::ball::SpawnedBalls>()
        .add_plugins(crate::scene::command_vectors::CommandVectorsPlugin);
    for member in session.team.members() {
        for name in ["Trunk", "Head_1", "Head_2"] {
            session.app.world_mut().spawn((
                crate::bevy_mujoco::MujocoBody::new(member.entity.unwrap(), name),
                Transform::default(),
            ));
        }
    }
    session.app.update();
    let shared = session.shared();
    session.physics.set_speed(crate::simulation::Speed::Octuple);
    shared.set_mode(SimulationMode::Running);
    let frame_period = Duration::from_secs_f64(1.0 / 60.0);
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_millis(600) {
        session.app.update();
        std::thread::sleep(frame_period);
    }
    let started = Instant::now();
    let simulation_start = session.team.now();
    let mut next_frame = started;
    let mut frames = Vec::new();
    let mut ages = Vec::new();
    let mut last_time = f64::NEG_INFINITY;
    let mut refreshed = 0;
    while started.elapsed() < Duration::from_secs(6) {
        std::thread::sleep(next_frame.saturating_duration_since(Instant::now()));
        let frame = Instant::now();
        std::hint::black_box(shared.view().mode); // Toolbar state.
        session.app.update(); // Actual model/pose synchronization systems.
        let view = shared.view();
        for member in session.team.members() {
            std::hint::black_box(view.object_pose(member.entity.unwrap()));
        }
        if view.time > last_time {
            refreshed += 1;
            last_time = view.time;
        }
        let now = (session.team.now().as_nanos() - session.origin) as f64 / 1e9;
        ages.push((now - view.time).max(0.0) * 1000.0);
        frames.push(frame.elapsed().as_secs_f64() * 1000.0);
        assert!(session.physics.poll().is_none());
        next_frame = (next_frame + frame_period).max(Instant::now());
    }
    let wall = started.elapsed().as_secs_f64();
    let sim = session
        .team
        .now()
        .duration_since(simulation_start)
        .as_secs_f64();
    let mut sorted = frames.clone();
    sorted.sort_by(f64::total_cmp);
    let p99 = sorted[((sorted.len() - 1) as f64 * 0.99).round() as usize];
    let max_age = ages.iter().copied().fold(0.0, f64::max);
    let result = json!({"requested_speed":8,"actual_speed":sim/wall,"updates_per_second":frames.len()as f64/wall,
        "frame_p99_ms":p99,"frame_max_ms":sorted.last(),"frames":frames.len(),"refreshed_frames":refreshed,
        "max_snapshot_age_simulation_ms":max_age,"raw_frame_ms":frames,"raw_snapshot_age_ms":ages});
    if let Some(output) = std::env::var_os("SIMULATOR_PANEL_TEST_OUTPUT") {
        std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    eprintln!(
        "Panel: actual={:.3}x updates={:.2}/s p99={p99:.3}ms max={:.3}ms refreshed={refreshed}/{} max_age={max_age:.3}sim-ms",
        sim / wall,
        frames.len() as f64 / wall,
        sorted.last().unwrap(),
        frames.len()
    );
    let pause = Instant::now();
    shared.set_mode(SimulationMode::Paused);
    assert!(pause.elapsed() < Duration::from_millis(50));
    let deadline = Instant::now() + Duration::from_secs(2);
    while shared.view().mode != SimulationMode::Paused {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let frozen = session.team.now();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(session.team.now(), frozen);
    crate::simulation::reset(session.app.world_mut());
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let selected = session.team.selected().unwrap();
        let pose = shared.view().object_pose(selected.entity.unwrap()).unwrap();
        if (pose.translation.x - selected.pose.translation.x).abs() < 1e-4
            && (pose.translation.z - selected.pose.translation.z).abs() < 1e-4
        {
            break;
        }
        assert!(Instant::now() < deadline, "reset edit was not applied");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(shared.take_edit_error().is_none());
    drop(session);
    assert!(crate::finish_recordings().is_empty());
    assert!(
        p99 < 50.0,
        "panel p99 {p99}ms exceeds its responsiveness budget"
    );
    assert!(
        refreshed * 2 > frames.len(),
        "view snapshots stopped refreshing"
    );
    assert!(
        max_age < 1000.0,
        "view snapshot remained stale for {max_age} simulated ms"
    );
}

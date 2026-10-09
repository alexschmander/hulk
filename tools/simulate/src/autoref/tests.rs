use super::*;
use engine::Effect;
use game_controller_core::{action::VAction, actions::*, types::*};
use std::time::Duration;

fn fixture() -> (Engine, Snapshot) {
    let field = crate::FieldConfiguration::load(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters"),
        "hsl_small",
    )
    .unwrap();
    let engine = Engine::new(Competition::Middle, field).unwrap();
    let mut snapshot = Snapshot {
        ball: Some(Ball {
            entity: 1,
            position: [0.0, 0.0, 0.11],
            velocity: [0.0; 3],
            epoch: 0,
        }),
        ..Default::default()
    };
    for team in crate::TeamId::ALL {
        for number in 1..=3 {
            let p = [
                if team == crate::TeamId::Hulks {
                    -3.0
                } else {
                    3.0
                },
                number as f64 * 0.5,
                0.65,
            ];
            snapshot.robots.push(Robot {
                id: crate::RobotId { team, number },
                position: p,
                feet: vec![[p[0], p[1]]],
                speed: 0.0,
                leg_speed: 0.0,
                fallen: false,
                penalized: false,
            });
        }
    }
    (engine, snapshot)
}
fn step(e: &mut Engine, s: &Snapshot, seconds: f64) -> Vec<Effect> {
    e.update(s, Duration::from_secs_f64(seconds))
}
fn playing(e: &mut Engine, s: &Snapshot) {
    step(e, s, 0.01);
    assert_eq!(e.core.get_game(false).state, State::Ready);
    step(e, s, 45.0);
    assert_eq!(e.core.get_game(false).state, State::Set);
    assert!(step(e, s, 2.1).contains(&Effect::Whistle));
    assert_eq!(e.core.get_game(false).state, State::Playing);
}
fn id(team: crate::TeamId, number: u8) -> crate::RobotId {
    crate::RobotId { team, number }
}
fn move_ball(s: &mut Snapshot, p: [f64; 3]) {
    let ball = s.ball.as_mut().unwrap();
    ball.position = p;
    ball.velocity = [0.2, 0.0, 0.0];
}
fn touch(e: &mut Engine, s: &mut Snapshot, robot: crate::RobotId) {
    s.contacts.clear();
    step(e, s, 0.01);
    s.contacts.insert(robot);
    step(e, s, 0.01);
    s.contacts.clear();
}
#[test]
fn kickoff_uses_true_state_and_delayed_packets_and_only_simulation_time() {
    let (mut e, s) = fixture();
    playing(&mut e, &s);
    assert_eq!(e.core.get_game(true).state, State::Set);
    step(&mut e, &s, 9.0);
    assert_eq!(e.core.get_game(true).state, State::Set);
    step(&mut e, &s, 1.0);
    assert_eq!(e.core.get_game(true).state, State::Playing);
}
#[test]
fn full_ball_crossing_and_last_deflection_determine_kick_in() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    touch(&mut e, &mut s, id(crate::TeamId::Hulks, 1));
    touch(&mut e, &mut s, id(crate::TeamId::Opponents, 2));
    move_ball(&mut s, [0.7, 3.05, 0.11]);
    step(&mut e, &s, 0.01);
    assert_ne!(e.core.get_game(false).set_play, SetPlay::ThrowIn);
    move_ball(&mut s, [0.7, 3.5, 0.11]);
    step(&mut e, &s, 0.01);
    let game = e.core.get_game(false);
    assert_eq!(game.set_play, SetPlay::ThrowIn);
    assert_eq!(game.kicking_side, Some(Side::Home));
    assert!(game.stopped);
    let effects = step(&mut e, &s, 0.6);
    assert!(
        effects
            .iter()
            .any(|effect| matches!(effect,Effect::Ball([_,y]) if *y==3.0))
    );
    assert!(!effects.contains(&Effect::Whistle));
    move_ball(&mut s, [0.7, 3.0, 0.11]);
    s.ball.as_mut().unwrap().epoch += 1;
    step(&mut e, &s, 0.4);
    assert!(!e.core.get_game(false).stopped);
}
#[test]
fn both_goal_lines_award_corners_and_goal_kicks() {
    for positive in [false, true] {
        for defender_touched in [false, true] {
            let (mut e, mut s) = fixture();
            playing(&mut e, &s);
            let defender = if positive {
                crate::TeamId::Opponents
            } else {
                crate::TeamId::Hulks
            };
            let toucher = if defender_touched {
                defender
            } else {
                if positive {
                    crate::TeamId::Hulks
                } else {
                    crate::TeamId::Opponents
                }
            };
            touch(&mut e, &mut s, id(toucher, 1));
            move_ball(&mut s, [if positive { 5.0 } else { -5.0 }, 2.0, 0.11]);
            step(&mut e, &s, 0.01);
            assert_eq!(
                e.core.get_game(false).set_play,
                if defender_touched {
                    SetPlay::CornerKick
                } else {
                    SetPlay::GoalKick
                }
            );
            let effects = step(&mut e, &s, 0.6);
            assert!(effects.iter().any(|effect| matches!(effect,Effect::Ball([x,_]) if (x.abs()-if defender_touched {4.5}else{3.5}).abs()<0.001)));
        }
    }
}
#[test]
fn kickoff_goal_requires_two_own_players_and_goals_are_counted_once() {
    for valid in [false, true] {
        let (mut e, mut s) = fixture();
        playing(&mut e, &s);
        touch(&mut e, &mut s, id(crate::TeamId::Hulks, 1));
        move_ball(&mut s, [0.5, 0.0, 0.11]);
        step(&mut e, &s, 0.01);
        if valid {
            touch(&mut e, &mut s, id(crate::TeamId::Hulks, 2));
        }
        move_ball(&mut s, [5.0, 0.0, 0.11]);
        step(&mut e, &s, 0.01);
        assert_eq!(
            e.core.get_game(false).teams[Side::Home].score,
            u8::from(valid)
        );
        step(&mut e, &s, 0.01);
        assert_eq!(
            e.core.get_game(false).teams[Side::Home].score,
            u8::from(valid)
        );
        if valid {
            assert_eq!(e.core.get_game(false).kicking_side, Some(Side::Away));
        } else {
            assert_eq!(e.core.get_game(false).set_play, SetPlay::GoalKick);
        }
    }
}
#[test]
fn penalty_placement_starts_timer_then_releases_without_teleporting_back() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    for team in crate::TeamId::ALL {
        for number in [1, 2] {
            e.commands.push(VAction::Penalize(Penalize {
                side: side(team),
                player: PlayerNumber::new(number),
                call: PenaltyCall::RequestForPickUp,
            }));
        }
    }
    step(&mut e, &s, 0.01);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)]
            .penalty_timer
            .get_remaining()
            .whole_seconds(),
        0
    );
    for robot in &mut s.robots {
        robot.penalized = true;
    }
    let effects = step(&mut e, &s, 0.1);
    let placed: Vec<_> = effects
        .iter()
        .filter_map(|effect| {
            if let Effect::Robot(id, p, _) = effect {
                Some((*id, *p))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(placed.len(), 4);
    for (id, p) in &placed {
        assert_eq!(p[0] > 0.0, id.team == crate::TeamId::Opponents);
        assert!(p[1].abs() > 3.2);
    }
    for i in 0..placed.len() {
        for j in i + 1..placed.len() {
            assert_ne!(placed[i].1, placed[j].1);
        }
    }
    step(&mut e, &s, 0.4);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)]
            .penalty_timer
            .get_remaining()
            .whole_seconds(),
        45
    );
    e.enabled = false;
    let effects = step(&mut e, &s, 45.0);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)].penalty,
        Penalty::NoPenalty
    );
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Robot(..)))
    );
}
#[test]
fn motion_in_set_stays_in_place_and_clock_freezes_on_brief_stop() {
    let (mut e, s) = fixture();
    step(&mut e, &s, 0.01);
    step(&mut e, &s, 45.0);
    e.enabled = false;
    e.commands.push(VAction::Penalize(Penalize {
        side: Side::Home,
        player: PlayerNumber::new(1),
        call: PenaltyCall::MotionInSet,
    }));
    let effects = step(&mut e, &s, 0.1);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Robot(..)))
    );
    e.manual_whistle = true;
    step(&mut e, &s, 0.1);
    e.commands
        .push(VAction::StopPlay(StopPlay { resume: false }));
    step(&mut e, &s, 0.1);
    let remaining = e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)]
        .penalty_timer
        .get_remaining();
    let primary = e.core.get_game(false).primary_timer.get_remaining();
    step(&mut e, &s, 20.0);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)]
            .penalty_timer
            .get_remaining(),
        remaining
    );
    assert!(e.core.get_game(false).primary_timer.get_remaining() < primary);
}
#[test]
fn manual_ball_move_cannot_score_and_unknown_last_touch_is_neutral() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    move_ball(&mut s, [5.0, 0.0, 0.11]);
    s.ball.as_mut().unwrap().epoch += 1;
    step(&mut e, &s, 0.01);
    assert_eq!(e.core.get_game(false).teams[Side::Home].score, 0);
    move_ball(&mut s, [0.0, 0.0, 0.11]);
    step(&mut e, &s, 0.01);
    move_ball(&mut s, [0.0, 4.0, 0.11]);
    step(&mut e, &s, 0.01);
    assert_eq!(e.core.get_game(false).state, State::Ready);
    assert_eq!(e.core.get_game(false).kicking_side, None);
}

#[test]
fn indirect_goal_and_second_touch_restrictions_survive_ball_free() {
    for second in [false, true] {
        let (mut e, mut s) = fixture();
        playing(&mut e, &s);
        e.restarts.push((SetPlay::ThrowIn, Side::Home));
        step(&mut e, &s, 0.01);
        let placement = step(&mut e, &s, 0.6)
            .into_iter()
            .find_map(|v| {
                if let Effect::Ball(p) = v {
                    Some(p)
                } else {
                    None
                }
            })
            .unwrap();
        move_ball(&mut s, [placement[0], placement[1], 0.11]);
        s.ball.as_mut().unwrap().epoch += 1;
        step(&mut e, &s, 0.4);
        touch(&mut e, &mut s, id(crate::TeamId::Hulks, 1));
        move_ball(&mut s, [1.0, 1.0, 0.11]);
        step(&mut e, &s, 0.01);
        assert_eq!(e.core.get_game(false).set_play, SetPlay::NoSetPlay);
        if second {
            touch(&mut e, &mut s, id(crate::TeamId::Hulks, 1));
            assert_eq!(e.core.get_game(false).set_play, SetPlay::IndirectFreeKick);
            assert_eq!(e.core.get_game(false).kicking_side, Some(Side::Away));
        } else {
            move_ball(&mut s, [5.0, 0.0, 0.11]);
            step(&mut e, &s, 0.01);
            assert_eq!(e.core.get_game(false).teams[Side::Home].score, 0);
            assert_eq!(e.core.get_game(false).set_play, SetPlay::GoalKick);
        }
    }
}
#[test]
fn timeout_and_global_stuck_use_the_core_restart_lifecycle() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    step(&mut e, &s, 10.0); // Kickoff expires; no robot within a meter.
    step(&mut e, &s, 30.0);
    assert_eq!(e.core.get_game(false).state, State::Ready);
    assert_eq!(e.core.get_game(false).kicking_side, None);
    step(&mut e, &s, 45.0);
    step(&mut e, &s, 2.1);
    s.robots[0].position = [0.3, 0.0, 0.65];
    s.robots[0].feet = vec![[0.3, 0.0]];
    step(&mut e, &s, 0.1);
    step(&mut e, &s, 10.1);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)].penalty,
        Penalty::LocalGameStuck
    );
}

#[test]
#[ignore = "requires MuJoCo, motion models and ONNX Runtime"]
fn automatic_referee_robot_roundtrip_and_physical_penalty_handling() {
    use crate::{
        bevy_mujoco::{MjcfObject, MujocoWorldPlugin, SharedPhysics, SimulationMode},
        simulation::PhysicsWorker,
        team::Team,
    };
    use bevy::prelude::*;
    use ros_z::prelude::*;
    use std::{thread, time::Instant};
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let router = runtime
        .block_on(
            ContextBuilder::default()
                .with_mode("router")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(["tcp/127.0.0.1:0"])
                .build(),
        )
        .unwrap();
    let endpoint =
        runtime.block_on(async { router.session().info().locators().await[0].to_string() });
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let parameters = root.join("../../etc/parameters");
    let config = crate::Configuration {
        referee: RefereeMode::Automatic,
        competition: Competition::Middle,
        field_configuration: Some(
            crate::FieldConfiguration::load(&parameters, "hsl_small").unwrap(),
        ),
        parameter_root: parameters,
        model_directory: root.join("../../etc/neural_networks"),
        router: Some(endpoint),
        namespace: crate::robot_namespace(crate::RobotId::FIRST),
        location: Some("hsl_small".into()),
        robot_count: 2,
        opponent_count: 2,
        profile: std::env::var("SIMULATOR_TEST_PROFILE")
            .ok()
            .map(|s| serde_json::from_value(serde_json::Value::String(s)).unwrap())
            .unwrap_or(crate::Profile::MotionBehavior),
        controller: crate::ControllerSource::External,
    };
    let team = runtime
        .block_on(Team::new(runtime.handle().clone(), config))
        .unwrap();
    {
        let mut e = team.referee().unwrap().engine.lock().unwrap();
        // Keep protocol delay real; shorten only fixture setup and penalty duration.
        e.core.params.competition.set_plays[SetPlay::KickOff].ready_duration =
            Duration::from_secs(2);
        e.core.params.competition.penalties[Penalty::PickedUp].duration = Duration::from_secs(2);
    }
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    for member in team.members() {
        let entity = app
            .world_mut()
            .spawn((
                MjcfObject::new(root.join("assets/k1_robot.xml"), "Trunk")
                    .with_free_joint("world_joint")
                    .grounded(),
                member.pose,
            ))
            .id();
        team.bind(member.id, entity);
    }
    let radius = 0.11;
    let ball_parameters = team.members()[0]
        .io
        .parameters
        .snapshot()
        .typed()
        .ball
        .clone();
    app.world_mut().spawn((
        MjcfObject::from_factory(
            move || crate::scene::ball::ball_spec(radius, &ball_parameters),
            "ball",
        )
        .with_free_joint("ball_free_joint")
        .grounded(),
        Transform::default(),
        crate::scene::ball::Ball,
    ));
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    let mut worker = PhysicsWorker::start_team(physics.clone(), team.clone());
    let members = team.members();
    let raw = runtime.block_on(async {
        let mut caches = Vec::new();
        for member in &members {
            caches.push(
                member
                    .io
                    .node()
                    .subscriber::<Option<types::game_controller_state::GameControllerState>>(
                        "game_controller_state",
                    )
                    .cache(1)
                    .build()
                    .await
                    .unwrap(),
            );
        }
        caches
    });
    let wait = |label: &str, worker: &mut PhysicsWorker, predicate: &mut dyn FnMut() -> bool| {
        let start = Instant::now();
        while !predicate() {
            assert!(worker.poll().is_none(), "worker failed");
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "{label}: condition timeout"
            );
            thread::sleep(Duration::from_millis(10));
        }
    };
    physics.lock().mode = SimulationMode::Running;
    eprintln!("started physics");
    wait("start", &mut worker, &mut || {
        team.referee()
            .unwrap()
            .engine
            .lock()
            .unwrap()
            .core
            .get_game(false)
            .state
            == State::Playing
    });
    {
        let e = team.referee().unwrap().engine.lock().unwrap();
        assert_eq!(e.core.get_game(true).state, State::Set);
    }
    eprintln!("reached Playing");
    wait("returns", &mut worker, &mut || {
        team.referee().unwrap().engine.lock().unwrap().returns.len() == 4
    });
    eprintln!("received all returns");
    wait("raw Playing", &mut worker, &mut || {
        raw.iter().all(|cache| {
            cache.get_latest().is_some_and(|v| {
                v.as_ref()
                    .as_ref()
                    .is_some_and(|g| g.game_state == hsl_network_messages::GameState::Playing)
            })
        })
    });
    eprintln!("raw Playing received");
    let referee = team.referee().unwrap();
    {
        let mut e = referee.engine.lock().unwrap();
        e.enabled = false;
        // Clear any positioning penalties before the targeted handler check.
        for member in &members {
            e.commands.push(VAction::Unpenalize(Unpenalize {
                side: side(member.id.team),
                player: PlayerNumber::new(member.id.number),
                force: true,
            }));
        }
    }
    thread::sleep(Duration::from_millis(100));
    let selected = members[0].entity.unwrap();
    let other_epochs: Vec<_> = members[1..]
        .iter()
        .map(|m| physics.lock().object_epoch(m.entity.unwrap()))
        .collect();
    referee
        .engine
        .lock()
        .unwrap()
        .commands
        .push(VAction::Penalize(Penalize {
            side: Side::Home,
            player: PlayerNumber::new(1),
            call: PenaltyCall::RequestForPickUp,
        }));
    wait("sideline", &mut worker, &mut || {
        physics
            .lock()
            .object_pose(selected)
            .unwrap()
            .translation
            .z
            .abs()
            > 3.3
    });
    assert!(physics.lock().object_pose(selected).unwrap().translation.x < 0.0);
    for (member, epoch) in members[1..].iter().zip(other_epochs) {
        assert_eq!(physics.lock().object_epoch(member.entity.unwrap()), epoch);
    }
    physics.lock().mode = SimulationMode::Paused;
    thread::sleep(Duration::from_millis(50));
    let frozen = referee.engine.lock().unwrap().core.get_time();
    let robot_time = team.now();
    thread::sleep(Duration::from_millis(250));
    assert_eq!(referee.engine.lock().unwrap().core.get_time(), frozen);
    assert_eq!(team.now(), robot_time);
    physics.lock().mode = SimulationMode::Running;
    wait("release", &mut worker, &mut || {
        referee.engine.lock().unwrap().core.get_game(false).teams[Side::Home][PlayerNumber::new(1)]
            .penalty
            == Penalty::NoPenalty
    });
    assert!(
        physics
            .lock()
            .object_pose(selected)
            .unwrap()
            .translation
            .z
            .abs()
            > 2.8,
        "robot must re-enter under its own control"
    );
    drop(worker);
    drop(raw);
    drop(members);
    drop(team);
    drop(app);
    router.shutdown().unwrap();
}

#[test]
fn goal_area_excess_uses_entry_order_and_set_border_distance() {
    for in_set in [false, true] {
        let (mut e, mut s) = fixture();
        if in_set {
            step(&mut e, &s, 0.01);
            step(&mut e, &s, 45.0);
        } else {
            playing(&mut e, &s);
            step(&mut e, &s, 10.0);
        }
        for r in &mut s.robots {
            if r.id.team == crate::TeamId::Hulks {
                r.position = [-4.0, 0.0, 0.65];
                r.feet = vec![[-4.0, 0.0]];
            }
        }
        step(&mut e, &s, 0.01);
        let mut extra = s.robots[0].clone();
        extra.id.number = 4;
        extra.position[0] = -3.55;
        extra.feet = vec![[-3.55, 0.0]];
        s.robots.push(extra);
        step(&mut e, &s, 0.1);
        step(&mut e, &s, 0.6);
        assert_eq!(
            e.core.get_game(false).teams[Side::Home][PlayerNumber::new(4)].penalty,
            Penalty::IllegalPositioning
        );
        for n in 1..=3 {
            assert_eq!(
                e.core.get_game(false).teams[Side::Home][PlayerNumber::new(n)].penalty,
                Penalty::NoPenalty
            );
        }
    }
}
#[test]
fn kickoff_allows_only_one_player_across_halfway() {
    let (mut e, mut s) = fixture();
    step(&mut e, &s, 0.01);
    step(&mut e, &s, 45.0);
    for r in &mut s.robots[..2] {
        r.position = [0.2, 0.1, 0.65];
        r.feet = vec![[0.2, 0.1]];
    }
    step(&mut e, &s, 1.1);
    step(&mut e, &s, 0.6);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)].penalty,
        Penalty::NoPenalty
    );
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(2)].penalty,
        Penalty::IllegalPositioning
    );
}
#[test]
fn halftime_waits_for_ball_then_switches_ends_and_observes_full_break() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    move_ball(&mut s, [0.2, 0.0, 0.11]);
    step(&mut e, &s, 610.0);
    assert_eq!(e.core.get_game(false).state, State::Playing);
    s.ball.as_mut().unwrap().velocity = [0.0; 3];
    assert!(step(&mut e, &s, 0.01).contains(&Effect::Whistle));
    assert_eq!(e.core.get_game(false).state, State::Finished);
    let effects = step(&mut e, &s, 301.0);
    assert!(
        effects.iter().any(
            |v| matches!(v,Effect::Robot(id,[x,_],_) if id.team==crate::TeamId::Hulks && *x>0.0)
        )
    );
    assert_eq!(e.core.get_game(false).state, State::Initial);
    step(&mut e, &s, 298.0);
    assert_eq!(e.core.get_game(false).state, State::Initial);
    step(&mut e, &s, 1.1);
    assert_eq!(e.core.get_game(false).state, State::Ready);
    assert_eq!(e.core.get_game(false).kicking_side, Some(Side::Away));
}

#[tokio::test]
async fn private_controller_rejects_foreign_packets_and_accounts_both_teams() {
    use tokio::{
        net::UdpSocket,
        time::{sleep, timeout},
    };
    let (mut e, s) = fixture();
    playing(&mut e, &s);
    let e = std::sync::Arc::new(std::sync::Mutex::new(e));
    let mut transport = transport::Transport::new(e.clone()).await.unwrap();
    let mut network = crate::network::Network::automatic(transport.endpoints())
        .await
        .unwrap();
    let connection = network.add_robot(crate::TeamId::Hulks).await.unwrap();
    let robot = UdpSocket::bind(("127.0.0.1", connection.ports.state))
        .await
        .unwrap();
    let foreign = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    foreign
        .send_to(b"foreign controller", network.address().unwrap())
        .await
        .unwrap();
    let mut buffer = [0; 2048];
    assert!(
        timeout(Duration::from_millis(100), robot.recv_from(&mut buffer))
            .await
            .is_err()
    );
    transport.connect(network.address().unwrap(), e.clone());
    let (n, _) = timeout(Duration::from_secs(2), robot.recv_from(&mut buffer))
        .await
        .unwrap()
        .unwrap();
    let parsed = hsl_network_messages::GameControllerStateMessage::try_from(&buffer[..n]).unwrap();
    assert_eq!(parsed.game_state, hsl_network_messages::GameState::Set);
    let before = e.lock().unwrap().core.get_game(false).teams[Side::Home].message_budget;
    for team in crate::TeamId::ALL {
        foreign
            .send_to(
                &[0; 512],
                ("127.0.0.1", transport.endpoints().budgets[&team]),
            )
            .await
            .unwrap();
    }
    timeout(Duration::from_secs(2), async {
        loop {
            if crate::TeamId::ALL.iter().all(|team| {
                e.lock().unwrap().core.get_game(false).teams[side(*team)].message_budget
                    == before - 1
            }) {
                break;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(!e.lock().unwrap().core.get_game(false).teams[Side::Home].illegal_communication);
    foreign
        .send_to(
            &[0; 513],
            (
                "127.0.0.1",
                transport.endpoints().budgets[&crate::TeamId::Opponents],
            ),
        )
        .await
        .unwrap();
    timeout(Duration::from_secs(2), async {
        while !e.lock().unwrap().core.get_game(false).teams[Side::Away].illegal_communication {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(!e.lock().unwrap().core.get_game(false).teams[Side::Home].illegal_communication);
    drop(connection);
    drop(network);
    drop(transport);
}
#[test]
fn all_upstream_competition_presets_parse() {
    let (_, s) = fixture();
    for preset in [Competition::Small, Competition::Middle, Competition::Large] {
        let field = crate::FieldConfiguration::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters"),
            "hsl_small",
        )
        .unwrap();
        let mut e = Engine::new(preset, field).unwrap();
        step(&mut e, &s, 0.01);
        assert_eq!(e.core.get_game(false).state, State::Ready);
    }
}

#[test]
fn two_player_kickoff_allows_continuous_contact_outside_circle() {
    let (mut e, mut s) = fixture();
    s.robots.retain(|r| r.id.number <= 2);
    playing(&mut e, &s);
    s.contacts.insert(id(crate::TeamId::Hulks, 1));
    step(&mut e, &s, 0.01);
    move_ball(&mut s, [0.6, 0.0, 0.11]);
    step(&mut e, &s, 0.01);
    move_ball(&mut s, [1.0, 0.0, 0.11]);
    step(&mut e, &s, 0.01);
    s.contacts.clear();
    move_ball(&mut s, [5.0, 0.0, 0.11]);
    step(&mut e, &s, 0.01);
    assert_eq!(e.core.get_game(false).teams[Side::Home].score, 1);
}
#[test]
fn manual_restart_still_places_ball_and_sent_off_player_stays_removed() {
    let (mut e, mut s) = fixture();
    playing(&mut e, &s);
    e.enabled = false;
    e.restarts.push((SetPlay::CornerKick, Side::Home));
    step(&mut e, &s, 0.01);
    assert!(
        step(&mut e, &s, 0.6)
            .iter()
            .any(|e| matches!(e, Effect::Ball([4.5, 3.0])))
    );
    step(&mut e, &s, 0.4);
    assert!(!e.core.get_game(false).stopped);
    e.commands.push(VAction::Penalize(Penalize {
        side: Side::Home,
        player: PlayerNumber::new(1),
        call: PenaltyCall::RequestForPickUp,
    }));
    s.robots[0].penalized = true;
    step(&mut e, &s, 0.01);
    step(&mut e, &s, 0.4);
    e.commands.push(VAction::Penalize(Penalize {
        side: Side::Home,
        player: PlayerNumber::new(1),
        call: PenaltyCall::SendOff,
    }));
    assert!(
        step(&mut e, &s, 0.01)
            .iter()
            .any(|e| matches!(e,Effect::Robot(_,[_,y],_) if y.abs()>4.5))
    );
    step(&mut e, &s, 60.0);
    assert_eq!(
        e.core.get_game(false).teams[Side::Home][PlayerNumber::new(1)].penalty,
        Penalty::SentOff
    );
}

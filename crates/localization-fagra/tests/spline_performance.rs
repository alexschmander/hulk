//! Reproduce timings with:
//! cargo test -p localization-fagra --release --test spline_performance -- --ignored --nocapture --test-threads=1

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    time::Instant,
};

use linear_algebra::Pose3;
use localization_fagra::{spline::PoseSpline, variables::PoseControl};
use nalgebra::{Isometry3, RealField, UnitQuaternion, Vector3};

struct CountingAllocator;

thread_local! {
    // Count only this test thread, not allocations made by the test harness.
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}

// SAFETY: forwards every allocation operation unchanged to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn controls<R: RealField + Copy>(angle_scale: f64) -> [PoseControl<R>; 4] {
    let c = |x| R::from_f64(x).unwrap();
    std::array::from_fn(|i| {
        let x = i as f64;
        PoseControl {
            pose: Pose3::wrap(Isometry3::from_parts(
                Vector3::new(c(0.2 * x * x), c(-0.1 * x), c(0.5 + 0.1 * x.sin())).into(),
                UnitQuaternion::from_euler_angles(
                    c(angle_scale * x),
                    c(angle_scale * x * x * 0.5),
                    c(-angle_scale * x),
                ),
            )),
        }
    })
}

fn exercise<R: RealField + Copy>() {
    let c = |x| R::from_f64(x).unwrap();
    for scale in [0.02, 0.6] {
        let controls = controls::<R>(scale);
        for i in 0..32 {
            let spline = PoseSpline::new(black_box(controls.each_ref()), c(0.2)).unwrap();
            let tau = black_box(c(i as f64 / 31.0));
            black_box(spline.pose(tau).unwrap());
            black_box(spline.velocity(tau).unwrap());
            black_box(spline.state(tau).unwrap());
            black_box(spline.kinematics(tau).unwrap());
            let linearized = spline.linearize().unwrap();
            black_box(linearized.pose(tau).unwrap());
            black_box(linearized.velocity(tau).unwrap());
            black_box(linearized.state(tau).unwrap());
            black_box(linearized.kinematics(tau).unwrap());
        }
    }
}

#[test]
fn preparation_and_evaluation_allocate_nothing() {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    exercise::<f32>();
    exercise::<f64>();
    let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(count, 0, "spline preparation/evaluation allocated");
}

fn measure<T>(name: &str, mut operation: impl FnMut(usize) -> T) {
    const ITERATIONS: usize = 20_000;
    for i in 0..1000 {
        black_box(operation(i));
    }
    let mut timings = [0.0; 7];
    for timing in &mut timings {
        let start = Instant::now();
        for i in 0..ITERATIONS {
            black_box(operation(i));
        }
        *timing = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
    }
    timings.sort_by(f64::total_cmp);
    println!("{name:24} {:8.1} ns (median of 7)", timings[3]);
}

fn timings<R: RealField + Copy>() {
    let c = |x| R::from_f64(x).unwrap();
    for scale in [0.02, 0.6] {
        println!("\n{} rotation scale={scale}", std::any::type_name::<R>());
        let controls = controls::<R>(scale);
        let spline = PoseSpline::new(controls.each_ref(), c(0.2)).unwrap();
        let linearized = spline.linearize().unwrap();
        let taus: [_; 32] = std::array::from_fn(|i| c(i as f64 / 31.0));
        measure("prepare spline", |_| {
            PoseSpline::new(black_box(controls.each_ref()), black_box(c(0.2))).unwrap()
        });
        measure("prepare derivatives", |_| {
            black_box(&spline).linearize().unwrap()
        });
        measure("pose", |i| {
            black_box(&spline).pose(black_box(taus[i % 32])).unwrap()
        });
        measure("velocity", |i| {
            black_box(&spline)
                .velocity(black_box(taus[i % 32]))
                .unwrap()
        });
        measure("state", |i| {
            black_box(&spline).state(black_box(taus[i % 32])).unwrap()
        });
        measure("kinematics", |i| {
            black_box(&spline)
                .kinematics(black_box(taus[i % 32]))
                .unwrap()
        });
        measure("pose + Jacobians", |i| {
            black_box(&linearized)
                .pose(black_box(taus[i % 32]))
                .unwrap()
        });
        measure("velocity + Jacobians", |i| {
            black_box(&linearized)
                .velocity(black_box(taus[i % 32]))
                .unwrap()
        });
        measure("state + Jacobians", |i| {
            black_box(&linearized)
                .state(black_box(taus[i % 32]))
                .unwrap()
        });
        measure("kinematics + Jacobians", |i| {
            black_box(&linearized)
                .kinematics(black_box(taus[i % 32]))
                .unwrap()
        });
    }
}

#[test]
#[ignore = "release microbenchmark; use --release --ignored --nocapture --test-threads=1"]
fn release_timings() {
    timings::<f32>();
    timings::<f64>();
}

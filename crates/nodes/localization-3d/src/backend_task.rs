use std::sync::Arc;

use color_eyre::{Result, eyre::Context as _};
use localization_factrs::VinsBackend;
use ros_z::{pubsub::Publisher, time::Time};
use tokio::{
    sync::watch,
    task::{JoinHandle, JoinSet},
};
use types::time_wrapper::TimeWrapper;

use crate::{diagnostics::SolveDiagnostics, publish::spawn_publisher};

pub(crate) fn spawn_backend_task(
    mut backend: VinsBackend,
    solve_diagnostics_publisher: Publisher<TimeWrapper<SolveDiagnostics>>,
    output_tasks: &mut JoinSet<Result<()>>,
) -> JoinHandle<Result<()>> {
    let (diagnostics, mut pending) = watch::channel::<Option<TimeWrapper<SolveDiagnostics>>>(None);
    let solve_diagnostics_publisher = Arc::new(solve_diagnostics_publisher);
    let publisher = Arc::clone(&solve_diagnostics_publisher);
    spawn_publisher(output_tasks, async move {
        while pending.changed().await.is_ok() {
            let snapshot = pending.borrow_and_update().clone();
            if let Some(snapshot) = snapshot {
                publisher
                    .publish(&snapshot)
                    .await
                    .wrap_err("failed to publish solve diagnostics")?;
            }
        }
        Ok(())
    });
    tokio::task::spawn_blocking(move || -> Result<()> {
        loop {
            let Some(result) = backend.solve_next_blocking()? else {
                continue;
            };

            // Diagnostics are observational: a slow subscriber must not stall the estimator.
            if solve_diagnostics_publisher.has_subscribers() {
                let snapshot = backend
                    .compute_last_solve_diagnostics()
                    .expect("diagnostics are available after a successful solve");
                diagnostics.send_replace(Some(TimeWrapper {
                    time: Time::from_wallclock(result.time),
                    inner: snapshot.into(),
                }));
            }
        }
    })
}

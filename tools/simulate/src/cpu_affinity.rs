//! Keep CPU-heavy robot tasks off the physics core on larger Linux hosts.
#[cfg(target_os = "linux")]
pub(crate) use linux::{configure_child, physics, restore_child, worker};

#[cfg(not(target_os = "linux"))]
pub(crate) fn physics() {}
#[cfg(not(target_os = "linux"))]
pub(crate) fn worker() {}
#[cfg(not(target_os = "linux"))]
pub(crate) fn configure_child(_: &mut tokio::process::Command) {}
#[cfg(not(target_os = "linux"))]
pub(crate) fn restore_child() {}

#[cfg(target_os = "linux")]
mod linux {
    use nix::{
        sched::{CpuSet, sched_getaffinity, sched_setaffinity},
        unistd::Pid,
    };
    use std::{cmp::Reverse, collections::BTreeSet, fs, path::PathBuf, sync::OnceLock};

    const INHERITED_CPUS: &str = "HULK_SIMULATOR_INHERITED_CPUS";
    struct Placement {
        inherited: CpuSet,
        physics: CpuSet,
        workers: CpuSet,
    }

    fn cpus(text: &str) -> Option<Vec<usize>> {
        let mut cpus = BTreeSet::new();
        for range in text.trim().split(',') {
            let (start, end) = range.split_once('-').unwrap_or((range, range));
            let start: usize = start.parse().ok()?;
            let end: usize = end.parse().ok()?;
            if start > end || end >= CpuSet::count() {
                return None;
            }
            cpus.extend(start..=end);
        }
        Some(cpus.into_iter().collect())
    }

    fn cache_size(cpu: usize) -> usize {
        let root = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cache"));
        fs::read_dir(root)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let level: usize = fs::read_to_string(path.join("level"))
                    .ok()?
                    .trim()
                    .parse()
                    .ok()?;
                let size = fs::read_to_string(path.join("size")).ok()?;
                let size = size.trim();
                let (number, multiplier) = if let Some(number) = size.strip_suffix('K') {
                    (number, 1024)
                } else if let Some(number) = size.strip_suffix('M') {
                    (number, 1024 * 1024)
                } else {
                    (size, 1)
                };
                Some((
                    level,
                    number.parse::<usize>().ok()?.checked_mul(multiplier)?,
                ))
            })
            .max()
            .map_or(0, |(_, size)| size)
    }

    fn discover() -> Option<Placement> {
        let allowed = sched_getaffinity(Pid::from_raw(0)).ok()?;
        let mut cores = BTreeSet::new();
        for cpu in 0..CpuSet::count() {
            if allowed.is_set(cpu).ok()? {
                let siblings = fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                ))
                .ok()?;
                cores.insert(cpus(&siblings)?);
            }
        }
        // Leave smaller or restricted hosts to the OS scheduler. Reserve a whole
        // physical core, including its SMT siblings, within the inherited mask.
        if cores.len() < 8 {
            return None;
        }
        let core = cores
            .into_iter()
            .filter(|core| {
                !core.is_empty() && core.iter().all(|&cpu| allowed.is_set(cpu).unwrap_or(false))
            })
            .max_by_key(|core| {
                let cpu = core[0];
                let frequency = fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
                ))
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .unwrap_or(0);
                (cache_size(cpu), frequency, Reverse(cpu))
            })?;
        let mut workers = allowed;
        for &cpu in &core {
            workers.unset(cpu).ok()?;
        }
        let mut physics = CpuSet::new();
        physics.set(core[0]).ok()?;
        Some(Placement {
            inherited: allowed,
            physics,
            workers,
        })
    }

    fn placement() -> Option<&'static Placement> {
        static PLACEMENT: OnceLock<Option<Placement>> = OnceLock::new();
        PLACEMENT.get_or_init(discover).as_ref()
    }

    // The owned MCP server also processes observations. Exclude it from the
    // physics core, then restore the original set before its child discovers
    // the same placement. No affinity is widened beyond that inherited set.
    pub(crate) fn configure_child(command: &mut tokio::process::Command) {
        command.env_remove(INHERITED_CPUS);
        if let Some(placement) = placement() {
            let inherited = (0..CpuSet::count())
                .filter(|&cpu| placement.inherited.is_set(cpu).unwrap_or(false))
                .map(|cpu| cpu.to_string())
                .collect::<Vec<_>>()
                .join(",");
            command.env(INHERITED_CPUS, inherited);
        }
    }
    pub(crate) fn restore_child() {
        let Some(cpus) = std::env::var(INHERITED_CPUS)
            .ok()
            .and_then(|text| cpus(&text))
        else {
            return;
        };
        let mut inherited = CpuSet::new();
        for cpu in cpus {
            if inherited.set(cpu).is_err() {
                return;
            }
        }
        let _ = sched_setaffinity(Pid::from_raw(0), &inherited);
    }

    pub(crate) fn worker() {
        if let Some(placement) = placement() {
            let _ = sched_setaffinity(Pid::from_raw(0), &placement.workers);
        }
    }
    pub(crate) fn physics() {
        if let Some(placement) = placement() {
            let _ = sched_setaffinity(Pid::from_raw(0), &placement.physics);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn parses_kernel_cpu_lists_and_rejects_invalid_ranges() {
            assert_eq!(cpus("0,16\n"), Some(vec![0, 16]));
            assert_eq!(cpus("2-4,9,3"), Some(vec![2, 3, 4, 9]));
            assert_eq!(cpus("4-2"), None);
            assert_eq!(cpus(""), None);
            assert_eq!(cpus(&CpuSet::count().to_string()), None);
        }
    }
}

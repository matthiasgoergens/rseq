//! Per-CPU counter benchmark, randomised complete block design.
//!
//! Never A-then-B: each *block* samples a parameter point θ = (affinity
//! domain, thread count), then runs every arm once at that θ in randomised
//! order. Background-load drift hits all arms of a block equally, so paired
//! within-block ratios are unbiased; quiet machines just tighten the CIs.
//!
//! Anytime property: `bench counter` APPENDS to the results CSV with
//! continuing block ids and fresh entropy per invocation — start and stop
//! whenever, results concatenate. `bench analyze <csv>` is order-independent
//! and reports paired medians, never pooled means.
//!
//! Arms, chosen so each adjacent pair isolates one effect:
//! - `rseq-jit-pad`:   JIT counter, 64-byte stride        (the headline)
//! - `rseq-jit-nopad`: JIT counter, 8-byte stride         (false sharing)
//! - `rseq-asm-nopad`: hand-written asm, 8-byte stride (call overhead; tallies aborts)
//! - `atomic-shard`:   per-CPU padded `lock xadd`         (lock prefix vs rseq)
//! - `atomic-shared`:  one shared `lock xadd` cache line  (contention)
//! - `mutex`:          `std::sync::Mutex<u64>`            (the conventional baseline)

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
mod real {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::io::Write as _;
    use std::sync::Barrier;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    use rseq::codegen::{CompiledSeq, RegionSet};
    use rseq::progs;
    use rseq::rt::{self, PerCpuCounter, RseqArea};
    use rseq::sys;

    const ARMS: &[&str] = &[
        "rseq-jit-pad",
        "rseq-jit-nopad",
        "rseq-asm-nopad",
        "atomic-shard",
        "atomic-shared",
        "mutex",
    ];

    // ---------- entropy ----------

    struct Rng(u64);

    impl Rng {
        fn fresh() -> Self {
            let mut seed = [0u8; 8];
            let data = fs::File::open("/dev/urandom")
                .and_then(|mut f| {
                    use std::io::Read;
                    f.read_exact(&mut seed).map(|()| seed)
                })
                .expect("read /dev/urandom");
            Self(u64::from_le_bytes(data) | 1)
        }

        fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Log-uniform integer in [lo, hi].
        fn log_uniform(&mut self, lo: usize, hi: usize) -> usize {
            let (l, h) = ((lo as f64).ln(), (hi as f64).ln());
            let v = (l + self.unit() * (h - l)).exp().round() as usize;
            v.clamp(lo, hi)
        }

        fn shuffle<T>(&mut self, xs: &mut [T]) {
            for i in (1..xs.len()).rev() {
                let j = (self.next() % (i as u64 + 1)) as usize;
                xs.swap(i, j);
            }
        }
    }

    // ---------- topology ----------

    #[derive(Clone)]
    struct Domain {
        name: &'static str,
        mask: u64,
    }

    fn parse_cpu_list(s: &str) -> u64 {
        let mut mask = 0u64;
        for part in s.trim().split(',') {
            if part.is_empty() {
                continue;
            }
            let mut ends = part.splitn(2, '-');
            let lo: u32 = ends.next().unwrap().trim().parse().unwrap_or(0);
            let hi: u32 = ends.next().map_or(lo, |h| h.trim().parse().unwrap_or(lo));
            for c in lo..=hi.min(63) {
                mask |= 1 << c;
            }
        }
        mask
    }

    fn domains() -> Vec<Domain> {
        let allowed = sys::sched_getaffinity_self();
        let mut out = vec![Domain {
            name: "all",
            mask: allowed,
        }];
        for (name, path) in [
            ("pcore", "/sys/devices/cpu_core/cpus"),
            ("ecore", "/sys/devices/cpu_atom/cpus"),
        ] {
            if let Ok(s) = fs::read_to_string(path) {
                let mask = parse_cpu_list(&s) & allowed;
                if mask != 0 {
                    out.push(Domain { name, mask });
                }
            }
        }
        out
    }

    // ---------- measurement ----------

    struct Sample {
        ns: u64,
        aborts: u64,
    }

    /// Run `body(ops)` on `threads` threads pinned to `mask`. Each worker
    /// times itself after the barrier; the measurement is the span from the
    /// earliest start to the latest end. (Timing from the coordinating
    /// thread is wrong: its own barrier wake-up can lag the workers by
    /// milliseconds on an idle powersave machine, making short workloads
    /// look arbitrarily fast.)
    fn timed<F>(threads: usize, mask: u64, ops: u64, body: F) -> u64
    where
        F: Fn(u64) + Sync,
    {
        let barrier = Barrier::new(threads);
        let mut spans: Vec<(Instant, Instant)> = Vec::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let ret = unsafe { sys::sched_setaffinity(0, mask) };
                        assert!(ret >= 0, "pin thread: {ret}");
                        barrier.wait();
                        let t0 = Instant::now();
                        body(ops);
                        (t0, Instant::now())
                    })
                })
                .collect();
            spans = handles.into_iter().map(|h| h.join().unwrap()).collect();
        });
        let start = spans
            .iter()
            .map(|s| s.0)
            .min()
            .expect("at least one thread");
        let end = spans
            .iter()
            .map(|s| s.1)
            .max()
            .expect("at least one thread");
        (end - start).as_nanos() as u64
    }

    fn area() -> *mut RseqArea {
        rt::current_area().expect("rseq required for benchmarks")
    }

    fn run_arm(arm: &str, threads: usize, mask: u64, ops: u64) -> Sample {
        let total = threads as u64 * ops;
        match arm {
            "rseq-jit-pad" | "rseq-jit-nopad" => {
                let stride = if arm == "rseq-jit-pad" { 8 } else { 1 };
                let (layout, prog, counters) = progs::counter_inc_strided(stride);
                let seq = CompiledSeq::compile(&prog, &layout).expect("compiles");
                let mut rs = RegionSet::new(&layout);
                let ns = timed(threads, mask, ops, |n| {
                    let a = area();
                    for _ in 0..n {
                        unsafe {
                            rs.call_cached(&seq, a, &[]);
                        }
                    }
                    unsafe {
                        core::ptr::write_volatile(&raw mut (*a).rseq_cs, 0);
                    }
                });
                let sum: u64 = rs.region_mut(counters).iter().sum();
                assert_eq!(sum, total, "{arm}: lost or doubled commits");
                Sample { ns, aborts: 0 }
            }
            "rseq-asm-nopad" => {
                let mut counter = PerCpuCounter::new();
                let aborts = AtomicU64::new(0);
                let ns = timed(threads, mask, ops, |n| {
                    let mut local = 0u64;
                    for _ in 0..n {
                        assert!(counter.inc(&mut local));
                    }
                    aborts.fetch_add(local, Ordering::Relaxed);
                });
                assert_eq!(counter.sum(), total, "{arm}: lost or doubled commits");
                Sample {
                    ns,
                    aborts: aborts.into_inner(),
                }
            }
            "atomic-shard" => {
                #[repr(align(64))]
                struct Padded(AtomicU64);
                let shards: Vec<Padded> = (0..rt::MAX_CPUS)
                    .map(|_| Padded(AtomicU64::new(0)))
                    .collect();
                let ns = timed(threads, mask, ops, |n| {
                    let a = area();
                    for _ in 0..n {
                        let cpu =
                            unsafe { core::ptr::read_volatile(&raw const (*a).cpu_id) } as usize;
                        shards[cpu].0.fetch_add(1, Ordering::Relaxed);
                    }
                });
                let sum: u64 = shards.iter().map(|p| p.0.load(Ordering::Relaxed)).sum();
                assert_eq!(sum, total, "{arm}: lost increments");
                Sample { ns, aborts: 0 }
            }
            "atomic-shared" => {
                let counter = AtomicU64::new(0);
                let ns = timed(threads, mask, ops, |n| {
                    for _ in 0..n {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                });
                assert_eq!(counter.into_inner(), total, "{arm}: lost increments");
                Sample { ns, aborts: 0 }
            }
            "mutex" => {
                let counter = Mutex::new(0u64);
                let ns = timed(threads, mask, ops, |n| {
                    for _ in 0..n {
                        *counter.lock().unwrap() += 1;
                    }
                });
                assert_eq!(*counter.lock().unwrap(), total, "{arm}: lost increments");
                Sample { ns, aborts: 0 }
            }
            _ => unreachable!("unknown arm {arm}"),
        }
    }

    // ---------- the block loop ----------

    fn metadata_lines() -> String {
        let mut out = String::new();
        let governor = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .unwrap_or_else(|_| "unknown".into());
        let model = fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split_once(':').map_or("?", |x| x.1).trim().to_string())
            })
            .unwrap_or_else(|| "unknown".into());
        let rev = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .map_or_else(
                || "unknown".into(),
                |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
            );
        let _ = writeln!(out, "# cpu: {model}");
        let _ = writeln!(out, "# governor: {}", governor.trim());
        let _ = writeln!(out, "# git: {rev}");
        let _ = writeln!(out, "# arms: {}", ARMS.join(","));
        out
    }

    fn next_block_id(path: &str) -> u64 {
        fs::read_to_string(path)
            .map(|s| {
                s.lines()
                    .filter(|l| !l.starts_with('#') && !l.starts_with("block"))
                    .filter_map(|l| l.split(',').next())
                    .filter_map(|b| b.parse::<u64>().ok())
                    .max()
                    .map_or(0, |m| m + 1)
            })
            .unwrap_or(0)
    }

    fn run_counter(blocks: u64, ops: u64, path: &str) {
        if rt::current_area().is_none() {
            eprintln!("rseq unavailable; cannot benchmark");
            std::process::exit(1);
        }
        let governor = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .unwrap_or_default();
        if governor.trim() != "performance" {
            eprintln!(
                "note: cpufreq governor is '{}', not 'performance' — noisier blocks, \
                 the pairing still holds",
                governor.trim()
            );
        }
        let doms = domains();
        let mut rng = Rng::fresh();
        let fresh = !std::path::Path::new(path).exists();
        let first = next_block_id(path);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .expect("open csv");
        let _ = write!(file, "{}", metadata_lines());
        if fresh {
            let _ = writeln!(file, "block,domain,threads,ops,arm,ns,mops,aborts");
        }
        let allowed_mask = sys::sched_getaffinity_self();
        for block in first..first + blocks {
            // θ = (domain, threads): domain uniform, threads log-uniform up
            // to 2x the domain size (the oversubscribed regime included).
            let dom = &doms[(rng.next() % doms.len() as u64) as usize];
            let ncpus = dom.mask.count_ones() as usize;
            let threads = rng.log_uniform(1, 2 * ncpus);
            let mut order: Vec<&str> = ARMS.to_vec();
            rng.shuffle(&mut order);
            let mut shown = String::new();
            for arm in order {
                let s = run_arm(arm, threads, dom.mask, ops);
                let mops = (threads as f64 * ops as f64) / (s.ns as f64 / 1e3);
                let _ = writeln!(
                    file,
                    "{block},{},{threads},{ops},{arm},{},{mops:.2},{}",
                    dom.name, s.ns, s.aborts
                );
                let _ = write!(shown, " {arm}={mops:.0}");
            }
            file.flush().expect("flush csv");
            eprintln!("block {block}: {}x{threads}:{shown} (Mops/s)", dom.name);
            // Restore main-thread-visible affinity for the next block.
            let _ = unsafe { sys::sched_setaffinity(0, allowed_mask) };
        }
        eprintln!("appended {blocks} blocks to {path}");
    }

    // ---------- analysis ----------

    struct Row {
        /// Index of the source file — block ids from different files both
        /// start at zero, so blocks are only identified by (file, block).
        file: usize,
        block: u64,
        domain: String,
        threads: usize,
        arm: String,
        mops: f64,
        aborts: u64,
        ops: u64,
    }

    fn read_rows(paths: &[String]) -> Vec<Row> {
        let mut rows = Vec::new();
        for (file, p) in paths.iter().enumerate() {
            let content = fs::read_to_string(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
            for line in content.lines() {
                if line.starts_with('#') || line.starts_with("block") || line.is_empty() {
                    continue;
                }
                let f: Vec<&str> = line.split(',').collect();
                if f.len() != 8 {
                    continue;
                }
                rows.push(Row {
                    file,
                    block: f[0].parse().unwrap_or(0),
                    domain: f[1].into(),
                    threads: f[2].parse().unwrap_or(0),
                    ops: f[3].parse().unwrap_or(0),
                    arm: f[4].into(),
                    mops: f[6].parse().unwrap_or(0.0),
                    aborts: f[7].parse().unwrap_or(0),
                });
            }
        }
        rows
    }

    fn median(xs: &mut [f64]) -> f64 {
        xs.sort_by(f64::total_cmp);
        if xs.is_empty() {
            return f64::NAN;
        }
        let n = xs.len();
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            f64::midpoint(xs[n / 2 - 1], xs[n / 2])
        }
    }

    fn bucket(threads: usize) -> &'static str {
        match threads {
            1 => "1",
            2..=4 => "2-4",
            5..=16 => "5-16",
            _ => "17+",
        }
    }

    /// Per-cell paired data: (throughputs, within-block ratios vs mutex).
    type Cell = (Vec<f64>, Vec<f64>);

    fn analyze(paths: &[String]) {
        let rows = read_rows(paths);
        if rows.is_empty() {
            eprintln!("no data");
            return;
        }
        // Group rows per block. A block is identified by (file, block id) —
        // ids restart at zero in each file — and shares θ across arms by
        // construction.
        let mut blocks: BTreeMap<(usize, u64), BTreeMap<String, f64>> = BTreeMap::new();
        let mut buckets: BTreeMap<(usize, u64), (String, &'static str)> = BTreeMap::new();
        let mut abort_rates: BTreeMap<(String, &'static str), Vec<f64>> = BTreeMap::new();
        for r in &rows {
            let key = (r.file, r.block);
            blocks.entry(key).or_default().insert(r.arm.clone(), r.mops);
            buckets
                .entry(key)
                .or_insert((r.domain.clone(), bucket(r.threads)));
            if r.arm == "rseq-asm-nopad" && r.ops > 0 {
                abort_rates
                    .entry((r.domain.clone(), bucket(r.threads)))
                    .or_default()
                    .push(r.aborts as f64 / (r.ops as f64 * r.threads as f64));
            }
        }
        // Paired within-block: throughput medians and ratio-vs-mutex per
        // (domain, thread-bucket). Incomplete blocks (an interrupted run
        // that did not finish every arm) are dropped: pairing requires the
        // whole block.
        let mut dropped = 0usize;
        let mut cells: BTreeMap<(String, &'static str, String), Cell> = BTreeMap::new();
        for (key, arms) in &blocks {
            if arms.len() != ARMS.len() {
                dropped += 1;
                continue;
            }
            let Some((domain, bkt)) = buckets.get(key).cloned() else {
                continue;
            };
            let mutex = arms.get("mutex").copied();
            for (arm, mops) in arms {
                let cell = cells.entry((domain.clone(), bkt, arm.clone())).or_default();
                cell.0.push(*mops);
                if let Some(m) = mutex
                    && m > 0.0
                {
                    cell.1.push(mops / m);
                }
            }
        }
        if dropped > 0 {
            eprintln!("dropped {dropped} incomplete block(s)");
        }
        println!(
            "{:<7} {:<6} {:<16} {:>7} {:>12} {:>12} {:>8}",
            "domain", "thr", "arm", "blocks", "med Mops/s", "x vs mutex", "faster%"
        );
        for ((domain, bkt, arm), (mut mops, mut ratios)) in cells {
            let n = mops.len();
            let med = median(&mut mops);
            let faster = if ratios.is_empty() {
                f64::NAN
            } else {
                100.0 * ratios.iter().filter(|&&r| r > 1.0).count() as f64 / ratios.len() as f64
            };
            let ratio = median(&mut ratios);
            println!(
                "{domain:<7} {bkt:<6} {arm:<16} {n:>7} {med:>12.1} {ratio:>12.2} {faster:>7.0}%"
            );
        }
        println!();
        println!("abort rate (rseq-asm-nopad, aborts per op):");
        for ((domain, bkt), mut rates) in abort_rates {
            let n = rates.len();
            let med = median(&mut rates);
            println!("  {domain:<7} {bkt:<6} n={n:<4} median {med:.2e}");
        }
    }

    // ---------- CLI ----------

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.first().map(String::as_str) {
            Some("counter") => {
                let mut blocks = 16u64;
                let mut ops = 300_000u64;
                let mut out = "bench-counter.csv".to_string();
                let mut it = args[1..].iter();
                while let Some(a) = it.next() {
                    match a.as_str() {
                        "--blocks" => blocks = it.next().expect("--blocks N").parse().expect("N"),
                        "--ops" => ops = it.next().expect("--ops N").parse().expect("N"),
                        "--out" => out.clone_from(it.next().expect("--out FILE")),
                        other => panic!("unknown flag {other}"),
                    }
                }
                run_counter(blocks, ops, &out);
            }
            Some("analyze") => analyze(&args[1..]),
            _ => {
                eprintln!("usage: bench counter [--blocks N] [--ops N] [--out FILE]");
                eprintln!("       bench analyze <csv>...");
                std::process::exit(2);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
fn main() {
    real::main();
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
fn main() {
    eprintln!("bench requires x86-64 Linux with glibc");
}

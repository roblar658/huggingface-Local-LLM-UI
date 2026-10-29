use anyhow::{ anyhow, Result };
use std::sync::atomic::{ AtomicBool, Ordering };
use sysinfo::System;

#[derive(Debug, Clone)]
pub struct HardwareSpec {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub gpu_name: String,
    pub dedicated_vram_gb: f64,
}

impl HardwareSpec {
    pub fn detect() -> Result<Self> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_cores = sys.cpus().len().max(1);
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_else(|| "Ukjent CPU".to_string());

        let total_ram_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        let (gpu_name, dedicated_vram_gb) = detect_windows_gpu()
            .unwrap_or_else(|_| ("Standard Windows GPU".to_string(), 0.0));

        Ok(HardwareSpec {
            cpu_brand,
            cpu_cores,
            total_ram_gb,
            available_ram_gb,
            gpu_name,
            dedicated_vram_gb,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceLevel {
    /// Normal drift: bruker ~75% av logiske kjerner (trygt under maks kapasitet)
    OptimalSubMax,
    /// Moderat ressursbruk eller når brukeren har byttet til et annet program (Alt+Tab)
    Balanced,
    /// Lite ledig RAM / høy systemlast: reduserer tråder og legger inn mikropauser
    LowResources,
    /// Kritisk lavt minne / overbelastet: frigjør working set og tar en midlertidig pause
    CriticalPause,
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceSnapshot {
    pub avail_phys_mb: u64,
    pub avail_page_file_mb: u64,
    pub memory_load_pct: u32,
    pub app_focused: bool,
    pub level: ResourceLevel,
    pub recommended_threads: usize,
    pub total_cores: usize,
    pub prefill_chunk_size: usize,
    pub throttle_sleep_ms: u64,
}

impl ResourceSnapshot {
    pub fn status_badge(&self) -> String {
        let mode_str = match self.level {
            ResourceLevel::OptimalSubMax => "Trygg sub-maks",
            ResourceLevel::Balanced => {
                if !self.app_focused {
                    "Bakgrunnsmodus (skånsom)"
                } else {
                    "Balansert"
                }
            }
            ResourceLevel::LowResources => "Lavt minne (redusert fart)",
            ResourceLevel::CriticalPause => "Kritisk minne (midlertidig pause)",
        };
        format!(
            "{} [{}/{} tråder | {:.1} GB ledig]",
            mode_str,
            self.recommended_threads,
            self.total_cores,
            self.avail_phys_mb as f64 / 1024.0
        )
    }
}

pub struct ResourceGovernor {
    total_cores: usize,
    submax_threads: usize,
    balanced_threads: usize,
    low_threads: usize,
    app_focused: AtomicBool,
    pool_submax: rayon::ThreadPool,
    pool_balanced: rayon::ThreadPool,
    pool_low: rayon::ThreadPool,
    pool_minimal: rayon::ThreadPool,
}

impl ResourceGovernor {
    pub fn new() -> Self {
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);

        // Trygt under maksimalt: ~75% av kjerner, og alltid minst 1-2 kjerner ledig til OS/UI
        let submax_threads = if total_cores <= 2 {
            1
        } else if total_cores <= 4 {
            total_cores - 1
        } else {
            ((total_cores * 3) / 4).min(total_cores.saturating_sub(2)).max(2)
        };

        let balanced_threads = (total_cores / 2).max(1).min(submax_threads);
        let low_threads = if total_cores >= 4 { 2 } else { 1 };

        let build_pool = |num_threads: usize, lowest_prio: bool| -> rayon::ThreadPool {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .start_handler(move |_| {
                    lower_current_thread_priority(lowest_prio);
                })
                .build()
                .unwrap_or_else(|_| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .expect("Kunne ikke opprette fallback Rayon-pool")
                })
        };

        let pool_submax = build_pool(submax_threads, false);
        let pool_balanced = build_pool(balanced_threads, false);
        let pool_low = build_pool(low_threads, true);
        let pool_minimal = build_pool(1, true);

        Self {
            total_cores,
            submax_threads,
            balanced_threads,
            low_threads,
            app_focused: AtomicBool::new(true),
            pool_submax,
            pool_balanced,
            pool_low,
            pool_minimal,
        }
    }

    pub fn set_app_focused(&self, focused: bool) {
        self.app_focused.store(focused, Ordering::Relaxed);
    }

    pub fn is_app_focused(&self) -> bool {
        self.app_focused.load(Ordering::Relaxed)
    }

    pub fn sample(&self) -> ResourceSnapshot {
        let (avail_phys_mb, avail_page_file_mb, memory_load_pct) = query_fast_memory_status();
        let app_focused = self.is_app_focused();

        let (level, recommended_threads, prefill_chunk_size, throttle_sleep_ms) =
            if avail_phys_mb < 500 || avail_page_file_mb < 500 || memory_load_pct >= 96 {
                (ResourceLevel::CriticalPause, 1, 4, 45)
            } else if avail_phys_mb < 1100 || avail_page_file_mb < 1000 || memory_load_pct >= 90 {
                (ResourceLevel::LowResources, self.low_threads, 8, 14)
            } else if !app_focused || avail_phys_mb < 2100 || avail_page_file_mb < 1900 || memory_load_pct >= 82 {
                let sleep_ms = if !app_focused { 2 } else { 4 };
                (ResourceLevel::Balanced, self.balanced_threads, 16, sleep_ms)
            } else {
                (ResourceLevel::OptimalSubMax, self.submax_threads, 24, 0)
            };

        ResourceSnapshot {
            avail_phys_mb,
            avail_page_file_mb,
            memory_load_pct,
            app_focused,
            level,
            recommended_threads,
            total_cores: self.total_cores,
            prefill_chunk_size,
            throttle_sleep_ms,
        }
    }

    pub fn run_in_optimal_pool<OP, R>(&self, level: ResourceLevel, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        match level {
            ResourceLevel::OptimalSubMax => self.pool_submax.install(op),
            ResourceLevel::Balanced => self.pool_balanced.install(op),
            ResourceLevel::LowResources => self.pool_low.install(op),
            ResourceLevel::CriticalPause => self.pool_minimal.install(op),
        }
    }

    /// Sjekker ressurser før et inferens-steg. Tar automatiske, avbrytbare mikropauser ved
    /// kritisk lavt minne / overbelastning, men låser aldri tråden permanent.
    pub fn pace_and_recover<F>(
        &self,
        cancel_flag: Option<&AtomicBool>,
        mut on_status: F,
    ) -> Result<ResourceSnapshot>
    where
        F: FnMut(&str),
    {
        if let Some(flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                return Err(anyhow!("Generering avbrutt av bruker."));
            }
        }

        let mut snap = self.sample();

        match snap.level {
            ResourceLevel::CriticalPause => {
                trim_process_working_set();
                // Vent i opptil 15 x 200ms (3.0s maks per steg) for at OS skal frigjøre RAM.
                // Fortsetter deretter på 1 tråd slik at genereringen aldri henger fast.
                for step in 0..15 {
                    if let Some(flag) = cancel_flag {
                        if flag.load(Ordering::Relaxed) {
                            return Err(anyhow!("Generering avbrutt av bruker."));
                        }
                    }
                    if step == 0 || step % 5 == 0 {
                        on_status(&format!(
                            "Lite ledig minne ({} MB RAM / {}% brukt) – tar en kort automatisk pause for å unngå heng...",
                            snap.avail_phys_mb, snap.memory_load_pct
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    snap = self.sample();
                    if snap.level != ResourceLevel::CriticalPause {
                        break;
                    }
                }
                if snap.level == ResourceLevel::CriticalPause {
                    std::thread::sleep(std::time::Duration::from_millis(snap.throttle_sleep_ms));
                }
            }
            ResourceLevel::LowResources => {
                trim_process_working_set();
                std::thread::sleep(std::time::Duration::from_millis(snap.throttle_sleep_ms));
            }
            ResourceLevel::Balanced => {
                if snap.throttle_sleep_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(snap.throttle_sleep_ms));
                }
            }
            ResourceLevel::OptimalSubMax => {}
        }

        Ok(snap)
    }
}

#[cfg(windows)]
#[repr(C)]
struct MEMORYSTATUSEX {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn SetThreadPriority(hThread: *mut std::ffi::c_void, nPriority: i32) -> i32;
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn SetProcessWorkingSetSize(
        hProcess: *mut std::ffi::c_void,
        dwMinimumWorkingSetSize: usize,
        dwMaximumWorkingSetSize: usize,
    ) -> i32;
}

#[cfg(windows)]
pub fn query_fast_memory_status() -> (u64, u64, u32) {
    unsafe {
        let mut mem: MEMORYSTATUSEX = std::mem::zeroed();
        mem.dw_length = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut mem) != 0 {
            let avail_phys_mb = mem.ull_avail_phys / (1024 * 1024);
            let avail_page_mb = mem.ull_avail_page_file / (1024 * 1024);
            return (avail_phys_mb, avail_page_mb, mem.dw_memory_load);
        }
    }
    (4096, 4096, 50)
}

#[cfg(not(windows))]
pub fn query_fast_memory_status() -> (u64, u64, u32) {
    (4096, 4096, 50)
}

#[cfg(windows)]
pub fn lower_current_thread_priority(lowest: bool) {
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    const THREAD_PRIORITY_LOWEST: i32 = -2;
    unsafe {
        let prio = if lowest {
            THREAD_PRIORITY_LOWEST
        } else {
            THREAD_PRIORITY_BELOW_NORMAL
        };
        let _ = SetThreadPriority(GetCurrentThread(), prio);
    }
}

#[cfg(not(windows))]
pub fn lower_current_thread_priority(_lowest: bool) {}

#[cfg(windows)]
pub fn trim_process_working_set() {
    unsafe {
        let _ = SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX);
    }
}

#[cfg(not(windows))]
pub fn trim_process_working_set() {}

#[cfg(windows)]
fn detect_windows_gpu() -> Result<(String, f64)> {
    use windows::Win32::Graphics::Dxgi::{ CreateDXGIFactory1, IDXGIFactory1 };

    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
        let mut best_name = "Ingen dedikert GPU funnet".to_string();
        let mut max_vram = 0usize;

        let mut index = 0;
        while let Ok(adapter) = factory.EnumAdapters1(index) {
            if let Ok(desc) = adapter.GetDesc1() {
                let vram = desc.DedicatedVideoMemory;
                if vram > max_vram {
                    max_vram = vram;
                    let end = desc
                        .Description
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(desc.Description.len());
                    best_name = String::from_utf16_lossy(&desc.Description[..end]);
                }
            }
            index += 1;
        }

        let vram_gb = max_vram as f64 / 1024.0 / 1024.0 / 1024.0;
        Ok((best_name, vram_gb))
    }
}

#[cfg(not(windows))]
fn detect_windows_gpu() -> Result<(String, f64)> {
    Ok(("Ikke-Windows".to_string(), 0.0))
}

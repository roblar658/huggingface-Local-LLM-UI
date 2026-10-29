use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
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

        let (mut gpu_name, mut dedicated_vram_gb) = detect_windows_gpu()
            .unwrap_or_else(|_| ("Standard Windows GPU".to_string(), 0.0));

        // Suppler med NVML/CUDA dersom DXGI rapporterer 4GB-tak på 32-bit struktur eller vi har direkte NVML
        if let Some(tel) = crate::gpu::query_gpu_telemetry() {
            if tel.vram_total_mb > 0 {
                let nvml_gb = tel.vram_total_mb as f64 / 1024.0;
                if nvml_gb > dedicated_vram_gb {
                    dedicated_vram_gb = nvml_gb;
                }
            }
        }
        if let Some(cuda) = crate::gpu::CudaEngine::global() {
            if !cuda.device_name.is_empty() {
                gpu_name = format!("{} (CUDA PTX aktiv)", cuda.device_name);
            }
        }

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
    /// Normal drift: bruker trygt sub-maks av GPU og ~75% av CPU-kjerner
    OptimalSubMax,
    /// Moderat ressursbruk eller når brukeren har byttet til et annet program (Alt+Tab)
    Balanced,
    /// Lite ledig RAM/VRAM eller høy temperatur: reduserer GPU-grid og CPU-tråder
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
    pub gpu_active: bool,
    pub gpu_util_pct: u32,
    pub gpu_vram_used_mb: u64,
    pub gpu_vram_free_mb: u64,
    pub gpu_vram_total_mb: u64,
    pub gpu_offloaded_mb: u64,
    pub gpu_temp_c: u32,
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
            ResourceLevel::LowResources => "Lavt minne/høy last (redusert)",
            ResourceLevel::CriticalPause => "Kritisk last (midlertidig pause)",
        };

        if self.gpu_active {
            let temp_part = if self.gpu_temp_c > 0 {
                format!(", {}°C", self.gpu_temp_c)
            } else {
                String::new()
            };
            format!(
                "{} [GPU: {:.1} GB i VRAM ({:.1} GB ledig, {}% last{}) | {}/{} tråder | {:.1} GB RAM ledig]",
                mode_str,
                self.gpu_offloaded_mb as f64 / 1024.0,
                self.gpu_vram_free_mb as f64 / 1024.0,
                self.gpu_util_pct,
                temp_part,
                self.recommended_threads,
                self.total_cores,
                self.avail_phys_mb as f64 / 1024.0
            )
        } else {
            format!(
                "{} [{}/{} tråder | {:.1} GB RAM ledig ({}% brukt)]",
                mode_str,
                self.recommended_threads,
                self.total_cores,
                self.avail_phys_mb as f64 / 1024.0,
                self.memory_load_pct
            )
        }
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
    last_pause: std::sync::Mutex<std::time::Instant>,
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
            ((total_cores * 3) / 4)
                .min(total_cores.saturating_sub(2))
                .max(2)
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
            last_pause: std::sync::Mutex::new(
                std::time::Instant::now() - std::time::Duration::from_secs(10),
            ),
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

        let tel = crate::gpu::query_gpu_telemetry().unwrap_or_default();
        let (gpu_active, gpu_offloaded_mb) = if let Some(cuda) = crate::gpu::CudaEngine::global() {
            let bytes = cuda.offloaded_bytes.load(Ordering::Relaxed);
            (true, (bytes / (1024 * 1024)) as u64)
        } else {
            (false, 0)
        };

        // Merk: GPU-inferens allokerer 0 bytes ny VRAM under kjøring (alt ligger i fast workspace).
        // Lav ledig VRAM skal derfor ALDRI låse tråden i CriticalPause, kun justere GPU-grid skånsomt.
        let vram_tight = tel.vram_total_mb > 0 && tel.vram_free_mb < 450;
        let gpu_hot_crit = tel.temp_c >= 89;
        let gpu_hot_low = tel.temp_c >= 84;
        let gpu_hot_bal = tel.temp_c >= 79;

        let (level, recommended_threads, prefill_chunk_size, throttle_sleep_ms) =
            if (avail_phys_mb < 350 && memory_load_pct >= 97)
                || avail_page_file_mb < 350
                || gpu_hot_crit
            {
                (ResourceLevel::CriticalPause, 1, 8, 15)
            } else if avail_phys_mb < 900
                || avail_page_file_mb < 800
                || memory_load_pct >= 92
                || gpu_hot_low
            {
                (ResourceLevel::LowResources, self.low_threads, 12, 6)
            } else if !app_focused
                || avail_phys_mb < 1800
                || avail_page_file_mb < 1600
                || memory_load_pct >= 85
                || gpu_hot_bal
                || vram_tight
            {
                let sleep_ms = if !app_focused { 1 } else { 2 };
                (ResourceLevel::Balanced, self.balanced_threads, 20, sleep_ms)
            } else {
                (ResourceLevel::OptimalSubMax, self.submax_threads, 32, 0)
            };

        // Oppdater dynamisk CUDA thread-block tak slik at GPU alltid holder seg trygt under maks
        if let Some(cuda) = crate::gpu::CudaEngine::global() {
            let grid_cap = match level {
                ResourceLevel::OptimalSubMax => {
                    if tel.gpu_util_pct >= 90 {
                        96
                    } else {
                        144
                    }
                }
                ResourceLevel::Balanced => 56,
                ResourceLevel::LowResources => 24,
                ResourceLevel::CriticalPause => 8,
            };
            cuda.max_grid_x.store(grid_cap, Ordering::Relaxed);
        }

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
            gpu_active,
            gpu_util_pct: tel.gpu_util_pct,
            gpu_vram_used_mb: tel.vram_used_mb,
            gpu_vram_free_mb: tel.vram_free_mb,
            gpu_vram_total_mb: tel.vram_total_mb,
            gpu_offloaded_mb,
            gpu_temp_c: tel.temp_c,
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

    /// Sjekker CPU-, RAM- og GPU-ressurser før et inferens-steg. Tar automatiske, avbrytbare
    /// mikropauser ved kritisk lavt minne / overbelastning uten å låse tråden permanent.
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
                let should_pause = if let Ok(mut last) = self.last_pause.lock() {
                    if last.elapsed() >= std::time::Duration::from_secs(4) {
                        *last = std::time::Instant::now();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if should_pause {
                    on_status(&format!(
                        "Frigjør systemminne ({} MB RAM ledig) – fortsetter i skånsom modus...",
                        snap.avail_phys_mb
                    ));
                    std::thread::sleep(std::time::Duration::from_millis(220));
                    snap = self.sample();
                } else {
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
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

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

use anyhow::Result;
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

        let cpu_cores = sys.cpus().len();
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
                    let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
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

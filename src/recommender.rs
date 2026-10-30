use crate::hardware::HardwareSpec;
use crate::hf_client::HfModel;

#[derive(Debug, Clone)]
pub enum FitLevel {
    FullGpuOffload,
    HybridGpuCpu,
    CpuRamFallback,
    TooLarge,
}

#[derive(Debug, Clone)]
pub struct ModelAssessment {
    pub model_id: String,
    pub est_param_size: String,
    pub req_memory_gb: f64,
    pub fit_level: FitLevel,
}

/// Finner parameter-størrelse (f.eks. 397B, 70B, 35B, 27B, 14B, 9B, 8B, 7B, 3B, 1.5B)
/// ved å sjekke separate ord/tokens i modellnavnet slik at "27b" ikke feiltolkes som "7b".
pub fn assess_model(model: &HfModel, hardware: &HardwareSpec) -> ModelAssessment {
    let id_lower = model.id.to_lowercase();
    let tokens: Vec<&str> = id_lower
        .split(|c: char| !c.is_alphanumeric() && c != '.')
        .filter(|s| !s.is_empty())
        .collect();

    let mut found: Option<(&str, f64)> = None;

    for t in &tokens {
        if let Some(num_str) = t.strip_suffix('b') {
            if let Ok(params) = num_str.parse::<f64>() {
                let req_gb = (params * 0.72).max(1.2);
                let label = if params >= 100.0 {
                    "100B+"
                } else if params >= 65.0 {
                    "70B"
                } else if params >= 30.0 {
                    "32B/35B"
                } else if params >= 20.0 {
                    "27B"
                } else if params >= 12.0 {
                    "14B"
                } else if params >= 8.5 {
                    "9B"
                } else if params >= 6.5 {
                    "7B/8B"
                } else if params >= 2.5 {
                    "3B"
                } else {
                    "1B/1.5B"
                };
                found = Some((label, req_gb));
                break;
            }
        }
    }

    let (param_str, req_gb) = found.unwrap_or(("7B (antatt)", 5.5));

    let fit_level = if hardware.dedicated_vram_gb >= req_gb + 0.5 {
        FitLevel::FullGpuOffload
    } else if hardware.dedicated_vram_gb >= 2.0
        && (hardware.dedicated_vram_gb + hardware.available_ram_gb >= req_gb + 1.0)
    {
        FitLevel::HybridGpuCpu
    } else if hardware.available_ram_gb >= req_gb + 1.0 {
        FitLevel::CpuRamFallback
    } else {
        FitLevel::TooLarge
    };

    ModelAssessment {
        model_id: model.id.clone(),
        est_param_size: param_str.to_string(),
        req_memory_gb: req_gb,
        fit_level,
    }
}

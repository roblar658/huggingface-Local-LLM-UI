use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct OcrEngine;

impl OcrEngine {
    fn get_python_script_path() -> Option<PathBuf> {
        let mut candidates = vec![
            PathBuf::from("tools/ocr_engine.py"),
            PathBuf::from("tools\\ocr_engine.py"),
        ];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates.push(parent.join("tools").join("ocr_engine.py"));
            }
        }

        for c in candidates {
            if c.exists() {
                return Some(c);
            }
        }
        None
    }

    fn get_native_tool_path() -> PathBuf {
        let candidates = [
            PathBuf::from("tools/win_ocr/bin/win_ocr.exe"),
            PathBuf::from("tools\\win_ocr\\bin\\win_ocr.exe"),
            PathBuf::from("target/release/win_ocr.exe"),
            PathBuf::from("target\\release\\win_ocr.exe"),
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|dir| dir.join("win_ocr.exe")))
                .unwrap_or_else(|| PathBuf::from("win_ocr.exe")),
        ];

        for c in &candidates {
            if c.exists() {
                return c.clone();
            }
        }
        PathBuf::from("tools/win_ocr/bin/win_ocr.exe")
    }

    /// Kjører OCR på et bilde fra utklippstavlen (f.eks. etter Win+Shift+S eller PrintScreen).
    /// Benytter RapidOCR (PP-OCRv4 ONNX) først for høyest presisjon og linjeskift,
    /// og faller tilbake til Windows Media OCR ved behov.
    pub fn recognize_clipboard() -> Result<String> {
        // 1. Forsøk RapidOCR (state-of-the-art dyp lærings-OCR)
        if let Some(script) = Self::get_python_script_path() {
            if let Ok(output) = Command::new("python").arg(&script).output() {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                } else {
                    let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    if err.starts_with("INGEN_BILDE") {
                        return Err(anyhow!(
                            "Ingen bilde i utklippstavlen. Ta et skjermbilde (f.eks. med Win+Shift+S) eller kopier et bilde først."
                        ));
                    }
                }
            }
        }

        // 2. Fallback til Windows native OCR
        let tool = Self::get_native_tool_path();
        if !tool.exists() {
            return Err(anyhow!(
                "Fant ikke OCR-motoren. Sjekk at Python med RapidOCR eller win_ocr.exe finnes i {:?}",
                tool
            ));
        }

        let output = Command::new(&tool)
            .output()
            .map_err(|e| anyhow!("Kunne ikke starte OCR-prosess: {}", e))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if err.starts_with("INGEN_BILDE") {
                return Err(anyhow!(
                    "Ingen bilde i utklippstavlen. Ta et skjermbilde (f.eks. med Win+Shift+S) eller kopier et bilde først."
                ));
            } else if err.starts_with("TOM_TEKST") {
                return Err(anyhow!("Bildet ble analysert, men ingen tekst ble funnet."));
            } else {
                return Err(anyhow!("OCR feilet: {}", err));
            }
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            return Err(anyhow!("Ingen tekst funnet i bildet."));
        }

        Ok(text)
    }

    /// Kjører OCR på en bildefil fra disk
    pub fn recognize_file(path: &Path) -> Result<String> {
        if let Some(script) = Self::get_python_script_path() {
            if let Ok(output) = Command::new("python").arg(&script).arg(path).output() {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                }
            }
        }

        let tool = Self::get_native_tool_path();
        if !tool.exists() {
            return Err(anyhow!("Fant ikke OCR-motoren på {:?}", tool));
        }

        let output = Command::new(&tool)
            .arg(path)
            .output()
            .map_err(|e| anyhow!("Kunne ikke starte OCR-prosess: {}", e))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(anyhow!("OCR feilet: {}", err));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            return Err(anyhow!("Ingen tekst funnet i bildefilen."));
        }

        Ok(text)
    }
}

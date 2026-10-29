use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
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

    /// Henter bilde fra utklippstavlen og lagrer fullversjon + thumbnail rådata
    pub fn grab_clipboard_image(dest_png: &Path, dest_thumb_raw: &Path) -> Result<(usize, usize)> {
        let script = Self::get_python_script_path().ok_or_else(|| {
            anyhow!("Fant ikke tools/ocr_engine.py for henting av utklippstavlebilde")
        })?;

        let output = Command::new("python")
            .arg(&script)
            .arg("--grab-clipboard")
            .arg(dest_png)
            .arg(dest_thumb_raw)
            .output()
            .map_err(|e| anyhow!("Kunne ikke starte python for utklippstavle: {}", e))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if err.starts_with("NO_IMAGE") {
                return Err(anyhow!(
                    "Ingen bilde i utklippstavlen. Ta et skjermbilde (f.eks. med Win+Shift+S) først."
                ));
            }
            return Err(anyhow!("Feil ved henting av bilde fra utklippstavlen: {}", err));
        }

        let out_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let parts: Vec<&str> = out_str.split_whitespace().collect();
        if parts.len() >= 3 && parts[0] == "OK" {
            let w: usize = parts[1].parse().unwrap_or(80);
            let h: usize = parts[2].parse().unwrap_or(80);
            return Ok((w, h));
        }

        Ok((80, 80))
    }

    /// Leser rå RGBA-pikseldata fra generert thumbnail
    pub fn load_thumbnail_raw(thumb_raw: &Path) -> Result<(usize, usize, Vec<u8>)> {
        let file = File::open(thumb_raw)?;
        let mut reader = BufReader::new(file);

        let mut header = String::new();
        reader.read_line(&mut header)?;
        let parts: Vec<&str> = header.trim().split(',').collect();
        if parts.len() < 2 {
            return Err(anyhow!("Ugyldig rådata-header for thumbnail"));
        }
        let w: usize = parts[0].parse()?;
        let h: usize = parts[1].parse()?;

        let mut pixels = Vec::new();
        reader.read_to_end(&mut pixels)?;

        if pixels.len() < w * h * 4 {
            return Err(anyhow!(
                "Ufullstendig rådata: forventet {} bytes, fant {}",
                w * h * 4,
                pixels.len()
            ));
        }

        Ok((w, h, pixels))
    }

    /// Kjører OCR på en bildefil (med RapidOCR PP-OCRv4 ONNX og fallback til Windows OCR)
    pub fn recognize_file(path: &Path) -> Result<String> {
        // 1. Prøv RapidOCR først
        if let Some(script) = Self::get_python_script_path() {
            if let Ok(output) = Command::new("python").arg(&script).arg("--ocr").arg(path).output() {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                }
            }
        }

        // 2. Fallback til Windows native OCR
        let tool = Self::get_native_tool_path();
        if tool.exists() {
            if let Ok(output) = Command::new(&tool).arg(path).output() {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                }
            }
        }

        Ok(String::new())
    }
}

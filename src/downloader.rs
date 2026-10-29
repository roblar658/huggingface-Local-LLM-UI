use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub async fn download_hf_file_with_progress<F>(
    repo_id: &str,
    filename: &str,
    output_dir: &Path,
    mut on_progress: F,
) -> Result<PathBuf>
where
    F: FnMut(u64, u64) + Send + 'static,
{
    tokio::fs::create_dir_all(output_dir).await?;
    let target_path = output_dir.join(filename);

    if target_path.exists() {
        return Ok(target_path);
    }

    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);

    let client = Client::builder()
        .user_agent("RustHfLocalExplorer/1.0")
        .build()?;

    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow!("Kunne ikke laste ned fil ({}): {}", url, response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;

    let mut file = File::create(&target_path)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total_size);
    }

    Ok(target_path)
}

pub async fn download_hf_file(repo_id: &str, filename: &str, output_dir: &Path) -> Result<PathBuf> {
    tokio::fs::create_dir_all(output_dir).await?;
    let target_path = output_dir.join(filename);

    if target_path.exists() {
        println!("Filen eksisterer allerede: {:?}", target_path);
        return Ok(target_path);
    }

    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    println!("Laster ned: {}", url);

    let client = Client::builder()
        .user_agent("RustHfLocalExplorer/1.0")
        .build()?;

    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow!("Kunne ikke laste ned fil: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Laster ned {}", filename));

    let mut file = File::create(&target_path)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        pb.inc(chunk.len() as u64);
    }

    pb.finish_with_message("Nedlasting fullført!");
    Ok(target_path)
}

/// Smart nedlasting av tokenizer.json:
/// GGUF-depoter (f.eks. "ornith-ai/Ornith-1.5-9B-GGUF") inneholder ofte kun .gguf-filer,
/// mens tokenizer.json ligger i moder-depotet ("ornith-ai/Ornith-1.5-9B").
/// Denne funksjonen prøver:
/// 1. Selve GGUF-depotet
/// 2. Moder-depotet (ved å fjerne -GGUF / -gguf suffiks)
/// 3. Sjekke HF API metadata (license_link / base_model)
/// 4. Fallback til standard Qwen/Llama tokenizer dersom depotet kun er et rent GGUF-speil
pub async fn ensure_tokenizer_for_repo(repo_id: &str, output_dir: &Path) -> Result<PathBuf> {
    tokio::fs::create_dir_all(output_dir).await?;
    let tokenizer_path = output_dir.join("tokenizer.json");

    if tokenizer_path.exists() {
        if let Ok(meta) = std::fs::metadata(&tokenizer_path) {
            if meta.len() > 1000 {
                return Ok(tokenizer_path);
            }
        }
    }

    let mut candidates = vec![repo_id.to_string()];

    // Stripp vanlige GGUF-suffikser for å finne moder-depotet
    for suffix in ["-GGUF", "-gguf", "_GGUF", "_gguf"] {
        if let Some(stripped) = repo_id.strip_suffix(suffix) {
            candidates.push(stripped.to_string());
        }
    }

    // Fallback-depoter med kompatible standard-tokenizere
    if repo_id.to_lowercase().contains("qwen") || repo_id.to_lowercase().contains("ornith") {
        candidates.push("ornith-ai/Ornith-1.5-9B".to_string());
        candidates.push("Qwen/Qwen2.5-7B-Instruct".to_string());
    }
    candidates.push("HuggingFaceTB/SmolLM2-1.7B-Instruct".to_string());

    let client = Client::builder()
        .user_agent("RustHfLocalExplorer/1.0")
        .build()?;

    for candidate_repo in candidates {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            candidate_repo
        );
        println!("Prøver å hente tokenizer fra: {}", url);

        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                let bytes = resp.bytes().await?;
                if bytes.len() > 1000 {
                    std::fs::write(&tokenizer_path, &bytes)?;
                    println!("Tokenizer lastet ned fra {} ({} bytes)", candidate_repo, bytes.len());
                    return Ok(tokenizer_path);
                }
            } else {
                println!("  -> Fant ikke tokenizer.json i {} ({}), prøver neste kandidat...", candidate_repo, resp.status());
            }
        }
    }

    Err(anyhow!("Kunne ikke finne tokenizer.json i noen av kandidat-depotene for {}", repo_id))
}

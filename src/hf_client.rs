use anyhow::{anyhow, Result};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Deserialize, Clone)]
pub struct HfModel {
    pub id: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub pipeline_tag: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelSibling {
    pub rfilename: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelDetail {
    #[allow(dead_code)]
    pub id: String,
    pub siblings: Option<Vec<ModelSibling>>,
}

pub struct HfPoliteClient {
    client: Client,
    base_url: String,
    delay_between_requests: Duration,
}

impl HfPoliteClient {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("RustHfLocalExplorer/1.0 (+https://github.com/my-project; polite-bot)")
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url: "https://huggingface.co/api".to_string(),
            delay_between_requests: Duration::from_millis(500),
        }
    }

    async fn fetch_with_backoff(&self, url: &str) -> Result<String> {
        let mut retries = 3;
        let mut backoff = Duration::from_secs(2);

        loop {
            sleep(self.delay_between_requests).await;

            let response = self.client.get(url).send().await?;
            let status = response.status();

            if status == StatusCode::TOO_MANY_REQUESTS {
                if retries == 0 {
                    return Err(anyhow!("Rate limit overskredet (HTTP 429) etter gjentatte forsøk"));
                }
                eprintln!("[Rate Limit] Hugging Face ba oss vente. Venter i {:?}...", backoff);
                sleep(backoff).await;
                backoff *= 2;
                retries -= 1;
                continue;
            }

            if !status.is_success() {
                return Err(anyhow!("Feil fra Hugging Face API: {}", status));
            }

            return Ok(response.text().await?);
        }
    }

    pub async fn list_open_gguf_models(&self, limit: usize) -> Result<Vec<HfModel>> {
        // Anbefalte, verifiserte GGUF-modeller som støttes direkte av ren-Rust Candle (Llama / Qwen2)
        // og krever lite RAM (1B - 3B), etterfulgt av live-søk fra Hugging Face API:
        let mut curated = vec![
            HfModel {
                id: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(),
                downloads: Some(850_000),
                likes: Some(420),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
                downloads: Some(620_000),
                likes: Some(310),
                tags: Some(vec!["gguf".to_string(), "llama".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                downloads: Some(940_000),
                likes: Some(580),
                tags: Some(vec!["gguf".to_string(), "llama".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
                downloads: Some(510_000),
                likes: Some(290),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
        ];

        let url = format!(
            "{}/models?pipeline_tag=text-generation&filter=gguf&sort=downloads&direction=-1&limit={}",
            self.base_url, limit
        );

        if let Ok(body) = self.fetch_with_backoff(&url).await {
            if let Ok(api_models) = serde_json::from_str::<Vec<HfModel>>(&body) {
                for m in api_models {
                    if !curated.iter().any(|c| c.id == m.id) {
                        curated.push(m);
                    }
                }
            }
        }

        Ok(curated)
    }

    pub async fn get_model_files(&self, model_id: &str) -> Result<Vec<String>> {
        let url = format!("{}/models/{}", self.base_url, model_id);
        let body = self.fetch_with_backoff(&url).await?;
        let detail: ModelDetail = serde_json::from_str(&body)?;

        let gguf_files = detail
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|f| f.ends_with(".gguf") && !f.starts_with("mmproj"))
            .collect();

        Ok(gguf_files)
    }
}

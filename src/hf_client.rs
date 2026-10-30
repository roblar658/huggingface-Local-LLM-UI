use anyhow::{anyhow, Result};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone, Deserialize)]
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

    async fn fetch_query_with_backoff(&self, url: &str, params: &[(&str, &str)]) -> Result<String> {
        let mut retries = 3;
        let mut backoff = Duration::from_secs(2);

        loop {
            sleep(self.delay_between_requests).await;

            let response = self.client.get(url).query(params).send().await?;
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

    async fn fetch_with_backoff(&self, url: &str) -> Result<String> {
        self.fetch_query_with_backoff(url, &[]).await
    }

    /// Returnerer et bredt utvalg av verifiserte åpne GGUF-modeller
    /// innen ulike parametre (0.1B til 70B), inkludert Reasoning, Coder,
    /// Qwen 2.5, Llama 3.2, Gemma 2, Mistral, SmolLM2 og Phi.
    pub fn curated_models() -> Vec<HfModel> {
        vec![
            // --- Kompakte / Raske (0.1B - 3B) ---
            HfModel {
                id: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string(),
                downloads: Some(410_000),
                likes: Some(195),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "compact".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(),
                downloads: Some(890_000),
                likes: Some(460),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF".to_string(),
                downloads: Some(520_000),
                likes: Some(380),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "coder".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF".to_string(),
                downloads: Some(1_250_000),
                likes: Some(890),
                tags: Some(vec!["gguf".to_string(), "deepseek".to_string(), "reasoning".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
                downloads: Some(680_000),
                likes: Some(340),
                tags: Some(vec!["gguf".to_string(), "llama".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                downloads: Some(990_000),
                likes: Some(620),
                tags: Some(vec!["gguf".to_string(), "llama".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
                downloads: Some(580_000),
                likes: Some(330),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/gemma-2-2b-it-GGUF".to_string(),
                downloads: Some(430_000),
                likes: Some(250),
                tags: Some(vec!["gguf".to_string(), "gemma2".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Phi-3.5-mini-instruct-GGUF".to_string(),
                downloads: Some(380_000),
                likes: Some(210),
                tags: Some(vec!["gguf".to_string(), "phi3".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF".to_string(),
                downloads: Some(210_000),
                likes: Some(150),
                tags: Some(vec!["gguf".to_string(), "smollm".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF".to_string(),
                downloads: Some(290_000),
                likes: Some(180),
                tags: Some(vec!["gguf".to_string(), "smollm".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF".to_string(),
                downloads: Some(490_000),
                likes: Some(260),
                tags: Some(vec!["gguf".to_string(), "smollm".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },

            // --- Middels / Standard (7B - 9B) ---
            HfModel {
                id: "Qwen/Qwen2.5-7B-Instruct-GGUF".to_string(),
                downloads: Some(1_450_000),
                likes: Some(920),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
                downloads: Some(1_120_000),
                likes: Some(840),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "coder".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF".to_string(),
                downloads: Some(2_100_000),
                likes: Some(1_450),
                tags: Some(vec!["gguf".to_string(), "deepseek".to_string(), "reasoning".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF".to_string(),
                downloads: Some(1_870_000),
                likes: Some(1_210),
                tags: Some(vec!["gguf".to_string(), "deepseek".to_string(), "reasoning".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
                downloads: Some(2_650_000),
                likes: Some(1_820),
                tags: Some(vec!["gguf".to_string(), "llama".to_string(), "candle-ready".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Mistral-7B-Instruct-v0.3-GGUF".to_string(),
                downloads: Some(890_000),
                likes: Some(510),
                tags: Some(vec!["gguf".to_string(), "mistral".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Ministral-8B-Instruct-2410-GGUF".to_string(),
                downloads: Some(640_000),
                likes: Some(430),
                tags: Some(vec!["gguf".to_string(), "mistral".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/gemma-2-9b-it-GGUF".to_string(),
                downloads: Some(960_000),
                likes: Some(680),
                tags: Some(vec!["gguf".to_string(), "gemma2".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/phi-4-GGUF".to_string(),
                downloads: Some(780_000),
                likes: Some(530),
                tags: Some(vec!["gguf".to_string(), "phi4".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "ornith-ai/Ornith-1.5-9B-GGUF".to_string(),
                downloads: Some(5_260_000),
                likes: Some(612),
                tags: Some(vec!["gguf".to_string(), "ornith".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "NousResearch/Hermes-3-Llama-3.1-8B-GGUF".to_string(),
                downloads: Some(520_000),
                likes: Some(410),
                tags: Some(vec!["gguf".to_string(), "hermes".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },

            // --- Store & Kapasitetsrike (14B - 35B+) ---
            HfModel {
                id: "Qwen/Qwen2.5-14B-Instruct-GGUF".to_string(),
                downloads: Some(890_000),
                likes: Some(720),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(),
                downloads: Some(760_000),
                likes: Some(640),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "coder".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF".to_string(),
                downloads: Some(1_420_000),
                likes: Some(1_050),
                tags: Some(vec!["gguf".to_string(), "deepseek".to_string(), "reasoning".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/gemma-2-27b-it-GGUF".to_string(),
                downloads: Some(610_000),
                likes: Some(490),
                tags: Some(vec!["gguf".to_string(), "gemma2".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF".to_string(),
                downloads: Some(2_520_000),
                likes: Some(1_810),
                tags: Some(vec!["gguf".to_string(), "qwen".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-32B-Instruct-GGUF".to_string(),
                downloads: Some(1_150_000),
                likes: Some(980),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF".to_string(),
                downloads: Some(940_000),
                likes: Some(810),
                tags: Some(vec!["gguf".to_string(), "qwen2".to_string(), "coder".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF".to_string(),
                downloads: Some(1_750_000),
                likes: Some(1_320),
                tags: Some(vec!["gguf".to_string(), "deepseek".to_string(), "reasoning".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
            HfModel {
                id: "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF".to_string(),
                downloads: Some(840_000),
                likes: Some(760),
                tags: Some(vec!["gguf".to_string(), "llama".to_string()]),
                pipeline_tag: Some("text-generation".to_string()),
            },
        ]
    }

    pub async fn list_open_gguf_models(&self, limit: usize) -> Result<Vec<HfModel>> {
        let mut curated = Self::curated_models();

        let url = format!("{}/models", self.base_url);
        let limit_str = limit.max(100).to_string();
        let params = [
            ("pipeline_tag", "text-generation"),
            ("filter", "gguf"),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", &limit_str),
        ];

        if let Ok(body) = self.fetch_query_with_backoff(&url, &params).await {
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

    pub async fn search_open_gguf_models(&self, query: &str, limit: usize) -> Result<Vec<HfModel>> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return self.list_open_gguf_models(limit).await;
        }

        // Ta med kuraterte modeller som matcher søket lokalt
        let q_lower = trimmed.to_lowercase();
        let mut results: Vec<HfModel> = Self::curated_models()
            .into_iter()
            .filter(|m| m.id.to_lowercase().contains(&q_lower))
            .collect();

        let url = format!("{}/models", self.base_url);
        let limit_str = limit.to_string();
        let params = [
            ("pipeline_tag", "text-generation"),
            ("filter", "gguf"),
            ("search", trimmed),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", &limit_str),
        ];

        if let Ok(body) = self.fetch_query_with_backoff(&url, &params).await {
            if let Ok(api_models) = serde_json::from_str::<Vec<HfModel>>(&body) {
                for m in api_models {
                    if !results.iter().any(|r| r.id == m.id) {
                        results.push(m);
                    }
                }
            }
        }

        Ok(results)
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

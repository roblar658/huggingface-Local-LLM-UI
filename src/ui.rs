use crate::hardware::{HardwareSpec, ResourceGovernor, ResourceLevel, ResourceSnapshot};
use crate::hf_client::{HfModel, HfPoliteClient};
use crate::inference::LocalLlm;
use crate::recommender::{assess_model, FitLevel};
use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(PartialEq)]
enum ActiveTab {
    Explore,
    Downloads,
    SystemPrompt,
    Chat,
}

pub enum AppEvent {
    HardwareDetected(HardwareSpec),
    ModelsLoaded(Result<Vec<HfModel>, String>),
    FilesLoaded(Result<Vec<String>, String>),
    DownloadProgress {
        filename: String,
        downloaded: u64,
        total: u64,
    },
    DownloadFinished(Result<PathBuf, String>),
    ModelLoadResult(Result<String, String>),
    InferenceToken(String),
    InferenceFinished,
    InferenceCancelled,
    InferenceError(String),
    CompositePromptReady(String),
    InferenceStatus(String),
    ImagePasted(Result<(usize, PathBuf, PathBuf, usize, usize), String>),
}

pub struct AttachedImage {
    pub id: usize,
    pub path: PathBuf,
    pub texture: Option<egui::TextureHandle>,
    pub width: usize,
    pub height: usize,
}

pub struct HfAssistantApp {
    tokio_handle: tokio::runtime::Handle,
    event_tx: Sender<AppEvent>,
    event_rx: Receiver<AppEvent>,

    active_tab: ActiveTab,
    hardware: Option<HardwareSpec>,
    status_msg: String,

    // Utforsk Hugging Face
    models: Vec<HfModel>,
    is_loading_models: bool,
    filter_search: String,
    filter_only_fitting: bool,
    selected_model: Option<HfModel>,
    model_files: Vec<String>,
    is_loading_files: bool,
    selected_file: Option<String>,

    // Nedlastinger
    downloading_filename: Option<String>,
    download_fraction: f32,
    download_bytes_str: String,
    downloaded_models: Vec<PathBuf>,

    // System Prompt
    system_prompt: String,
    disable_thinking_tags: bool,
    system_prompt_approved: bool,

    // Chat / Prompting
    loaded_llm: Arc<Mutex<Option<LocalLlm>>>,
    selected_llm_path: Option<PathBuf>,
    is_loading_llm: bool,
    chat_history: Vec<(String, String)>, // (sender, text)
    current_prompt: String,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    use_context: bool,
    enable_ocr_paste: bool,
    auto_evaluate_and_continue: bool,
    attached_images: Vec<AttachedImage>,
    next_image_id: usize,
    is_pasting_image: bool,
    is_generating: bool,
    cancel_flag: Arc<AtomicBool>,
    egui_ctx: egui::Context,
    resource_governor: Arc<ResourceGovernor>,
    resource_snapshot: ResourceSnapshot,
    last_resource_refresh: std::time::Instant,
}

impl HfAssistantApp {
    pub fn new(cc: &eframe::CreationContext<'_>, tokio_handle: tokio::runtime::Handle) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let (event_tx, event_rx) = channel();

        let mut app = Self {
            tokio_handle: tokio_handle.clone(),
            event_tx: event_tx.clone(),
            event_rx,

            active_tab: ActiveTab::Explore,
            hardware: None,
            status_msg: "Starter Hugging Face Assistant...".to_string(),

            models: Vec::new(),
            is_loading_models: true,
            filter_search: String::new(),
            filter_only_fitting: false,
            selected_model: None,
            model_files: Vec::new(),
            is_loading_files: false,
            selected_file: None,

            downloading_filename: None,
            download_fraction: 0.0,
            download_bytes_str: String::new(),
            downloaded_models: Vec::new(),

            system_prompt: "Du er en ekstremt kortfattet assistent. Svar kun med selve svaret på norsk i maksimalt 1–2 korte setninger uten innledning. Om ikke mulig, øk antall setninger med 1 inntil det gir mening.".to_string(),
            disable_thinking_tags: true,
            system_prompt_approved: false,

            loaded_llm: Arc::new(Mutex::new(None)),
            selected_llm_path: None,
            is_loading_llm: false,
            chat_history: Vec::new(),
            current_prompt: String::new(),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 500,
            use_context: true,
            enable_ocr_paste: true,
            auto_evaluate_and_continue: true,
            attached_images: Vec::new(),
            next_image_id: 1,
            is_pasting_image: false,
            is_generating: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            egui_ctx: cc.egui_ctx.clone(),
            resource_governor: {
                let gov = Arc::new(ResourceGovernor::new());
                gov
            },
            resource_snapshot: ResourceGovernor::new().sample(),
            last_resource_refresh: std::time::Instant::now(),
        };
        app.resource_snapshot = app.resource_governor.sample();

        // 1. Detekter maskinvare asynkront
        let tx = event_tx.clone();
        thread::spawn(move || {
            if let Ok(hw) = HardwareSpec::detect() {
                let _ = tx.send(AppEvent::HardwareDetected(hw));
            }
        });

        // 2. Scan eksisterende nedlastinger i models/ og last automatisk inn første modell
        app.refresh_local_models();
        if let Some(first_model) = app.downloaded_models.first().cloned() {
            app.selected_llm_path = Some(first_model);
            app.load_selected_model();
        }

        // 3. Hent modeller fra HF
        app.fetch_hf_models();

        app
    }

    fn refresh_local_models(&mut self) {
        let mut list = Vec::new();
        for dir in crate::downloader::get_all_models_dirs() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                        if !list.contains(&path) {
                            list.push(path);
                        }
                    }
                }
            }
        }
        self.downloaded_models = list;
    }

    fn fetch_hf_models(&self) {
        let tx = self.event_tx.clone();
        self.tokio_handle.spawn(async move {
            let client = HfPoliteClient::new();
            match client.list_open_gguf_models(20).await {
                Ok(models) => {
                    let _ = tx.send(AppEvent::ModelsLoaded(Ok(models)));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::ModelsLoaded(Err(e.to_string())));
                }
            }
        });
    }

    fn fetch_model_files(&self, model_id: String) {
        let tx = self.event_tx.clone();
        self.tokio_handle.spawn(async move {
            let client = HfPoliteClient::new();
            match client.get_model_files(&model_id).await {
                Ok(files) => {
                    let _ = tx.send(AppEvent::FilesLoaded(Ok(files)));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::FilesLoaded(Err(e.to_string())));
                }
            }
        });
    }

    fn start_download(&mut self, repo_id: String, filename: String) {
        self.downloading_filename = Some(filename.clone());
        self.download_fraction = 0.0;
        self.download_bytes_str = "Klargjør nedlasting...".to_string();
        self.status_msg = format!("Laster ned {}...", filename);

        let tx = self.event_tx.clone();
        let fname_clone = filename.clone();
        let repo_clone = repo_id.clone();

        self.tokio_handle.spawn(async move {
            let models_dir = crate::downloader::get_default_models_dir();
            let tx_prog = tx.clone();
            let fname_for_prog = fname_clone.clone();

            let res = crate::downloader::download_hf_file_with_progress(
                &repo_clone,
                &fname_clone,
                &models_dir,
                move |downloaded, total| {
                    let _ = tx_prog.send(AppEvent::DownloadProgress {
                        filename: fname_for_prog.clone(),
                        downloaded,
                        total,
                    });
                },
            )
            .await;

            // Smart nedlasting av tokenizer.json (sjekker både GGUF-depot og moder-depot)
            let _ = crate::downloader::ensure_tokenizer_for_repo(&repo_clone, &models_dir).await;

            match res {
                Ok(path) => {
                    let _ = tx.send(AppEvent::DownloadFinished(Ok(path)));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::DownloadFinished(Err(e.to_string())));
                }
            }
        });
    }

    fn cancel_generation(&mut self) {
        if self.is_generating {
            self.cancel_flag.store(true, Ordering::SeqCst);
            self.status_msg = "Avbryter generering...".to_string();
        }
    }

    fn paste_clipboard_image(&mut self, is_explicit: bool) {
        if self.is_pasting_image {
            return;
        }
        self.is_pasting_image = true;
        let id = self.next_image_id;
        self.next_image_id += 1;
        let tx = self.event_tx.clone();

        let cache_dir = std::env::temp_dir().join("hf_assistant_cache");
        let _ = std::fs::create_dir_all(&cache_dir);
        let dest_png = cache_dir.join(format!("img_{}.png", id));
        let dest_thumb_raw = cache_dir.join(format!("img_{}_thumb.raw", id));

        self.status_msg = format!("Henter bilde #{} fra utklippstavlen...", id);

        thread::spawn(move || {
            let res = crate::ocr::OcrEngine::grab_clipboard_image(&dest_png, &dest_thumb_raw);
            match res {
                Ok((w, h)) => {
                    let _ = tx.send(AppEvent::ImagePasted(Ok((id, dest_png, dest_thumb_raw, w, h))));
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if !is_explicit && err_str.contains("NO_IMAGE") {
                        let _ = tx.send(AppEvent::ImagePasted(Err(String::new())));
                    } else {
                        let _ = tx.send(AppEvent::ImagePasted(Err(err_str)));
                    }
                }
            }
        });
    }

    fn send_prompt(&mut self) {
        if !self.system_prompt_approved {
            self.status_msg =
                "Vurder og godkjenn System Prompt i fanen 'System Prompt' før du sender melding."
                    .to_string();
            return;
        }

        let prompt = self.current_prompt.trim().to_string();
        if prompt.is_empty() && self.attached_images.is_empty() {
            return;
        }
        if self.is_generating {
            return;
        }

        self.cancel_flag.store(false, Ordering::SeqCst);

        let images_to_process = std::mem::take(&mut self.attached_images);
        let num_images = images_to_process.len();

        let user_display_text = if num_images > 0 {
            if prompt.is_empty() {
                format!("[Kjører OCR på {} vedlagt(e) bilde(r)...]", num_images)
            } else {
                format!("{}\n\n[Kjører OCR på {} vedlagt(e) bilde(r)...]", prompt, num_images)
            }
        } else {
            prompt.clone()
        };

        self.chat_history.push(("Du".to_string(), user_display_text));
        self.chat_history.push(("Modell".to_string(), String::new()));
        self.current_prompt.clear();
        self.is_generating = true;
        self.status_msg = if num_images > 0 {
            format!("Gjenkjenner tekst fra {} bilde(r) med OCR...", num_images)
        } else {
            "Genererer svar...".to_string()
        };

        let llm_arc = self.loaded_llm.clone();
        let tx = self.event_tx.clone();
        let max_toks = self.max_tokens;
        let temp = self.temperature as f64;
        let p = self.top_p as f64;
        let base_system_prompt = self.system_prompt.trim().to_string();
        let disable_think = self.disable_thinking_tags;
        let cancel_arc = self.cancel_flag.clone();
        let use_context = self.use_context;
        let auto_eval = self.auto_evaluate_and_continue;
        let history_snapshot = self.chat_history.clone();
        let ui_ctx = self.egui_ctx.clone();

        thread::spawn(move || {
            // 1. Kjør OCR på alle vedlagte bilder med linjeskift og mellomrom bevart
            let mut composite_parts = Vec::new();
            for (i, img) in images_to_process.iter().enumerate() {
                if cancel_arc.load(Ordering::Relaxed) {
                    let _ = tx.send(AppEvent::InferenceCancelled);
                    return;
                }
                let ocr_res = crate::ocr::OcrEngine::recognize_file(&img.path);
                let text = ocr_res.unwrap_or_default().trim().to_string();
                if num_images > 1 {
                    if text.is_empty() {
                        composite_parts.push(format!("[Bilde {}]:\n(Ingen tekst gjenkjent)", i + 1));
                    } else {
                        composite_parts.push(format!("[Bilde {}]:\n{}", i + 1, text));
                    }
                } else if text.is_empty() {
                    composite_parts.push("[Bilde]:\n(Ingen tekst gjenkjent)".to_string());
                } else if prompt.is_empty() {
                    composite_parts.push(text);
                } else {
                    composite_parts.push(format!("[Bilde]:\n{}", text));
                }
            }

            if !prompt.is_empty() {
                composite_parts.push(prompt.clone());
            }

            let final_prompt = composite_parts.join("\n\n");
            let _ = tx.send(AppEvent::CompositePromptReady(final_prompt.clone()));

            // 2. Bygg prompt og system-prompt med eller uten kontekst
            let (prompt_to_send, sys_to_send) = if use_context {
                let mut parts: Vec<String> = Vec::new();
                let history_len = history_snapshot.len();
                let prior = if history_len >= 2 { &history_snapshot[..history_len - 2] } else { &[] };
                for (sender, text) in prior {
                    let t = text.trim();
                    if t.is_empty() || t == "[Avbrutt av bruker]" || sender == "System" {
                        continue;
                    }
                    if sender == "Du" {
                        parts.push(format!("[bruker] {}", t));
                    } else if sender == "Modell" {
                        parts.push(format!("[AI] {}", t));
                    }
                }
                parts.push(format!("[bruker] {}", final_prompt));
                let full_prompt = parts.join(" ");

                let context_instruction = "Samtalehistorikk og kontekst er strukturert kronologisk der [bruker] angir meldinger fra brukeren og [AI] angir svar fra assistenten, adskilt med mellomrom. Tolk denne konteksten og svar som [AI] på den siste [bruker]-meldingen.";
                let full_sys = if base_system_prompt.is_empty() {
                    context_instruction.to_string()
                } else {
                    format!("{} {}", base_system_prompt, context_instruction)
                };

                (full_prompt, full_sys)
            } else {
                (final_prompt.clone(), base_system_prompt)
            };

            // 3. Utfør inferens med dynamisk ressursstyring og anti-hang
            let mut guard = llm_arc.lock().unwrap();
            if let Some(ref mut llm) = *guard {
                let tx_token = tx.clone();
                let tx_status = tx.clone();
                let ctx_tok = ui_ctx.clone();
                let ctx_stat = ui_ctx.clone();
                let cancel_for_cb = cancel_arc.clone();
                let first_answer_ref = Arc::new(Mutex::new(String::new()));
                let first_ans_cb = first_answer_ref.clone();

                let res = llm.generate_with_callback_and_status(
                    &sys_to_send,
                    disable_think,
                    &prompt_to_send,
                    max_toks,
                    temp,
                    p,
                    Some(cancel_arc.as_ref()),
                    move |status_str| {
                        let _ = tx_status.send(AppEvent::InferenceStatus(status_str.to_string()));
                        ctx_stat.request_repaint();
                    },
                    move |token| {
                        if cancel_for_cb.load(Ordering::Relaxed) {
                            return false;
                        }
                        if let Ok(mut g) = first_ans_cb.lock() {
                            g.push_str(token);
                        }
                        let _ = tx_token.send(AppEvent::InferenceToken(token.to_string()));
                        ctx_tok.request_repaint();
                        true
                    },
                );

                if cancel_arc.load(Ordering::Relaxed) {
                    let _ = tx.send(AppEvent::InferenceCancelled);
                    ui_ctx.request_repaint();
                    return;
                } else if let Err(e) = res {
                    let msg = e.to_string();
                    if msg.contains("Avbrutt av bruker") {
                        let _ = tx.send(AppEvent::InferenceCancelled);
                    } else {
                        let _ = tx.send(AppEvent::InferenceError(msg));
                    }
                    ui_ctx.request_repaint();
                    return;
                }

                // 4. Lynrask tilstandsbasert vurdering og automatisk fortsettelse (uten re-prefill/hang)
                let first_ans = first_answer_ref.lock().unwrap().clone();
                if auto_eval && !first_ans.trim().is_empty() {
                    let tx_eval_token = tx.clone();
                    let tx_eval_status = tx.clone();
                    let ctx_eval_tok = ui_ctx.clone();
                    let ctx_eval_stat = ui_ctx.clone();
                    let cancel_eval = cancel_arc.clone();

                    let eval_res = llm.evaluate_and_continue_with_callback(
                        &first_ans,
                        max_toks,
                        temp,
                        p,
                        Some(cancel_arc.as_ref()),
                        move |status_str| {
                            let _ = tx_eval_status.send(AppEvent::InferenceStatus(status_str.to_string()));
                            ctx_eval_stat.request_repaint();
                        },
                        move |token| {
                            if cancel_eval.load(Ordering::Relaxed) {
                                return false;
                            }
                            let _ = tx_eval_token.send(AppEvent::InferenceToken(token.to_string()));
                            ctx_eval_tok.request_repaint();
                            true
                        },
                    );

                    if cancel_arc.load(Ordering::Relaxed) {
                        let _ = tx.send(AppEvent::InferenceCancelled);
                        ui_ctx.request_repaint();
                        return;
                    } else if let Err(e) = eval_res {
                        let msg = e.to_string();
                        if msg.contains("Avbrutt av bruker") {
                            let _ = tx.send(AppEvent::InferenceCancelled);
                            ui_ctx.request_repaint();
                            return;
                        }
                    }
                }

                let _ = tx.send(AppEvent::InferenceFinished);
                ui_ctx.request_repaint();
            } else {
                let _ = tx.send(AppEvent::InferenceError(
                    "Ingen modell er lastet inn i minnet enda. Gå til 'Nedlastinger & Lokale Modeller' og klikk 'Last inn & Åpne i Chat'.".to_string(),
                ));
                ui_ctx.request_repaint();
            }
        });
    }
}

impl eframe::App for HfAssistantApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let is_focused = ctx.input(|i| i.focused);
        self.resource_governor.set_app_focused(is_focused);
        if self.last_resource_refresh.elapsed() >= std::time::Duration::from_millis(600) {
            self.resource_snapshot = self.resource_governor.sample();
            self.last_resource_refresh = std::time::Instant::now();
        }
        if self.enable_ocr_paste && !self.is_pasting_image {
            let wants_paste = ctx.input(|i| {
                (i.modifiers.ctrl && i.key_pressed(egui::Key::V))
                    || (i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::V))
            });
            if wants_paste {
                self.paste_clipboard_image(false);
            }
        }
        if self.is_generating && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.cancel_generation();
        }

        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::HardwareDetected(hw) => {
                    self.status_msg = format!(
                        "Maskinvare klar: {} ({:.2} GB VRAM)",
                        hw.gpu_name, hw.dedicated_vram_gb
                    );
                    self.hardware = Some(hw);
                }
                AppEvent::ModelsLoaded(res) => {
                    self.is_loading_models = false;
                    match res {
                        Ok(list) => {
                            self.status_msg = format!(
                                "Fant {} åpne GGUF-modeller fra Hugging Face.",
                                list.len()
                            );
                            self.models = list;
                        }
                        Err(err) => {
                            self.status_msg = format!("Feil ved henting av modeller: {}", err);
                        }
                    }
                }
                AppEvent::FilesLoaded(res) => {
                    self.is_loading_files = false;
                    match res {
                        Ok(files) => {
                            self.model_files = files;
                            self.selected_file = self
                                .model_files
                                .iter()
                                .find(|f| f.to_lowercase().contains("q4_k_m"))
                                .cloned()
                                .or_else(|| self.model_files.first().cloned());
                        }
                        Err(err) => {
                            self.status_msg = format!("Feil ved henting av filer: {}", err);
                        }
                    }
                }
                AppEvent::DownloadProgress {
                    filename: _,
                    downloaded,
                    total,
                } => {
                    if total > 0 {
                        self.download_fraction = downloaded as f32 / total as f32;
                        let mb_down = downloaded as f64 / 1024.0 / 1024.0;
                        let mb_tot = total as f64 / 1024.0 / 1024.0;
                        self.download_bytes_str = format!(
                            "{:.1} MB / {:.1} MB ({:.1}%)",
                            mb_down,
                            mb_tot,
                            self.download_fraction * 100.0
                        );
                    }
                }
                AppEvent::DownloadFinished(res) => {
                    self.downloading_filename = None;
                    match res {
                        Ok(path) => {
                            self.status_msg = format!(
                                "Nedlasting fullført: {:?}",
                                path.file_name().unwrap_or_default()
                            );
                            self.refresh_local_models();
                            self.selected_llm_path = Some(path);
                        }
                        Err(e) => {
                            self.status_msg = format!("Nedlasting feilet: {}", e);
                        }
                    }
                }
                AppEvent::ModelLoadResult(res) => {
                    self.is_loading_llm = false;
                    match res {
                        Ok(msg) => {
                            self.status_msg = msg.clone();
                            self.chat_history.push(("System".to_string(), msg));
                        }
                        Err(err) => {
                            self.status_msg = format!("Kunne ikke laste modell: {}", err);
                            self.chat_history
                                .push(("System".to_string(), format!("⚠️ {}", err)));
                        }
                    }
                }
                AppEvent::InferenceStatus(status) => {
                    self.status_msg = status;
                }
                AppEvent::InferenceToken(token) => {
                    if let Some((_, text)) = self.chat_history.last_mut() {
                        text.push_str(&token);
                    }
                }
                AppEvent::InferenceFinished => {
                    self.is_generating = false;
                    if self.status_msg.starts_with("Generer") || self.status_msg.starts_with("Vurder") {
                        self.status_msg = "Generering og vurdering fullført.".to_string();
                    }
                }
                AppEvent::InferenceCancelled => {
                    self.is_generating = false;
                    self.status_msg = "Generering avbrutt.".to_string();
                    if let Some((_, text)) = self.chat_history.last_mut() {
                        if text.is_empty() {
                            text.push_str("[Avbrutt av bruker]");
                        } else {
                            text.push_str(" [Avbrutt av bruker]");
                        }
                    }
                }
                AppEvent::InferenceError(err) => {
                    self.is_generating = false;
                    self.status_msg = format!("Inferens-feil: {}", err);
                    if let Some((_, text)) = self.chat_history.last_mut() {
                        text.push_str(&format!("\n[Feil: {}]", err));
                    }
                }
                AppEvent::CompositePromptReady(composite) => {
                    let len = self.chat_history.len();
                    if len >= 2 {
                        self.chat_history[len - 2].1 = composite;
                    }
                    self.status_msg = "Genererer svar...".to_string();
                }
                AppEvent::ImagePasted(res) => {
                    self.is_pasting_image = false;
                    match res {
                        Ok((id, png_path, thumb_path, w, h)) => {
                            let texture = if let Ok((tw, th, pixels)) = crate::ocr::OcrEngine::load_thumbnail_raw(&thumb_path) {
                                let color_img = egui::ColorImage::from_rgba_unmultiplied([tw, th], &pixels);
                                Some(ctx.load_texture(format!("thumb_{}", id), color_img, Default::default()))
                            } else {
                                None
                            };
                            self.attached_images.push(AttachedImage {
                                id,
                                path: png_path,
                                texture,
                                width: w,
                                height: h,
                            });
                            self.status_msg = format!("Bilde #{} lagt til (totalt {} bilde(r) vedlagt).", id, self.attached_images.len());
                        }
                        Err(err) => {
                            if !err.is_empty() {
                                self.status_msg = format!("Kunne ikke lime inn bilde: {}", err);
                            }
                        }
                    }
                }
            }
        }

        // Topp-panel med maskinvareoversikt (samme stil som før)
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new("🤖 Hugging Face Local Assistant")
                        .color(egui::Color32::WHITE)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new("Windows / Rust GUI")
                        .color(egui::Color32::from_rgb(0, 180, 255))
                        .strong(),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("🔄 Oppdater HF-liste").clicked() {
                        self.is_loading_models = true;
                        self.fetch_hf_models();
                    }
                });
            });

            ui.add_space(4.0);

            ui.group(|ui| {
                ui.horizontal(|ui| {
                    if let Some(ref hw) = self.hardware {
                        ui.label(format!(
                            "🖥️ CPU: {} ({} kjerner)",
                            hw.cpu_brand, hw.cpu_cores
                        ));
                        ui.separator();
                        ui.label(format!(
                            "💾 RAM: {:.1} GB (ledig {:.1} GB)",
                            hw.total_ram_gb, hw.available_ram_gb
                        ));
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "🎮 GPU: {} | {:.2} GB VRAM",
                                hw.gpu_name, hw.dedicated_vram_gb
                            ))
                            .strong()
                            .color(egui::Color32::from_rgb(50, 220, 100)),
                        );
                        ui.separator();
                        let badge_color = match self.resource_snapshot.level {
                            ResourceLevel::OptimalSubMax => egui::Color32::from_rgb(80, 210, 255),
                            ResourceLevel::Balanced => egui::Color32::from_rgb(180, 200, 255),
                            ResourceLevel::LowResources => egui::Color32::from_rgb(245, 190, 60),
                            ResourceLevel::CriticalPause => egui::Color32::from_rgb(255, 110, 90),
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "⚡ Ressursstyring: {}",
                                self.resource_snapshot.status_badge()
                            ))
                            .color(badge_color)
                            .strong(),
                        );
                    } else {
                        ui.spinner();
                        ui.label("Analyserer lokal maskinvare...");
                    }
                });
            });

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.active_tab,
                    ActiveTab::Explore,
                    "🔍 Utforsk HF Modeller",
                );
                ui.selectable_value(
                    &mut self.active_tab,
                    ActiveTab::Downloads,
                    "📥 Nedlastinger & Lokale Modeller",
                );
                let sys_label = if self.system_prompt_approved {
                    "⚙️ System Prompt (✓)"
                } else {
                    "⚙️ System Prompt"
                };
                ui.selectable_value(
                    &mut self.active_tab,
                    ActiveTab::SystemPrompt,
                    sys_label,
                );
                ui.selectable_value(&mut self.active_tab, ActiveTab::Chat, "💬 Chat & Prompting");
            });
            ui.add_space(6.0);
        });

        // Bunn-panel med status og fremdriftsindikator
        egui::TopBottomPanel::bottom("status_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            if let Some(ref fname) = self.downloading_filename {
                ui.horizontal(|ui| {
                    ui.label(format!("⏳ Laster ned {}:", fname));
                    ui.add(egui::ProgressBar::new(self.download_fraction).show_percentage());
                    ui.label(&self.download_bytes_str);
                });
            }
            if self.is_loading_llm {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Laster inn GGUF-modell og tokenizer i minnet...");
                });
            }
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Status:").strong());
                ui.label(&self.status_msg);
            });
            ui.add_space(4.0);
        });

        // Hovedpanel
        egui::CentralPanel::default().show(ctx, |ui| match self.active_tab {
            ActiveTab::Explore => self.render_explore_tab(ui),
            ActiveTab::Downloads => self.render_downloads_tab(ui),
            ActiveTab::SystemPrompt => self.render_system_prompt_tab(ui),
            ActiveTab::Chat => self.render_chat_tab(ui),
        });

        if self.downloading_filename.is_some() || self.is_generating || self.is_loading_llm {
            let repaint_ms = if is_focused { 33 } else { 150 };
            ctx.request_repaint_after(std::time::Duration::from_millis(repaint_ms));
        }
    }
}

impl HfAssistantApp {
    fn render_explore_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("🔎 Søk:");
            ui.text_edit_singleline(&mut self.filter_search);
            ui.checkbox(
                &mut self.filter_only_fitting,
                "Vis kun modeller som passer min PC",
            );
        });

        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let hw_opt = self.hardware.clone();
            let selected_model_clone = self.selected_model.clone();

            for m in self.models.clone() {
                if !self.filter_search.is_empty()
                    && !m
                        .id
                        .to_lowercase()
                        .contains(&self.filter_search.to_lowercase())
                {
                    continue;
                }

                let assessment = hw_opt.as_ref().map(|hw| assess_model(&m, hw));

                if self.filter_only_fitting {
                    if let Some(ref a) = assessment {
                        if matches!(a.fit_level, FitLevel::TooLarge) {
                            continue;
                        }
                    }
                }

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(&m.id)
                                    .color(egui::Color32::WHITE)
                                    .strong()
                                    .size(15.0),
                            );

                            ui.horizontal(|ui| {
                                if let Some(ref a) = assessment {
                                    let (text, color) = match a.fit_level {
                                        FitLevel::FullGpuOffload => (
                                            "🟢 Rask (Full GPU VRAM)",
                                            egui::Color32::from_rgb(50, 220, 100),
                                        ),
                                        FitLevel::CpuRamFallback => (
                                            "🟡 Medium (System-RAM / CPU)",
                                            egui::Color32::from_rgb(240, 200, 50),
                                        ),
                                        FitLevel::TooLarge => (
                                            "🔴 For stor for maskinvaren",
                                            egui::Color32::from_rgb(240, 80, 80),
                                        ),
                                    };
                                    ui.colored_label(color, text);
                                    ui.label(format!("(Est. {})", a.est_param_size));
                                }
                            });
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Velg & Se GGUF-filer").clicked() {
                                self.selected_model = Some(m.clone());
                                self.is_loading_files = true;
                                self.fetch_model_files(m.id.clone());
                            }
                        });
                    });

                    if let Some(ref sel) = selected_model_clone {
                        if sel.id == m.id {
                            ui.separator();
                            ui.label("Tilgjengelige GGUF-filer:");

                            if self.is_loading_files {
                                ui.spinner();
                                ui.label("Henter filoversikt fra Hugging Face...");
                            } else if self.model_files.is_empty() {
                                ui.label("Ingen GGUF-filer funnet.");
                            } else {
                                let mut trigger_download: Option<(String, String)> = None;
                                ui.horizontal(|ui| {
                                    egui::ComboBox::from_label("Velg kvantisering")
                                        .selected_text(
                                            self.selected_file
                                                .as_deref()
                                                .unwrap_or("Velg fil"),
                                        )
                                        .show_ui(ui, |ui| {
                                            for f in &self.model_files {
                                                ui.selectable_value(
                                                    &mut self.selected_file,
                                                    Some(f.clone()),
                                                    f,
                                                );
                                            }
                                        });

                                    if let Some(ref target_file) = self.selected_file {
                                        if ui.button("📥 Start nedlasting").clicked() {
                                            trigger_download =
                                                Some((sel.id.clone(), target_file.clone()));
                                        }
                                    }
                                });

                                if let Some((repo, file)) = trigger_download {
                                    self.start_download(repo, file);
                                }
                            }
                        }
                    }
                });
                ui.add_space(4.0);
            }
        });
    }

    fn render_downloads_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading(
            egui::RichText::new("Nedlastede modeller i C:\\dev\\hf_local_assistant\\models")
                .color(egui::Color32::WHITE)
                .strong(),
        );
        ui.colored_label(
            egui::Color32::from_gray(225),
            "Her vises alle .gguf-modeller du har lastet ned lokalt på maskinen.",
        );
        ui.separator();

        let mut local_files = self.downloaded_models.clone();
        if local_files.is_empty() {
            self.refresh_local_models();
            local_files = self.downloaded_models.clone();
        }

        if local_files.is_empty() {
            ui.colored_label(
                egui::Color32::WHITE,
                "Ingen modeller er lastet ned enda. Gå til fanen 'Utforsk HF Modeller' for å laste ned en modell.",
            );
        } else {
            for path in local_files {
                let fname = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let meta = std::fs::metadata(&path).ok();
                let size_gb = meta
                    .map(|m| m.len() as f64 / 1024.0 / 1024.0 / 1024.0)
                    .unwrap_or(0.0);

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("📦 {} ({:.2} GB)", fname, size_gb))
                                .color(egui::Color32::WHITE)
                                .strong(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("💬 Last inn & Åpne i Chat").clicked() {
                                self.selected_llm_path = Some(path.clone());
                                self.load_selected_model();
                                self.active_tab = ActiveTab::Chat;
                            }
                        });
                    });
                });
                ui.add_space(4.0);
            }
        }
    }

    fn render_system_prompt_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading(
            egui::RichText::new("System Prompt (Vurdering før bruk)")
                .color(egui::Color32::WHITE)
                .strong(),
        );
        ui.colored_label(
            egui::Color32::from_gray(225),
            "Her kan du vurdere, tilpasse og godkjenne System Promptet som styrer modellens oppførsel i chatten.",
        );
        ui.separator();

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Status:")
                        .color(egui::Color32::WHITE)
                        .strong(),
                );
                if self.system_prompt_approved {
                    ui.colored_label(
                        egui::Color32::from_rgb(50, 220, 100),
                        "🟢 Vurdert og godkjent for bruk i Chat",
                    );
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(240, 200, 50),
                        "🟡 Ikke godkjent enda – se gjennom under og klikk 'Godkjenn System Prompt'",
                    );
                }
            });
        });

        ui.add_space(6.0);

        ui.group(|ui| {
            ui.label(
                egui::RichText::new("Hurtigmaler:")
                    .color(egui::Color32::WHITE)
                    .strong(),
            );
            ui.horizontal_wrapped(|ui| {
                if ui.button("Standard (Kortfattet 1–2 setninger)").clicked() {
                    self.system_prompt = "Du er en ekstremt kortfattet assistent. Svar kun med selve svaret på norsk i maksimalt 1–2 korte setninger uten innledning. Om ikke mulig, øk antall setninger med 1 inntil det gir mening.".to_string();
                    self.system_prompt_approved = false;
                }
                if ui.button("Konsis Norsk Assistent").clicked() {
                    self.system_prompt = "Du er en hjelpsom, kunnskapsrik og konsis norskspråklig assistent. Svar alltid direkte og presist på norsk med mindre brukeren ber om et annet språk.".to_string();
                    self.system_prompt_approved = false;
                }
                if ui.button("Rust- & Kodeekspert").clicked() {
                    self.system_prompt = "Du er en erfaren systemutvikler og Rust-ekspert. Gi presise tekniske svar og ren kode på norsk.".to_string();
                    self.system_prompt_approved = false;
                }
                if ui.button("Grundig & Forklarende").clicked() {
                    self.system_prompt = "Du er en grundig fagperson. Forklar ting steg for steg på en oversiktlig måte på norsk.".to_string();
                    self.system_prompt_approved = false;
                }
            });

            ui.separator();
            ui.label(
                egui::RichText::new("System Prompt-tekst:")
                    .color(egui::Color32::WHITE)
                    .strong(),
            );
            let resp = ui.add_sized(
                [ui.available_width(), 110.0],
                egui::TextEdit::multiline(&mut self.system_prompt)
                    .text_color(egui::Color32::WHITE)
                    .hint_text("Skriv inn System Prompt her..."),
            );
            if resp.changed() {
                self.system_prompt_approved = false;
            }

            ui.add_space(4.0);
            if ui
                .checkbox(
                    &mut self.disable_thinking_tags,
                    egui::RichText::new(
                        "Hopp over intern tenkeblokk (<think>...</think>) for raskere, direkte svar",
                    )
                    .color(egui::Color32::from_gray(230)),
                )
                .changed()
            {
                self.system_prompt_approved = false;
            }
            if ui
                .checkbox(
                    &mut self.use_context,
                    egui::RichText::new(
                        "Bruk samtalekontekst (append meldinger med [bruker] og [AI] samt tolkningsinstruks)",
                    )
                    .color(egui::Color32::from_gray(230)),
                )
                .changed()
            {
                self.system_prompt_approved = false;
            }
        });

        ui.add_space(6.0);

        let words = self.system_prompt.split_whitespace().count();
        let chars = self.system_prompt.chars().count();
        let est_tokens = ((chars as f32 / 3.8).ceil() as usize).max(1) + 12;
        let lower = self.system_prompt.to_lowercase();
        let has_lang = lower.contains("norsk") || lower.contains("engelsk") || lower.contains("språk");

        ui.group(|ui| {
            ui.label(
                egui::RichText::new("Vurdering av System Prompt:")
                    .color(egui::Color32::WHITE)
                    .strong(),
            );
            ui.colored_label(
                egui::Color32::from_gray(230),
                format!(
                    "• Størrelse: {} ord ({} tegn, ca. {} prefill-tokens per melding)",
                    words, chars, est_tokens
                ),
            );
            ui.colored_label(
                egui::Color32::from_gray(230),
                format!(
                    "• Språkinstruks: {}",
                    if has_lang {
                        "Spesifisert (sikrer at modellen holder seg til valgt språk)"
                    } else {
                        "Ikke spesifisert (modellen kan svare på engelsk)"
                    }
                ),
            );
            ui.colored_label(
                egui::Color32::from_gray(230),
                format!(
                    "• Tenkeblokk (<think>): {}",
                    if self.disable_thinking_tags {
                        "Deaktivert (gir raskest mulig svar)"
                    } else {
                        "Aktivert (modellen tenker høyt før svaret)"
                    }
                ),
            );
            ui.colored_label(
                egui::Color32::from_gray(230),
                format!(
                    "• Samtalekontekst: {}",
                    if self.use_context {
                        "Aktivert (appendes med mellomrom, [bruker] og [AI] + instruks i prompt)"
                    } else {
                        "Deaktivert (enkelt-forespørsler uten historikk)"
                    }
                ),
            );
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("✅ Godkjenn System Prompt & Åpne Chat").clicked() {
                self.system_prompt_approved = true;
                self.status_msg = "System Prompt er vurdert og godkjent.".to_string();
                self.active_tab = ActiveTab::Chat;
            }
            if ui.button("✔️ Godkjenn (bli i fanen)").clicked() {
                self.system_prompt_approved = true;
                self.status_msg = "System Prompt er vurdert og godkjent.".to_string();
            }
        });
    }

    fn load_selected_model(&mut self) {
        if self.is_loading_llm {
            return;
        }
        if let Some(ref path) = self.selected_llm_path {
            let p_clone = path.clone();
            let arc_llm = self.loaded_llm.clone();
            let tx = self.event_tx.clone();
            let handle = self.tokio_handle.clone();
            let gov = self.resource_governor.clone();
            let ui_ctx = self.egui_ctx.clone();

            self.is_loading_llm = true;
            self.status_msg = format!(
                "Laster inn {:?} i minnet...",
                path.file_name().unwrap_or_default()
            );

            thread::spawn(move || {
                let models_dir = p_clone.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| crate::downloader::get_default_models_dir());
                // Sørg for at tokenizer.json er på plass før vi laster GGUF
                let tok_res = handle.block_on(async {
                    crate::downloader::ensure_tokenizer_for_repo(
                        "ornith-ai/Ornith-1.5-9B",
                        &models_dir,
                    )
                    .await
                });

                let tok_path = match tok_res {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.send(AppEvent::ModelLoadResult(Err(format!(
                            "Tokenizer-feil: {}",
                            e
                        ))));
                        return;
                    }
                };

                {
                    let mut guard = arc_llm.lock().unwrap();
                    *guard = None;
                }
                match LocalLlm::load(&p_clone, &tok_path) {
                    Ok(mut model) => {
                        model.set_resource_governor(gov);
                        let arch = model.architecture.clone();
                        let gpu_info = if let Some(cuda) = crate::gpu::CudaEngine::global() {
                            let gb = cuda.offloaded_bytes.load(Ordering::Relaxed) as f64
                                / (1024.0 * 1024.0 * 1024.0);
                            let tens = cuda.offloaded_tensors.load(Ordering::Relaxed);
                            format!(
                                " [{:.2} GB / {} tensorer i GPU VRAM på {}]",
                                gb, tens, cuda.device_name
                            )
                        } else {
                            String::new()
                        };
                        let mut guard = arc_llm.lock().unwrap();
                        *guard = Some(model);
                        ui_ctx.request_repaint();
                        let _ = tx.send(AppEvent::ModelLoadResult(Ok(format!(
                            "Modell {:?} (arkitektur: {}){} er ferdig lastet inn og klar for prompting!",
                            p_clone.file_name().unwrap_or_default(),
                            arch,
                            gpu_info
                        ))));
                    }
                    Err(e) => {
                        let _ = tx.send(AppEvent::ModelLoadResult(Err(e.to_string())));
                    }
                }
            });
        }
    }

    fn render_chat_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new("Lokal Chat & Prompting")
                    .color(egui::Color32::WHITE)
                    .strong(),
            );
            if let Some(ref p) = self.selected_llm_path {
                ui.colored_label(
                    egui::Color32::from_rgb(0, 200, 100),
                    format!(
                        "(Valgt modell: {})",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    ),
                );
            } else {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "(Ingen modell lastet inn - velg fra 'Nedlastinger & Lokale Modeller')",
                );
            }
        });

        ui.separator();

        if !self.system_prompt_approved {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(240, 200, 50),
                        "⚠️ System Prompt må vurderes før du starter chat:",
                    );
                    if ui.button("⚙️ Vurder i System Prompt-fanen").clicked() {
                        self.active_tab = ActiveTab::SystemPrompt;
                    }
                    if ui.button("✅ Godkjenn standard System Prompt").clicked() {
                        self.system_prompt_approved = true;
                        self.status_msg = "System Prompt godkjent.".to_string();
                    }
                });
            });
            ui.separator();
        }

        ui.horizontal(|ui| {
            ui.colored_label(egui::Color32::WHITE, "Temperatur:");
            ui.add(egui::Slider::new(&mut self.temperature, 0.1..=1.5));
            ui.colored_label(egui::Color32::WHITE, "Max Tokens:");
            ui.add(egui::Slider::new(&mut self.max_tokens, 50..=2000));
            ui.checkbox(
                &mut self.use_context,
                egui::RichText::new("Bruk kontekst").color(egui::Color32::WHITE),
            )
            .on_hover_text(
                "Inkluderer tidligere meldinger formatert med [bruker] og [AI] adskilt med mellomrom samt instruksjon i System Prompt.",
            );
            ui.checkbox(
                &mut self.enable_ocr_paste,
                egui::RichText::new("Bilde-OCR").color(egui::Color32::WHITE),
            )
            .on_hover_text(
                "Tillater innliming av bilder (fra utklippstavlen eller skjermklipp) som oversettes til tekst via Windows OCR og brukes som prompt.",
            );
            ui.checkbox(
                &mut self.auto_evaluate_and_continue,
                egui::RichText::new("Auto-vurder & fortsett").color(egui::Color32::WHITE),
            )
            .on_hover_text(
                "Når AI er ferdig, vurderer den automatisk om svaret er godt nok. Hvis ikke, bes den automatisk om å fortsette og fullføre svaret.",
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(egui::RichText::new("🗑️ Nullstill chat").color(egui::Color32::WHITE))
                    .on_hover_text("Tømmer samtalehistorikken og starter på nytt.")
                    .clicked()
                {


            if self.is_generating {
                        self.cancel_generation();
                    }
                    self.chat_history.clear();
                    self.status_msg = "Samtalehistorikk er nullstilt.".to_string();
                }
                if ui.button("⚙️ System Prompt").clicked() {
                    self.active_tab = ActiveTab::SystemPrompt;
                }
            });
        });

        ui.separator();

        let text_style = egui::TextStyle::Body;
        let row_height = ui.text_style_height(&text_style);
        let num_rows = self.chat_history.len();

        egui::ScrollArea::vertical()
            .max_height(430.0)
            .stick_to_bottom(true)
            .show_rows(ui, row_height, num_rows, |ui, _row_range| {
                for (sender, msg) in &self.chat_history {
                    ui.group(|ui| {
                        if sender == "Du" {
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 180, 255),
                                format!("👤 {}:", sender),
                            );
                        } else if sender == "System" {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 200, 80),
                                format!("ℹ️ {}:", sender),
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 255, 150),
                                format!("🤖 {}:", sender),
                            );
                        }
                        // Lysere, ren hvit tekst for høy kontrast i chat
                        ui.colored_label(egui::Color32::WHITE, msg);
                    });
                    ui.add_space(4.0);
                }
            });

        ui.separator();

        if !self.attached_images.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new("📷 Vedlagte bilder:")
                        .color(egui::Color32::from_rgb(120, 200, 255))
                        .strong(),
                );

                let mut to_remove = None;
                for (idx, img) in self.attached_images.iter().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            if let Some(ref tex) = img.texture {
                                ui.add(egui::Image::new(tex).max_size(egui::vec2(52.0, 52.0)));
                            } else {
                                ui.label("🖼️ Bilde");
                            }

                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("#{}", img.id))
                                        .color(egui::Color32::WHITE)
                                        .small(),
                                );
                                if ui
                                    .button(
                                        egui::RichText::new("✕")
                                            .color(egui::Color32::from_rgb(255, 100, 100))
                                            .strong(),
                                    )
                                    .on_hover_text("Fjern dette bildet")
                                    .clicked()
                                {
                                    to_remove = Some(idx);
                                }
                            });
                        });
                    });
                    ui.add_space(4.0);
                }

                if let Some(idx) = to_remove {
                    let removed = self.attached_images.remove(idx);
                    let _ = std::fs::remove_file(&removed.path);
                    let cache_dir = std::env::temp_dir().join("hf_assistant_cache");
                    let thumb_raw = cache_dir.join(format!("img_{}_thumb.raw", removed.id));
                    let _ = std::fs::remove_file(thumb_raw);
                    self.status_msg = format!("Fjernet bilde #{}.", removed.id);
                }
            });
            ui.add_space(4.0);
        }

        ui.horizontal(|ui| {
            let mut right_reserve = if self.is_generating { 115.0 } else { 80.0 };
            if self.enable_ocr_paste {
                right_reserve += 175.0;
            }
            let response = ui.add_sized(
                [ui.available_width() - right_reserve, if self.current_prompt.contains('\n') { 70.0 } else { 34.0 }],
                egui::TextEdit::multiline(&mut self.current_prompt)
                    .text_color(egui::Color32::WHITE)
                    .hint_text("Skriv melding... (eller lim inn bilder med Ctrl+V)"),
            );

            if response.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)
                && !self.is_generating
            {
                self.send_prompt();
            }

            if self.enable_ocr_paste {
                if self.is_pasting_image {
                    ui.spinner();
                    ui.colored_label(egui::Color32::from_rgb(255, 200, 80), "Limer inn...");
                } else if ui
                    .button(egui::RichText::new("📷 Lim inn bilde (+)").color(egui::Color32::WHITE))
                    .on_hover_text("Lim inn bilde fra utklippstavlen (Win+Shift+S eller kopi). Du kan legge til flere bilder!")
                    .clicked()
                {
                    self.paste_clipboard_image(true);
                }
            }

            if self.is_generating {
                ui.spinner();
                if ui.button("⏹ Avbryt").clicked() {
                    self.cancel_generation();
                }
            } else if ui.button("Send ➔").clicked() {
                self.send_prompt();
            }
        });
    }
}

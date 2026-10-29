use crate::hardware::HardwareSpec;
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
    is_generating: bool,
    cancel_flag: Arc<AtomicBool>,
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
            max_tokens: 250,
            is_generating: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        };

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
        let models_dir = Path::new("models");
        let mut list = Vec::new();
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    list.push(path);
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
            let models_dir = Path::new("models");
            let tx_prog = tx.clone();
            let fname_for_prog = fname_clone.clone();

            let res = crate::downloader::download_hf_file_with_progress(
                &repo_clone,
                &fname_clone,
                models_dir,
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
            let _ = crate::downloader::ensure_tokenizer_for_repo(&repo_clone, models_dir).await;

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

    fn send_prompt(&mut self) {
        if !self.system_prompt_approved {
            self.status_msg =
                "Vurder og godkjenn System Prompt i fanen 'System Prompt' før du sender melding."
                    .to_string();
            return;
        }

        let prompt = self.current_prompt.trim().to_string();
        if prompt.is_empty() || self.is_generating {
            return;
        }

        self.cancel_flag.store(false, Ordering::SeqCst);

        self.chat_history.push(("Du".to_string(), prompt.clone()));
        self.chat_history.push(("Modell".to_string(), String::new()));
        self.current_prompt.clear();
        self.is_generating = true;
        self.status_msg = "Genererer svar...".to_string();

        let llm_arc = self.loaded_llm.clone();
        let tx = self.event_tx.clone();
        let max_toks = self.max_tokens;
        let temp = self.temperature as f64;
        let p = self.top_p as f64;
        let sys_prompt = self.system_prompt.clone();
        let disable_think = self.disable_thinking_tags;
        let cancel_arc = self.cancel_flag.clone();

        thread::spawn(move || {
            let mut guard = llm_arc.lock().unwrap();
            if let Some(ref mut llm) = *guard {
                let tx_token = tx.clone();
                let cancel_for_cb = cancel_arc.clone();
                let res = llm.generate_with_callback(
                    &sys_prompt,
                    disable_think,
                    &prompt,
                    max_toks,
                    temp,
                    p,
                    Some(cancel_arc.as_ref()),
                    move |token| {
                        if cancel_for_cb.load(Ordering::Relaxed) {
                            return false;
                        }
                        let _ = tx_token.send(AppEvent::InferenceToken(token.to_string()));
                        true
                    },
                );

                if cancel_arc.load(Ordering::Relaxed) {
                    let _ = tx.send(AppEvent::InferenceCancelled);
                } else if let Err(e) = res {
                    let msg = e.to_string();
                    if msg.contains("Avbrutt av bruker") {
                        let _ = tx.send(AppEvent::InferenceCancelled);
                    } else {
                        let _ = tx.send(AppEvent::InferenceError(msg));
                    }
                } else {
                    let _ = tx.send(AppEvent::InferenceFinished);
                }
            } else {
                let _ = tx.send(AppEvent::InferenceError(
                    "Ingen modell er lastet inn i minnet enda. Gå til 'Nedlastinger & Lokale Modeller' og klikk 'Last inn & Åpne i Chat'.".to_string(),
                ));
            }
        });
    }
}

impl eframe::App for HfAssistantApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                AppEvent::InferenceToken(token) => {
                    if let Some((_, text)) = self.chat_history.last_mut() {
                        text.push_str(&token);
                    }
                }
                AppEvent::InferenceFinished => {
                    self.is_generating = false;
                    self.status_msg = "Generering fullført.".to_string();
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
            ctx.request_repaint();
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

        let models_dir = Path::new("models");
        let mut local_files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    local_files.push(p);
                }
            }
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
        if let Some(ref path) = self.selected_llm_path {
            let p_clone = path.clone();
            let arc_llm = self.loaded_llm.clone();
            let tx = self.event_tx.clone();
            let handle = self.tokio_handle.clone();

            self.is_loading_llm = true;
            self.status_msg = format!(
                "Laster inn {:?} i minnet...",
                path.file_name().unwrap_or_default()
            );

            thread::spawn(move || {
                let models_dir = Path::new("models");
                // Sørg for at tokenizer.json er på plass før vi laster GGUF
                let tok_res = handle.block_on(async {
                    crate::downloader::ensure_tokenizer_for_repo(
                        "ornith-ai/Ornith-1.5-9B",
                        models_dir,
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

                match LocalLlm::load(&p_clone, &tok_path) {
                    Ok(model) => {
                        let arch = model.architecture.clone();
                        let mut guard = arc_llm.lock().unwrap();
                        *guard = Some(model);
                        let _ = tx.send(AppEvent::ModelLoadResult(Ok(format!(
                            "Modell {:?} (arkitektur: {}) er ferdig lastet inn og klar for prompting!",
                            p_clone.file_name().unwrap_or_default(),
                            arch
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
            ui.add(egui::Slider::new(&mut self.max_tokens, 50..=1000));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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

        ui.horizontal(|ui| {
            let right_reserve = if self.is_generating { 115.0 } else { 80.0 };
            let response = ui.add_sized(
                [ui.available_width() - right_reserve, 32.0],
                egui::TextEdit::singleline(&mut self.current_prompt)
                    .text_color(egui::Color32::WHITE)
                    .hint_text("Skriv din prompt her og trykk Enter..."),
            );

            if response.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                && !self.is_generating
            {
                self.send_prompt();
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

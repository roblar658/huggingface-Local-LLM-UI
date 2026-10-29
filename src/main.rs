mod downloader;
mod hardware;
mod hf_client;
mod inference;
mod recommender;
mod ui;

use anyhow::Result;
use hardware::HardwareSpec;
use hf_client::HfPoliteClient;
use recommender::{assess_model, FitLevel};
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--cli") {
        run_cli()
    } else {
        run_gui()
    }
}

fn run_gui() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let handle = rt.handle().clone();

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1120.0, 780.0])
            .with_min_inner_size([850.0, 600.0])
            .with_title("Hugging Face Local Assistant - Windows"),
        ..Default::default()
    };

    eframe::run_native(
        "Hugging Face Local Assistant",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(ui::HfAssistantApp::new(cc, handle)))
        }),
    ).map_err(|e| anyhow::anyhow!("Feil ved oppstart av GUI: {}", e))
}

fn run_cli() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        println!("=======================================================");
        println!("   Hugging Face Local Assistant (Windows / Rust CLI)   ");
        println!("=======================================================\n");

        println!("Analyserer lokal PC...");
        let hw = HardwareSpec::detect()?;
        println!("  CPU:           {} ({} kjerner)", hw.cpu_brand, hw.cpu_cores);
        println!("  RAM:           Total: {:.1} GB (Tilgjengelig: {:.1} GB)", hw.total_ram_gb, hw.available_ram_gb);
        println!("  GPU:           {}", hw.gpu_name);
        println!("  Dedikert VRAM: {:.2} GB\n", hw.dedicated_vram_gb);

        println!("Henter åpne GGUF-modeller fra Hugging Face (høflig API-tilgang)...");
        let client = HfPoliteClient::new();
        let models = client.list_open_gguf_models(10).await?;

        println!("\nAnbefalte modeller for din maskinvare:");
        println!("{:<4} {:<40} {:<10} {:<20}", "Nr", "Modell ID", "Størrelse", "Kjøringsmodus");
        println!("{}", "-".repeat(80));

        let mut assessments = Vec::new();
        for (i, m) in models.iter().enumerate() {
            let assessment = assess_model(m, &hw);
            let fit_desc = match assessment.fit_level {
                FitLevel::FullGpuOffload => "Rask (Full GPU VRAM)",
                FitLevel::CpuRamFallback => "Medium (System-RAM / CPU)",
                FitLevel::TooLarge => "For stor for PC-en",
            };
            println!("{:<4} {:<40} {:<10} {:<20}", i + 1, m.id, assessment.est_param_size, fit_desc);
            assessments.push((m, assessment));
        }

        print!("\nVelg modellnummer for å se filer og laste ned (eller 'q' for å avslutte): ");
        io::stdout().flush()?;
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim();

        if choice == "q" || choice.is_empty() {
            println!("Avslutter.");
            return Ok(());
        }

        let idx: usize = choice.parse::<usize>()? - 1;
        let (selected_model, _) = &assessments[idx];

        println!("\nHenter filer for {}...", selected_model.id);
        let files = client.get_model_files(&selected_model.id).await?;
        
        let target_file = files
            .iter()
            .find(|f| f.to_lowercase().contains("q4_k_m"))
            .or_else(|| files.first())
            .ok_or_else(|| anyhow::anyhow!("Ingen GGUF-filer funnet i dette depotet"))?;

        println!("Valgt GGUF-kvantiseringsfil: {}", target_file);

        let models_dir = Path::new("models");
        let downloaded_path = downloader::download_hf_file(&selected_model.id, target_file, models_dir).await?;

        println!("Sikrer at tokenizer.json er tilgjengelig (sjekker moder-depot ved behov)...");
        let tokenizer_path = downloader::ensure_tokenizer_for_repo(&selected_model.id, models_dir).await?;

        println!("\nKlargjør inferensmotor...");
        let mut llm = inference::LocalLlm::load(&downloaded_path, &tokenizer_path)?;

        let mut history: Vec<(String, String)> = Vec::new();
        loop {
            print!("\nSkriv inn prompt (eller 'exit' for å avslutte, 'reset' for å nullstille chat): ");
            io::stdout().flush()?;
            let mut prompt = String::new();
            io::stdin().read_line(&mut prompt)?;
            let prompt = prompt.trim();

            if prompt == "exit" {
                break;
            }
            if prompt == "reset" {
                history.clear();
                println!("Samtalehistorikk er nullstilt.");
                continue;
            }
            if prompt.is_empty() {
                continue;
            }

            let mut parts: Vec<String> = Vec::new();
            for (sender, text) in &history {
                if sender == "Du" {
                    parts.push(format!("[bruker] {}", text));
                } else if sender == "Modell" {
                    parts.push(format!("[AI] {}", text));
                }
            }
            parts.push(format!("[bruker] {}", prompt));
            let full_prompt = parts.join(" ");

            let sys_prompt = "Du er en ekstremt kortfattet assistent. Svar kun med selve svaret på norsk i maksimalt 1–2 korte setninger uten innledning. Om ikke mulig, øk antall setninger med 1 inntil det gir mening. Samtalehistorikk og kontekst er strukturert kronologisk der [bruker] angir meldinger fra brukeren og [AI] angir svar fra assistenten, adskilt med mellomrom. Tolk denne konteksten og svar som [AI] på den siste [bruker]-meldingen.";

            println!("\nGenererer svar...");
            let ans = llm.generate_with_callback(
                sys_prompt,
                true,
                &full_prompt,
                200,
                0.7,
                0.9,
                None,
                |tok| {
                    print!("{}", tok);
                    let _ = io::stdout().flush();
                    true
                },
            )?;
            history.push(("Du".to_string(), prompt.to_string()));
            history.push(("Modell".to_string(), ans));
        }

        Ok(())
    })
}

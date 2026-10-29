# 🤖 Hugging Face Local Assistant (Windows / Rust GUI & CLI)

En rask, brukervennlig lokal AI-assistent for Windows skrevet i **Rust**, med **100 % egenutviklet inferensmotor** (uten `llama.cpp`), **zero-copy minnekartlegging (`MapViewOfFile`)** og **AVX2 + FMA SIMD-akselererte kvantiseringskjerner**.

---

## ✨ Hovedfunksjoner

1. **Lokal maskinvareanalyse (Windows DXGI & System-RAM)**
   - Detekterer automatisk CPU-modell og antall logiske kjerner, total/tilgjengelig RAM og dedikert GPU VRAM direkte via Windows `IDXGIFactory1` (`DXGI_ADAPTER_DESC1`).
2. **Høflig Hugging Face Hub-utforsker**
   - Søker etter åpne GGUF-modeller via Hugging Face Hub API med egendefinert `User-Agent`, automatisk pause mellom forespørsler (rate-limit-vennlig) og eksponentiell backoff ved HTTP 429.
   - Vurderer hver modell mot PC-ens tilgjengelige GPU VRAM og system-RAM (`🟢 Rask (Full GPU VRAM)`, `🟡 Medium (System-RAM / CPU)`, `🔴 For stor for maskinvaren`).
3. **Smart GGUF- og Tokenizer-nedlaster**
   - Strømmer `.gguf`-filer med sanntids fremdriftsindikator.
   - Finner og laster automatisk ned tilhørende `tokenizer.json` (inkludert automatisk oppslag i moder-depotet når GGUF-depotet kun inneholder `.gguf`-vektene).
4. **100 % Egenutviklet Rust Inferensmotor (`src/inference.rs`)**
   - **Zero-Copy Windows Minnekartlegging (`CreateFileMappingW` + `MapViewOfFile`)**: Kartlegger fler-gigabyte `.gguf`-modeller direkte fra disk på under 5 ms uten å spise av Windows Commit Charge / sidevekslingsfil.
   - **Egen GGUF v1/v2/v3-parser**: Leser metadata og tensor-offsets direkte fra minnekartet.
   - **Egne AVX2 + FMA 256-bit SIMD-kjerner**: Maskinvareakselerert matrisemultiplikasjon (`Q4_K`, `Q6_K`, `Q5_K`, `Q8_0`, `Q4_0`, `F16`, `BF16`, `F32`) parallellisert over alle CPU-tråder med `rayon`.
   - **Støtter både `qwen35` (Gated DeltaNet SSM + Full Attention) og standard Transformer-modeller (`llama`, `qwen2`, `mistral`, `gemma`, `phi3`)**.
5. **Brukervennlig Windows GUI (`eframe` / `egui`) og CLI-modus (`--cli`)**
   - **🔍 Utforsk HF Modeller**: Søk, filtrer etter maskinvaretilpasning og last ned kvantiseringer (`Q4_K_M` osv.).
   - **📥 Nedlastinger & Lokale Modeller**: Oversikt over lokale `.gguf`-filer i `models/` med ett-klikks innlasting.
   - **⚙️ System Prompt**: Egen fane for vurdering, forhåndsmaler, estimering av prefill-kostnad og godkjenning av System Prompt før bruk.
   - **💬 Chat & Prompting**: Høykontrast hvit tekst, justerbar temperatur/max tokens og umiddelbar avbryting (`⏹ Avbryt` / `Esc`).

---

## 🚀 Kjøring og bygging

### Krav
- Windows 10 / 11 (x86_64 med AVX2/FMA)
- Rust (`cargo` / `rustc`)

### Bygg og start Windows GUI (standard)
```powershell
cargo run --release
```

### Start i terminal / CLI-modus
```powershell
cargo run --release -- --cli
```

---

## 📁 Prosjektstruktur

- `src/main.rs` – Oppstart av Windows GUI (standard) eller CLI (`--cli`).
- `src/ui.rs` – Windows-brukergrensesnitt i `eframe`/`egui`.
- `src/inference.rs` – 100 % egenutviklet GGUF-parser, Win32 `MapViewOfFile`, AVX2+FMA kvantiseringskjerner, `Qwen35` Gated DeltaNet SSM + Full Attention, standard Transformer og Top-P/Temperatur-sampler.
- `src/hardware.rs` – Deteksjon av CPU, RAM og dedikert GPU VRAM via Windows DXGI.
- `src/hf_client.rs` – Høflig klient mot Hugging Face Hub API.
- `src/recommender.rs` – Estimering av modellstørrelse og minnekrav mot lokal maskinvare.
- `src/downloader.rs` – Nedlasting av `.gguf` og automatisk oppløsning av `tokenizer.json`.

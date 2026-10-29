# Hugging Face Local Assistant

En Windows-applikasjon i Rust for utforsking, nedlasting og lokal kjøring av GGUF-språkmodeller med en egenutviklet inferensmotor, minnekartlegging og SIMD-optimaliserte kvantiseringskjerner.

---

## Oversikt

Applikasjonen tilbyr et komplett miljø for å finne, vurdere og kjøre åpne språkmodeller lokalt på Windows. Løsningen inkluderer maskinvareanalyse via Windows DXGI, integrasjon mot Hugging Face Hub API, automatisk nedlasting av vekter og tokenizere, samt en egenutviklet inferensmotor implementert direkte i Rust.

---

## Funksjonalitet og arkitektur

### 1. Maskinvareanalyse (Windows DXGI og systemminne)
- Registrerer CPU-arkitektur, antall logiske kjerner, totalt og ledig systemminne (RAM).
- Identifiserer dedikert GPU-videominne (VRAM) direkte via Windows `IDXGIFactory1` (`DXGI_ADAPTER_DESC1`).
- Beregner kompatibilitet for modeller og kvantiseringer opp mot tilgjengelig minne før nedlasting.

### 2. Hugging Face Hub-integrasjon
- Søk og filtrering i åpne GGUF-modeller via Hugging Face REST API.
- Strukturert forespørselshåndtering med egendefinert User-Agent, rate-limiting og eksponentiell tilbakeholdelse ved HTTP 429.
- Automatisk analyse og klassifisering av modeller basert på om de får plass i dedikert VRAM, system-RAM eller overskrider maskinvarens kapasitet.

### 3. Nedlasting og tokenisering
- Asynkron strømming av GGUF-filer med sanntids fremdriftsrapportering.
- Automatisk deteksjon og nedlasting av kompatibel `tokenizer.json` for valgt modell, inkludert oppslag mot opprinnelig basemodell der GGUF-depotet kun inneholder binærvekter.

### 4. Egenutviklet inferensmotor
- **Zero-Copy minnekartlegging**: Benytter Win32 API (`CreateFileMappingW` og `MapViewOfFile`) for å kartlegge fler-gigabyte `.gguf`-filer direkte fra disk uten unødig allokering i sidevekslingsfilen.
- **GGUF-parser**: Fullstendig parser for GGUF-formatversjoner 1, 2 og 3, inkludert metadatastruktur, arraytyper og tensordeskriptorer.
- **SIMD-optimaliserte beregningskjerner**: AVX2- og FMA-akselerert vektor- og matrisemultiplikasjon for vanlige kvantiseringsformater (`Q4_K`, `Q6_K`, `Q5_K`, `Q8_0`, `Q4_0`, `F16`, `BF16`, `F32`), parallellisert med `rayon`.
- **Modellarkitekturer**:
  - `qwen35`: Hybridarkitektur med Gated DeltaNet State Space Model (SSM) og Full Attention.
  - Standard Transformer: Støtte for modeller basert på RoPE, RMSNorm og SiLU/SwiGLU (inkludert Llama-, Mistral-, Gemma- og Phi-varianter).
- **Sampling og kontroll**: Temperatur- og Top-P-sampling samt umiddelbar avbruddshåndtering under generering og prefill via atomiske flagg.

### 5. Brukergrensesnitt
- **Grafisk grensesnitt (GUI)**: Bygget med `eframe` og `egui`.
  - Modellutforsker med søk og filter basert på minnekapasitet.
  - Lokal filoversikt med status for nedlastede vekter og tokenizere.
  - Dedikert fane for vurdering, justering og forhåndsgodkjenning av systeminstruksjoner (System Prompt).
  - Samtalepanel med høykontrast-tekst, justerbare inferensparametere og avbryt-funksjon.
- **Kommandolinjegrensesnitt (CLI)**: Støtte for ren terminalbasert interaksjon via flagget `--cli`.

---

## Systemkrav

- Operativsystem: Windows 10 eller Windows 11 (64-bit x86_64)
- Prosessor: x86_64-prosessor med støtte for AVX2 og FMA instruksjonssett
- Kompilator: Rust 1.75 eller nyere (ved kompilering fra kildekode)

---

## Kompilering og oppstart

### Kompilering (Release)
```powershell
cargo build --release
```

### Starte grafisk grensesnitt (GUI)
```powershell
cargo run --release
```

### Starte kommandolinjegrensesnitt (CLI)
```powershell
cargo run --release -- --cli
```

---

## Prosjektstruktur

```
src/
  main.rs         Inngangspunkt for applikasjonen (velger GUI eller CLI)
  ui.rs           Grafisk brukergrensesnitt implementert med egui/eframe
  inference.rs    Egenutviklet inferensmotor (GGUF-parser, Win32 mmap, SIMD-kjerner, sampling)
  hardware.rs     Maskinvaredeteksjon via Windows DXGI og systemkall
  hf_client.rs    Klient mot Hugging Face Hub API med hastighetsbegrensning
  recommender.rs  Beregning av minnebehov og maskinvarekompatibilitet
  downloader.rs   Håndtering av filnedlastinger og oppløsning av tokenizer
```

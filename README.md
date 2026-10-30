# Hugging Face Local Assistant

En Windows-applikasjon i Rust for utforsking, nedlasting og lokal kjøring av GGUF-språkmodeller med en egenutviklet GPU- og CPU-inferensmotor (CUDA Driver API PTX-kjerner og AVX2/FMA SIMD), minnekartlegging og dynamisk ressursstyring.

<img width="957" height="545" alt="image" src="https://github.com/user-attachments/assets/921df855-9dfc-40ee-9b9a-735c6f3c4573" />
<img width="956" height="536" alt="image" src="https://github.com/user-attachments/assets/8ceb6532-02de-437f-a3f5-a36627cdbdc6" />
<img width="957" height="526" alt="image" src="https://github.com/user-attachments/assets/d9086947-ecf9-47a9-bc20-bcd3083200a7" />
<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/b4eb4f28-3999-4c4c-8e9e-17a18a5c2453" />



---

## Oversikt

Applikasjonen tilbyr et komplett miljø for å finne, vurdere og kjøre åpne språkmodeller lokalt på Windows. Løsningen inkluderer maskinvareanalyse via Windows DXGI og NVIDIA NVML, integrasjon mot Hugging Face Hub API, automatisk nedlasting av vekter og tokenizere, samt en egenutviklet hybrid GPU/CPU-inferensmotor med dynamisk ressursregulering implementert direkte i Rust.

---

## Funksjonalitet og arkitektur

### 1. Maskinvareanalyse og dynamisk ressursstyring (`ResourceGovernor`)
- Registrerer CPU-arkitektur, antall logiske kjerner, totalt og ledig systemminne (RAM) samt tilgjengelig sidevekslingsfil via Win32 `GlobalMemoryStatusEx`.
- Identifiserer dedikert GPU-videominne (VRAM), sanntids GPU-last, VRAM-bruk og GPU-temperatur via Windows DXGI (`IDXGIFactory1`) og NVIDIA Management Library (`nvml.dll`).
- **Trygg sub-maksimal ressursbruk (GPU og CPU)**:
  - **VRAM-sikkerhetsmargin**: Ved modellinnlasting reserveres alltid minst 1,2 GB (eller ~15 % av total VRAM) som fri margin slik at Windows Desktop Window Manager (DWM), brukergrensesnittet og andre åpne programmer aldri går tomme for videominne.
  - **Dynamisk GPU- og CPU-skalering**: Under normal drift reguleres antall samtidige CUDA thread-blocks på GPU og aktive `rayon`-arbeidstråder på CPU (~75 % av logiske kjerner med `THREAD_PRIORITY_BELOW_NORMAL`) slik at maskinen yter maksimalt innenfor trygge rammer.
- **Fokusbevisst skalering ved programbytte (`Alt+Tab`)**: Når applikasjonsvinduet mister fokus, skaleres både GPU-grid-størrelse, aktive CPU-tråder og UI-oppdateringsfrekvens automatisk ned for å gi full prioritet til det aktive programmet, og skaleres opp igjen når brukeren vender tilbake.
- **Automatisk håndtering av lavt minne og overbelastning**: Ved lavt tilgjengelig RAM/VRAM eller høy GPU-temperatur reduseres ressursbruken dynamisk. Ved kritisk minnepress frigjøres prosessens working set via `SetProcessWorkingSetSize`, og motoren tar korte, avbrytbare pauser inntil belastningen synker uten at genereringen henger seg opp eller stopper før svaret er ferdig.

### 2. Hugging Face Hub-integrasjon
- Søk og filtrering i åpne GGUF-modeller via Hugging Face REST API.
- Strukturert forespørselshåndtering med egendefinert User-Agent, rate-limiting og eksponentiell tilbakeholdelse ved HTTP 429.
- Automatisk analyse og klassifisering av modeller basert på om de får plass i dedikert VRAM, system-RAM eller overskrider maskinvarens kapasitet.

### 3. Nedlasting og tokenisering
- Asynkron strømming av GGUF-filer med sanntids fremdriftsrapportering.
- Automatisk deteksjon og nedlasting av kompatibel `tokenizer.json` for valgt modell, inkludert oppslag mot opprinnelig basemodell der GGUF-depotet kun inneholder binærvekter.

### 4. Egenutviklet GPU- og CPU-inferensmotor
- **Direkte GPU-akselerasjon via CUDA Driver API (`nvcuda.dll`) og PTX**:
  - Egenutviklede warp-reduserende CUDA PTX-kjerner (`matmul_q4k`, `matmul_q6k`, `matmul_q5k`, `matmul_q8_0`, `matmul_q4_0`, `matmul_f16`, `matmul_bf16`, `matmul_f32` og `swiglu_f32`) lastes direkte inn via `nvcuda.dll` uten krav om eksterne biblioteker ved kjøring.
  - **Fusjonerte GPU-pipelines**: Q/K/V-projeksjoner og SSM-projeksjoner som deler samme input kjøres i grupperte GPU-kall med kun én PCIe-overføring, og hele SwiGLU Feed-Forward Network (`ffn_gate`, `ffn_up`, `swiglu_f32`, `ffn_down`) kjøres fusjonert i VRAM/L2-cache uten mellomliggende kopiering til CPU.
  - **Dynamisk hybrid GPU/CPU-fordeling (lagvis allokering og sanntidstilpasning)**:
    - Modeller som er for store til å passe i VRAM alene (f.eks. 9B, 14B eller 32B på 6 GB / 8 GB GPU), kjøres som en sømløs hybrid der en del av transformator-lagene akselereres i GPU VRAM og resterende lag kjøres på CPU.
    - Hvert lag håndteres som en koherent beregningsenhet slik at GPU-lagene fullt ut utnytter fusjonerte PTX-kjerner (f.eks. samtidig Q/K/V og VRAM-intern SwiGLU).
    - **Dynamisk tilpasning ved oppstart og sanntid**: Fordelingen er ikke fastlåst. Ved innlasting allokeres så mange lag som trygt får plass i VRAM med minst 1600 MB fri sikkerhetsmargin. Under kjøring overvåker motoren kontinuerlig faktisk ledig VRAM og systemtrykk via NVML. Ved økt VRAM-press (f.eks. hvis brukeren åpner andre programmer) flyttes lag dynamisk fra GPU til CPU for å unngå krasj, og forfremmes automatisk tilbake til GPU når ressurser frigjøres.
- **Zero-Copy minnekartlegging**: Benytter Win32 API (`CreateFileMappingW` og `MapViewOfFile`) for å kartlegge fler-gigabyte `.gguf`-filer direkte fra disk uten unødig allokering i sidevekslingsfilen.
- **Dynamisk oppdelt prefill (Chunked Prefill)**: Innkommende prompts prosesseres i dynamiske del-batches tilpasset ledig RAM og VRAM.
- **GGUF-parser**: Fullstendig parser for GGUF-formatversjoner 1, 2 og 3, inkludert metadatastruktur, arraytyper og tensordeskriptorer.
- **Modellarkitekturer**:
  - `qwen35`: Hybridarkitektur med Gated DeltaNet State Space Model (SSM) og Full Attention (støtter opptil 8192 tokens kontekstlengde).
  - Standard Transformer: Støtte for modeller basert på RoPE, RMSNorm og SiLU/SwiGLU (inkludert Llama-, Mistral-, Gemma- og Phi-varianter).
- **Sampling, repetisjonsvern og kontroll**: Temperatur- og Top-P-sampling, automatisk deteksjon av n-gram-repetisjonsløkker samt umiddelbar avbruddshåndtering via atomiske flagg.

### 5. Brukergrensesnitt
- **Grafisk grensesnitt (GUI)**: Bygget med `eframe` og `egui`.
  - Topp-panel med sanntidsoversikt over maskinvare, aktiv ressursmodus, mengde modellvekter i GPU VRAM, GPU-last (%), GPU-temperatur, aktive CPU-tråder og tilgjengelig systemminne.
  - Modellutforsker med søk og filter basert på minnekapasitet.
  - Lokal filoversikt med status for nedlastede vekter og tokenizere.
  - Dedikert fane for vurdering, justering og forhåndsgodkjenning av systeminstruksjoner (System Prompt).
  - Samtalepanel med høykontrast-tekst, justerbare inferensparametere og avbryt-funksjon.
  - Samtalekontekst med støtte for fler-stegs samtaler formatert kronologisk med `[bruker]` og `[AI]` adskilt med mellomrom, samt tilhørende tolkningsinstruks i System Prompt.
  - **Avansert OCR og bildevedlegg i prompt**:
    - Mulighet for å lime inn flere bilder direkte inn i promptfeltet (via utklippstavle med `Ctrl+V`, skjermklipp via `Win+Shift+S`, eller eksplisitt knapp).
    - Hvert vedlagte bilde vises som et miniatyrbilde (thumbnail) med knapp for fjerning.
    - Brukeren kan skrive inn ledetekst sammen med bildene.
    - Ved sending kjøres OCR sekvensielt på bildene med RapidOCR (PP-OCRv4 neural ONNX-modell med fallback til Windows Media OCR). Tekstboks-gruppering rekonstruerer oppsett fra websider, kort, tabeller og kolonner, med korrekt bevaring av linjeskift, avsnitt og mellomrom mellom tall, og settes sammen til en helhetlig prompt for modellen.
  - **Tilstandsbasert automatisk selvvurdering og fortsettelse (Auto-vurder & fortsett)**:
    - Dersom genereringen treffer token-grensen midt i en setning, fullføres den påbegynte setningen sømløst direkte fra eksisterende token-posisjon.
    - Deretter gjenbrukes eksisterende KV-cache og SSM-tilstand til å raskt vurdere om svaret er fullstendig og godt nok uten å måtte prosessere hele samtalen på nytt. Hvis svaret er ufullstendig, fortsetter modellen automatisk inntil svaret er komplett.
  - Egen funksjon for øyeblikkelig nullstilling av samtalelogg (Reset chat).
- **Kommandolinjegrensesnitt (CLI)**: Støtte for ren terminalbasert interaksjon via flagget `--cli` med støtte for samtalehistorikk og `reset`-kommando.

---

## Systemkrav

- Operativsystem: Windows 10 eller Windows 11 (64-bit x86_64)
- Prosessor: x86_64-prosessor med støtte for AVX2 og FMA instruksjonssett
- Grafikkort (valgfritt, anbefalt): NVIDIA GPU med CUDA Driver API (`nvcuda.dll`) for maskinvareakselerert GGUF-inferens i VRAM
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

### Bygge Windows MSI-installasjonspakke
For å bygge en frittstående Windows MSI-installasjonspakke (`HuggingFaceLocalAssistant.msi`):
```powershell
powershell -ExecutionPolicy Bypass -File installer\build_msi.ps1
```
Installasjonspakken oppretter automatisk snarveier på skrivebordet og i Start-menyen, installerer applikasjonen og nødvendige biblioteker i `Program Files`, og gir en ren avinstallasjonsmulighet via Windows *Legg til eller fjern programmer*.

---

## Prosjektstruktur

```
installer/
  Package.wxs         WiX Toolset v4 definisjon for MSI-installasjonspakken
  build_msi.ps1       Byggescript for release-kompilering og MSI-pakking
src/
  main.rs             Inngangspunkt for applikasjonen (velger GUI eller CLI)
  ui.rs               Grafisk brukergrensesnitt implementert med egui/eframe
  gpu.rs              CUDA Driver API (nvcuda.dll) PTX-akselerator og NVML (nvml.dll) GPU-telemetri
  gguf_kernels.ptx    Forhåndskompilerte CUDA PTX-kjerner for kvantiserte GGUF-matriseoperasjoner og SwiGLU
  inference.rs        Egenutviklet inferensmotor (GGUF-parser, Win32 mmap, GPU/CPU-dispatch, tilstandsbasert fortsettelse)
  ocr.rs              OCR-integrasjon og utklippstavlehåndtering
  hardware.rs         Maskinvaredeteksjon via Windows DXGI/NVML og dynamisk ressursstyring (ResourceGovernor)
  hf_client.rs        Klient mot Hugging Face Hub API med hastighetsbegrensning
  recommender.rs      Beregning av minnebehov og maskinvarekompatibilitet
  downloader.rs       Håndtering av filnedlastinger og oppløsning av tokenizer
tools/
  ocr_engine.py       RapidOCR (PP-OCRv4 ONNX) layout- og linjeskiftrekonstruksjon
```

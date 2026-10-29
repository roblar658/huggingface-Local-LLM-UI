# Hugging Face Local Assistant

En Windows-applikasjon i Rust for utforsking, nedlasting og lokal kjøring av GGUF-språkmodeller med en egenutviklet inferensmotor, minnekartlegging, dynamisk ressursstyring og SIMD-optimaliserte kvantiseringskjerner.

<img width="833" height="539" alt="image" src="https://github.com/user-attachments/assets/358cd437-1927-45d1-afdc-f8a607513aaf" />
<img width="833" height="536" alt="image" src="https://github.com/user-attachments/assets/515dd373-1091-44ba-ad74-f93c82161eeb" />
<img width="956" height="514" alt="image" src="https://github.com/user-attachments/assets/b71f2a8a-56ec-47e6-b27d-08c534be350d" />
<img width="956" height="574" alt="image" src="https://github.com/user-attachments/assets/6c29b817-9c34-499a-a552-5e54a4664f20" />


---

## Oversikt

Applikasjonen tilbyr et komplett miljø for å finne, vurdere og kjøre åpne språkmodeller lokalt på Windows. Løsningen inkluderer maskinvareanalyse via Windows DXGI, integrasjon mot Hugging Face Hub API, automatisk nedlasting av vekter og tokenizere, samt en egenutviklet inferensmotor med dynamisk ressursregulering implementert direkte i Rust.

---

## Funksjonalitet og arkitektur

### 1. Maskinvareanalyse og dynamisk ressursstyring (`ResourceGovernor`)
- Registrerer CPU-arkitektur, antall logiske kjerner, totalt og ledig systemminne (RAM) samt tilgjengelig sidevekslingsfil via Win32 `GlobalMemoryStatusEx`.
- Identifiserer dedikert GPU-videominne (VRAM) direkte via Windows `IDXGIFactory1` (`DXGI_ADAPTER_DESC1`).
- **Trygg sub-maksimal ressursbruk**: Forhåndskonfigurerte `rayon`-trådpooler kjører med redusert trådprioritet (`THREAD_PRIORITY_BELOW_NORMAL` / `THREAD_PRIORITY_LOWEST`) og benytter under normal drift omtrent 75 % av tilgjengelige logiske kjerner slik at operativsystem, vindushåndtering og andre programmer alltid forblir responsive.
- **Fokusbevisst skalering ved programbytte (`Alt+Tab`)**: Når applikasjonsvinduet mister fokus, skaleres antall aktive arbeidstråder og UI-oppdateringsfrekvens automatisk ned for å unngå ressurskonflikt med andre aktive programmer, og skaleres opp igjen når brukeren vender tilbake.
- **Automatisk håndtering av lavt minne og overbelastning**: Ved lavt tilgjengelig minne eller høy systemlast reduseres trådantall og prefill-batchstørrelse dynamisk. Ved kritisk lavt minne frigjøres prosessens working set via `SetProcessWorkingSetSize`, og motoren tar korte, avbrytbare pauser inntil minnetrykket synker uten at genereringen henger seg opp eller stopper før svaret er ferdig.

### 2. Hugging Face Hub-integrasjon
- Søk og filtrering i åpne GGUF-modeller via Hugging Face REST API.
- Strukturert forespørselshåndtering med egendefinert User-Agent, rate-limiting og eksponentiell tilbakeholdelse ved HTTP 429.
- Automatisk analyse og klassifisering av modeller basert på om de får plass i dedikert VRAM, system-RAM eller overskrider maskinvarens kapasitet.

### 3. Nedlasting og tokenisering
- Asynkron strømming av GGUF-filer med sanntids fremdriftsrapportering.
- Automatisk deteksjon og nedlasting av kompatibel `tokenizer.json` for valgt modell, inkludert oppslag mot opprinnelig basemodell der GGUF-depotet kun inneholder binærvekter.

### 4. Egenutviklet inferensmotor
- **Zero-Copy minnekartlegging**: Benytter Win32 API (`CreateFileMappingW` og `MapViewOfFile`) for å kartlegge fler-gigabyte `.gguf`-filer direkte fra disk uten unødig allokering i sidevekslingsfilen.
- **Dynamisk oppdelt prefill (Chunked Prefill)**: Innkommende prompts prosesseres i dynamiske del-batches tilpasset ledig RAM. Dette reduserer midlertidige minneallokeringer drastisk og tillater kontinuerlig ressurssjekk og avbruddshåndtering.
- **GGUF-parser**: Fullstendig parser for GGUF-formatversjoner 1, 2 og 3, inkludert metadatastruktur, arraytyper og tensordeskriptorer.
- **SIMD-optimaliserte beregningskjerner**: AVX2- og FMA-akselerert vektor- og matrisemultiplikasjon for vanlige kvantiseringsformater (`Q4_K`, `Q6_K`, `Q5_K`, `Q8_0`, `Q4_0`, `F16`, `BF16`, `F32`), parallellisert via `ResourceGovernor`.
- **Modellarkitekturer**:
  - `qwen35`: Hybridarkitektur med Gated DeltaNet State Space Model (SSM) og Full Attention (støtter opptil 8192 tokens kontekstlengde).
  - Standard Transformer: Støtte for modeller basert på RoPE, RMSNorm og SiLU/SwiGLU (inkludert Llama-, Mistral-, Gemma- og Phi-varianter).
- **Sampling, repetisjonsvern og kontroll**: Temperatur- og Top-P-sampling, automatisk deteksjon av n-gram-repetisjonsløkker samt umiddelbar avbruddshåndtering via atomiske flagg.

### 5. Brukergrensesnitt
- **Grafisk grensesnitt (GUI)**: Bygget med `eframe` og `egui`.
  - Topp-panel med sanntidsoversikt over maskinvare, aktiv ressursmodus, antall aktive tråder og tilgjengelig systemminne.
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
  inference.rs    Egenutviklet inferensmotor (GGUF-parser, Win32 mmap, SIMD-kjerner, tilstandsbasert fortsettelse)
  ocr.rs          OCR-integrasjon og utklippstavlehåndtering
  hardware.rs     Maskinvaredeteksjon via Windows DXGI og dynamisk ressursstyring (ResourceGovernor)
  hf_client.rs    Klient mot Hugging Face Hub API med hastighetsbegrensning
  recommender.rs  Beregning av minnebehov og maskinvarekompatibilitet
  downloader.rs   Håndtering av filnedlastinger og oppløsning av tokenizer
tools/
  ocr_engine.py   RapidOCR (PP-OCRv4 ONNX) layout- og linjeskiftrekonstruksjon
```

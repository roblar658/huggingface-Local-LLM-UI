# Arkitektur og Kommunikasjonsflyt

Dette dokumentet gir en grundig teknisk gjennomgang av arkitekturen, trådmodellen, dataflyten og maskinvareintegrasjonen i **Hugging Face Local Assistant**.

---

## Innholdsfortegnelse
1. [Overordnet systemarkitektur](#1-overordnet-systemarkitektur)
2. [Trådmodell og asynkron kommunikasjonsflyt](#2-trådmodell-og-asynkron-kommunikasjonsflyt)
3. [Hybrid GPU/CPU-allokering og dynamisk minneflyt](#3-hybrid-gpucpu-allokering-og-dynamisk-minneflyt)
4. [Modellutvalg, Hugging Face Hub-søk og FitLevel-beslutningstre](#4-modellutvalg-hugging-face-hub-søk-og-fitlevel-beslutningstre)
5. [Multimodal OCR og sammensatt prompt-pipeline](#5-multimodal-ocr-og-sammensatt-prompt-pipeline)
6. [Tekniske detaljer om delsystemer](#6-tekniske-detaljer-om-delsystemer)

---

## 1. Overordnet systemarkitektur

Systemet er inndelt i modulære lag der brukergrensesnitt, maskinvareovervåking, modellutforsking, bildebehandling og inferensmotor opererer koordinert uten å blokkere hverandre.

```mermaid
flowchart TD
    subgraph UI["Brukergrensesnitt"]
        GUI["Windows GUI (eframe / egui)"]
        CLI["CLI-grensesnitt (Terminal)"]
        EventLoop["Ikke-blokkerende Event-Loop (60 FPS, try_lock)"]
        Hourglass["Animert Timeglass-indikator (⏳ / ⌛)"]
    end

    subgraph Telemetry["Maskinvareovervåking & Ressursstyring"]
        DXGI["Windows DXGI (GPU-deteksjon)"]
        NVML["NVIDIA NVML (VRAM, Last %, Temp °C)"]
        WinMem["Win32 GlobalMemoryStatusEx (RAM & Paging)"]
        Gov["ResourceGovernor (Rayon trådpooler & Prioritering)"]
    end

    subgraph Catalog["Modellkatalog & Hugging Face Hub"]
        CuratedList["Kuratert bibliotek (30+ toppmodeller)"]
        HFHubApi["Hugging Face REST API (Topp 120 + Live-søk)"]
        Recommender["Recommender & FitLevel (🟢 Full GPU / 🟡 Partiell / 🔴 For stor)"]
        Downloader["Downloader (SHA-256 & Fremdrift)"]
        TokResolver["Tokenizer Resolver (Automatisk oppslag)"]
    end

    subgraph Preproc["Multimodal Bilde- og Tekstbehandling"]
        Clipboard["Utklippstavle / Skjermklipp (Ctrl+V / Win+Shift+S)"]
        Upscaler["2x Bikubisk oppskalering & Kontrast"]
        OCR["RapidOCR / Windows Media OCR"]
        TrapCorrection["Gjenoppretting av seriftall (f.eks. '100' vs 'ioo')"]
        PromptBuilder["Sammensatt prompt (Bilder + Kontekst + Instruksjon)"]
    end

    subgraph Engine["Egenutviklet Hybrid GPU/CPU Inferensmotor"]
        Mmap["Zero-Copy Win32 MapViewOfFile (.gguf)"]
        Parser["GGUF Parser (Metadata, Tensordata)"]
        DynamicPlanner["Dynamisk lagplanlegger (VRAM-margin & Flyt)"]
        CUDA_PTX["CUDA Driver API (nvcuda.dll) PTX-kjerner"]
        CPU_AVX["CPU AVX2+FMA SIMD (Rayon trådpool)"]
        ModelArch["Modellarkitektur (Qwen3.5 SSM DeltaNet / Standard Transformer)"]
        Sampler["Sampler (Temperatur, Top-P, Repetisjonssperre)"]
        AutoEval["Auto-vurder & Fortsett (Gjenbruk av KV-cache)"]
    end

    GUI --> EventLoop
    CLI --> Engine
    EventLoop --> Telemetry
    Telemetry --> Gov
    Gov --> DXGI
    Gov --> NVML
    Gov --> WinMem

    EventLoop --> Catalog
    Catalog --> HFHubApi
    Catalog --> CuratedList
    HFHubApi --> Recommender
    Recommender --> Downloader
    Downloader --> TokResolver
    Downloader --> Mmap

    EventLoop --> Preproc
    Clipboard --> Upscaler
    Upscaler --> OCR
    OCR --> TrapCorrection
    TrapCorrection --> PromptBuilder
    PromptBuilder --> Engine

    Mmap --> Parser
    Parser --> DynamicPlanner
    DynamicPlanner -->|GPU-lag| CUDA_PTX
    DynamicPlanner -->|CPU-lag| CPU_AVX
    CUDA_PTX --> ModelArch
    CPU_AVX --> ModelArch
    ModelArch --> Sampler
    Sampler --> AutoEval
    AutoEval -.->|Strømmer tokens| EventLoop
    EventLoop -.-> Hourglass
```

---

## 2. Trådmodell og asynkron kommunikasjonsflyt

For å sikre at Windows-brukergrensesnittet aldri fryser eller havner i *"Not Responding"*-tilstand under krevende matriseoperasjoner, benytter applikasjonen en strengt ikke-blokkerende meldingskø (`std::sync::mpsc::channel`) mellom hovedtråden og bakgrunnstrådene.

```mermaid
sequenceDiagram
    autonumber
    actor Bruker as Bruker
    participant UI as GUI-tråd (eframe/egui)
    participant Channel as mpsc::channel (AppEvent)
    participant Worker as Bakgrunnstråd (thread::spawn)
    participant Engine as Inferensmotor (LocalLlm)
    participant Hardware as Maskinvare (NVML/DXGI)

    Bruker->>UI: Skriver melding / limer inn bilde (Ctrl+V) og trykker Send
    UI->>UI: Viser animert timeglass (⏳) og oppretter melding i logg
    UI->>Worker: Starter asynkron oppgave (thread::spawn)
    
    opt Bilde vedlagt
        Worker->>Worker: 2x Bikubisk oppskalering & OCR-prosessering
        Worker->>Worker: Korrigerer seriftall (f.eks. 'ioo' -> '100')
        Worker->>Channel: Send CompositePromptReady
        Channel-->>UI: Oppdaterer promptvisning i chatlogg
    end

    Worker->>Engine: Låser modellmuteks (guard = llm.lock())
    
    loop Prefill & Token-generering
        Worker->>Hardware: Henter hurtigbufret telemetri (sample_cached 150ms)
        opt Hvert 4. sekund
            Worker->>Engine: dynamically_adjust_placement(snap)
        end
        Worker->>Engine: forward(&[token], pos) (PTX på GPU / AVX2 på CPU)
        Engine-->>Worker: Neste token generert
        Worker->>Channel: Send AppEvent::InferenceToken(token)
        Worker->>UI: ctx.request_repaint() (vekker UI-tråden)
        
        Note over UI: UI-tråd kaller try_lock() (ikke-blokkerende)<br/>Tegner 60 FPS og animerer timeglass (⏳ ⟷ ⌛)
        Channel-->>UI: try_recv() henter token og legger til i visning
    end

    opt Auto-vurder & Fortsett
        Worker->>Engine: evaluate_and_continue (gjenbruker KV-cache)
        Worker->>Channel: Send resterende tokens
        Channel-->>UI: Oppdaterer chat
    end

    Worker->>Channel: Send AppEvent::InferenceFinished
    Worker-->>Engine: Frigjør modellmuteks (drop guard)
    Channel-->>UI: Status = "Ferdig", fjerner timeglass
    UI-->>Bruker: Komplett svar synlig og klar for ny prompt
```

---

## 3. Hybrid GPU/CPU-allokering og dynamisk minneflyt

Når en modell er for stor til å få plass i dedikert GPU-minne alene, fordeles lagene sømløst mellom GPU VRAM og system-RAM. Motoren overvåker faktisk VRAM-bruk i sanntid og flytter lag dynamisk med innebygd hysterese for å forhindre PCIe-metning.

```mermaid
flowchart TD
    subgraph Storage["Disk / Lagring"]
        DiskFile["GGUF-modellfil på disk (f.eks. 9B / 14B / 32B)"]
    end

    subgraph MemoryMapping["Minnekartlegging"]
        Mmap["Win32 CreateFileMappingW & MapViewOfFile (Zero-Copy)"]
    end

    subgraph Placement["Dynamisk Hybrid Allokeringsmotor"]
        CheckVram["Sjekk faktisk VRAM via NVML (Trekk fra 1600 MB sikkerhetsmargin)"]
        SplitCalc["Beregn optimal splitting: N lag på GPU, M lag på CPU"]
        GpuLayers["GPU-lag (0..K): Fusjonerte PTX Q/K/V og SwiGLU FFN"]
        CpuLayers["CPU-lag (K..Totalt): Rayon AVX2+FMA SIMD-arbeidstråder"]
    end

    subgraph RealtimeFeedback["Sanntids Ressursregulering (Under Inferens)"]
        Monitor["NVML Sanntidsovervåking (Hvert 150ms, Cooldown: 4 sekunder)"]
        Condition{"VRAM-press eller minnemangel?"}
        Evict["Flytt lag fra GPU til CPU VRAM (cuMemFree)"]
        Promote["Forfrem lag fra CPU til GPU VRAM (cuMemAlloc + HtoD)"]
        Stable["Behold stabil lagfordeling"]
    end

    DiskFile --> Mmap
    Mmap --> CheckVram
    CheckVram --> SplitCalc
    SplitCalc --> GpuLayers
    SplitCalc --> CpuLayers

    GpuLayers -.-> Monitor
    CpuLayers -.-> Monitor
    Monitor --> Condition
    Condition -- "VRAM under press (> 90% eller < 450 MB ledig)" --> Evict
    Condition -- "Rikelig med ledig VRAM & lav last" --> Promote
    Condition -- "Normale rammebetingelser" --> Stable
    Evict -.->|Oppdaterer status| CpuLayers
    Promote -.->|Oppdaterer status| GpuLayers
```

---

## 4. Modellutvalg, Hugging Face Hub-søk og FitLevel-beslutningstre

Modellkatalogen kombinerer en kuratert samling av verifiserte modeller med live-oppslag mot Hugging Face Hub API. Hver modell analyseres matematisk mot brukerens lokale maskinvare og klassifiseres med fargekoder.

```mermaid
flowchart TD
    Start["Henting av modeller"] --> CatalogSources

    subgraph CatalogSources["Kilder for modeller"]
        Curated["Kuratert bibliotek (30+ modeller fra 0.1B til 70B)"]
        HFHubTop["Hugging Face API: Topp 120 GGUF-modeller"]
        HFHubSearch["Hugging Face API: Live-søk (f.eks. 'deepseek', 'coder', 'norwegian')"]
    end

    CatalogSources --> Merge["Flett og dedupliser basert på modell-ID"]
    Merge --> Parse["Analyser parameternavn (f.eks. 135m, 1.5b, 7b, 9b, 14b, 32b, 70b)"]
    Parse --> CalcMemory["Estimer minnebehov: req_gb = (params * 0.72)"]
    
    CalcMemory --> HardwareCheck["Sammenlign med lokal maskinvare (DXGI + Win32 RAM)"]
    
    HardwareCheck --> Decision{"Hvor passer modellen?"}

    Decision -- "safe_vram >= req_gb + 0.3 GB" --> Green["🟢 Rask (Full GPU VRAM)<br/>Passer 100% i dedikert GPU-minne"]
    Decision -- "combined_safe >= req_gb + 0.5 GB" --> Yellow["🟡 Partiell (Delvis GPU + RAM)<br/>Kjøres sømløst som hybrid lagfordeling"]
    Decision -- "Overskrider samlet minne" --> Red["🔴 For stor for maskinvaren<br/>Vil medføre overbelastning"]

    Green --> UI_Display["Vis med fargekode, parameterstørrelse og nedlastingstall i GUI"]
    Yellow --> UI_Display
    Red --> UI_Display
```

---

## 5. Multimodal OCR og sammensatt prompt-pipeline

Bildeinnliming og skjermklipp forbehandles med bildeoppskalering og optisk tegngjenkjenning før teksten kombineres med samtalehistorikk og sendes til inferensmotoren.

```mermaid
flowchart LR
    subgraph Input["Bildeinput"]
        Paste["Ctrl+V fra utklippstavle"]
        Snip["Win+Shift+S skjermklipp"]
        Btn["'Lim inn bilde'-knapp"]
    end

    subgraph Preprocessing["Forbehandling"]
        Upscale["2x Bikubisk oppskalering"]
        Contrast["Gråtone & Kontrastforbedring"]
        Deskew["Oppretting av skjev tekst"]
    end

    subgraph TextExtraction["Tekstgjenkjenning & Rekonstruksjon"]
        OCR_Engine["RapidOCR / Windows Media OCR"]
        Layout["Layout-analyse: Bevar linjeskift, tabeller & avsnitt"]
        NumericTrap["Gjenoppretting av seriftall (f.eks. 'ioo' -> '100')"]
    end

    subgraph Composition["Prompt-sammensetting"]
        History["Samtalehistorikk formatert med [bruker] og [AI]"]
        UserText["Valgfri manuell ledetekst fra brukeren"]
        SysPrompt["System Prompt instruksjoner"]
        FinalPrompt["Komplett strukturert prompt"]
    end

    subgraph InferenceStream["Inferens & Selvvurdering"]
        Prefill["Chunked Prefill"]
        Streaming["Token-strømming med animert timeglass (⏳)"]
        Eval["Auto-vurder & fortsett uten re-prefill"]
    end

    Input --> Preprocessing
    Preprocessing --> TextExtraction
    TextExtraction --> Composition
    Composition --> InferenceStream
```

---

## 6. Tekniske detaljer om delsystemer

### A. Ikke-blokkerende UI og responsivitet
- **`try_lock()`-mønster**: Hovedtråden i Windows kaller utelukkende `self.loaded_llm.try_lock()` for å avlese modellstatus og lagplassering. Dersom inferenstråden arbeider, returneres umiddelbart sist hurtigbufrede `HybridStatus`. Hovedtråden venter aldri på en `Mutex`-lås, noe som eliminerer vindusfrys og *"Not Responding"*.
- **Animert timeglass**: Under ventetid (prefill og token-generering) animeres timeglasset (`⏳` ⟷ `⌛`) jevnt i prompt-feltet, chat-boblen og statuslinjen ved å hente tidsstempel fra `egui::Context::input` og forespørre opptegning med 50 ms intervall.

### B. Dynamisk ressursregulering (`ResourceGovernor`)
- **Fokusbevissthet**: Applikasjonen detekterer om vinduet er aktivt via `egui::Context::input(|i| i.focused)`. Ved `Alt+Tab` reduseres GPU-grid-størrelse, CPU-tråder og bildefrekvens for å overlate systemressurser til andre applikasjoner.
- **Hurtigbufret telemetri**: `ResourceGovernor::sample_cached` begrenser direkte kall mot Win32 `GlobalMemoryStatusEx` og NVIDIA NVML C API til maksimalt ett oppslag per 150 ms under aktiv inferens, noe som eliminerer unødvendig kjerne-overhead.
- **Minnesikkerhet**: Working set for prosessen tømmes aldri under aktiv token-generering, slik at minnekartlagte vekter forblir varme i minnet uten sidefeil (page faults).

### C. Hybrid inferensmotor
- **CUDA PTX-kjerner**: Egendefinerte PTX-kjerner for kvantiserte matriseoperasjoner (`matmul_q4k`, `matmul_q6k`, `matmul_q5k`, `matmul_q8_0`, `matmul_q4_0`, `matmul_f16`, `matmul_bf16`, `matmul_f32`) og fusjonert SwiGLU FFN (`swiglu_f32`) lastes direkte via `nvcuda.dll` uten eksterne runtime-avhengigheter.
- **AVX2+FMA CPU-kjerner**: Resterende lag kjøres over Rayon parallell trådpool konfigurert med sub-maksimal trådmengde og `THREAD_PRIORITY_BELOW_NORMAL`.
- **Tilstandsbevarende auto-vurdering**: Modellen gjenbruker sin eksisterende KV-cache og State Space Model (SSM)-tilstand for å vurdere svaret og fullføre avkuttede setninger uten å måtte kjøre oppstarts-prefill på nytt.

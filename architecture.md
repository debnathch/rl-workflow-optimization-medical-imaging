# Medical Imaging Digital Twin Workflow Optimizer
## Architectural Design & Flow Diagrams

---

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph INPUT["📥 Inputs"]
        YAML["config/default.yaml\nPydantic Config"]
        CLI["CLI Entry Points\nmedimg-*"]
    end

    subgraph DATAGEN["📊 Data Generation Layer"]
        GEN["SyntheticDataGenerator\n100k+ Encounters"]
        HL7G["HL7EventGenerator\nADT · ORM · SIU · ORU"]
        DICOMG["DICOMMetadataGenerator\nCT · MRI · XRAY Params"]
    end

    subgraph SIMCORE["🏥 Simulation Core  (SimPy DES)"]
        HOSP["HospitalSimulation\nSimPy Environment"]
        ARRIVE["Patient Arrival\nPoisson + Diurnal"]
        PATHWAY["Patient Pathway Process\nRegistration → Queue → Scan → Report"]
        QUEUE["Priority Queues\nCT · MRI · XRAY · Emergency"]
        SCAN["Scanner Resources\nPriorityResource × 9"]
        REPORT["Radiologist Report Queue\n7 Radiologists, Specialty-matched"]
        STATS["SimulationStats\nWait-times · TAT · Utilization"]
    end

    subgraph POLICIES["🎯 Scheduling Policies"]
        FIFO["FIFOPolicy\nFirst-In-First-Out"]
        PRI["PriorityTriagePolicy\nEmergency > Urgent > Routine"]
        PPO_POL["PPOPolicy\nML-learned Dispatch"]
    end

    subgraph RLENV["🤖 RL Environment (Gymnasium)"]
        ENV["MedicalImagingEnv\ngym.Env wrapper"]
        OBS["Observation Space\n21-dim float32 vector"]
        ACT["Action Space\nDiscrete(3)"]
        REW["Reward Function\nWait · TAT · Utilization · Imbalance"]
    end

    subgraph TRAINING["🧠 RL Training (SB3 PPO)"]
        TRAINER["PPOTrainer\nStable-Baselines3"]
        CALLBACK["KPILoggingCallback\nTensorBoard KPIs"]
        MODEL["PPO Model\n.zip checkpoint"]
    end

    subgraph ANALYTICS["📈 Analytics & Reporting"]
        METRICS["MetricsComputer\nKPIs · Gini · Percentiles"]
        PMETS["PolicyMetrics\nComparison Dataclass"]
        REPORT_GEN["ReportGenerator\nPlots · HTML · PDF"]
    end

    subgraph OUTPUTS["📤 Outputs"]
        PARQUET["Parquet Files\npatients · hl7 · dicom"]
        FIGS["Publication Figures\nPNG · PDF"]
        TB["TensorBoard Logs"]
        DASH["Streamlit Dashboard\n5-Tab Live View"]
    end

    YAML --> HOSP
    YAML --> GEN
    CLI --> GEN
    CLI --> HOSP
    CLI --> TRAINER
    CLI --> DASH

    GEN --> HL7G
    GEN --> DICOMG
    GEN --> PARQUET

    HOSP --> ARRIVE --> PATHWAY
    PATHWAY --> QUEUE
    QUEUE --> SCAN
    SCAN --> REPORT
    REPORT --> STATS

    FIFO --> HOSP
    PRI --> HOSP
    PPO_POL --> HOSP

    HOSP --> ENV
    ENV --> OBS
    ENV --> ACT
    ENV --> REW

    ENV --> TRAINER
    TRAINER --> CALLBACK --> TB
    TRAINER --> MODEL
    MODEL --> PPO_POL

    STATS --> METRICS --> PMETS
    PMETS --> REPORT_GEN --> FIGS
    ENV --> DASH
    STATS --> DASH
```

---

## 2. Patient Journey Flow (Full Pathway)

```mermaid
flowchart TD
    A([👤 Patient Arrives\nPoisson Process]) --> B{Operating Hours?\n06:00–22:00}
    B -- No --> SKIP([Patient Skipped\nOutside Hours])
    B -- Yes --> C[Determine Priority\nEMERGENCY / URGENT / ROUTINE]

    C --> D[Assign Modality\nCT 45% · MRI 30% · XRAY 25%]
    D --> E[Sample Body Part\nChest · Brain · Knee ...]

    E --> F["HL7: ADT^A01\nPatient Admission"]
    F --> G[Registration\n~2 min delay]

    G --> H["HL7: ORM^O01\nImaging Order"]
    H --> I[Join Modality Queue\nPriorityResource]

    I --> J{Scanner\nAvailable?}
    J -- No --> WAIT[⏳ Wait in Queue\nLognormal distribution]
    WAIT --> J
    J -- Yes --> K["HL7: SIU^S12\nSchedule Appointment"]

    K --> L[Setup & Prep\n~5 min Normal]
    L --> M["🔬 Scan in Progress\nCT: 25 min · MRI: 45 min\nXRAY: 12 min"]
    M --> N["Generate DICOM Study\nUID · Params · Metadata"]

    N --> O[Join Report Queue\nRadiologist Assignment]
    O --> P{Radiologist\nAvailable?\nSpecialty Match}
    P -- No --> RWAIT[⏳ Report Wait\n~10 min Exponential]
    RWAIT --> P
    P -- Yes --> Q["📝 Radiologist Reports\nCT: 20 min · MRI: 30 min\nXRAY: 8 min"]

    Q --> R["HL7: ORU^R01\nReport Transmitted"]
    R --> S[Patient Discharge\n~5 min]
    S --> T["HL7: ADT^A03\nDischarge Event"]
    T --> U([✅ Encounter Complete\nMetrics Recorded])

    style A fill:#4CAF50,color:#fff
    style U fill:#2196F3,color:#fff
    style SKIP fill:#FF5722,color:#fff
    style M fill:#9C27B0,color:#fff
    style Q fill:#FF9800,color:#fff
```

---

## 3. RL Training Loop

```mermaid
sequenceDiagram
    participant PPO as PPO Agent<br/>(SB3)
    participant ENV as MedicalImagingEnv<br/>(Gymnasium)
    participant SIM as HospitalSimulation<br/>(SimPy)
    participant CB as KPILoggingCallback
    participant TB as TensorBoard

    PPO->>ENV: reset(seed=42)
    ENV->>SIM: HospitalSimulation(config, policy, seed)
    SIM->>SIM: reset() — init env at t=360min (6AM)
    SIM->>SIM: start _patient_arrival_generator()
    SIM->>SIM: start _stats_collector()
    ENV-->>PPO: obs[21], info{}

    loop Every Decision Epoch (5 min)
        PPO->>ENV: step(action ∈ {0,1,2})
        Note over ENV: action=0: FIFO<br/>action=1: Priority<br/>action=2: Emergency-first
        ENV->>SIM: _apply_action(action)
        ENV->>SIM: run_until(now + 5min)
        SIM->>SIM: Process arrivals, scans, reports
        SIM-->>ENV: SimulationSnapshot
        ENV->>ENV: _compute_reward()
        Note over ENV: reward = -α·wait_time<br/>- β·emergency_TAT<br/>+ γ·utilization<br/>- δ·imbalance
        ENV-->>PPO: obs[21], reward, terminated, truncated, info

        PPO->>PPO: Store in rollout buffer
    end

    Note over PPO: Every 2048 steps → Update policy
    PPO->>CB: on_step()
    CB->>TB: Log avg_wait_time, avg_emergency_tat
    CB->>PPO: check eval_freq
    PPO->>ENV: evaluate_policy(n_episodes=5)
    ENV-->>PPO: mean_reward, std_reward

    loop Full Training (500k steps)
        PPO->>PPO: policy_gradient_update
        PPO->>PPO: value_function_update
        PPO->>PPO: entropy_regularization
    end

    PPO->>PPO: save("best_model.zip")
```

---

## 4. Data Generation Pipeline

```mermaid
flowchart LR
    subgraph CONFIG["⚙️ Config"]
        C1["n_patients: 100,000\nstart: 2024-01-01\nend: 2024-12-31\nseed: 42"]
    end

    subgraph GEN["🏭 SyntheticDataGenerator"]
        AT["_generate_arrival_times()\nPoisson Process\nTotal span: 525,600 min"]
        ENC["_generate_encounters()\n100k iterations"]
        DF["_to_dataframes()\nPandas conversion"]
    end

    subgraph PER_PT["Per-Patient Generation"]
        P1["Sample Priority\nEMG 8% · URG 15% · ROUTINE 77%"]
        P2["Sample Modality\nCT 45% · MRI 30% · XRAY 25%"]
        P3["Sample Body Part\nWeighted per modality"]
        P4["Sample Timestamps\nLognormal durations"]
        P5["Create Patient\nDataclass with all fields"]
        P6["DICOM Study\nDICOMMetadataGenerator"]
        P7["5 HL7 Messages\nHL7EventGenerator"]
    end

    subgraph OUT["📁 outputs/dataset/"]
        O1["patients.parquet\n15.6 MB\n100k rows × 35 cols"]
        O2["hl7_messages.parquet\n52.8 MB\n500k rows\nADT·ORM·SIU·ORU·ADT"]
        O3["dicom_studies.parquet\n16.1 MB\n100k rows\nUID·params·metadata"]
        O4["metadata.json\nSeed · timestamp · config"]
    end

    CONFIG --> GEN
    AT --> ENC
    ENC --> P1 --> P2 --> P3 --> P4 --> P5
    P5 --> P6
    P5 --> P7
    P6 --> DF
    P7 --> DF
    DF --> O1
    DF --> O2
    DF --> O3
    DF --> O4
```

---

## 5. Class Hierarchy & Relationships

```mermaid
classDiagram
    class Config {
        +SimulationConfig simulation
        +ArrivalConfig arrivals
        +ModalitiesConfig modalities
        +RadiologistsConfig radiologists
        +RLConfig rl
        +DatasetConfig dataset
        +AnalyticsConfig analytics
        +load_config() Config
    }

    class Patient {
        +str patient_id
        +str mrn
        +Priority priority
        +Modality modality
        +str body_part
        +int age
        +float arrival_time
        +float scan_start_time
        +PatientStatus status
        +DICOMStudy dicom_study
        +list hl7_messages
        +wait_time() float
        +scan_duration() float
        +total_turnaround() float
        +to_dict() dict
    }

    class HL7Message {
        +HL7MessageType message_type
        +str message_control_id
        +str patient_id
        +float timestamp
        +dict segments
        +create() HL7Message
        +to_dict() dict
    }

    class DICOMStudy {
        +str study_instance_uid
        +Modality modality
        +str body_part
        +str manufacturer
        +int number_of_series
        +int number_of_images
        +dict acquisition_params
        +to_dict() dict
    }

    class HospitalSimulation {
        +Config config
        +simpy.Environment env
        +dict patients
        +dict scanners
        +dict radiologists
        +SimulationStats stats
        +reset()
        +run(duration) SimulationStats
        +run_until(until)
        +get_snapshot() SimulationSnapshot
        +scanner_utilizations() dict
        +radiologist_workloads() dict
    }

    class SimulationStats {
        +list wait_times
        +list scan_durations
        +list total_turnarounds
        +list emergency_turnarounds
        +dict modality_throughput
        +int n_arrived
        +int n_completed
        +record_patient_complete(patient)
        +summary(duration) dict
    }

    class MedicalImagingEnv {
        +Box observation_space
        +Discrete action_space
        +reset() tuple
        +step(action) tuple
        +render()
        +close()
    }

    class MetricsComputer {
        +compute(policy, stats, ...) PolicyMetrics
        +compare(metrics_list) DataFrame
        +_gini_coefficient(loads) float
    }

    class PolicyMetrics {
        +str policy_name
        +float avg_wait_time_min
        +float p95_wait_time_min
        +float throughput_per_hour
        +float avg_scanner_utilization
        +float workload_gini
        +float avg_emergency_tat_min
    }

    class SyntheticDataGenerator {
        +Config config
        +generate(n_patients) dict
        +_generate_encounters() tuple
        +_generate_arrival_times() ndarray
        +_sample_queue_wait() float
        +_sample_scan_duration() float
    }

    Patient "1" --> "1" DICOMStudy
    Patient "1" --> "5" HL7Message
    HospitalSimulation "1" --> "1" SimulationStats
    HospitalSimulation "1" --> "*" Patient
    MedicalImagingEnv "1" --> "1" HospitalSimulation
    MetricsComputer --> PolicyMetrics
    Config --> HospitalSimulation
    Config --> SyntheticDataGenerator
    Config --> MedicalImagingEnv
```

---

## 6. SimPy Simulation Internal Timeline

```mermaid
gantt
    title Patient Encounter Timeline (Approximate Durations)
    dateFormat  mm
    axisFormat  %M min

    section Patient Arrives
    Arrival event          :milestone, 00, 0min

    section Registration
    HL7 ADT-A01 sent       :milestone, 00, 0min
    Registration delay     :reg, 00, 2min

    section Ordering
    HL7 ORM-O01 sent       :milestone, 02, 0min
    Order processing       :ord, 02, 3min

    section Queue Wait
    HL7 SIU-S12 sent       :milestone, 05, 0min
    Waiting for scanner    :crit, wait, 05, 45min

    section Scan
    Setup / prep           :setup, 50, 5min
    CT Scan in progress    :active, scan, 55, 25min

    section DICOM Generation
    Study metadata created :milestone, 80, 0min

    section Reporting
    Report queue wait      :rwait, 80, 10min
    Radiologist reporting  :active, report, 90, 20min

    section Discharge
    HL7 ORU-R01 sent       :milestone, 110, 0min
    Discharge processing   :disc, 110, 5min
    HL7 ADT-A03 sent       :milestone, 115, 0min
```

---

## 7. Deployment & CLI Interface

```mermaid
graph LR
    subgraph USER["👤 Researcher / Clinician"]
        direction TB
        CLI_GEN["medimg-generate\n--n-patients 100000"]
        CLI_EXP["medimg-experiment\n--duration 480 --runs 5"]
        CLI_TRN["medimg-train\n--timesteps 500000"]
        CLI_REP["medimg-report\n--input outputs/"]
        CLI_DASH["medimg-dashboard\nStreamlit :8501"]
    end

    subgraph SCRIPTS["🖥️ Scripts Layer"]
        S1["generate_dataset.py\nTyper app"]
        S2["run_experiment.py\nTyper app"]
        S3["train_agent.py\nTyper app"]
        S4["reporting_script.py\nTyper app"]
        S5["dashboard.py\nStreamlit app"]
    end

    subgraph CORE["⚙️ Core Engine"]
        C1["SyntheticDataGenerator"]
        C2["HospitalSimulation\n+ Policies"]
        C3["MedicalImagingEnv\n+ PPOTrainer"]
        C4["MetricsComputer\n+ ReportGenerator"]
        C5["Streamlit Dashboard\n5 Tabs"]
    end

    subgraph STORE["💾 Storage"]
        F1["outputs/dataset/\n*.parquet + metadata.json"]
        F2["outputs/experiments/\nresults.json + comparison.csv"]
        F3["outputs/training/\nbest_model.zip + tensorboard/"]
        F4["outputs/figures/\n*.png + *.pdf"]
        F5["outputs/analytics/\nreport.html"]
    end

    CLI_GEN --> S1 --> C1 --> F1
    CLI_EXP --> S2 --> C2 --> F2
    CLI_TRN --> S3 --> C3 --> F3
    CLI_REP --> S4 --> C4 --> F4
    CLI_DASH --> S5 --> C5

    F1 -.->|Loads dataset| C5
    F2 -.->|Loads results| C5
    F3 -.->|Loads model| C2
    C5 --> F5

    style USER fill:#E3F2FD,stroke:#1565C0
    style SCRIPTS fill:#E8F5E9,stroke:#2E7D32
    style CORE fill:#FFF3E0,stroke:#E65100
    style STORE fill:#FCE4EC,stroke:#880E4F
```

---

## 8. Reward Function Decomposition

```mermaid
graph TD
    subgraph REWARD["🎯 Reward = Σ weighted KPIs"]
        R["reward(t)"]

        subgraph NEGATIVE["Penalties (negative)"]
            W["− α · avg_wait_time\nα = 1.0\nReduces patient wait"]
            E["− β · emergency_TAT\nβ = 3.0\nStrongest penalty"]
            I["− δ · workload_imbalance\nδ = 0.8\nGini coefficient of radiologist loads"]
        end

        subgraph POSITIVE["Incentives (positive)"]
            U["+ γ · |utilization − target|\nγ = 0.5, target = 0.85\nRewards scanner efficiency"]
            TP["+ ε · throughput_per_hour\nε = 0.3\nRewards completing patients"]
        end
    end

    R --> W
    R --> E
    R --> I
    R --> U
    R --> TP

    subgraph OBS["📡 Observation Vector (21 features)"]
        O1["[0]   CT queue length / 50"]
        O2["[1]   MRI queue length / 50"]
        O3["[2]   XRAY queue length / 50"]
        O4["[3]   Emergency queue length / 50"]
        O5["[4]   CT scanner utilization"]
        O6["[5]   MRI scanner utilization"]
        O7["[6]   XRAY scanner utilization"]
        O8["[7–13] 7 radiologist workload scores"]
        O9["[14]  Hour of day (0–1)"]
        O10["[15]  Day of week (0–1)"]
        O11["[16]  Routine patient count"]
        O12["[17]  Urgent patient count"]
        O13["[18]  Emergency patient count"]
        O14["[19]  Patients completed last epoch"]
        O15["[20]  Avg wait time last epoch / 300"]
    end

    subgraph ACT["🕹️ Action Space — Discrete(3)"]
        A0["0 → FIFO\nQueue order by arrival time"]
        A1["1 → Priority Triage\nEmergency > Urgent > Routine"]
        A2["2 → Emergency First\nMax priority to EMERGENCY queue"]
    end
```

---

## 9. HL7 Message Flow (Healthcare Interoperability)

```mermaid
sequenceDiagram
    participant REG as Registration System
    participant RIS as RIS (Order System)
    participant SIS as Scheduling System
    participant PACS as PACS / Scanner
    participant RAD as Radiologist Workstation

    Note over REG,RAD: Patient Arrives at Hospital

    REG->>RIS: ADT^A01 (Admit/Visit Notify)
    Note right of REG: MSH · PID · PV1 · EVN segments<br/>Patient demographics + visit info

    RIS->>PACS: ORM^O01 (Imaging Order)
    Note right of RIS: MSH · PID · ORC · OBR segments<br/>Modality, body part, priority, LOINC code

    SIS->>PACS: SIU^S12 (Schedule Appointment)
    Note right of SIS: SCH · PID · AIL · AIG segments<br/>Scanner ID, scheduled time, duration

    PACS->>PACS: Perform Scan
    Note over PACS: DICOM Study Generated<br/>Study UID · Series · SOP instances<br/>Acquisition parameters

    PACS->>RAD: ORU^R01 (Report Result)
    Note right of PACS: MSH · PID · OBR · OBX segments<br/>Report text, result_status='F' (Final)

    RAD->>REG: ADT^A03 (Discharge)
    Note right of RAD: MSH · PID · PV1 segments<br/>Discharge datetime

    Note over REG,RAD: Encounter Complete → Metrics Recorded
```

---

## Key Design Decisions

| Design Choice | Rationale |
|---------------|-----------|
| **SimPy DES** | Causal, event-driven; enables exact replay with same seed |
| **PriorityResource** | Native SimPy priority queuing for emergency pre-emption |
| **6 AM clock start** | Simulation env initialised at operating hours start so arrivals aren't filtered |
| **Gymnasium wrapper** | Industry standard; enables plug-in of any SB3 algorithm |
| **Discrete(3) action space** | Maps cleanly to interpretable scheduling heuristics |
| **21-dim observation** | Captures all clinically relevant queue + resource state |
| **Parquet output** | Columnar, compressed, fast for 100k rows analytics |
| **Pydantic config** | Type-safe YAML loading with validation and defaults |
| **`numpy.default_rng`** | Modern numpy Generator API; reproducible across all stochastic components |
| **Modality-specific DICOM params** | CT/MRI/XRAY have physically distinct parameter sets (kV, TR/TE, SID) |

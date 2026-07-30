from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from medimg_twin.config.settings import Config, load_config
from medimg_twin.simulation.entities import (
    Patient, Priority, Modality, PatientStatus,
    HL7Message, HL7MessageType, DICOMStudy,
    RadiologistSpecialty, ScannerSpec, RadiologistSpec, ScannerStatus
)
from medimg_twin.simulation.hl7_events import HL7EventGenerator
from medimg_twin.simulation.dicom_meta import DICOMMetadataGenerator

logger = logging.getLogger(__name__)

REFERRING_DEPARTMENTS = [
    'Emergency Department', 'Internal Medicine', 'Orthopedics', 
    'Neurology', 'Oncology', 'Cardiology', 'General Surgery', 
    'Pediatrics', 'Pulmonology', 'Gastroenterology'
]

CLINICAL_INDICATIONS_CT = [
    'Chest pain', 'Pulmonary embolism', 'Trauma assessment', 
    'Abdominal pain', 'Stroke evaluation', 'Tumor staging', 
    'Infection workup', 'Post-op follow-up', 'Bowel obstruction', 
    'Aortic aneurysm screening', 'Headache evaluation', 
    'Fever of unknown origin', 'Lymphoma staging', 'Appendicitis', 'Kidney stone'
]

CLINICAL_INDICATIONS_MRI = [
    'Low back pain', 'Knee injury', 'Brain tumor', 'Multiple sclerosis', 
    'Shoulder pain', 'Spinal cord compression', 'Seizure disorder', 
    'Dementia evaluation', 'Soft tissue mass', 'ACL tear', 'Meniscal tear', 
    'Herniated disc'
]

CLINICAL_INDICATIONS_XRAY = [
    'Cough', 'Dyspnea', 'Rib pain', 'Follow-up pneumonia', 
    'Cardiomegaly screening', 'Bone fracture', 'Joint pain', 
    'Scoliosis screening', 'Foreign body', 'Post-procedure check'
]

class SyntheticDataGenerator:
    def __init__(self, config: Config, seed: int | None = None):
        self.config = config
        effective_seed = seed if seed is not None else config.simulation.seed
        self.rng = np.random.default_rng(effective_seed)
        
        self.hl7_gen = HL7EventGenerator()
        self.dicom_gen = DICOMMetadataGenerator(rng=self.rng)
        
        self.start_date = datetime.fromisoformat(config.dataset.start_date)
        self.end_date = datetime.fromisoformat(config.dataset.end_date)
        self.total_days = (self.end_date - self.start_date).days
        
        # Parse modality distribution from config (config.modalities.distribution)
        dist = config.modalities.distribution
        self.modality_types = [Modality.CT, Modality.MRI, Modality.XRAY]
        raw_probs = [dist.CT, dist.MRI, dist.XRAY]
        prob_sum = sum(raw_probs)
        self.modality_probs = [p / prob_sum for p in raw_probs]

        # Load body part distributions from config.dataset.body_parts
        # Format: {"CT": [["CHEST", 0.35], ["ABDOMEN", 0.30], ...], ...}
        self.body_parts: dict[Modality, tuple[list[str], list[float]]] = {}
        for mod_key, parts_list in config.dataset.body_parts.items():
            mod_enum = Modality(mod_key)
            parts = [entry[0] for entry in parts_list]
            probs = [float(entry[1]) for entry in parts_list]
            p_sum = sum(probs)
            probs = [p / p_sum for p in probs]
            self.body_parts[mod_enum] = (parts, probs)

        # Fallback body parts if not in config
        _default_body_parts: dict[Modality, tuple[list[str], list[float]]] = {
            Modality.CT: (["CHEST", "ABDOMEN", "HEAD"], [0.40, 0.35, 0.25]),
            Modality.MRI: (["BRAIN", "SPINE", "KNEE"], [0.40, 0.35, 0.25]),
            Modality.XRAY: (["CHEST", "EXTREMITY", "SPINE"], [0.55, 0.30, 0.15]),
        }
        for m, default in _default_body_parts.items():
            if m not in self.body_parts:
                self.body_parts[m] = default

        self.clinical_indications = {
            Modality.CT: CLINICAL_INDICATIONS_CT,
            Modality.MRI: CLINICAL_INDICATIONS_MRI,
            Modality.XRAY: CLINICAL_INDICATIONS_XRAY,
        }
        self.referring_departments = REFERRING_DEPARTMENTS

    def generate(self, n_patients: int | None = None, output_dir: Path | str | None = None, show_progress: bool = True) -> dict[str, Path]:
        if n_patients is None:
            n_patients = self.config.dataset.n_patients
            
        if output_dir is None:
            output_dir = Path(self.config.dataset.output_dir)
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting generation of {n_patients} patients. Saving to {output_dir}")
        
        patients_list, hl7_list, dicom_list = self._generate_encounters(n_patients, show_progress)
        
        patients_df, hl7_df, dicom_df = self._to_dataframes(patients_list, hl7_list, dicom_list)
        
        patients_path = output_dir / "patients.parquet"
        hl7_path = output_dir / "hl7_messages.parquet"
        dicom_path = output_dir / "dicom_studies.parquet"
        
        patients_df.to_parquet(patients_path, index=False)
        hl7_df.to_parquet(hl7_path, index=False)
        dicom_df.to_parquet(dicom_path, index=False)
        
        metadata = {
            "n_patients": n_patients,
            "seed": self.config.simulation.seed,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "generation_timestamp": datetime.now().isoformat(),
            "config_summary": "Auto-generated metadata" # Can be expanded
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Generation complete.")
        
        return {
            "patients": patients_path,
            "hl7_messages": hl7_path,
            "dicom_studies": dicom_path,
            "metadata": metadata_path
        }

    def _generate_encounters(self, n_patients: int, show_progress: bool) -> tuple[list, list, list]:
        arrival_times = self._generate_arrival_times(n_patients)

        patients_list: list = []
        hl7_list: list = []
        dicom_list: list = []

        iterator = range(n_patients)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating encounters")

        # Pre-compute priority probabilities from config
        emg_r = self.config.arrivals.emergency_ratio
        urg_r = self.config.arrivals.urgent_ratio
        rou_r = max(1.0 - emg_r - urg_r, 0.0)
        priority_list = [Priority.EMERGENCY, Priority.URGENT, Priority.ROUTINE]
        priority_probs = np.array([emg_r, urg_r, rou_r], dtype=float)
        priority_probs /= priority_probs.sum()

        for i in iterator:
            patient_id = f"PT{i:08d}"
            mrn = f"MRN{i:010d}"
            visit_number = str(uuid.uuid4())

            # Sample priority by index (avoids numpy.str_ conversion)
            priority_idx = int(self.rng.choice(len(priority_list), p=priority_probs))
            priority = priority_list[priority_idx]

            # Sample modality by index
            modality_idx = int(self.rng.choice(len(self.modality_types), p=self.modality_probs))
            modality = self.modality_types[modality_idx]

            parts, part_probs = self.body_parts[modality]
            body_part = str(self.rng.choice(parts, p=part_probs))

            age = int(self.rng.integers(1, 95))
            sex = str(self.rng.choice(["M", "F", "O"], p=[0.49, 0.49, 0.02]))

            referring_department = str(self.rng.choice(self.referring_departments))
            clinical_indication = str(self.rng.choice(self.clinical_indications[modality]))

            arrival_dt = self.start_date + timedelta(minutes=float(arrival_times[i]))

            registration_time = arrival_times[i] + self.rng.exponential(2.0)
            queue_wait = self._sample_queue_wait(priority, modality)
            scan_duration = self._sample_scan_duration(modality)

            setup_mean = self.config.modalities.setup_time.mean
            setup_sigma = self.config.modalities.setup_time.sigma
            setup_time = float(self.rng.normal(setup_mean, setup_sigma))

            scan_start = registration_time + queue_wait + max(setup_time, 0.0)
            scan_end = scan_start + scan_duration

            report_wait = float(self.rng.exponential(10.0))
            report_duration = self._sample_report_duration(modality, priority)

            report_start = scan_end + report_wait
            report_end = report_start + report_duration
            discharge_time = report_end + float(self.rng.exponential(5.0))

            patient = Patient(
                patient_id=patient_id,
                mrn=mrn,
                visit_number=visit_number,
                age=age,
                sex=sex,
                priority=priority,
                modality=modality,
                body_part=body_part,
                referring_department=referring_department,
                clinical_indication=clinical_indication,
                arrival_time=float(arrival_times[i]),
                registration_time=float(registration_time),
                scan_start_time=float(scan_start),
                scan_end_time=float(scan_end),
                report_start_time=float(report_start),
                report_end_time=float(report_end),
                discharge_time=float(discharge_time),
                status=PatientStatus.DISCHARGED,
            )

            scanner_suffix = int(self.rng.integers(1, 5))
            patient.assigned_scanner_id = f"{modality.value}_SCANNER_0{scanner_suffix}"
            patient.assigned_radiologist_id = f"RAD{int(self.rng.integers(1, 8)):03d}"

            # Generate DICOM study with correct signature
            study_dt = arrival_dt + timedelta(minutes=float(scan_start - arrival_times[i]))
            accession = self.dicom_gen._generate_accession(patient_id, study_dt)
            dicom_study = self.dicom_gen.generate_study(
                patient_id=patient_id,
                modality=modality,
                body_part=body_part,
                study_datetime=study_dt,
                accession_number=accession,
            )
            patient.dicom_study = dicom_study

            # Generate HL7 messages using float sim_times (minutes from epoch)
            adt_a01 = self.hl7_gen.generate_adt_a01(patient, sim_time=float(registration_time))
            orm_o01 = self.hl7_gen.generate_orm_o01(
                patient,
                scanner_id=patient.assigned_scanner_id,
                sim_time=float(registration_time + self.rng.uniform(1, 5)),
            )
            siu_s12 = self.hl7_gen.generate_siu_s12(
                patient,
                scanner_id=patient.assigned_scanner_id,
                scheduled_time=float(scan_start),
                sim_time=float(registration_time + self.rng.uniform(2, 8)),
            )
            report_text = f"Radiological report for patient {patient_id}, {modality.value} {body_part}. Findings within normal limits."
            oru_r01 = self.hl7_gen.generate_oru_r01(
                patient,
                dicom_study=dicom_study,
                report_text=report_text,
                radiologist_id=patient.assigned_radiologist_id,
                sim_time=float(report_end),
            )
            adt_a03 = self.hl7_gen.generate_adt_a03(patient, sim_time=float(discharge_time))

            messages = [adt_a01, orm_o01, siu_s12, oru_r01, adt_a03]
            patient.hl7_messages = messages

            patients_list.append(patient)
            hl7_list.extend(messages)
            dicom_list.append(dicom_study)

        return patients_list, hl7_list, dicom_list

    def _generate_arrival_times(self, n_patients: int) -> np.ndarray:
        # Simplistic Poisson approach
        total_time_minutes = self.total_days * 24 * 60
        rate = n_patients / total_time_minutes
        
        inter_arrivals = self.rng.exponential(1/rate, size=int(n_patients * 1.5))
        arrival_times = np.cumsum(inter_arrivals)
        arrival_times = arrival_times[arrival_times <= total_time_minutes]
        
        if len(arrival_times) < n_patients:
            extra = n_patients - len(arrival_times)
            more_inter = self.rng.exponential(1/rate, size=int(extra * 1.5))
            more_arrivals = arrival_times[-1] + np.cumsum(more_inter) if len(arrival_times) > 0 else np.cumsum(more_inter)
            arrival_times = np.concatenate([arrival_times, more_arrivals])
            
        return np.sort(arrival_times[:n_patients])

    def _sample_queue_wait(self, priority: Priority, modality: Modality) -> float:
        if priority == Priority.EMERGENCY:
            wait = self.rng.lognormal(np.log(5), 0.4)
            wait = np.clip(wait, 1, 30)
        elif priority == Priority.URGENT:
            wait = self.rng.lognormal(np.log(20), 0.5)
            wait = np.clip(wait, 5, 120)
        else:
            wait = self.rng.lognormal(np.log(45), 0.6)
            wait = np.clip(wait, 10, 300)
            
        if modality == Modality.CT:
            wait *= 1.2
            
        return float(wait)

    def _sample_scan_duration(self, modality: Modality) -> float:
        mean = self.config.modalities.scan_duration[modality.value].mean
        sigma = self.config.modalities.scan_duration[modality.value].sigma
        
        duration = self.rng.lognormal(np.log(mean), sigma)
        duration = np.clip(duration, mean * 0.3, mean * 3)
        return float(duration)

    def _sample_report_duration(self, modality: Modality, priority: Priority) -> float:
        """Sample report duration from config parameters."""
        rad_cfg = self.config.radiologists.reporting_duration
        mod_key = modality.value
        if mod_key in rad_cfg:
            mean = rad_cfg[mod_key].mean
            sigma = rad_cfg[mod_key].sigma
        else:
            # Fallback defaults
            defaults = {Modality.CT: (20.0, 0.4), Modality.MRI: (30.0, 0.45), Modality.XRAY: (8.0, 0.3)}
            mean, sigma = defaults.get(modality, (15.0, 0.4))

        duration = float(self.rng.lognormal(np.log(mean), sigma))
        duration = float(np.clip(duration, mean * 0.2, mean * 5))

        if priority == Priority.EMERGENCY:
            speedup = self.config.radiologists.emergency_priority_speedup
            duration *= speedup

        return duration

    def _to_dataframes(self, patients: list, hl7_messages: list, dicom_studies: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        patients_df = pd.DataFrame([p.to_dict() for p in patients])
        
        hl7_records = []
        for msg in hl7_messages:
            rec = {
                'patient_id': msg.patient_id,
                'visit_number': msg.visit_number,
                'message_type': msg.message_type.value,
                'message_control_id': msg.message_control_id,
                'timestamp_minutes': float(msg.timestamp)  # Already in minutes
            }
            for k, v in msg.segments.items():
                rec[f'segment_{k}'] = v
            hl7_records.append(rec)
        hl7_df = pd.DataFrame(hl7_records)
        
        dicom_df = pd.DataFrame([d.to_dict() for d in dicom_studies if d is not None])
        
        return patients_df, hl7_df, dicom_df

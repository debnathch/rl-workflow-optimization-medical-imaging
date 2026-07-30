"""
Domain entities for the hospital imaging simulation.

All domain objects are immutable dataclasses or Pydantic models where possible,
with mutable state tracked in SimPy resources and explicit state containers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class Modality(str, Enum):
    """Medical imaging modality types."""

    CT = "CT"
    MRI = "MRI"
    XRAY = "XRAY"


class Priority(int, Enum):
    """Patient priority levels (higher value = higher priority)."""

    ROUTINE = 0
    URGENT = 1
    EMERGENCY = 2


class PatientStatus(str, Enum):
    """Patient encounter lifecycle states."""

    REGISTERED = "REGISTERED"
    WAITING_SCAN = "WAITING_SCAN"
    IN_SCAN = "IN_SCAN"
    WAITING_REPORT = "WAITING_REPORT"
    REPORTED = "REPORTED"
    DISCHARGED = "DISCHARGED"


class ScannerStatus(str, Enum):
    """Scanner operational states."""

    IDLE = "IDLE"
    BUSY = "BUSY"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"


class HL7MessageType(str, Enum):
    """HL7 v2 message type codes used in the simulation."""

    ADT_A01 = "ADT^A01"   # Admit
    ADT_A03 = "ADT^A03"   # Discharge
    ADT_A08 = "ADT^A08"   # Update patient info
    ORM_O01 = "ORM^O01"   # Imaging order
    ORU_R01 = "ORU^R01"   # Result / report
    SIU_S12 = "SIU^S12"   # Schedule new appointment
    SIU_S15 = "SIU^S15"   # Cancel appointment


class RadiologistSpecialty(str, Enum):
    """Radiologist subspecialties for workload routing."""

    CHEST = "chest"
    MUSCULOSKELETAL = "musculoskeletal"
    NEURORADIOLOGY = "neuroradiology"
    ABDOMINAL = "abdominal"
    EMERGENCY = "emergency"


# ─────────────────────────────────────────────────────────────────────────────
# HL7 Message
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HL7Message:
    """Represents a synthetic HL7 v2 message."""

    message_type: HL7MessageType
    message_control_id: str
    sending_application: str
    receiving_application: str
    timestamp: float          # SimPy simulation time (minutes)
    patient_id: str
    visit_number: str
    segments: dict[str, Any]  # Segment name → field dict

    @classmethod
    def create(
        cls,
        message_type: HL7MessageType,
        timestamp: float,
        patient_id: str,
        visit_number: str,
        segments: dict[str, Any],
        sending_app: str = "RADIOLOGY_HIS",
        receiving_app: str = "EHR_SYSTEM",
    ) -> "HL7Message":
        return cls(
            message_type=message_type,
            message_control_id=str(uuid.uuid4()).replace("-", "")[:20].upper(),
            sending_application=sending_app,
            receiving_application=receiving_app,
            timestamp=timestamp,
            patient_id=patient_id,
            visit_number=visit_number,
            segments=segments,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "message_control_id": self.message_control_id,
            "sending_application": self.sending_application,
            "receiving_application": self.receiving_application,
            "timestamp_minutes": self.timestamp,
            "patient_id": self.patient_id,
            "visit_number": self.visit_number,
            "segments": self.segments,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DICOM Study Metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DICOMStudy:
    """Synthetic DICOM study metadata (no pixel data)."""

    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: Modality
    body_part: str
    study_description: str
    series_description: str
    protocol_name: str
    manufacturer: str
    manufacturer_model: str
    station_name: str
    study_date: str           # YYYYMMDD
    study_time: str           # HHMMSS
    patient_id: str
    accession_number: str
    referring_physician: str
    # Modality-specific acquisition parameters
    acquisition_params: dict[str, Any]
    number_of_series: int
    number_of_images: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "sop_instance_uid": self.sop_instance_uid,
            "modality": self.modality.value,
            "body_part": self.body_part,
            "study_description": self.study_description,
            "series_description": self.series_description,
            "protocol_name": self.protocol_name,
            "manufacturer": self.manufacturer,
            "manufacturer_model": self.manufacturer_model,
            "station_name": self.station_name,
            "study_date": self.study_date,
            "study_time": self.study_time,
            "patient_id": self.patient_id,
            "accession_number": self.accession_number,
            "referring_physician": self.referring_physician,
            "acquisition_params": self.acquisition_params,
            "number_of_series": self.number_of_series,
            "number_of_images": self.number_of_images,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Patient:
    """Represents a patient encounter in the simulation."""

    patient_id: str
    mrn: str                        # Medical record number
    visit_number: str               # Encounter ID
    arrival_time: float             # SimPy time (minutes)
    priority: Priority
    modality: Modality
    body_part: str
    age: int
    sex: str                        # "M" / "F" / "O"
    referring_department: str
    clinical_indication: str

    # Mutable state (updated throughout encounter)
    status: PatientStatus = PatientStatus.REGISTERED
    hl7_messages: list[HL7Message] = field(default_factory=list)
    dicom_study: DICOMStudy | None = None
    assigned_scanner_id: str | None = None
    assigned_radiologist_id: str | None = None

    # Timestamps (minutes since simulation start)
    registration_time: float | None = None
    queue_entry_time: float | None = None
    scan_start_time: float | None = None
    scan_end_time: float | None = None
    report_start_time: float | None = None
    report_end_time: float | None = None
    discharge_time: float | None = None

    @property
    def wait_time(self) -> float | None:
        """Time from arrival to scan start."""
        if self.scan_start_time is not None:
            return self.scan_start_time - self.arrival_time
        return None

    @property
    def scan_duration(self) -> float | None:
        """Actual scan duration."""
        if self.scan_start_time is not None and self.scan_end_time is not None:
            return self.scan_end_time - self.scan_start_time
        return None

    @property
    def report_duration(self) -> float | None:
        """Radiologist reporting duration."""
        if self.report_start_time is not None and self.report_end_time is not None:
            return self.report_end_time - self.report_start_time
        return None

    @property
    def total_turnaround(self) -> float | None:
        """Total time from arrival to report complete."""
        if self.report_end_time is not None:
            return self.report_end_time - self.arrival_time
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "mrn": self.mrn,
            "visit_number": self.visit_number,
            "arrival_time": self.arrival_time,
            "priority": self.priority.name,
            "priority_value": self.priority.value,
            "modality": self.modality.value,
            "body_part": self.body_part,
            "age": self.age,
            "sex": self.sex,
            "referring_department": self.referring_department,
            "clinical_indication": self.clinical_indication,
            "status": self.status.value,
            "assigned_scanner_id": self.assigned_scanner_id,
            "assigned_radiologist_id": self.assigned_radiologist_id,
            "registration_time": self.registration_time,
            "queue_entry_time": self.queue_entry_time,
            "scan_start_time": self.scan_start_time,
            "scan_end_time": self.scan_end_time,
            "report_start_time": self.report_start_time,
            "report_end_time": self.report_end_time,
            "discharge_time": self.discharge_time,
            "wait_time": self.wait_time,
            "scan_duration": self.scan_duration,
            "report_duration": self.report_duration,
            "total_turnaround": self.total_turnaround,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scanner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScannerSpec:
    """Static specification of a scanner (immutable after creation)."""

    scanner_id: str
    modality: Modality
    manufacturer: str
    model_name: str
    station_name: str
    location: str               # e.g. "Radiology Suite A"

    # Track utilization over time as (start, end) tuples
    busy_intervals: list[tuple[float, float]] = field(default_factory=list)
    maintenance_intervals: list[tuple[float, float]] = field(default_factory=list)

    status: ScannerStatus = ScannerStatus.IDLE
    current_patient_id: str | None = None
    total_studies: int = 0

    def utilization(self, total_available_time: float) -> float:
        """Compute utilization fraction over the simulation window."""
        if total_available_time <= 0:
            return 0.0
        busy_total = sum(end - start for start, end in self.busy_intervals)
        return min(busy_total / total_available_time, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner_id": self.scanner_id,
            "modality": self.modality.value,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "station_name": self.station_name,
            "location": self.location,
            "total_studies": self.total_studies,
            "status": self.status.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Radiologist
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RadiologistSpec:
    """Radiologist definition with specialty and schedule."""

    radiologist_id: str
    specialty: RadiologistSpecialty
    shift_start_hour: int   # 0-23
    shift_end_hour: int     # 1-24
    max_daily_reads: int

    # Runtime tracking
    total_reads: int = 0
    total_report_minutes: float = 0.0
    queue: list[str] = field(default_factory=list)   # patient_ids
    busy_intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_on_shift(self) -> bool:
        """Check if radiologist is currently on shift (hour-of-day based)."""
        # This is checked against simulation clock — done in hospital.py
        return True  # placeholder; actual check uses sim.now

    @property
    def workload_score(self) -> float:
        """Fraction of max capacity used today."""
        return self.total_reads / max(self.max_daily_reads, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "radiologist_id": self.radiologist_id,
            "specialty": self.specialty.value,
            "shift_start_hour": self.shift_start_hour,
            "shift_end_hour": self.shift_end_hour,
            "max_daily_reads": self.max_daily_reads,
            "total_reads": self.total_reads,
            "total_report_minutes": self.total_report_minutes,
            "workload_score": self.workload_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation State Snapshot (used for RL observations)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationSnapshot:
    """Point-in-time snapshot of the simulation state for RL observation."""

    sim_time: float
    ct_queue_length: int
    mri_queue_length: int
    xray_queue_length: int
    emergency_queue_length: int
    ct_utilization: float
    mri_utilization: float
    xray_utilization: float
    radiologist_workloads: list[float]   # One per radiologist
    hour_of_day: float                   # Normalized 0-1
    day_of_week: float                   # Normalized 0-1
    priority_counts: list[int]           # [routine, urgent, emergency] counts in queue
    completed_last_epoch: int
    avg_wait_last_epoch: float

    def to_observation(self) -> list[float]:
        """Flatten to observation vector for Gymnasium."""
        obs = [
            float(self.ct_queue_length),
            float(self.mri_queue_length),
            float(self.xray_queue_length),
            float(self.emergency_queue_length),
            self.ct_utilization,
            self.mri_utilization,
            self.xray_utilization,
            *self.radiologist_workloads,
            self.hour_of_day,
            self.day_of_week,
            float(self.priority_counts[0]),
            float(self.priority_counts[1]),
            float(self.priority_counts[2]),
            float(self.completed_last_epoch),
            self.avg_wait_last_epoch,
        ]
        return obs

"""Unit tests for domain entities."""

from __future__ import annotations

import pytest
from medimg_twin.simulation.entities import (
    HL7Message,
    HL7MessageType,
    Modality,
    Patient,
    PatientStatus,
    Priority,
    RadiologistSpec,
    RadiologistSpecialty,
    ScannerSpec,
    ScannerStatus,
    SimulationSnapshot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_patient() -> Patient:
    return Patient(
        patient_id="PT00000001",
        mrn="MRN0000000001",
        visit_number="VN000000000001",
        arrival_time=10.0,
        priority=Priority.URGENT,
        modality=Modality.CT,
        body_part="CHEST",
        age=50,
        sex="M",
        referring_department="ED",
        clinical_indication="Chest pain",
    )


@pytest.fixture
def sample_scanner() -> ScannerSpec:
    return ScannerSpec(
        scanner_id="CT_SCANNER_01",
        modality=Modality.CT,
        manufacturer="GE Healthcare",
        model_name="Revolution CT",
        station_name="CT_SCANNER_01",
        location="Radiology Suite A Room 1",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Priority enum
# ─────────────────────────────────────────────────────────────────────────────


def test_priority_enum_values() -> None:
    """Test that priority enum has correct integer values."""
    assert Priority.ROUTINE.value == 0
    assert Priority.URGENT.value == 1
    assert Priority.EMERGENCY.value == 2


def test_priority_ordering() -> None:
    """EMERGENCY > URGENT > ROUTINE by numeric value."""
    assert Priority.EMERGENCY.value > Priority.URGENT.value
    assert Priority.URGENT.value > Priority.ROUTINE.value


def test_priority_name_access() -> None:
    """Priority names are accessible."""
    assert Priority.EMERGENCY.name == "EMERGENCY"
    assert Priority.ROUTINE.name == "ROUTINE"


# ─────────────────────────────────────────────────────────────────────────────
# Modality enum
# ─────────────────────────────────────────────────────────────────────────────


def test_modality_ct_value() -> None:
    assert Modality.CT.value == "CT"


def test_modality_mri_value() -> None:
    assert Modality.MRI.value == "MRI"


def test_modality_xray_value() -> None:
    assert Modality.XRAY.value == "XRAY"


def test_modality_all_three_exist() -> None:
    assert len(Modality) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────


def test_patient_creation(sample_patient: Patient) -> None:
    """Patient dataclass stores all fields correctly."""
    assert sample_patient.patient_id == "PT00000001"
    assert sample_patient.priority == Priority.URGENT
    assert sample_patient.modality == Modality.CT
    assert sample_patient.arrival_time == 10.0
    assert sample_patient.status == PatientStatus.REGISTERED


def test_wait_time_none_before_scan(sample_patient: Patient) -> None:
    """wait_time is None when scan_start_time is not set."""
    assert sample_patient.scan_start_time is None
    assert sample_patient.wait_time is None


def test_wait_time_computed(sample_patient: Patient) -> None:
    """wait_time = scan_start_time - arrival_time."""
    sample_patient.scan_start_time = 35.0
    assert sample_patient.wait_time == pytest.approx(25.0)


def test_scan_duration_computed(sample_patient: Patient) -> None:
    """scan_duration = scan_end_time - scan_start_time."""
    sample_patient.scan_start_time = 35.0
    sample_patient.scan_end_time = 60.0
    assert sample_patient.scan_duration == pytest.approx(25.0)


def test_scan_duration_none_when_incomplete(sample_patient: Patient) -> None:
    """scan_duration is None if scan_end_time is not set."""
    sample_patient.scan_start_time = 35.0
    assert sample_patient.scan_duration is None


def test_total_turnaround(sample_patient: Patient) -> None:
    """total_turnaround = report_end_time - arrival_time."""
    sample_patient.report_end_time = 110.0
    assert sample_patient.total_turnaround == pytest.approx(100.0)


def test_total_turnaround_none_when_not_reported(sample_patient: Patient) -> None:
    """total_turnaround is None before report completion."""
    assert sample_patient.total_turnaround is None


def test_to_dict_contains_expected_keys(sample_patient: Patient) -> None:
    """to_dict() returns all required keys."""
    d = sample_patient.to_dict()
    expected_keys = {
        "patient_id", "mrn", "visit_number", "arrival_time",
        "priority", "modality", "status", "wait_time", "scan_duration",
        "body_part", "age", "sex",
    }
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"


def test_to_dict_priority_is_name(sample_patient: Patient) -> None:
    """to_dict() returns priority as name string."""
    d = sample_patient.to_dict()
    assert d["priority"] == "URGENT"


def test_report_duration_computed(sample_patient: Patient) -> None:
    """report_duration = report_end_time - report_start_time."""
    sample_patient.report_start_time = 80.0
    sample_patient.report_end_time = 100.0
    assert sample_patient.report_duration == pytest.approx(20.0)


# ─────────────────────────────────────────────────────────────────────────────
# ScannerSpec
# ─────────────────────────────────────────────────────────────────────────────


def test_utilization_zero_with_no_intervals(sample_scanner: ScannerSpec) -> None:
    """Utilization is 0.0 when no busy intervals recorded."""
    assert sample_scanner.utilization(total_available_time=100.0) == pytest.approx(0.0)


def test_utilization_computed_correctly(sample_scanner: ScannerSpec) -> None:
    """Utilization = busy_time / total_time."""
    sample_scanner.busy_intervals.append((0.0, 60.0))
    assert sample_scanner.utilization(total_available_time=120.0) == pytest.approx(0.5)


def test_utilization_capped_at_1(sample_scanner: ScannerSpec) -> None:
    """Utilization is capped at 1.0 even if busy time exceeds total time."""
    sample_scanner.busy_intervals.append((0.0, 150.0))
    assert sample_scanner.utilization(total_available_time=100.0) == pytest.approx(1.0)


def test_utilization_zero_when_total_time_zero(sample_scanner: ScannerSpec) -> None:
    """Utilization returns 0.0 when total_available_time is 0."""
    assert sample_scanner.utilization(total_available_time=0.0) == pytest.approx(0.0)


def test_scanner_to_dict_keys(sample_scanner: ScannerSpec) -> None:
    """ScannerSpec.to_dict() contains required keys."""
    d = sample_scanner.to_dict()
    for key in ["scanner_id", "modality", "manufacturer", "total_studies", "status"]:
        assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# SimulationSnapshot
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot(
    ct_q: int = 2,
    mri_q: int = 1,
    xray_q: int = 3,
    emg_q: int = 0,
) -> SimulationSnapshot:
    return SimulationSnapshot(
        sim_time=60.0,
        ct_queue_length=ct_q,
        mri_queue_length=mri_q,
        xray_queue_length=xray_q,
        emergency_queue_length=emg_q,
        ct_utilization=0.7,
        mri_utilization=0.6,
        xray_utilization=0.8,
        radiologist_workloads=[0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.2],
        hour_of_day=0.5,
        day_of_week=0.3,
        priority_counts=[5, 2, 1],
        completed_last_epoch=3,
        avg_wait_last_epoch=25.0,
    )


def test_to_observation_length() -> None:
    """to_observation() produces a vector of length 21."""
    snap = _make_snapshot()
    obs = snap.to_observation()
    assert len(obs) == 21, f"Expected 21 features, got {len(obs)}"


def test_to_observation_all_numeric() -> None:
    """All observation values are numeric."""
    obs = _make_snapshot().to_observation()
    for i, val in enumerate(obs):
        assert isinstance(val, (int, float)), f"Non-numeric at index {i}: {val}"


def test_to_observation_queue_values_positive() -> None:
    """Queue counts are non-negative in observation."""
    obs = _make_snapshot(ct_q=5, mri_q=3, xray_q=7).to_observation()
    assert obs[0] >= 0  # CT queue
    assert obs[1] >= 0  # MRI queue
    assert obs[2] >= 0  # XRAY queue

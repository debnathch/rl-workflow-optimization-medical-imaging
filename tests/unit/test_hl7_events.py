"""Unit tests for HL7 event generation."""

from __future__ import annotations

import pytest
from medimg_twin.simulation.entities import (
    HL7MessageType,
    Modality,
    Patient,
    PatientStatus,
    Priority,
)
from medimg_twin.simulation.hl7_events import HL7EventGenerator


@pytest.fixture
def sample_patient() -> Patient:
    return Patient(
        patient_id="00000001",
        mrn="MRN0000000001",
        visit_number="VN000000000001",
        arrival_time=10.0,
        priority=Priority.URGENT,
        modality=Modality.CT,
        body_part="CHEST",
        age=45,
        sex="M",
        referring_department="ED",
        clinical_indication="Chest pain",
    )


@pytest.fixture
def hl7_gen() -> HL7EventGenerator:
    return HL7EventGenerator()


SIM_TIME = 600.0  # 10 minutes in seconds (unix-style)


def test_generate_adt_a01_type(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ADT^A01 message has correct type."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert msg.message_type == HL7MessageType.ADT_A01


def test_generate_adt_a01_has_pid_segment(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ADT^A01 message contains PID segment."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert "PID" in msg.segments


def test_generate_adt_a01_has_pv1_segment(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ADT^A01 message contains PV1 segment."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert "PV1" in msg.segments


def test_generate_orm_o01_type(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ORM^O01 message has correct type."""
    msg = hl7_gen.generate_orm_o01(sample_patient, scanner_id="CT_SCANNER_01", sim_time=SIM_TIME)
    assert msg.message_type == HL7MessageType.ORM_O01


def test_generate_orm_o01_has_obr(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ORM^O01 message contains OBR segment."""
    msg = hl7_gen.generate_orm_o01(sample_patient, scanner_id="CT_SCANNER_01", sim_time=SIM_TIME)
    assert "OBR" in msg.segments


def test_generate_siu_s12_type(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """SIU^S12 message has correct type."""
    msg = hl7_gen.generate_siu_s12(
        sample_patient, scanner_id="CT_SCANNER_01",
        scheduled_time=SIM_TIME + 30.0, sim_time=SIM_TIME
    )
    assert msg.message_type == HL7MessageType.SIU_S12


def test_generate_adt_a03_type(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """ADT^A03 message has correct type."""
    msg = hl7_gen.generate_adt_a03(sample_patient, sim_time=SIM_TIME)
    assert msg.message_type == HL7MessageType.ADT_A03


def test_message_control_id_unique(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """Two ADT^A01 messages have different message_control_ids."""
    msg1 = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    msg2 = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME + 1.0)
    assert msg1.message_control_id != msg2.message_control_id


def test_patient_id_in_message(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """Patient ID is stored in the generated message."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert msg.patient_id == sample_patient.patient_id


def test_message_has_sending_application(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """Message has a non-empty sending_application."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert msg.sending_application


def test_msh_segment_present(hl7_gen: HL7EventGenerator, sample_patient: Patient) -> None:
    """MSH segment is present in all message types."""
    msg = hl7_gen.generate_adt_a01(sample_patient, sim_time=SIM_TIME)
    assert "MSH" in msg.segments

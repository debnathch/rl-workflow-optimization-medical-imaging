from __future__ import annotations
import uuid
import logging
import datetime
from typing import Any

from medimg_twin.simulation.entities import (
    HL7Message, HL7MessageType, Patient, Modality, DICOMStudy
)

logger = logging.getLogger(__name__)

# Module-level constants
LOINC_CODES = {
    (Modality.CT, "chest"): "36643-5^CT chest^LN",
    (Modality.CT, "head"): "36636-9^CT head^LN",
    (Modality.MRI, "brain"): "36622-9^MRI brain^LN",
    (Modality.MRI, "spine"): "36632-8^MRI spine^LN",
    (Modality.XRAY, "chest"): "36643-5^Xray chest^LN",
    (Modality.XRAY, "extremity"): "36643-5^Xray extremity^LN",
}

REPORT_TEMPLATES = {
    Modality.CT: [
        "CT {body_part} protocol. Patient ID: {patient_id}. Findings: Normal study. No acute abnormality.",
        "CT {body_part} protocol. Patient ID: {patient_id}. Findings: Mild degenerative changes noted.",
        "CT {body_part} protocol. Patient ID: {patient_id}. Findings: Questionable opacity, correlate clinically.",
        "CT {body_part} protocol. Patient ID: {patient_id}. Findings: Chronic changes without acute process.",
        "CT {body_part} protocol. Patient ID: {patient_id}. Findings: Incidental finding of small nodule, recommend follow-up."
    ],
    Modality.MRI: [
        "MRI {body_part} protocol. Patient ID: {patient_id}. Findings: Unremarkable study.",
        "MRI {body_part} protocol. Patient ID: {patient_id}. Findings: No evidence of acute ischemia or hemorrhage.",
        "MRI {body_part} protocol. Patient ID: {patient_id}. Findings: Mild to moderate osteoarthritis.",
        "MRI {body_part} protocol. Patient ID: {patient_id}. Findings: Stable appearance compared to prior exam.",
        "MRI {body_part} protocol. Patient ID: {patient_id}. Findings: Motion artifact limits evaluation, otherwise normal."
    ],
    Modality.XRAY: [
        "X-ray {body_part}. Patient ID: {patient_id}. Findings: No fracture or dislocation.",
        "X-ray {body_part}. Patient ID: {patient_id}. Findings: Clear lungs, normal heart size.",
        "X-ray {body_part}. Patient ID: {patient_id}. Findings: Mild degenerative joint disease.",
        "X-ray {body_part}. Patient ID: {patient_id}. Findings: No acute cardiopulmonary process.",
        "X-ray {body_part}. Patient ID: {patient_id}. Findings: Old healed fracture noted, no acute changes."
    ]
}

REFERRING_PHYSICIANS = [
    "Smith^John^A", "Doe^Jane^M", "Johnson^Robert^L", "Williams^Mary^K",
    "Brown^James^T", "Jones^Patricia^R", "Garcia^Michael^E", "Miller^Linda^S",
    "Davis^William^B", "Rodriguez^Barbara^C", "Martinez^David^F", "Hernandez^Susan^G",
    "Lopez^Joseph^H", "Gonzalez^Jessica^J", "Wilson^Charles^W", "Anderson^Sarah^N",
    "Thomas^Thomas^P", "Taylor^Karen^D", "Moore^Christopher^V", "Jackson^Nancy^Z"
]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph",
    "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa",
    "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret", "Mark",
    "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily",
    "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Carol", "Kevin",
    "Amanda", "Brian", "Melissa", "George", "Deborah", "Timothy", "Stephanie"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts"
]

class HL7EventGenerator:
    """Generates synthetic HL7 v2 messages for a hospital imaging workflow simulation."""

    def __init__(self, facility_name: str = "CITY_GENERAL_HOSPITAL", sending_app: str = "RIS_SYSTEM") -> None:
        """
        Initialize the HL7 generator with facility and application configurations.

        Args:
            facility_name: Name of the sending facility.
            sending_app: Name of the sending application.
        """
        self.facility_name = facility_name
        self.sending_app = sending_app
        logger.info(f"Initialized HL7EventGenerator with facility={facility_name}, app={sending_app}")

    def _generate_fake_name(self, patient_id: str) -> str:
        """
        Deterministically generate Last^First name from patient_id.
        """
        try:
            seed = int(patient_id)
        except ValueError:
            seed = hash(patient_id)
        last_idx = seed % len(LAST_NAMES)
        first_idx = (seed // len(LAST_NAMES)) % len(FIRST_NAMES)
        return f"{LAST_NAMES[last_idx]}^{FIRST_NAMES[first_idx]}"

    def _get_service_id(self, modality: Modality, body_part: str) -> str:
        """
        Return a LOINC-like code string for the given modality and body part.
        """
        key = (modality, body_part.lower())
        return LOINC_CODES.get(key, f"99999-9^{modality.name} {body_part}^LN")

    def _generate_pid_segment(self, patient: Patient) -> dict[str, Any]:
        """
        Private helper to build PID segment from patient object.
        """
        name = self._generate_fake_name(patient.patient_id)
        return {
            "patient_id": patient.patient_id,
            "mrn": patient.patient_id,
            "patient_name": name,
            "dob": patient.dob.strftime("%Y%m%d") if hasattr(patient, 'dob') else "19800101",
            "sex": getattr(patient, 'sex', "U"),
            "address": getattr(patient, 'address', "123 Main St^^City^ST^12345"),
            "phone": getattr(patient, 'phone', "555-555-5555")
        }

    def _generate_msh_segment(self, message_type: str, sim_time: float) -> dict[str, Any]:
        """Generate common MSH segment."""
        dt = datetime.datetime.fromtimestamp(sim_time)
        return {
            "field_separator": "|",
            "encoding_chars": "^~\\&",
            "sending_app": self.sending_app,
            "receiving_app": "EHR_SYSTEM",
            "datetime": dt.strftime("%Y%m%d%H%M%S"),
            "message_type": message_type,
            "control_id": str(uuid.uuid4()).replace("-", ""),
            "processing_id": "P",
            "version": "2.5.1"
        }

    def generate_adt_a01(self, patient: Patient, sim_time: float) -> HL7Message:
        """
        Generate Admission (ADT^A01) message.
        """
        msh = self._generate_msh_segment("ADT^A01", sim_time)
        pid = self._generate_pid_segment(patient)
        dt = datetime.datetime.fromtimestamp(sim_time).strftime("%Y%m%d%H%M%S")
        
        try:
            seed = int(patient.patient_id)
        except ValueError:
            seed = hash(patient.patient_id)
            
        pv1 = {
            "patient_class": "I",
            "location": "WARD1^ROOM2^BED3",
            "admission_type": "E",
            "attending_doctor": REFERRING_PHYSICIANS[seed % len(REFERRING_PHYSICIANS)],
            "visit_number": f"V{patient.patient_id}",
            "admit_datetime": dt
        }
        
        evn = {
            "event_type_code": "A01",
            "datetime": dt
        }
        
        segments = {"MSH": msh, "PID": pid, "PV1": pv1, "EVN": evn}
        return HL7Message.create(
            message_type=HL7MessageType.ADT_A01,
            timestamp=sim_time,
            patient_id=patient.patient_id,
            visit_number=patient.visit_number,
            segments=segments,
            sending_app=self.sending_app,
        )

    def generate_orm_o01(self, patient: Patient, scanner_id: str, sim_time: float) -> HL7Message:
        """
        Generate Imaging Order (ORM^O01) message.
        """
        msh = self._generate_msh_segment("ORM^O01", sim_time)
        pid = self._generate_pid_segment(patient)
        dt = datetime.datetime.fromtimestamp(sim_time).strftime("%Y%m%d%H%M%S")
        
        placer_order = f"ORD{int(sim_time)}"
        filler_order = f"FIL{int(sim_time)}"
        
        try:
            seed = int(patient.patient_id)
        except ValueError:
            seed = hash(patient.patient_id)
        doctor = REFERRING_PHYSICIANS[seed % len(REFERRING_PHYSICIANS)]
        
        orc = {
            "order_control": "NW",
            "placer_order_number": placer_order,
            "filler_order_number": filler_order,
            "order_status": "IP",
            "order_datetime": dt,
            "ordering_provider": doctor
        }
        
        obr = {
            "set_id": "1",
            "placer_order_number": placer_order,
            "filler_order_number": filler_order,
            "universal_service_id": self._get_service_id(Modality.CT, "chest"), 
            "priority": "R",
            "requested_datetime": dt,
            "observation_datetime": dt,
            "ordering_provider": doctor,
            "reason_for_study": "Pain",
            "scheduled_datetime": dt,
            "result_status": "O"
        }
        
        segments = {"MSH": msh, "PID": pid, "ORC": orc, "OBR": obr}
        return HL7Message.create(
            message_type=HL7MessageType.ORM_O01,
            timestamp=sim_time,
            patient_id=patient.patient_id,
            visit_number=patient.visit_number,
            segments=segments,
            sending_app=self.sending_app,
        )

    def generate_siu_s12(self, patient: Patient, scanner_id: str, scheduled_time: float, sim_time: float) -> HL7Message:
        """
        Generate Schedule New Appointment (SIU^S12) message.
        """
        msh = self._generate_msh_segment("SIU^S12", sim_time)
        pid = self._generate_pid_segment(patient)
        
        sch = {
            "placer_appt_id": f"APT{int(scheduled_time)}",
            "filler_appt_id": f"FIL{int(scheduled_time)}",
            "appointment_reason": "Follow-up",
            "appointment_type": "NORMAL",
            "duration": "30",
            "duration_units": "MIN",
            "appointment_timing": datetime.datetime.fromtimestamp(scheduled_time).strftime("%Y%m%d%H%M%S"),
            "filler_status_code": "Booked"
        }
        
        ail = {
            "set_id": "1",
            "location_resource_id": scanner_id
        }
        
        aig = {
            "set_id": "1",
            "resource_type": "EQUIPMENT",
            "resource_id": scanner_id
        }
        
        segments = {"MSH": msh, "SCH": sch, "PID": pid, "AIL": ail, "AIG": aig}
        return HL7Message.create(
            message_type=HL7MessageType.SIU_S12,
            timestamp=sim_time,
            patient_id=patient.patient_id,
            visit_number=patient.visit_number,
            segments=segments,
            sending_app=self.sending_app,
        )

    def generate_oru_r01(self, patient: Patient, dicom_study: DICOMStudy, report_text: str, radiologist_id: str, sim_time: float) -> HL7Message:
        """
        Generate Result/Report (ORU^R01) message.
        """
        msh = self._generate_msh_segment("ORU^R01", sim_time)
        pid = self._generate_pid_segment(patient)
        dt = datetime.datetime.fromtimestamp(sim_time).strftime("%Y%m%d%H%M%S")
        
        try:
            seed = int(patient.patient_id)
        except ValueError:
            seed = hash(patient.patient_id)
        doctor = REFERRING_PHYSICIANS[seed % len(REFERRING_PHYSICIANS)]
        
        obr = {
            "set_id": "1",
            "placer_order_number": dicom_study.accession_number,
            "filler_order_number": f"FIL{dicom_study.accession_number}",
            "universal_service_id": self._get_service_id(dicom_study.modality, getattr(dicom_study, 'body_part', 'unknown')),
            "priority": "R",
            "requested_datetime": dt,
            "observation_datetime": dt,
            "ordering_provider": doctor,
            "reason_for_study": "Diagnosis",
            "scheduled_datetime": dt,
            "result_status": "F"
        }
        
        obx = {
            "set_id": "1",
            "value_type": "TX",
            "observation_id": "REPORT",
            "observation_value": report_text,
            "result_status": "F",
            "observation_datetime": dt
        }
        
        segments = {"MSH": msh, "PID": pid, "OBR": obr, "OBX": obx}
        return HL7Message.create(
            message_type=HL7MessageType.ORU_R01,
            timestamp=sim_time,
            patient_id=patient.patient_id,
            visit_number=patient.visit_number,
            segments=segments,
            sending_app=self.sending_app,
        )

    def generate_adt_a03(self, patient: Patient, sim_time: float) -> HL7Message:
        """
        Generate Discharge (ADT^A03) message.
        """
        msh = self._generate_msh_segment("ADT^A03", sim_time)
        pid = self._generate_pid_segment(patient)
        dt = datetime.datetime.fromtimestamp(sim_time).strftime("%Y%m%d%H%M%S")
        
        try:
            seed = int(patient.patient_id)
        except ValueError:
            seed = hash(patient.patient_id)
            
        pv1 = {
            "patient_class": "I",
            "location": "WARD1^ROOM2^BED3",
            "admission_type": "E",
            "attending_doctor": REFERRING_PHYSICIANS[seed % len(REFERRING_PHYSICIANS)],
            "visit_number": f"V{patient.patient_id}",
            "admit_datetime": dt,
            "discharge_datetime": dt
        }
        
        segments = {"MSH": msh, "PID": pid, "PV1": pv1}
        return HL7Message.create(
            message_type=HL7MessageType.ADT_A03,
            timestamp=sim_time,
            patient_id=patient.patient_id,
            visit_number=patient.visit_number,
            segments=segments,
            sending_app=self.sending_app,
        )

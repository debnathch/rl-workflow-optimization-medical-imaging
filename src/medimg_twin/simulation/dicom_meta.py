from __future__ import annotations
import uuid
import logging
import datetime
import random
from typing import Any
import numpy as np

from medimg_twin.simulation.entities import DICOMStudy, Modality

logger = logging.getLogger(__name__)

# Module-level constants
STUDY_DESCRIPTIONS = {
    Modality.CT: {
        "chest": "CT CHEST W/O CONTRAST",
        "head": "CT HEAD W/O CONTRAST",
        "abdomen": "CT ABDOMEN/PELVIS"
    },
    Modality.MRI: {
        "brain": "MRI BRAIN W/O CONTRAST",
        "spine": "MRI LUMBAR SPINE",
        "knee": "MRI KNEE W/O CONTRAST"
    },
    Modality.XRAY: {
        "chest": "XR CHEST 2 VIEWS",
        "extremity": "XR EXTREMITY",
        "abdomen": "XR ABDOMEN 1 VIEW"
    }
}

SERIES_DESCRIPTIONS = {
    Modality.CT: ["SCOUT", "AXIAL NON-CONTRAST", "CORONAL RECON", "SAGITTAL RECON", "THIN SLICE"],
    Modality.MRI: ["LOCALIZER", "T1 SAGITTAL", "T2 AXIAL", "TIRM CORONAL", "DWI", "ADC MAP"],
    Modality.XRAY: ["PA", "LATERAL", "AP", "OBLIQUE"]
}

RECONSTRUCTION_KERNELS = ['B30f', 'B50f', 'I30f', 'Br40']

MANUFACTURERS = {
    'GE Healthcare': {
        'CT': ['Revolution CT', 'Optima CT660'],
        'MRI': ['Signa Premier', 'Discovery MR750'],
        'XRAY': ['Optima XR220']
    },
    'Siemens Healthineers': {
        'CT': ['SOMATOM Force', 'SOMATOM Drive'],
        'MRI': ['MAGNETOM Vida', 'MAGNETOM Skyra'],
        'XRAY': ['Luminos Fusion', 'Ysio Max']
    },
    'Philips Healthcare': {
        'CT': ['IQon Elite', 'Incisive CT'],
        'MRI': ['Ingenia Ambition', 'Ingenia Elition'],
        'XRAY': ['DigitalDiagnost C90']
    },
    'Canon Medical': {
        'CT': ['Aquilion ONE', 'Aquilion Prime SP'],
        'MRI': ['Vantage Galan', 'Vantage Orian'],
        'XRAY': ['CXDI-710C']
    }
}

class DICOMMetadataGenerator:
    """Generates synthetic DICOM study metadata (no pixel data)."""

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the generator.

        Args:
            rng: NumPy random number generator for reproducible generation.
        """
        self.rng = rng
        logger.info("Initialized DICOMMetadataGenerator.")

    def _generate_uid(self, prefix: str = "1.2.840.10008") -> str:
        """Generate a valid DICOM UID string using numpy RNG for reproducibility."""
        # Generate 128-bit integer using rng for reproducibility
        hi = int(self.rng.integers(0, 2**63))
        lo = int(self.rng.integers(0, 2**63))
        suffix = str(hi * (2**63) + lo)
        uid = f"{prefix}.5.1.4.1.1.{suffix}"
        return uid[:64]

    def _generate_accession(self, patient_id: str, timestamp: datetime.datetime) -> str:
        """Return accession number like 'ACC20240115001234'."""
        date_str = timestamp.strftime("%Y%m%d")
        rand_num = self.rng.integers(1000, 9999)
        return f"ACC{date_str}{rand_num}"

    def _generate_ct_params(self, body_part: str) -> dict[str, Any]:
        """Generate CT specific parameters."""
        return {
            "kv": float(self.rng.choice([80, 100, 120, 140])),
            "mas": float(self.rng.uniform(100, 400)),
            "slice_thickness_mm": float(self.rng.uniform(0.625, 5.0)),
            "pitch": float(self.rng.uniform(0.5, 1.5)),
            "fov_mm": float(self.rng.uniform(200, 500)),
            "reconstruction_kernel": str(self.rng.choice(RECONSTRUCTION_KERNELS)),
            "window_center": int(self.rng.uniform(-50, 50)),
            "window_width": int(self.rng.uniform(100, 400))
        }

    def _generate_mri_params(self, body_part: str) -> dict[str, Any]:
        """Generate MRI specific parameters."""
        return {
            "tr_ms": float(self.rng.uniform(300, 5000)),
            "te_ms": float(self.rng.uniform(10, 120)),
            "flip_angle_deg": float(self.rng.uniform(5, 90)),
            "field_strength_T": float(self.rng.choice([1.5, 3.0])),
            "matrix_size": [int(self.rng.choice([256, 512])), int(self.rng.choice([256, 512]))],
            "fov_mm": float(self.rng.uniform(150, 400)),
            "slice_thickness_mm": float(self.rng.uniform(1.0, 5.0)),
            "number_of_excitations": int(self.rng.choice([1, 2, 3]))
        }

    def _generate_xray_params(self, body_part: str) -> dict[str, Any]:
        """Generate X-ray specific parameters."""
        return {
            "kv": float(self.rng.uniform(50, 125)),
            "mas": float(self.rng.uniform(2, 100)),
            "sid_mm": float(self.rng.uniform(1000, 1800)),
            "detector_size_mm": [int(self.rng.choice([350, 430])), int(self.rng.choice([350, 430]))],
            "grid_used": bool(self.rng.choice([True, False])),
            "aec_mode": str(self.rng.choice(["ON", "OFF"]))
        }

    def generate_study(self, patient_id: str, modality: Modality, body_part: str, study_datetime: datetime.datetime, accession_number: str) -> DICOMStudy:
        """
        Generate a realistic DICOMStudy object with metadata.
        """
        study_uid = self._generate_uid()
        
        mfrs = list(MANUFACTURERS.keys())
        mfr = str(self.rng.choice(mfrs))
        mod_key = modality.name
        models = MANUFACTURERS[mfr].get(mod_key, ["Generic Model"])
        model = str(self.rng.choice(models))
        
        if modality == Modality.CT:
            station = f"CT_SCANNER_0{self.rng.integers(1, 4)}"
            num_series = int(self.rng.integers(3, 6))
            num_images = int(self.rng.integers(150, 501))
            params = self._generate_ct_params(body_part)
            protocol = f"{body_part.upper()}_CT_WITH_CONTRAST"
        elif modality == Modality.MRI:
            station = f"MRI_SCANNER_0{self.rng.integers(1, 3)}"
            num_series = int(self.rng.integers(6, 13))
            num_images = int(self.rng.integers(200, 601))
            params = self._generate_mri_params(body_part)
            protocol = f"{body_part.upper()}_MRI_DIFFUSION_WB"
        else:
            station = f"XRAY_ROOM_0{self.rng.integers(1, 5)}"
            num_series = int(self.rng.integers(1, 3))
            num_images = int(self.rng.integers(1, 5))
            params = self._generate_xray_params(body_part)
            protocol = f"{body_part.upper()}_PA_LAT"

        desc_dict = STUDY_DESCRIPTIONS.get(modality, {})
        study_desc = desc_dict.get(body_part.lower(), f"{modality.name} {body_part.upper()}")
        series_list = SERIES_DESCRIPTIONS.get(modality, ["SERIES1"])
        series_desc = str(self.rng.choice(series_list)) if series_list else "SERIES1"

        # Use the referring physicians from hl7_events — just pick one based on patient_id hash
        referring_physician = "Smith^John^A"

        study = DICOMStudy(
            study_instance_uid=study_uid,
            series_instance_uid=self._generate_uid("1.2.840.10008.series"),
            sop_instance_uid=self._generate_uid("1.2.840.10008.sop"),
            patient_id=patient_id,
            accession_number=accession_number,
            modality=modality,
            body_part=body_part.upper(),
            study_description=study_desc,
            series_description=series_desc,
            manufacturer=mfr,
            manufacturer_model=model,
            station_name=station,
            study_date=study_datetime.strftime("%Y%m%d"),
            study_time=study_datetime.strftime("%H%M%S"),
            protocol_name=protocol,
            number_of_series=num_series,
            number_of_images=num_images,
            acquisition_params=params,
            referring_physician=referring_physician,
        )
        return study

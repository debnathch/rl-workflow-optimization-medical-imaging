"""Unit tests for DICOMMetadataGenerator."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from medimg_twin.simulation.entities import Modality
from medimg_twin.simulation.dicom_meta import DICOMMetadataGenerator

STUDY_DT = datetime.datetime(2024, 6, 15, 10, 30, 0)


@pytest.fixture
def dicom_gen() -> DICOMMetadataGenerator:
    return DICOMMetadataGenerator(rng=np.random.default_rng(42))


def _make_study(gen: DICOMMetadataGenerator, modality: Modality = Modality.CT, body_part: str = "CHEST"):
    return gen.generate_study(
        patient_id="00000001",
        modality=modality,
        body_part=body_part,
        study_datetime=STUDY_DT,
        accession_number="ACC20240615001",
    )


def test_generate_ct_study_modality(dicom_gen: DICOMMetadataGenerator) -> None:
    """CT study has correct modality."""
    study = _make_study(dicom_gen, Modality.CT, "CHEST")
    assert study.modality == Modality.CT


def test_generate_mri_study_series_count(dicom_gen: DICOMMetadataGenerator) -> None:
    """MRI study has 6-12 series."""
    study = _make_study(dicom_gen, Modality.MRI, "BRAIN")
    assert 6 <= study.number_of_series <= 12


def test_generate_xray_study_images(dicom_gen: DICOMMetadataGenerator) -> None:
    """XRAY study has 1-4 images."""
    study = _make_study(dicom_gen, Modality.XRAY, "CHEST")
    assert 1 <= study.number_of_images <= 4


def test_ct_params_has_required_keys(dicom_gen: DICOMMetadataGenerator) -> None:
    """CT acquisition params contain required keys."""
    study = _make_study(dicom_gen, Modality.CT, "CHEST")
    for key in ["kv", "mas", "slice_thickness_mm"]:
        assert key in study.acquisition_params, f"Missing CT param: {key}"


def test_mri_params_has_required_keys(dicom_gen: DICOMMetadataGenerator) -> None:
    """MRI acquisition params contain required keys."""
    study = _make_study(dicom_gen, Modality.MRI, "BRAIN")
    for key in ["tr_ms", "te_ms", "field_strength_T"]:
        assert key in study.acquisition_params, f"Missing MRI param: {key}"


def test_xray_params_has_required_keys(dicom_gen: DICOMMetadataGenerator) -> None:
    """XRAY acquisition params contain required keys."""
    study = _make_study(dicom_gen, Modality.XRAY, "CHEST")
    for key in ["kv", "mas", "sid_mm"]:
        assert key in study.acquisition_params, f"Missing XRAY param: {key}"


def test_uid_format(dicom_gen: DICOMMetadataGenerator) -> None:
    """study_instance_uid starts with '1.' and contains dots."""
    study = _make_study(dicom_gen)
    assert study.study_instance_uid.startswith("1.")
    assert study.study_instance_uid.count(".") >= 2


def test_accession_number_format(dicom_gen: DICOMMetadataGenerator) -> None:
    """Provided accession number is stored in the study."""
    study = _make_study(dicom_gen)
    assert study.accession_number == "ACC20240615001"


def test_to_dict_completeness(dicom_gen: DICOMMetadataGenerator) -> None:
    """to_dict() contains all required keys."""
    study = _make_study(dicom_gen)
    d = study.to_dict()
    for key in ["study_instance_uid", "accession_number", "modality", "manufacturer"]:
        assert key in d, f"Missing key: {key}"


def test_reproducibility() -> None:
    """Two generators with same seed produce identical study UIDs."""
    gen1 = DICOMMetadataGenerator(rng=np.random.default_rng(42))
    gen2 = DICOMMetadataGenerator(rng=np.random.default_rng(42))
    s1 = _make_study(gen1)
    s2 = _make_study(gen2)
    assert s1.study_instance_uid == s2.study_instance_uid


def test_manufacturer_is_known_vendor(dicom_gen: DICOMMetadataGenerator) -> None:
    """Generated manufacturer is one of the known vendors."""
    known = {"GE Healthcare", "Siemens Healthineers", "Philips Healthcare", "Canon Medical"}
    study = _make_study(dicom_gen)
    assert study.manufacturer in known

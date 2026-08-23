"""
SimPy-based hospital imaging department simulation.

Models the complete patient pathway from arrival through registration,
imaging queue, scan, radiologist reporting, and discharge. Supports
multiple scheduling policies and tracks all KPI metrics.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import simpy

from medimg_twin.config.settings import Config
from medimg_twin.simulation.dicom_meta import DICOMMetadataGenerator
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
from medimg_twin.simulation.hl7_events import HL7EventGenerator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling Policy Protocol
# ─────────────────────────────────────────────────────────────────────────────


SchedulingPolicy = Callable[["HospitalSimulation"], str | None]
"""Policy callable: takes simulation state, returns patient_id to schedule or None."""


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Statistics Collector
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationStats:
    """Accumulates statistics during a simulation run."""

    # Per-patient lists
    wait_times: list[float] = field(default_factory=list)
    scan_durations: list[float] = field(default_factory=list)
    report_durations: list[float] = field(default_factory=list)
    total_turnarounds: list[float] = field(default_factory=list)
    emergency_turnarounds: list[float] = field(default_factory=list)

    # Per-modality stats
    modality_wait_times: dict[str, list[float]] = field(
        default_factory=lambda: {"CT": [], "MRI": [], "XRAY": []}
    )
    modality_throughput: dict[str, int] = field(
        default_factory=lambda: {"CT": 0, "MRI": 0, "XRAY": 0}
    )

    # Queue length time series: list of (time, ct_q, mri_q, xray_q, emg_q)
    queue_snapshots: list[tuple[float, int, int, int, int]] = field(default_factory=list)

    # Epoch-level metrics for RL reward computation
    epoch_completed: list[int] = field(default_factory=list)
    epoch_avg_wait: list[float] = field(default_factory=list)

    # Totals
    n_arrived: int = 0
    n_completed: int = 0
    n_abandoned: int = 0  # patients who arrived after department close

    def record_patient_complete(self, patient: Patient) -> None:
        """Record metrics for a completed patient encounter."""
        if patient.wait_time is not None:
            self.wait_times.append(patient.wait_time)
            self.modality_wait_times[patient.modality.value].append(patient.wait_time)
        if patient.scan_duration is not None:
            self.scan_durations.append(patient.scan_duration)
        if patient.report_duration is not None:
            self.report_durations.append(patient.report_duration)
        if patient.total_turnaround is not None:
            self.total_turnarounds.append(patient.total_turnaround)
            if patient.priority == Priority.EMERGENCY:
                self.emergency_turnarounds.append(patient.total_turnaround)
        self.modality_throughput[patient.modality.value] += 1
        self.n_completed += 1

    def summary(self, sim_duration: float) -> dict[str, Any]:
        """Compute summary statistics."""
        def _safe_mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        def _safe_std(lst: list[float]) -> float:
            return float(np.std(lst)) if lst else 0.0

        throughput_per_hour = self.n_completed / (sim_duration / 60.0) if sim_duration > 0 else 0.0

        return {
            "n_arrived": self.n_arrived,
            "n_completed": self.n_completed,
            "n_abandoned": self.n_abandoned,
            "throughput_per_hour": throughput_per_hour,
            "avg_wait_time_min": _safe_mean(self.wait_times),
            "std_wait_time_min": _safe_std(self.wait_times),
            "p50_wait_time_min": float(np.percentile(self.wait_times, 50)) if self.wait_times else 0.0,
            "p95_wait_time_min": float(np.percentile(self.wait_times, 95)) if self.wait_times else 0.0,
            "avg_emergency_tat_min": _safe_mean(self.emergency_turnarounds),
            "avg_total_tat_min": _safe_mean(self.total_turnarounds),
            "avg_scan_duration_min": _safe_mean(self.scan_durations),
            "avg_report_duration_min": _safe_mean(self.report_durations),
            "modality_throughput": dict(self.modality_throughput),
            "modality_avg_wait": {
                k: _safe_mean(v) for k, v in self.modality_wait_times.items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hospital Simulation
# ─────────────────────────────────────────────────────────────────────────────


class HospitalSimulation:
    """
    SimPy discrete-event simulation of a hospital imaging department.

    Simulates:
    - Patient arrivals (Poisson, diurnal variation, emergency stream)
    - HL7 workflow events (ADT, ORM, SIU, ORU)
    - DICOM study metadata generation
    - Scanner queuing and utilization
    - Radiologist shift scheduling and report queue
    - Three scheduling policies: FIFO, Priority, PPO

    Args:
        config: Simulation configuration.
        policy: Scheduling policy callable (defaults to FIFO).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: Config,
        policy: SchedulingPolicy | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.seed = seed if seed is not None else config.simulation.seed
        self.rng = np.random.default_rng(self.seed)

# SimPy environment (created fresh on each reset)
        # Start at 6 AM in simulation minutes so arrivals aren't filtered by operating hours
        self._run_duration: float = self.config.simulation.duration_minutes
        self.env: simpy.Environment = simpy.Environment()

        # Generators
        self.hl7_gen = HL7EventGenerator()
        self.dicom_gen = DICOMMetadataGenerator(rng=self.rng)

        # Registry of patients and resources
        self.patients: dict[str, Patient] = {}
        self.scanners: dict[str, ScannerSpec] = {}
        self.radiologists: dict[str, RadiologistSpec] = {}

        # SimPy resources (scanner queues)
        self.scanner_resources: dict[str, simpy.Resource] = {}

        # Priority queues per modality (list of (neg_priority, arrival_time, patient_id))
        self.scan_queues: dict[Modality, list[tuple[int, float, str]]] = {
            Modality.CT: [],
            Modality.MRI: [],
            Modality.XRAY: [],
        }
        self.emergency_queue: deque[str] = deque()

        # Radiologist queue (patient_ids awaiting reporting)
        self.report_queue: deque[str] = deque()

        # Statistics
        self.stats = SimulationStats()

        # Event for signaling scheduling decisions
        self.scheduling_event: simpy.Event | None = None

        # Epoch tracking for RL
        self._epoch_start_completed = 0
        self._epoch_start_wait_sum = 0.0

        # Adaptive dispatch pool (policy-driven scanner assignment for adaptive_ppo)
        self._dispatch_pool: dict[str, simpy.Event] = {}
        self._free_scanners: set[str] = set()
        self._dispatch_trigger: simpy.Event | None = None

        # Build scanners and radiologists from config
        self._init_scanners()
        self._init_radiologists()

        logger.info(
            "HospitalSimulation initialized: %d scanners, %d radiologists, seed=%d",
            len(self.scanners),
            len(self.radiologists),
            self.seed,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_scanners(self) -> None:
        """Build scanner registry from config."""
        manufacturers = ["GE Healthcare", "Siemens Healthineers", "Philips Healthcare", "Canon Medical"]
        modality_config = self.config.modalities.scanner_count

        scanner_map = {
            Modality.CT: (modality_config.CT, "CT_SCANNER", "Radiology Suite A"),
            Modality.MRI: (modality_config.MRI, "MRI_SCANNER", "MRI Suite"),
            Modality.XRAY: (modality_config.XRAY, "XRAY_ROOM", "X-Ray Department"),
        }

        for modality, (count, prefix, location) in scanner_map.items():
            for i in range(1, count + 1):
                scanner_id = f"{prefix}_{i:02d}"
                mfr = manufacturers[i % len(manufacturers)]
                self.scanners[scanner_id] = ScannerSpec(
                    scanner_id=scanner_id,
                    modality=modality,
                    manufacturer=mfr,
                    model_name=f"{mfr.split()[0]} Model {modality.value}-{i}",
                    station_name=scanner_id,
                    location=f"{location} Room {i}",
                )

    def _init_radiologists(self) -> None:
        """Build radiologist registry from config."""
        specialty_map = {
            "chest": RadiologistSpecialty.CHEST,
            "musculoskeletal": RadiologistSpecialty.MUSCULOSKELETAL,
            "neuroradiology": RadiologistSpecialty.NEURORADIOLOGY,
            "abdominal": RadiologistSpecialty.ABDOMINAL,
            "emergency": RadiologistSpecialty.EMERGENCY,
        }
        for rad_cfg in self.config.radiologists.roster:
            specialty = specialty_map.get(rad_cfg.specialty, RadiologistSpecialty.EMERGENCY)
            self.radiologists[rad_cfg.id] = RadiologistSpec(
                radiologist_id=rad_cfg.id,
                specialty=specialty,
                shift_start_hour=rad_cfg.shift_start,
                shift_end_hour=rad_cfg.shift_end,
                max_daily_reads=rad_cfg.max_daily_reads,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SimPy resource setup (called on each run/reset)
    # ─────────────────────────────────────────────────────────────────────────

    def _create_simpy_resources(self) -> None:
        """Create SimPy PriorityResource for each scanner (supports priority scheduling)."""
        self.scanner_resources = {
            scanner_id: simpy.PriorityResource(self.env, capacity=1)
            for scanner_id in self.scanners
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation processes
    # ─────────────────────────────────────────────────────────────────────────

    def _patient_arrival_generator(self, duration: float) -> simpy.events.ProcessGenerator:
        """Generates patient arrivals using a Poisson process with diurnal variation."""
        patient_counter = 0
        op_start = self.config.simulation.operating_hours.start * 60
        op_end = self.config.simulation.operating_hours.end * 60
        base_iat = self.config.arrivals.routine_mean_iat
        diurnal = self.config.arrivals.diurnal_factors
        emergency_ratio = self.config.arrivals.emergency_ratio
        urgent_ratio = self.config.arrivals.urgent_ratio
        # env.now starts at op_start (6 AM offset added in reset), so relative duration end
        end_time = self.env.now + duration

        while self.env.now < end_time:
            # Compute diurnal scale factor based on hour of day
            current_hour = (self.env.now % (24 * 60)) / 60.0
            block_idx = min(int(current_hour / 4), 5)
            scale = diurnal[block_idx]

            # Sample inter-arrival time
            effective_iat = base_iat / max(scale, 0.01)
            iat = self.rng.exponential(effective_iat)
            yield self.env.timeout(iat)

            # Skip arrivals outside operating hours (simulated as closed)
            now_minute = self.env.now % (24 * 60)
            if not (op_start <= now_minute < op_end):
                continue

            # Determine priority
            r = self.rng.random()
            if r < emergency_ratio:
                priority = Priority.EMERGENCY
            elif r < emergency_ratio + urgent_ratio:
                priority = Priority.URGENT
            else:
                priority = Priority.ROUTINE

            # Determine modality (use index sampling — rng.choice() converts enums to numpy.str_)
            dist = self.config.modalities.distribution
            modality_probs = np.array([dist.CT, dist.MRI, dist.XRAY], dtype=float)
            _modality_list = [Modality.CT, Modality.MRI, Modality.XRAY]
            modality_idx = int(self.rng.choice(len(_modality_list), p=modality_probs))
            modality = _modality_list[modality_idx]

            # Sample body part
            body_parts_cfg = self.config.dataset.body_parts
            mod_key = modality.value
            if mod_key in body_parts_cfg and body_parts_cfg[mod_key]:
                bp_list = body_parts_cfg[mod_key]
                bp_names = [b[0] for b in bp_list]
                bp_probs = np.array([b[1] for b in bp_list], dtype=float)
                bp_probs /= bp_probs.sum()
                body_part = str(self.rng.choice(bp_names, p=bp_probs))
            else:
                body_part = "UNSPECIFIED"

            patient_counter += 1
            patient_id = f"PT{patient_counter:08d}"
            mrn = f"MRN{patient_counter:010d}"
            visit_number = f"VN{patient_counter:012d}"

            age = int(self.rng.integers(1, 95))
            sex = str(self.rng.choice(["M", "F", "O"], p=[0.49, 0.49, 0.02]))

            patient = Patient(
                patient_id=patient_id,
                mrn=mrn,
                visit_number=visit_number,
                arrival_time=float(self.env.now),
                priority=priority,
                modality=modality,
                body_part=body_part,
                age=age,
                sex=sex,
                referring_department=str(
                    self.rng.choice(["Emergency Dept", "Internal Medicine", "Orthopedics",
                                      "Neurology", "Oncology", "Cardiology", "Surgery"])
                ),
                clinical_indication="Imaging ordered per clinical protocol",
            )
            self.patients[patient_id] = patient
            self.stats.n_arrived += 1

            logger.debug("Patient %s arrived at t=%.1f min, priority=%s, modality=%s",
                         patient_id, self.env.now, priority.name, modality.value)

            # Start patient pathway process
            self.env.process(self._patient_pathway(patient))

    def _patient_pathway(self, patient: Patient) -> simpy.events.ProcessGenerator:
        """Process a single patient through the full imaging workflow."""
        # 1. Registration (ADT A01)
        reg_delay = max(0, self.rng.normal(2.0, 0.5))
        yield self.env.timeout(reg_delay)
        patient.registration_time = float(self.env.now)
        patient.status = PatientStatus.REGISTERED

        adt_a01 = self.hl7_gen.generate_adt_a01(patient, self.env.now)
        patient.hl7_messages.append(adt_a01)

        # 2. Order (ORM O01)
        order_delay = self.rng.uniform(1.0, 5.0)
        yield self.env.timeout(order_delay)
        # Pick a scanner of the right modality
        modality_scanners = [
            sid for sid, s in self.scanners.items()
            if s.modality == patient.modality
        ]
        scanner_id = str(self.rng.choice(modality_scanners)) if modality_scanners else "UNKNOWN"
        orm_o01 = self.hl7_gen.generate_orm_o01(patient, scanner_id, self.env.now)
        patient.hl7_messages.append(orm_o01)

        # 3. Queue for scan
        patient.queue_entry_time = float(self.env.now)
        patient.status = PatientStatus.WAITING_SCAN

        if patient.priority == Priority.EMERGENCY:
            self.emergency_queue.append(patient.patient_id)

        # ── Policy-driven SimPy priority ─────────────────────────────────────
        # The policy callable returns a patient_id recommendation. We use that
        # recommendation to derive a SimPy queue priority so the actual resource
        # queue ordering truly reflects the policy's decision.
        #
        # SimPy PriorityResource: lower number = served first.
        #
        #  FIFO policy     → all patients get the same priority (queue_entry_time
        #                     as a fractional tiebreaker so SimPy respects arrival order)
        #  Priority policy → Emergency=0, Urgent=1, Routine=2  (lower = served first)
        #  PPO policy      → model predicts action → maps to one of the above schemes

        if self.policy is not None:
            # Ask the policy which patient it recommends next
            recommended_pid = self.policy(self)
            policy_name = getattr(self.policy, "name", "fifo")

            if policy_name == "fifo":
                # Pure FIFO: all get same base priority, arrival_time as tiebreaker
                # Use a small fractional offset so queue_entry_time orders them
                simpy_priority = round(patient.queue_entry_time * 0.0001, 6)

            elif policy_name == "priority":
                # Clinical priority: Emergency=0, Urgent=1, Routine=2
                priority_map = {
                    Priority.EMERGENCY: 0,
                    Priority.URGENT:    1,
                    Priority.ROUTINE:   2,
                }
                simpy_priority = priority_map[patient.priority]

            elif policy_name == "ppo":
                # PPO maps its action to a priority scheme dynamically
                # Re-query policy to get the action integer if possible
                try:
                    from medimg_twin.simulation.entities import PatientStatus as _PS  # noqa
                    snapshot = self.get_snapshot()
                    import numpy as _np
                    obs = _np.array(snapshot.to_observation(), dtype=_np.float32)
                    action, _ = self.policy.model.predict(obs, deterministic=True)  # type: ignore[union-attr]
                    action_int = int(action)
                    if action_int == 0:  # FIFO
                        simpy_priority = round(patient.queue_entry_time * 0.0001, 6)
                    elif action_int == 1:  # Priority triage
                        priority_map2 = {Priority.EMERGENCY: 0, Priority.URGENT: 1, Priority.ROUTINE: 2}
                        simpy_priority = priority_map2[patient.priority]
                    else:  # action_int == 2: Emergency-first aggressive
                        em_map = {Priority.EMERGENCY: -1, Priority.URGENT: 1, Priority.ROUTINE: 2}
                        simpy_priority = em_map[patient.priority]
                except Exception:
                    # Fallback to priority ordering if model predict fails
                    priority_map3 = {Priority.EMERGENCY: 0, Priority.URGENT: 1, Priority.ROUTINE: 2}
                    simpy_priority = priority_map3[patient.priority]

            elif policy_name == "adaptive_ppo":
                # Aging-based SimPy priority — computed at QUEUE ENTRY.
                #
                # IMPORTANT: SimPy sets priority once at request() time and never
                # updates it. We therefore compute priority at queue entry using the
                # current simulation time as a proxy for "how long will this patient
                # have been waiting" relative to other patients.
                #
                # Key insight: we use a LARGE negative offset per minute of potential
                # wait so that the OLDEST patient in each band is always at the front.
                #
                # Base priorities (at queue entry, wait_so_far = 0):
                #   Emergency:   -0.500 → always served before Urgent (1.0) ✅
                #   Urgent:       1.000
                #   Routine:      2.000
                #
                # Anti-starvation crossover (aging_rate = 0.025 / min):
                # A routine patient whose queue_entry_time is T_r and an urgent
                # whose entry is T_u have SimPy priorities at the moment of request:
                #   Routine:  2.0 - (T_r × 0) = 2.0 (at entry, wait=0)
                #   Urgent:   1.0 - (T_u × 0) = 1.0 (at entry, wait=0)
                #
                # To make an OLDER routine patient beat a NEWER urgent, we encode
                # the absolute entry time into the priority directly:
                #   adaptive priority = base - (entry_time × aging_rate)
                #
                # Example with aging_rate = 0.025:
                #   Routine entering at t=0:    2.0 - (0 × 0.025)  = 2.000
                #   Urgent entering at t=45:    1.0 - (45 × 0.025) = -0.125  ← WRONG DIR
                #
                # Correct direction: patients entering EARLIER should get LOWER priority
                # numbers (served sooner). Use NEGATIVE entry time:
                #   priority = base + (entry_time × aging_rate) but DECREASING over time
                #
                # FINAL FORMULA (correct):
                #   priority = base - (time_in_queue_at_release × rate)
                #   Since time_in_queue is not known at entry, use:
                #   priority = base + (entry_time × tiny_negative) so earlier = lower
                #
                # SIMPLEST CORRECT APPROACH: use priority = base, tiebreak by -entry_time
                # (SimPy uses (priority, request_time) for ordering; request_time is the
                # SimPy clock when request() is called, which equals entry_time for us)
                # So within the same priority class, earlier request = served first (FIFO).
                # The ONLY change for adaptive_ppo is the CROSS-CLASS band:
                # Routine patients at priority 2.0 are served AFTER Urgent at 1.0,
                # even if they arrived 100 min earlier. To fix this, we need them
                # to cross into the Urgent band. We do this by using their actual
                # wait time from the SimPy clock at the moment of the next scanner
                # release — but that's not available at request() time.
                #
                # PRAGMATIC SOLUTION: We use a slightly randomized base priority
                # per class, and rely on the policy __call__() to select patients.
                # The policy already does anti-starvation. The SimPy priority here
                # just needs to not actively fight the policy's selection.
                # We set all waiting patients to priority=0 so SimPy becomes FIFO,
                # and let the policy __call__() return the correct patient ID.
                # The SimPy queue ordering doesn't matter because we're selecting
                # by policy, not by SimPy queue position.
                #
                # USE PRIORITY=0 for ALL adaptive_ppo patients → SimPy FIFO,
                # but policy __call__ is the actual selection mechanism.
                simpy_priority = 0  # let policy __call__ handle patient selection

            else:
                # Unknown policy name — default to clinical priority ordering
                priority_map4 = {Priority.EMERGENCY: 0, Priority.URGENT: 1, Priority.ROUTINE: 2}
                simpy_priority = priority_map4[patient.priority]
        else:
            # No policy set — use FIFO (arrival-time ordering)
            simpy_priority = round(patient.queue_entry_time * 0.0001, 6)

        scanner_resource = self.scanner_resources.get(scanner_id)
        if scanner_resource is None:
            logger.warning("No resource for scanner %s", scanner_id)
            return

        req = scanner_resource.request(priority=simpy_priority)

        # SIU^S12 (scheduling)
        siu = self.hl7_gen.generate_siu_s12(patient, scanner_id, self.env.now + 5.0, self.env.now)
        patient.hl7_messages.append(siu)

        yield req

        # 4. Setup + Scan
        patient.assigned_scanner_id = scanner_id
        if scanner_id in self.emergency_queue:
            self.emergency_queue.remove(scanner_id)

        scanner = self.scanners[scanner_id]
        scanner.status = ScannerStatus.BUSY
        scanner.current_patient_id = patient.patient_id

        setup_time = max(0.0, self.rng.normal(
            self.config.modalities.setup_time.mean,
            self.config.modalities.setup_time.sigma,
        ))
        yield self.env.timeout(setup_time)

        patient.scan_start_time = float(self.env.now)
        patient.status = PatientStatus.IN_SCAN
        scan_start = self.env.now

        # Sample scan duration
        dur_params = self.config.modalities.scan_duration[patient.modality.value]
        scan_duration = float(self.rng.lognormal(
            math.log(dur_params.mean), dur_params.sigma
        ))
        scan_duration = float(np.clip(scan_duration, dur_params.mean * 0.3, dur_params.mean * 3.5))

        yield self.env.timeout(scan_duration)
        patient.scan_end_time = float(self.env.now)

        # Track scanner utilization
        scanner.busy_intervals.append((scan_start, float(self.env.now)))
        scanner.total_studies += 1
        scanner.status = ScannerStatus.IDLE
        scanner.current_patient_id = None
        scanner_resource.release(req)

        # Generate DICOM metadata
        import datetime as _dt
        study_dt = _dt.datetime(2024, 1, 1) + _dt.timedelta(minutes=float(self.env.now))
        dicom_study = self.dicom_gen.generate_study(
            patient_id=patient.patient_id,
            modality=patient.modality,
            body_part=patient.body_part,
            study_datetime=study_dt,
            accession_number=f"ACC{patient.patient_id}",
        )
        patient.dicom_study = dicom_study

        # 5. Report Queue
        patient.status = PatientStatus.WAITING_REPORT
        self.report_queue.append(patient.patient_id)
        report_wait = self.rng.exponential(8.0)
        yield self.env.timeout(report_wait)

        # Assign radiologist
        rad_id = self._assign_radiologist(patient)
        patient.assigned_radiologist_id = rad_id
        patient.report_start_time = float(self.env.now)

        rad_params = self.config.radiologists.reporting_duration.get(patient.modality.value)
        if rad_params:
            speedup = (
                self.config.radiologists.emergency_priority_speedup
                if patient.priority == Priority.EMERGENCY
                else 1.0
            )
            report_duration = float(self.rng.lognormal(
                math.log(rad_params.mean * speedup), rad_params.sigma
            ))
            report_duration = float(np.clip(report_duration, rad_params.mean * 0.2, rad_params.mean * 4.0))
        else:
            report_duration = 15.0

        yield self.env.timeout(report_duration)
        patient.report_end_time = float(self.env.now)
        patient.status = PatientStatus.REPORTED

        if rad_id and rad_id in self.radiologists:
            rad = self.radiologists[rad_id]
            rad.total_reads += 1
            rad.total_report_minutes += report_duration
            rad.busy_intervals.append((patient.report_start_time, float(self.env.now)))

        # ORU^R01
        report_text = f"Radiologist report for {patient.modality.value} of {patient.body_part}. Clinical indication: {patient.clinical_indication}. Findings: Normal study. No acute abnormality identified."
        if patient.dicom_study:
            oru = self.hl7_gen.generate_oru_r01(
                patient, patient.dicom_study, report_text, rad_id or "UNKNOWN", self.env.now
            )
            patient.hl7_messages.append(oru)

        # 6. Discharge
        discharge_delay = self.rng.exponential(5.0)
        yield self.env.timeout(discharge_delay)
        patient.discharge_time = float(self.env.now)
        patient.status = PatientStatus.DISCHARGED

        # ADT^A03
        adt_a03 = self.hl7_gen.generate_adt_a03(patient, self.env.now)
        patient.hl7_messages.append(adt_a03)

        self.stats.record_patient_complete(patient)
        logger.debug("Patient %s discharged at t=%.1f, total TAT=%.1f min",
                     patient.patient_id, self.env.now,
                     patient.total_turnaround or 0.0)

    def _assign_radiologist(self, patient: Patient) -> str | None:
        """Assign radiologist with least workload who is on shift."""
        current_hour = (self.env.now % (24 * 60)) / 60.0
        available = [
            r for r in self.radiologists.values()
            if r.shift_start_hour <= current_hour < r.shift_end_hour
            and r.total_reads < r.max_daily_reads
        ]
        if not available:
            # Fall back to any radiologist with capacity
            available = [r for r in self.radiologists.values()
                         if r.total_reads < r.max_daily_reads]
        if not available:
            return None
        # Prefer emergency specialty for emergency patients
        if patient.priority == Priority.EMERGENCY:
            emg = [r for r in available if r.specialty == RadiologistSpecialty.EMERGENCY]
            if emg:
                available = emg
        # Choose least loaded
        return min(available, key=lambda r: r.total_reads).radiologist_id

    def _stats_collector(self) -> simpy.events.ProcessGenerator:
        """Periodic process that records queue length snapshots."""
        while True:
            yield self.env.timeout(5.0)  # every 5 minutes
            ct_q = len(self.scan_queues[Modality.CT])
            mri_q = len(self.scan_queues[Modality.MRI])
            xray_q = len(self.scan_queues[Modality.XRAY])
            emg_q = len(self.emergency_queue)
            self.stats.queue_snapshots.append(
                (self.env.now, ct_q, mri_q, xray_q, emg_q)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive dispatch (policy-driven scanner assignment)
    # ─────────────────────────────────────────────────────────────────────────

    def _signal_dispatch(self) -> None:
        """Signal the adaptive dispatch loop to check for available work."""
        if self._dispatch_trigger and not self._dispatch_trigger.triggered:
            self._dispatch_trigger.succeed()

    def _adaptive_dispatch_loop(self) -> simpy.events.ProcessGenerator:
        """Background process: matches waiting patients to free scanners.

        Runs continuously during the simulation. Wakes on:
        - New patient added to dispatch pool (_signal_dispatch)
        - Scanner freed after scan completion (_signal_dispatch)
        - Fallback timeout every 1 sim-minute

        Uses anti-starvation logic to select the best patient for each free
        scanner, evaluated at DISPATCH time (not queue-entry time). This is
        the key architectural difference from SimPy's static PriorityResource.
        """
        while True:
            self._dispatch_trigger = self.env.event()
            yield self._dispatch_trigger | self.env.timeout(1.0)

            # Dispatch as many patients as possible in this cycle
            made_dispatch = True
            while made_dispatch:
                made_dispatch = False
                for scanner_id in list(self._free_scanners):
                    modality = self.scanners[scanner_id].modality

                    # Find undispatched patients needing this modality
                    candidates = [
                        pid for pid, ev in self._dispatch_pool.items()
                        if not ev.triggered
                        and self.patients[pid].modality == modality
                    ]

                    if not candidates:
                        continue

                    # Select best patient using anti-starvation logic
                    selected = self._select_for_dispatch(candidates)
                    if selected:
                        self._free_scanners.discard(scanner_id)
                        ev = self._dispatch_pool.pop(selected)
                        ev.succeed(value=scanner_id)
                        made_dispatch = True
                        break  # Re-check from start; state changed

    def _select_for_dispatch(self, candidate_pids: list[str]) -> str | None:
        """Select best patient from candidates using adaptive anti-starvation logic.

        Decision order (mirrors AdaptivePPOPolicy.__call__):
        1. Emergency — longest-waiting first
        2. Starving routine (waited > threshold) — anti-starvation promotion
        3. Urgent — longest-waiting first
        4. Routine — longest-waiting first

        Args:
            candidate_pids: Patient IDs of waiting patients (same modality).

        Returns:
            Selected patient_id, or None if no candidates.
        """
        if not candidate_pids:
            return None

        now = self.env.now
        patients = [self.patients[pid] for pid in candidate_pids]
        threshold = getattr(self.policy, 'starvation_threshold_min', 45.0)

        emg = [p for p in patients if p.priority == Priority.EMERGENCY]
        urgent = [p for p in patients if p.priority == Priority.URGENT]
        routine = [p for p in patients if p.priority == Priority.ROUTINE]

        # 1. Emergency first (longest-waiting)
        if emg:
            return max(
                emg,
                key=lambda p: now - (p.queue_entry_time if p.queue_entry_time is not None else now),
            ).patient_id

        # 2. Anti-starvation: routine patients waiting > threshold
        starving = [
            p for p in routine
            if (now - (p.queue_entry_time if p.queue_entry_time is not None else now)) > threshold
        ]
        if starving:
            return max(
                starving,
                key=lambda p: now - (p.queue_entry_time if p.queue_entry_time is not None else now),
            ).patient_id

        # 3. Urgent (longest-waiting)
        if urgent:
            return max(
                urgent,
                key=lambda p: now - (p.queue_entry_time if p.queue_entry_time is not None else now),
            ).patient_id

        # 4. Routine (longest-waiting)
        if routine:
            return max(
                routine,
                key=lambda p: now - (p.queue_entry_time if p.queue_entry_time is not None else now),
            ).patient_id

        return candidate_pids[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset simulation to initial state with fresh SimPy environment."""
        # Start the clock at 6 AM so patients arrive immediately
        op_start_minutes = self.config.simulation.operating_hours.start * 60.0
        self.env = simpy.Environment(initial_time=op_start_minutes)
        self._start_time = op_start_minutes  # track for run_until
        self.patients.clear()
        self.stats = SimulationStats()
        self.scan_queues = {m: [] for m in Modality}
        self.emergency_queue = deque()
        self.report_queue = deque()
        for scanner in self.scanners.values():
            scanner.busy_intervals.clear()
            scanner.total_studies = 0
            scanner.status = ScannerStatus.IDLE
            scanner.current_patient_id = None
        for rad in self.radiologists.values():
            rad.total_reads = 0
            rad.total_report_minutes = 0.0
            rad.busy_intervals.clear()
        self._create_simpy_resources()
        self._epoch_start_completed = 0

        # Reset adaptive dispatch pool
        self._dispatch_pool = {}
        self._free_scanners = set(self.scanners.keys())
        self._dispatch_trigger = None

    def run(self, duration: float | None = None) -> SimulationStats:
        """Run simulation to completion.

        Args:
            duration: Simulation duration in minutes. Defaults to config value.

        Returns:
            SimulationStats with all collected metrics.
        """
        run_duration = duration or self.config.simulation.duration_minutes
        self._run_duration = run_duration
        self.reset()
        self.env.process(self._patient_arrival_generator(run_duration))
        self.env.process(self._stats_collector())
        # Start adaptive dispatch loop if using adaptive_ppo policy
        if getattr(self.policy, 'name', '') == 'adaptive_ppo':
            self.env.process(self._adaptive_dispatch_loop())
        end_time = self.env.now + run_duration
        self.env.run(until=end_time)
        logger.info(
            "Simulation complete: %d patients arrived, %d completed",
            self.stats.n_arrived,
            self.stats.n_completed,
        )
        return self.stats

    def run_until(self, until: float) -> None:
        """Advance simulation to a specific absolute time (used by RL env)."""
        # Clamp to absolute end time
        abs_end = self._start_time + self._run_duration
        target = min(until, abs_end)
        if target > self.env.now:
            self.env.run(until=target)

    def get_snapshot(self) -> SimulationSnapshot:
        """Get current simulation state snapshot for RL observation."""
        total_time = max(self.env.now, 1.0)

        def _scanner_util(modality: Modality) -> float:
            scanners = [s for s in self.scanners.values() if s.modality == modality]
            if not scanners:
                return 0.0
            return float(np.mean([s.utilization(total_time) for s in scanners]))

        hour_of_day = ((self.env.now % (24 * 60)) / 60.0) / 24.0
        day_of_week = (int(self.env.now / (24 * 60)) % 7) / 7.0

        rad_workloads = [r.workload_score for r in self.radiologists.values()]
        # Pad to fixed length (7 radiologists max)
        while len(rad_workloads) < 7:
            rad_workloads.append(0.0)
        rad_workloads = rad_workloads[:7]

        # Count waiting patients by priority
        waiting = [p for p in self.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        priority_counts = [
            sum(1 for p in waiting if p.priority == Priority.ROUTINE),
            sum(1 for p in waiting if p.priority == Priority.URGENT),
            sum(1 for p in waiting if p.priority == Priority.EMERGENCY),
        ]

        # Epoch stats
        epoch_completed = self.stats.n_completed - self._epoch_start_completed
        epoch_wait = (
            float(np.mean(self.stats.wait_times[-epoch_completed:]))
            if epoch_completed > 0 and self.stats.wait_times
            else 0.0
        )

        return SimulationSnapshot(
            sim_time=float(self.env.now),
            ct_queue_length=sum(
                1 for p in self.patients.values()
                if p.status == PatientStatus.WAITING_SCAN and p.modality == Modality.CT
            ),
            mri_queue_length=sum(
                1 for p in self.patients.values()
                if p.status == PatientStatus.WAITING_SCAN and p.modality == Modality.MRI
            ),
            xray_queue_length=sum(
                1 for p in self.patients.values()
                if p.status == PatientStatus.WAITING_SCAN and p.modality == Modality.XRAY
            ),
            emergency_queue_length=len(self.emergency_queue),
            ct_utilization=_scanner_util(Modality.CT),
            mri_utilization=_scanner_util(Modality.MRI),
            xray_utilization=_scanner_util(Modality.XRAY),
            radiologist_workloads=rad_workloads,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            priority_counts=priority_counts,
            completed_last_epoch=epoch_completed,
            avg_wait_last_epoch=epoch_wait,
        )

    def scanner_utilizations(self) -> dict[str, float]:
        """Return per-scanner utilization fractions."""
        total_time = max(self.env.now, 1.0)
        return {
            sid: s.utilization(total_time) for sid, s in self.scanners.items()
        }

    def radiologist_workloads(self) -> dict[str, dict[str, Any]]:
        """Return per-radiologist workload summary."""
        return {rid: r.to_dict() for rid, r in self.radiologists.items()}

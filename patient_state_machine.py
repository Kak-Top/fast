"""
Patient State Machine — ICU Digital Twin Simulator
Each patient has a trajectory state that drives physiologically-correlated vitals.
States: stable → deteriorating → critical → recovering → stable (cycle)
No random numbers without clinical meaning — every value is derived.
"""

import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PatientState(Enum):
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"
    RECOVERING = "recovering"


# Clinical base profiles per diagnosis
DIAGNOSIS_PROFILES = {
    "Respiratory Failure": {
        "base_hr": 105,
        "base_sys": 102,
        "base_dia": 65,
        "base_rr": 24,
        "base_spo2": 91,
        "base_temp": 38.4,
        "base_glucose": 148,
        "base_creatinine": 1.3,
        "base_wbc": 13.2,
        "base_lactate": 2.1,
    },
    "Acute MI": {
        "base_hr": 98,
        "base_sys": 108,
        "base_dia": 70,
        "base_rr": 20,
        "base_spo2": 93,
        "base_temp": 37.6,
        "base_glucose": 162,
        "base_creatinine": 1.1,
        "base_wbc": 11.8,
        "base_lactate": 1.8,
    },
    "Sepsis": {
        "base_hr": 118,
        "base_sys": 88,
        "base_dia": 54,
        "base_rr": 26,
        "base_spo2": 90,
        "base_temp": 39.1,
        "base_glucose": 140,
        "base_creatinine": 1.6,
        "base_wbc": 18.4,
        "base_lactate": 3.2,
    },
    "Pneumonia": {
        "base_hr": 96,
        "base_sys": 112,
        "base_dia": 72,
        "base_rr": 22,
        "base_spo2": 92,
        "base_temp": 38.8,
        "base_glucose": 135,
        "base_creatinine": 1.0,
        "base_wbc": 16.2,
        "base_lactate": 1.6,
    },
    "default": {
        "base_hr": 95,
        "base_sys": 110,
        "base_dia": 70,
        "base_rr": 20,
        "base_spo2": 94,
        "base_temp": 38.0,
        "base_glucose": 130,
        "base_creatinine": 1.1,
        "base_wbc": 11.0,
        "base_lactate": 1.8,
    },
}

# How much each vital drifts per state transition
STATE_MODIFIERS = {
    PatientState.STABLE: {
        "hr_delta": 0,
        "sys_delta": 0,
        "dia_delta": 0,
        "rr_delta": 0,
        "spo2_delta": 0,
        "temp_delta": 0.0,
        "glucose_delta": 0,
        "creatinine_delta": 0.0,
        "wbc_delta": 0.0,
        "lactate_delta": 0.0,
    },
    PatientState.DETERIORATING: {
        "hr_delta": +18,
        "sys_delta": -20,
        "dia_delta": -12,
        "rr_delta": +7,
        "spo2_delta": -5,
        "temp_delta": +0.6,
        "glucose_delta": +30,
        "creatinine_delta": +0.4,
        "wbc_delta": +4.0,
        "lactate_delta": +1.2,
    },
    PatientState.CRITICAL: {
        "hr_delta": +32,
        "sys_delta": -38,
        "dia_delta": -22,
        "rr_delta": +12,
        "spo2_delta": -10,
        "temp_delta": +1.1,
        "glucose_delta": +60,
        "creatinine_delta": +0.9,
        "wbc_delta": +7.5,
        "lactate_delta": +2.8,
    },
    PatientState.RECOVERING: {
        "hr_delta": -10,
        "sys_delta": +10,
        "dia_delta": +6,
        "rr_delta": -5,
        "spo2_delta": +4,
        "temp_delta": -0.4,
        "glucose_delta": -20,
        "creatinine_delta": -0.2,
        "wbc_delta": -2.5,
        "lactate_delta": -0.8,
    },
}

# State transition probabilities per tick (10s tick)
# From -> To: probability
STATE_TRANSITIONS = {
    PatientState.STABLE: {
        PatientState.STABLE: 0.97,
        PatientState.DETERIORATING: 0.03,
    },
    PatientState.DETERIORATING: {
        PatientState.DETERIORATING: 0.85,
        PatientState.CRITICAL: 0.08,
        PatientState.RECOVERING: 0.07,
    },
    PatientState.CRITICAL: {
        PatientState.CRITICAL: 0.88,
        PatientState.RECOVERING: 0.12,
    },
    PatientState.RECOVERING: {
        PatientState.RECOVERING: 0.80,
        PatientState.STABLE: 0.20,
    },
}


@dataclass
class PatientSimulator:
    patient_id: str
    name: str
    age: int
    diagnosis: str
    state: PatientState = PatientState.STABLE
    tick_count: int = 0
    lab_tick_count: int = 0  # labs update slower

    # Internal smoothed values (prevent jumps)
    _smooth_hr: float = field(default=0.0, init=False)
    _smooth_sys: float = field(default=0.0, init=False)
    _smooth_dia: float = field(default=0.0, init=False)
    _smooth_rr: float = field(default=0.0, init=False)
    _smooth_spo2: float = field(default=0.0, init=False)
    _smooth_temp: float = field(default=0.0, init=False)

    def __post_init__(self):
        profile = DIAGNOSIS_PROFILES.get(self.diagnosis, DIAGNOSIS_PROFILES["default"])
        # Age factor: older patients have higher baseline instability
        age_factor = 1.0 + max(0, (self.age - 50)) * 0.005

        self._smooth_hr = profile["base_hr"] * age_factor
        self._smooth_sys = profile["base_sys"]
        self._smooth_dia = profile["base_dia"]
        self._smooth_rr = profile["base_rr"]
        self._smooth_spo2 = min(100, profile["base_spo2"])
        self._smooth_temp = profile["base_temp"]

    def _get_profile(self) -> dict:
        return DIAGNOSIS_PROFILES.get(self.diagnosis, DIAGNOSIS_PROFILES["default"])

    def _transition_state(self):
        """Probabilistically advance to next state."""
        transitions = STATE_TRANSITIONS[self.state]
        states = list(transitions.keys())
        probs = list(transitions.values())
        self.state = random.choices(states, weights=probs, k=1)[0]

    def _noise(self, magnitude: float) -> float:
        """Small physiological noise — gaussian, bounded."""
        return random.gauss(0, magnitude)

    def _smooth(self, current: float, target: float, alpha: float = 0.15) -> float:
        """Exponential smoothing — no sudden jumps."""
        return current + alpha * (target - current)

    def tick_vitals(self) -> dict:
        """
        Advance one tick (10s). Returns vitals dict ready to POST to FastAPI.
        All values are physiologically correlated:
        - MAP derived from sys/dia
        - SpO2 inversely correlated with RR (as RR rises, SpO2 falls)
        - HR and sys move in opposite directions during deterioration
        - Temperature drives WBC in labs
        """
        self.tick_count += 1
        self._transition_state()

        profile = self._get_profile()
        mod = STATE_MODIFIERS[self.state]

        # Target values for this state
        target_hr = profile["base_hr"] + mod["hr_delta"]
        target_sys = profile["base_sys"] + mod["sys_delta"]
        target_dia = profile["base_dia"] + mod["dia_delta"]
        target_rr = profile["base_rr"] + mod["rr_delta"]
        target_spo2 = profile["base_spo2"] + mod["spo2_delta"]
        target_temp = profile["base_temp"] + mod["temp_delta"]

        # Smooth toward target (no instant jumps)
        alpha = 0.12  # slow drift — realistic for ICU vitals
        self._smooth_hr = self._smooth(self._smooth_hr, target_hr, alpha)
        self._smooth_sys = self._smooth(self._smooth_sys, target_sys, alpha)
        self._smooth_dia = self._smooth(self._smooth_dia, target_dia, alpha)
        self._smooth_rr = self._smooth(self._smooth_rr, target_rr, alpha)
        self._smooth_spo2 = self._smooth(self._smooth_spo2, target_spo2, alpha)
        self._smooth_temp = self._smooth(self._smooth_temp, target_temp, alpha)

        # Add physiological noise
        hr = self._smooth_hr + self._noise(1.5)
        sys_bp = self._smooth_sys + self._noise(2.0)
        dia_bp = self._smooth_dia + self._noise(1.5)
        rr = self._smooth_rr + self._noise(0.8)
        temp = self._smooth_temp + self._noise(0.05)

        # SpO2: correlated with RR (high RR = respiratory distress = lower SpO2)
        rr_penalty = max(0, (rr - 20) * 0.3)
        spo2 = self._smooth_spo2 - rr_penalty + self._noise(0.4)

        # Clamp to physiological limits
        hr = max(40, min(200, round(hr, 1)))
        sys_bp = max(60, min(220, round(sys_bp, 1)))
        dia_bp = max(30, min(130, round(dia_bp, 1)))
        rr = max(8, min(45, round(rr, 1)))
        spo2 = max(70, min(100, round(spo2, 1)))
        temp = max(35.0, min(42.0, round(temp, 1)))

        # MAP is always derived (not simulated separately)
        # MAP = diastolic + (pulse_pressure / 3)
        map_val = round(dia_bp + (sys_bp - dia_bp) / 3, 1)

        return {
            "heart_rate": hr,
            "blood_pressure_sys": sys_bp,
            "blood_pressure_dia": dia_bp,
            "respiratory_rate": rr,
            "spo2": spo2,
            "temperature": temp,
            # map is metadata, not sent to vitals endpoint (it derives it)
            "_meta": {
                "patient_id": self.patient_id,
                "state": self.state.value,
                "map_derived": map_val,
                "tick": self.tick_count,
            },
        }

    def tick_labs(self) -> Optional[dict]:
        """
        Labs update every 30 minutes (180 vitals ticks at 10s each).
        Returns None if it's not time for a lab update.
        Lab values are correlated with current patient state.
        """
        self.lab_tick_count += 1
        # Labs update every 180 vitals ticks (30 min), but output on tick 1 to populate the UI initially.
        if self.lab_tick_count != 1 and self.lab_tick_count % 180 != 0:
            return None

        profile = self._get_profile()
        mod = STATE_MODIFIERS[self.state]

        glucose = profile["base_glucose"] + mod["glucose_delta"] + self._noise(8)
        creatinine = profile["base_creatinine"] + mod["creatinine_delta"] + self._noise(0.08)
        wbc = profile["base_wbc"] + mod["wbc_delta"] + self._noise(0.5)
        lactate = profile["base_lactate"] + mod["lactate_delta"] + self._noise(0.15)

        # Temperature drives WBC (fever response)
        temp_wbc_boost = max(0, (self._smooth_temp - 38.0) * 1.5)
        wbc += temp_wbc_boost

        # MAP drives lactate (low MAP = poor perfusion = lactate rises)
        map_val = self._smooth_dia + (self._smooth_sys - self._smooth_dia) / 3
        if map_val < 65:
            lactate += (65 - map_val) * 0.05

        glucose = max(50, min(600, round(glucose, 1)))
        creatinine = max(0.4, min(15.0, round(creatinine, 2)))
        wbc = max(1.0, min(50.0, round(wbc, 1)))
        lactate = max(0.5, min(20.0, round(lactate, 1)))

        return {
            "glucose": glucose,
            "creatinine": creatinine,
            "wbc": wbc,
            "lactate": lactate,
            "_meta": {
                "patient_id": self.patient_id,
                "state": self.state.value,
                "tick": self.tick_count,
            },
        }

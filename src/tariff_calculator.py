"""Tariff calculations. Estimates only - not an official electricity bill.

Cost model:
  energy_charge = flat (kwh * price) OR tiered slabs
  subtotal      = energy_charge + fixed_monthly_charge
  total         = subtotal * (1 + additional_percent / 100)
Tier 'upto' values are cumulative kWh limits; the last tier uses upto=None.
"""
from dataclasses import dataclass, field
from typing import List

DISCLAIMER = ("Estimate only. Actual bills may include fixed charges, taxes, "
              "fuel adjustments and other fees not modelled here.")


@dataclass
class Tariff:
    currency: str = "\u20b9"
    price_per_kwh: float = 6.0
    tiers: List[dict] = field(default_factory=list)  # [{"upto": 100, "rate": 3.0}, ..., {"upto": None, "rate": 6}]
    fixed_monthly_charge: float = 0.0
    additional_percent: float = 0.0

    def validate(self):
        if self.price_per_kwh < 0 or self.fixed_monthly_charge < 0 or self.additional_percent < 0:
            raise ValueError("Tariff values must be non-negative.")
        prev = 0.0
        for i, t in enumerate(self.tiers):
            if t["rate"] < 0:
                raise ValueError("Tier rates must be non-negative.")
            if t["upto"] is None:
                if i != len(self.tiers) - 1:
                    raise ValueError("Only the last tier may be unlimited (upto=None).")
            else:
                if t["upto"] <= prev:
                    raise ValueError("Tier limits must be strictly increasing.")
                prev = t["upto"]
        if self.tiers and self.tiers[-1]["upto"] is not None:
            raise ValueError("The last tier must be unlimited (upto=None).")

    @property
    def model_name(self) -> str:
        return "Tiered" if self.tiers else "Flat rate"


def energy_charge(kwh: float, tariff: Tariff) -> float:
    if kwh < 0:
        raise ValueError("kWh cannot be negative.")
    if not tariff.tiers:
        return kwh * tariff.price_per_kwh
    total, lower = 0.0, 0.0
    for t in tariff.tiers:
        upper = float("inf") if t["upto"] is None else t["upto"]
        if kwh > lower:
            total += (min(kwh, upper) - lower) * t["rate"]
        lower = upper
    return total


def estimate_bill(kwh: float, tariff: Tariff, include_fixed: bool = True) -> dict:
    tariff.validate()
    energy = energy_charge(kwh, tariff)
    fixed = tariff.fixed_monthly_charge if include_fixed else 0.0
    subtotal = energy + fixed
    extra = subtotal * tariff.additional_percent / 100.0
    return {"kwh": kwh, "energy_charge": energy, "fixed_charge": fixed,
            "additional_charge": extra, "total": subtotal + extra,
            "tariff_model": tariff.model_name, "disclaimer": DISCLAIMER}
